#!/usr/bin/env python3
"""Stream a pi05_droid replay demo from a remote Jetson host.

The server reads a raw DROID episode, queries an already-running OpenPI policy
server, and streams model input images plus timing metrics to browser clients.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import dataclasses
import datetime as dt
import hashlib
import http
import io
import json
import logging
import pathlib
import statistics
import time
import urllib.parse
from typing import Any

import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
from PIL import Image
import websockets
import websockets.asyncio.server as ws_server

LOGGER = logging.getLogger("pi05_droid_demo")

DROID_CONTROL_FREQUENCY = 15


class FreshUnavailableError(RuntimeError):
    code = "fresh_unavailable"


@dataclasses.dataclass(frozen=True)
class EpisodeCandidate:
    episode_dir: pathlib.Path
    prompt: str
    dataset_success: bool | None

    @property
    def task_id(self) -> str:
        return _task_id(self.episode_dir, self.prompt)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pathlib.Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _decode(value.item())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _first_present(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    raise KeyError(f"Missing all expected keys: {keys}")


def _as_float32_array(value: Any, *, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(_decode(value), dtype=np.float32)
    if shape is not None:
        array = array.reshape(shape)
    return array


def _jpeg_b64(image: np.ndarray, *, quality: int = 82) -> str:
    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    values_sorted = sorted(values)
    return {
        "count": float(len(values)),
        "mean": float(statistics.fmean(values)),
        "min": float(values_sorted[0]),
        "p50": float(np.quantile(values_sorted, 0.50)),
        "p90": float(np.quantile(values_sorted, 0.90)),
        "p95": float(np.quantile(values_sorted, 0.95)),
        "max": float(values_sorted[-1]),
    }


def _load_json(path: pathlib.Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _find_annotations_path(data_root: pathlib.Path, explicit_path: pathlib.Path | None) -> pathlib.Path | None:
    if explicit_path is not None:
        return explicit_path
    candidate = data_root / "aggregated-annotations-030724.json"
    if candidate.exists():
        return candidate
    matches = list(data_root.glob("**/aggregated-annotations-030724.json"))
    return matches[0] if matches else None


def _episode_id_from_metadata(episode_dir: pathlib.Path) -> str | None:
    metadata_files = sorted(episode_dir.glob("metadata_*.json"))
    if not metadata_files:
        return None
    return metadata_files[0].stem.split("_")[-1]


def _prompt_for_episode(
    episode_dir: pathlib.Path,
    data_root: pathlib.Path,
    annotations_path: pathlib.Path | None,
    fallback_prompt: str,
) -> str:
    episode_id = _episode_id_from_metadata(episode_dir)
    annotations = _load_json(_find_annotations_path(data_root, annotations_path))
    if episode_id and episode_id in annotations:
        annotation = annotations[episode_id]
        for key in ("language_instruction1", "language_instruction", "language_instruction_1"):
            if key in annotation:
                return str(annotation[key])
    return fallback_prompt


def _dataset_success_from_path(episode_dir: pathlib.Path) -> bool | None:
    parts = {part.lower() for part in episode_dir.parts}
    if "success" in parts:
        return True
    if "failure" in parts or "fail" in parts:
        return False
    return None


def _task_id(episode_dir: pathlib.Path, prompt: str) -> str:
    text = f"{episode_dir.resolve()}::{prompt}"
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _episode_dirs(data_root: pathlib.Path) -> list[pathlib.Path]:
    matches = sorted([*data_root.glob("**/trajectory.h5"), *data_root.glob("**/trajectory.hdf5")])
    episode_dirs = []
    seen: set[pathlib.Path] = set()
    for match in matches:
        episode_dir = match.parent.resolve()
        if episode_dir in seen:
            continue
        if (episode_dir / "recordings" / "MP4").exists():
            episode_dirs.append(episode_dir)
            seen.add(episode_dir)
    return episode_dirs


def _episode_catalog(
    data_root: pathlib.Path,
    annotations_path: pathlib.Path | None,
    fallback_prompt: str,
) -> list[EpisodeCandidate]:
    return [
        EpisodeCandidate(
            episode_dir=episode_dir,
            prompt=_prompt_for_episode(episode_dir, data_root, annotations_path, fallback_prompt),
            dataset_success=_dataset_success_from_path(episode_dir),
        )
        for episode_dir in _episode_dirs(data_root)
    ]


def _parse_websocket_query(websocket: ws_server.ServerConnection) -> dict[str, str]:
    request = getattr(websocket, "request", None)
    path = getattr(request, "path", None) or getattr(websocket, "path", "") or ""
    parsed = urllib.parse.urlparse(path)
    values = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
    return {key: value[-1] for key, value in values.items()}


def _query_bool(query: dict[str, str], key: str) -> bool:
    return query.get(key, "").lower() in {"1", "true", "yes", "on"}


def _resolved_query_path(value: str | None) -> pathlib.Path | None:
    if not value:
        return None
    return pathlib.Path(value).expanduser().resolve()


def _choose_next_candidate(
    catalog: list[EpisodeCandidate],
    pool: list[EpisodeCandidate],
    excluded_episode: pathlib.Path | None,
) -> EpisodeCandidate:
    if excluded_episode is None:
        return pool[0]

    pool_dirs = {candidate.episode_dir.resolve() for candidate in pool}
    catalog_dirs = [candidate.episode_dir.resolve() for candidate in catalog]
    try:
        start_index = catalog_dirs.index(excluded_episode.resolve()) + 1
    except ValueError:
        start_index = 0

    for offset in range(len(catalog)):
        candidate = catalog[(start_index + offset) % len(catalog)]
        if candidate.episode_dir.resolve() in pool_dirs:
            return candidate
    return pool[0]


def _select_episode_for_request(args: argparse.Namespace, query: dict[str, str]) -> tuple[pathlib.Path, dict[str, Any]]:
    data_root = pathlib.Path(args.data_root).resolve()
    annotations_path = pathlib.Path(args.annotations_path) if args.annotations_path else None
    fresh_requested = _query_bool(query, "fresh")
    excluded_episode = _resolved_query_path(query.get("exclude_episode"))
    excluded_prompt = query.get("exclude_prompt") or None

    if args.episode_dir:
        episode_dir = pathlib.Path(args.episode_dir).expanduser().resolve()
        prompt = _prompt_for_episode(episode_dir, data_root, annotations_path, args.prompt)
        if fresh_requested:
            raise FreshUnavailableError(
                "Fresh is unavailable because remote_demo_server.py was started with --episode-dir. "
                "Start it with --data-root containing at least two raw DROID episodes."
            )
        return episode_dir, {
            "task_id": _task_id(episode_dir, prompt),
            "episode_dir": str(episode_dir),
            "prompt": prompt,
            "episode_count": 1,
            "fresh_requested": False,
            "excluded_episode_dir": str(excluded_episode) if excluded_episode else None,
            "excluded_prompt": excluded_prompt,
        }

    catalog = _episode_catalog(data_root, annotations_path, args.prompt)
    if not catalog:
        raise FileNotFoundError(
            f"No usable DROID episode found under {data_root}. Expected trajectory.h5 or trajectory.hdf5 "
            "plus a recordings/MP4 directory."
        )

    if not fresh_requested:
        chosen = catalog[0]
    else:
        alternatives = [
            candidate
            for candidate in catalog
            if excluded_episode is None or candidate.episode_dir.resolve() != excluded_episode.resolve()
        ]
        if not alternatives:
            raise FreshUnavailableError(
                "Fresh requires at least two distinct raw DROID episodes under --data-root; "
                "the server found no replacement for the current episode."
            )
        prompt_alternatives = [
            candidate for candidate in alternatives if excluded_prompt is None or candidate.prompt != excluded_prompt
        ]
        chosen = _choose_next_candidate(catalog, prompt_alternatives or alternatives, excluded_episode)

    return chosen.episode_dir, {
        "task_id": chosen.task_id,
        "episode_dir": str(chosen.episode_dir),
        "prompt": chosen.prompt,
        "episode_count": len(catalog),
        "fresh_requested": fresh_requested,
        "excluded_episode_dir": str(excluded_episode) if excluded_episode else None,
        "excluded_prompt": excluded_prompt,
    }


def _find_episode_dir(data_root: pathlib.Path, episode_dir: pathlib.Path | None) -> pathlib.Path:
    if episode_dir is not None:
        return episode_dir.resolve()
    matches = _episode_dirs(data_root)
    if not matches:
        raise FileNotFoundError(
            f"No usable DROID episode found under {data_root}. Expected trajectory.h5 or trajectory.hdf5 "
            "plus a recordings/MP4 directory."
        )
    return matches[0].resolve()


def _trajectory_path(episode_dir: pathlib.Path) -> pathlib.Path:
    for name in ("trajectory.h5", "trajectory.hdf5"):
        candidate = episode_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No trajectory.h5 or trajectory.hdf5 found in {episode_dir}")


def _recording_dir(episode_dir: pathlib.Path) -> pathlib.Path:
    candidate = episode_dir / "recordings" / "MP4"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"No recordings/MP4 directory found in {episode_dir}")


def _hdf5_length(group: Any, *, keys_to_ignore: tuple[str, ...] = ("videos",)) -> int:
    for key in group:
        if key in keys_to_ignore:
            continue
        item = group[key]
        if hasattr(item, "keys"):
            with contextlib.suppress(ValueError, TypeError):
                return _hdf5_length(item, keys_to_ignore=keys_to_ignore)
        elif getattr(item, "shape", None):
            return int(item.shape[0])
    raise ValueError("Could not determine trajectory length from HDF5 file")


def _load_hdf5_index(group: Any, index: int, *, keys_to_ignore: tuple[str, ...] = ("videos",)) -> Any:
    if hasattr(group, "keys"):
        return {
            key: _load_hdf5_index(group[key], index, keys_to_ignore=keys_to_ignore)
            for key in group
            if key not in keys_to_ignore
        }
    try:
        return group[index]
    except (ValueError, TypeError):
        return group[()]


class Mp4Reader:
    def __init__(self, path: pathlib.Path) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("remote_demo_server.py requires OpenCV on the Jetson: uv pip install opencv-python") from exc

        self._cv2 = cv2
        self.path = path
        self.stem = path.stem
        self._capture = cv2.VideoCapture(str(path))
        if not self._capture.isOpened():
            raise RuntimeError(f"Could not open MP4 file: {path}")

    def read(self, index: int) -> np.ndarray:
        self._capture.set(self._cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self._capture.read()
        if not ok:
            raise EOFError(f"Could not read frame {index} from {self.path}")
        if "stereo" in self.stem.lower():
            frame = frame[:, : frame.shape[1] // 2]
        return frame[..., ::-1].copy()

    def close(self) -> None:
        self._capture.release()


class DroidRawEpisode:
    def __init__(
        self,
        data_root: pathlib.Path,
        episode_dir: pathlib.Path | None = None,
        *,
        annotations_path: pathlib.Path | None = None,
        fallback_prompt: str = "do something",
        exterior_camera_stem: str | None = None,
        wrist_camera_stem: str | None = None,
    ) -> None:
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError("remote_demo_server.py requires h5py on the Jetson: uv pip install h5py") from exc

        self.data_root = data_root.resolve()
        self.episode_dir = _find_episode_dir(self.data_root, episode_dir)
        self.trajectory_path = _trajectory_path(self.episode_dir)
        self.recording_dir = _recording_dir(self.episode_dir)
        self.prompt = _prompt_for_episode(self.episode_dir, self.data_root, annotations_path, fallback_prompt)
        self.task_id = _task_id(self.episode_dir, self.prompt)
        self.dataset_success = _dataset_success_from_path(self.episode_dir)

        self._h5 = h5py.File(self.trajectory_path, "r")
        self.length = _hdf5_length(self._h5)
        first_step = _load_hdf5_index(self._h5, 0)
        camera_type_map = self._camera_type_map(first_step)
        self._readers = [Mp4Reader(path) for path in sorted(self.recording_dir.glob("*.mp4"))]
        if not self._readers:
            raise FileNotFoundError(f"No MP4 files found in {self.recording_dir}")

        self.exterior_reader = self._select_reader(camera_type_map, exterior_camera_stem, preferred_role="exterior")
        self.wrist_reader = self._select_reader(camera_type_map, wrist_camera_stem, preferred_role="wrist")

    def _camera_type_map(self, step: dict[str, Any]) -> dict[str, int]:
        camera_type = step.get("observation", {}).get("camera_type", {})
        result = {}
        if isinstance(camera_type, dict):
            for key, value in camera_type.items():
                decoded = _decode(value)
                with contextlib.suppress(TypeError, ValueError):
                    result[str(key)] = int(decoded)
        return result

    def _role_for_reader(self, reader: Mp4Reader, camera_type_map: dict[str, int]) -> str | None:
        stem = reader.stem
        lower = stem.lower()
        if stem in camera_type_map:
            return "wrist" if camera_type_map[stem] == 0 else "exterior"
        if "hand" in lower or "wrist" in lower:
            return "wrist"
        if "varied" in lower or "exterior" in lower or "camera_1" in lower or "camera_2" in lower:
            return "exterior"
        return None

    def _select_reader(
        self, camera_type_map: dict[str, int], explicit_stem: str | None, *, preferred_role: str
    ) -> Mp4Reader:
        if explicit_stem:
            for reader in self._readers:
                if reader.stem == explicit_stem:
                    return reader
            raise ValueError(f"Could not find requested camera stem {explicit_stem!r}")

        for reader in self._readers:
            if self._role_for_reader(reader, camera_type_map) == preferred_role:
                return reader
        available = ", ".join(reader.stem for reader in self._readers)
        raise ValueError(f"Could not infer {preferred_role} camera from MP4 files: {available}")

    def read_step(self, index: int) -> dict[str, Any]:
        step = _load_hdf5_index(self._h5, index)
        observation = step.get("observation", {})
        robot_state = observation.get("robot_state", {})
        joint_position = _as_float32_array(
            _first_present(robot_state | observation, ("joint_positions", "joint_position")), shape=(7,)
        )
        gripper_position = _as_float32_array(
            _first_present(robot_state | observation, ("gripper_position",)), shape=(1,)
        )

        exterior_image = self.exterior_reader.read(index)
        wrist_image = self.wrist_reader.read(index)
        model_exterior_image = image_tools.resize_with_pad(exterior_image, 224, 224)
        model_wrist_image = image_tools.resize_with_pad(wrist_image, 224, 224)

        dataset_action = None
        action = step.get("action", step.get("action_dict", {}))
        with contextlib.suppress(KeyError, ValueError, TypeError):
            joint_velocity = _as_float32_array(_first_present(action, ("joint_velocity",)), shape=(7,))
            gripper_action = _as_float32_array(_first_present(action, ("gripper_position",)), shape=(1,))
            dataset_action = np.concatenate([joint_velocity, gripper_action], dtype=np.float32)

        return {
            "frame_index": index,
            "exterior_image": model_exterior_image,
            "wrist_image": model_wrist_image,
            "joint_position": joint_position,
            "gripper_position": gripper_position,
            "dataset_action": dataset_action,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "episode_dir": str(self.episode_dir),
            "trajectory_path": str(self.trajectory_path),
            "recording_dir": str(self.recording_dir),
            "length": self.length,
            "prompt": self.prompt,
            "dataset_demo_success": self.dataset_success,
            "exterior_camera": self.exterior_reader.stem,
            "wrist_camera": self.wrist_reader.stem,
        }

    def close(self) -> None:
        self._h5.close()
        for reader in self._readers:
            reader.close()


class DemoRun:
    def __init__(self, args: argparse.Namespace, query: dict[str, str] | None = None) -> None:
        self.args = args
        self.query = query or {}
        self.run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.output_dir = pathlib.Path(args.output_dir) / self.run_id
        self.selection: dict[str, Any] = {}
        self.client_roundtrip_ms: list[float] = []
        self.server_infer_ms: list[float] = []
        self.server_prev_total_ms: list[float] = []
        self.policy_infer_ms: list[float] = []
        self.video_frames: list[np.ndarray] = []
        self.query_count = 0
        self.frame_count = 0
        self.last_action_shape: tuple[int, ...] | None = None
        self.run_start_time: float | None = None

    async def send(self, websocket: ws_server.ServerConnection, event: dict[str, Any]) -> None:
        await websocket.send(json.dumps(event, default=_json_default))

    def metrics(self, *, completed: bool, episode: DroidRawEpisode, video_path: str | None = None) -> dict[str, Any]:
        elapsed_s = max((time.perf_counter() - self.run_start_time) if self.run_start_time is not None else 0.0, 1e-9)
        return {
            "run_id": self.run_id,
            "task_id": episode.task_id,
            "episode_dir": str(episode.episode_dir),
            "prompt": episode.prompt,
            "episode_count": self.selection.get("episode_count"),
            "fresh_requested": self.selection.get("fresh_requested", False),
            "excluded_episode_dir": self.selection.get("excluded_episode_dir"),
            "run_completed": completed,
            "dataset_demo_success": episode.dataset_success,
            "policy_task_success": None,
            "elapsed_s": elapsed_s,
            "frame_count": self.frame_count,
            "query_count": self.query_count,
            "action_shape": list(self.last_action_shape) if self.last_action_shape else None,
            "client_roundtrip_ms": _stats(self.client_roundtrip_ms),
            "server_infer_ms": _stats(self.server_infer_ms),
            "server_prev_total_ms": _stats(self.server_prev_total_ms),
            "policy_infer_ms": _stats(self.policy_infer_ms),
            "effective_fps": self.frame_count / elapsed_s,
            "metrics_path": str(self.output_dir / "metrics.json"),
            "video_path": video_path,
        }

    def write_video(self) -> tuple[str | None, str | None]:
        if not self.args.write_video or not self.video_frames:
            return None, None
        video_path = self.output_dir / "trajectory_preview.mp4"
        try:
            import imageio.v3 as iio

            iio.imwrite(video_path, np.asarray(self.video_frames), fps=self.args.fps)
            return str(video_path), None
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Could not write MP4 preview: %s", exc)
            return None, str(exc)

    async def run(self, websocket: ws_server.ServerConnection) -> None:
        selected_episode_dir, self.selection = _select_episode_for_request(self.args, self.query)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        episode = DroidRawEpisode(
            pathlib.Path(self.args.data_root),
            selected_episode_dir,
            annotations_path=pathlib.Path(self.args.annotations_path) if self.args.annotations_path else None,
            fallback_prompt=self.args.prompt,
            exterior_camera_stem=self.args.exterior_camera_stem,
            wrist_camera_stem=self.args.wrist_camera_stem,
        )
        try:
            policy = websocket_client_policy.WebsocketClientPolicy(self.args.policy_host, self.args.policy_port)
            await self.send(
                websocket,
                {
                    "type": "run_started",
                    "run_id": self.run_id,
                    "task_id": episode.task_id,
                    "episode_dir": str(episode.episode_dir),
                    "prompt": episode.prompt,
                    "episode_count": self.selection.get("episode_count"),
                    "fresh_requested": self.selection.get("fresh_requested", False),
                    "excluded_episode_dir": self.selection.get("excluded_episode_dir"),
                    "episode": episode.metadata(),
                    "selection": self.selection,
                    "policy_server_metadata": policy.get_server_metadata(),
                    "stream_mode": self.args.stream_mode,
                    "open_loop_horizon": self.args.open_loop_horizon,
                },
            )

            stop_frame = min(episode.length, self.args.start_frame + self.args.max_steps)
            action_chunk = None
            actions_from_chunk_completed = self.args.open_loop_horizon
            run_start = time.perf_counter()
            self.run_start_time = run_start

            for frame_index in range(self.args.start_frame, stop_frame):
                loop_start = time.perf_counter()
                step = episode.read_step(frame_index)
                should_query = action_chunk is None or actions_from_chunk_completed >= self.args.open_loop_horizon

                if should_query:
                    request_data = {
                        "observation/exterior_image_1_left": step["exterior_image"],
                        "observation/wrist_image_left": step["wrist_image"],
                        "observation/joint_position": step["joint_position"],
                        "observation/gripper_position": step["gripper_position"],
                        "prompt": episode.prompt,
                    }
                    infer_start = time.perf_counter()
                    result = policy.infer(request_data)
                    roundtrip_ms = (time.perf_counter() - infer_start) * 1000
                    action_chunk = np.asarray(result["actions"])
                    self.query_count += 1
                    self.last_action_shape = tuple(action_chunk.shape)
                    actions_from_chunk_completed = 0
                    self.client_roundtrip_ms.append(roundtrip_ms)
                    server_timing = result.get("server_timing", {})
                    policy_timing = result.get("policy_timing", {})
                    if "infer_ms" in server_timing:
                        self.server_infer_ms.append(float(server_timing["infer_ms"]))
                    if "prev_total_ms" in server_timing:
                        self.server_prev_total_ms.append(float(server_timing["prev_total_ms"]))
                    if "infer_ms" in policy_timing:
                        self.policy_infer_ms.append(float(policy_timing["infer_ms"]))
                    await self.send(
                        websocket,
                        {
                            "type": "action_chunk",
                            "run_id": self.run_id,
                            "frame_index": frame_index,
                            "query_index": self.query_count,
                            "action_shape": list(action_chunk.shape),
                            "first_action": action_chunk[0].tolist(),
                            "client_roundtrip_ms": roundtrip_ms,
                            "server_timing": server_timing,
                            "policy_timing": policy_timing,
                        },
                    )

                action_index = min(actions_from_chunk_completed, len(action_chunk) - 1)
                current_action = action_chunk[action_index]
                actions_from_chunk_completed += 1
                should_stream_frame = self.args.stream_mode == "every-frame" or should_query
                if should_stream_frame:
                    self.frame_count += 1
                    self.video_frames.append(step["exterior_image"])
                    await self.send(
                        websocket,
                        {
                            "type": "frame",
                            "run_id": self.run_id,
                            "frame_index": frame_index,
                            "frame_count": self.frame_count,
                            "elapsed_s": time.perf_counter() - run_start,
                            "prompt": episode.prompt,
                            "exterior_jpeg": _jpeg_b64(step["exterior_image"], quality=self.args.jpeg_quality),
                            "wrist_jpeg": _jpeg_b64(step["wrist_image"], quality=self.args.jpeg_quality),
                            "action": current_action.tolist(),
                            "dataset_action": (
                                step["dataset_action"].tolist() if step["dataset_action"] is not None else None
                            ),
                            "metrics": self.metrics(completed=False, episode=episode),
                        },
                    )

                if self.args.realtime:
                    elapsed = time.perf_counter() - loop_start
                    await asyncio.sleep(max(0.0, (1.0 / self.args.fps) - elapsed))

            video_path, video_error = self.write_video()
            metrics = self.metrics(completed=True, episode=episode, video_path=video_path)
            if video_error is not None:
                metrics["video_error"] = video_error
            with (self.output_dir / "metrics.json").open("w", encoding="utf-8") as file:
                json.dump(metrics, file, indent=2, default=_json_default)
            await self.send(websocket, {"type": "run_finished", "run_id": self.run_id, "metrics": metrics})
        finally:
            episode.close()


def _health_check(connection: ws_server.ServerConnection, request: ws_server.Request) -> ws_server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


async def _handler(websocket: ws_server.ServerConnection, args: argparse.Namespace) -> None:
    LOGGER.info("Viewer connected: %s", websocket.remote_address)
    try:
        query = _parse_websocket_query(websocket)
        await DemoRun(args, query).run(websocket)
    except websockets.ConnectionClosed:
        LOGGER.info("Viewer disconnected: %s", websocket.remote_address)
    except FreshUnavailableError as exc:
        LOGGER.info("Fresh request unavailable: %s", exc)
        with contextlib.suppress(Exception):
            await websocket.send(json.dumps({"type": "error", "code": exc.code, "message": str(exc)}))
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Demo run failed")
        with contextlib.suppress(Exception):
            await websocket.send(json.dumps({"type": "error", "message": str(exc)}))
        raise


async def _serve(args: argparse.Namespace) -> None:
    async with ws_server.serve(
        lambda websocket: _handler(websocket, args),
        args.host,
        args.port,
        compression=None,
        max_size=None,
        process_request=_health_check,
    ) as server:
        LOGGER.info("DROID demo server listening on ws://%s:%s/ws", args.host, args.port)
        await server.serve_forever()


def _dry_run(args: argparse.Namespace) -> None:
    episode = DroidRawEpisode(
        pathlib.Path(args.data_root),
        pathlib.Path(args.episode_dir) if args.episode_dir else None,
        annotations_path=pathlib.Path(args.annotations_path) if args.annotations_path else None,
        fallback_prompt=args.prompt,
        exterior_camera_stem=args.exterior_camera_stem,
        wrist_camera_stem=args.wrist_camera_stem,
    )
    try:
        step = episode.read_step(args.start_frame)
        result = {
            "episode": episode.metadata(),
            "first_frame_index": args.start_frame,
            "exterior_image_shape": list(step["exterior_image"].shape),
            "wrist_image_shape": list(step["wrist_image"].shape),
            "joint_position_shape": list(step["joint_position"].shape),
            "gripper_position_shape": list(step["gripper_position"].shape),
            "dataset_action_shape": (
                list(step["dataset_action"].shape) if step["dataset_action"] is not None else None
            ),
        }
        print(json.dumps(result, indent=2, default=_json_default))
    finally:
        episode.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="Directory containing raw DROID episodes")
    parser.add_argument("--episode-dir", default=None, help="Specific raw DROID episode directory")
    parser.add_argument("--annotations-path", default=None, help="Path to aggregated-annotations-030724.json")
    parser.add_argument("--prompt", default="do something", help="Fallback prompt if no DROID annotation is found")
    parser.add_argument("--policy-host", default="127.0.0.1", help="Host for the existing OpenPI policy server")
    parser.add_argument("--policy-port", type=int, default=8000, help="Port for the existing OpenPI policy server")
    parser.add_argument("--host", default="0.0.0.0", help="Host for this streaming server")
    parser.add_argument("--port", type=int, default=8765, help="Port for this streaming server")
    parser.add_argument("--output-dir", default="demo_runs", help="Directory for metrics/video artifacts")
    parser.add_argument("--start-frame", type=int, default=0, help="First DROID frame to replay")
    parser.add_argument("--max-steps", type=int, default=120, help="Maximum DROID frames to scan")
    parser.add_argument("--open-loop-horizon", type=int, default=8, help="Frames executed before the next policy query")
    parser.add_argument("--fps", type=float, default=DROID_CONTROL_FREQUENCY, help="Playback/video FPS")
    parser.add_argument(
        "--stream-mode",
        choices=("action-chunk", "every-frame"),
        default="action-chunk",
        help="Stream one frame per policy query or every DROID frame",
    )
    parser.add_argument("--realtime", action="store_true", help="Sleep to match --fps during replay")
    parser.add_argument("--jpeg-quality", type=int, default=82, help="JPEG quality for browser frames")
    parser.add_argument("--exterior-camera-stem", default=None, help="Exact MP4 stem for the exterior camera")
    parser.add_argument("--wrist-camera-stem", default=None, help="Exact MP4 stem for the wrist camera")
    parser.add_argument("--no-write-video", dest="write_video", action="store_false", help="Skip MP4 preview output")
    parser.add_argument("--dry-run", action="store_true", help="Read metadata and the first frame, then exit")
    parser.set_defaults(write_video=True)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    if args.dry_run:
        _dry_run(args)
        return
    asyncio.run(_serve(args))


if __name__ == "__main__":
    main()
