#!/usr/bin/env python3
"""Smoke-test a converted pi05_droid PyTorch checkpoint on Jetson."""

import dataclasses
import logging
import pathlib
import time

import numpy as np
import torch
import tyro

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

torch._dynamo.config.suppress_errors = True  # noqa: SLF001


@dataclasses.dataclass
class Args:
    checkpoint_dir: pathlib.Path = pathlib.Path("torch_pi05_droid")
    config_name: str = "pi05_droid"
    prompt: str = "do something"
    warmup_steps: int = 2
    num_steps: int = 10
    denoise_steps: int = 10
    seed: int = 0
    pytorch_device: str | None = None


def _random_observation(rng: np.random.Generator, prompt: str) -> dict:
    return {
        "observation/exterior_image_1_left": rng.integers(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": rng.integers(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": rng.random(7, dtype=np.float32),
        "observation/gripper_position": rng.random(1, dtype=np.float32),
        "prompt": prompt,
    }


def main(args: Args) -> None:
    logging.info("Loading config %s from checkpoint %s", args.config_name, args.checkpoint_dir)
    train_config = _config.get_config(args.config_name)
    policy = _policy_config.create_trained_policy(
        train_config,
        args.checkpoint_dir,
        sample_kwargs={"num_steps": args.denoise_steps},
        pytorch_device=args.pytorch_device,
    )

    rng = np.random.default_rng(args.seed)
    for step in range(args.warmup_steps):
        logging.info("Warmup inference %d/%d", step + 1, args.warmup_steps)
        policy.infer(_random_observation(rng, args.prompt))

    timings_ms = []
    actions = None
    for step in range(args.num_steps):
        start = time.perf_counter()
        output = policy.infer(_random_observation(rng, args.prompt))
        elapsed_ms = (time.perf_counter() - start) * 1000
        timings_ms.append(elapsed_ms)
        actions = output["actions"]
        logging.info("Inference %d/%d: %.1f ms", step + 1, args.num_steps, elapsed_ms)

    assert actions is not None
    print(f"actions_shape={actions.shape}")
    print(f"actions_dtype={actions.dtype}")
    print(f"mean_infer_ms={float(np.mean(timings_ms)):.1f}")
    print(f"p50_infer_ms={float(np.quantile(timings_ms, 0.5)):.1f}")
    print(f"p90_infer_ms={float(np.quantile(timings_ms, 0.9)):.1f}")
    print(np.array2string(actions[0], precision=4, suppress_small=True))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
