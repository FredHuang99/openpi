# pi05_droid Browser Demo

This demo keeps the model and DROID replay environment on the remote Jetson and
streams model input frames plus metrics to a lightweight local browser UI.

It is an offline DROID trajectory replay demo. It does not run a real DROID
robot or RoboArena evaluation, so `policy_task_success` is reported as `null`.

## Components

- `scripts/serve_policy.py`: existing OpenPI policy server on the Jetson, port `8000`.
- `examples/droid_demo/remote_demo_server.py`: Jetson-side DROID raw replay and browser stream, port `8765`.
- `examples/droid_demo/local_viewer.py`: local Win11/mac browser UI, port `7860`.

## Fresh Tasks

The local viewer has `Connect`, `Stop`, and `Fresh` controls in one toolbar.
`Fresh` closes the current WebSocket connection and asks the remote demo server
for a new DROID replay task. The new task must use a different `episode_dir`
from the current or previous task; when multiple prompts are available, the
server prefers an episode with a different prompt too.

Fresh requires the remote server to be started with `--data-root` and at least
two usable raw DROID episodes under that directory. If the server is pinned with
`--episode-dir`, or if only one episode is available, Fresh returns an explicit
`fresh_unavailable` error instead of replaying the same task again.

Check the remote episode count with:

```bash
find /home/openpi/droid_demo_data \( -name trajectory.h5 -o -name trajectory.hdf5 \) \
  -exec dirname {} \; | sort -u | wc -l
```

## Remote Jetson

Install the one extra raw-DROID reader dependency if needed:

```bash
cd /home/openpi
. .pi0.5/bin/activate
uv pip install h5py
```

Prepare a small DROID raw subset:

```bash
mkdir -p /home/openpi/droid_demo_data
gsutil -m cp -r gs://gresearch/robotics/droid_raw/1.0.1/IRIS/success/2023-12-04 \
  /home/openpi/droid_demo_data/
gsutil -m cp gs://gresearch/robotics/droid_raw/1.0.1/aggregated-annotations-030724.json \
  /home/openpi/droid_demo_data/
```

Start the policy server:

```bash
cd /home/openpi
.pi0.5/bin/python3.10 scripts/serve_policy.py \
  --port=8000 \
  policy:checkpoint \
  --policy.config=pi05_droid \
  --policy.dir=./torch_pi05_droid
```

In a second remote shell, smoke-test the raw episode reader:

```bash
cd /home/openpi
.pi0.5/bin/python3.10 examples/droid_demo/remote_demo_server.py \
  --data-root /home/openpi/droid_demo_data \
  --annotations-path /home/openpi/droid_demo_data/aggregated-annotations-030724.json \
  --dry-run
```

Then start the stream:

```bash
cd /home/openpi
.pi0.5/bin/python3.10 examples/droid_demo/remote_demo_server.py \
  --data-root /home/openpi/droid_demo_data \
  --annotations-path /home/openpi/droid_demo_data/aggregated-annotations-030724.json \
  --policy-host 127.0.0.1 \
  --policy-port 8000 \
  --host 0.0.0.0 \
  --port 8765 \
  --max-steps 120 \
  --stream-mode action-chunk
```

## Local Viewer

On Win11:

```powershell
cd C:\openpi
py -3 examples\droid_demo\local_viewer.py --remote-ws ws://JETSON_IP:8765/ws --port 7860
```

On macOS:

```bash
cd /path/to/openpi
python3 examples/droid_demo/local_viewer.py --remote-ws ws://JETSON_IP:8765/ws --port 7860
```

Open `http://127.0.0.1:7860` if the browser does not open automatically.

For an SSH tunnel, forward the remote stream port first:

```powershell
ssh -L 8765:127.0.0.1:8765 hhy@JETSON_IP
```

Then run the viewer with:

```powershell
py -3 examples\droid_demo\local_viewer.py --remote-ws ws://127.0.0.1:8765/ws --port 7860
```
