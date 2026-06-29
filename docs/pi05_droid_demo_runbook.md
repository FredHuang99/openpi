# pi05_droid 展示 Demo 快速复现

这份文档只讲展示 demo 的运行方式。Jetson 上的 pi0.5 环境、`.pi0.5`
虚拟环境、`torch_pi05_droid` checkpoint 已默认准备好；完整 Jetson 安装流程见
`docs/jetson_pi05_orin_setup.md`。

## 1. 组件和端口

```text
Jetson shell 1
  scripts/serve_policy.py
  127.0.0.1:8000
  pi05_droid model server

Jetson shell 2
  examples/droid_demo/remote_demo_server.py
  0.0.0.0:8765
  DROID raw replay + image/metrics stream

Local Win11/mac
  examples/droid_demo/local_viewer.py
  127.0.0.1:7860
  Browser UI
```

浏览器访问的是本地 `http://127.0.0.1:7860`。本地 viewer 再通过 WebSocket
连接远端 `ws://JETSON_IP:8765/ws`，或者通过 SSH tunnel 连接
`ws://127.0.0.1:8765/ws`。

这是离线 DROID raw 轨迹回放 demo，不是真实 DROID 机器人闭环评测，也不是
RoboArena benchmark。最终 JSON 会包含：

- `run_completed`: demo 是否正常跑完。
- `dataset_demo_success`: DROID raw episode 路径中是否来自 success 目录。
- `policy_task_success`: 固定为 `null`，因为离线回放不能判断真实机器人任务成功率。

## 2. 本地 Win11 最小安装

本地不需要安装 OpenPI、torch、uv、CUDA 或 DROID 数据集。只需要 Python 3.10+
和浏览器。推荐 Python 3.11 或 3.12。

PowerShell:

```powershell
py -3 --version
```

如果没有 Python，任选一种方式安装：

```powershell
winget install -e --id Python.Python.3.12 --source winget
```

或者从 `https://www.python.org/downloads/windows/` 安装，并勾选 `Add python.exe
to PATH`。

验证：

```powershell
py -3 --version
cd C:\openpi
py -3 examples\droid_demo\local_viewer.py --help
```

## 3. 本地 macOS 最小安装

macOS 同样不需要 OpenPI、torch、uv、CUDA 或 DROID 数据集。

```bash
python3 --version
```

如果没有可用 Python，可以用 Homebrew：

```bash
brew install python
python3 --version
```

验证：

```bash
cd /path/to/openpi
python3 examples/droid_demo/local_viewer.py --help
```

## 4. 远端准备 DROID raw 数据

如果远端还没有 DROID raw 小样本，先下载 OpenPI 文档推荐的 30-demo 子集和语言标注。
这不是完整 DROID 数据集，不需要下载 1.8TB RLDS。

Jetson:

```bash
cd /home/openpi
. .pi0.5/bin/activate
uv pip install h5py

mkdir -p /home/openpi/droid_demo_data
gsutil -m cp -r gs://gresearch/robotics/droid_raw/1.0.1/IRIS/success/2023-12-04 \
  /home/openpi/droid_demo_data/
gsutil -m cp gs://gresearch/robotics/droid_raw/1.0.1/aggregated-annotations-030724.json \
  /home/openpi/droid_demo_data/
```

如果 MP4 读取失败，再补系统 ffmpeg：

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

检查数据目录里能找到 episode：

```bash
find /home/openpi/droid_demo_data \( -name trajectory.h5 -o -name trajectory.hdf5 \) | head
find /home/openpi/droid_demo_data -path "*/recordings/MP4" | head
```

Fresh 按钮要求至少 2 个可用 episode。用下面的命令检查数量：

```bash
find /home/openpi/droid_demo_data \( -name trajectory.h5 -o -name trajectory.hdf5 \) \
  -exec dirname {} \; | sort -u | wc -l
```

如果输出是 `1`，`Connect` 可以正常跑，但 `Fresh` 会返回 `fresh_unavailable`，
因为不能重复旧任务。

## 5. 远端启动 policy server

Jetson shell 1:

```bash
cd /home/openpi
.pi0.5/bin/python3.10 scripts/serve_policy.py \
  --port=8000 \
  policy:checkpoint \
  --policy.config=pi05_droid \
  --policy.dir=./torch_pi05_droid
```

保持这个 shell 不要关闭。

## 6. 远端启动 demo streaming server

先做 dry run，确认能读取 episode metadata 和第一帧：

```bash
cd /home/openpi
.pi0.5/bin/python3.10 examples/droid_demo/remote_demo_server.py \
  --data-root /home/openpi/droid_demo_data \
  --annotations-path /home/openpi/droid_demo_data/aggregated-annotations-030724.json \
  --dry-run
```

如果 dry run 输出 `exterior_image_shape`、`wrist_image_shape`、
`joint_position_shape`，就启动 streaming server。

Jetson shell 2:

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

`--stream-mode action-chunk` 表示每次 policy query 返回一帧和一个 action chunk。
如果想按 DROID 15Hz 逐帧播放，改成：

```bash
--stream-mode every-frame --realtime
```

不要用 `--episode-dir` 固定单个 episode 来启动 Fresh demo。固定 `--episode-dir`
时 `Connect` 可以跑，但 `Fresh` 会明确返回 `fresh_unavailable`。

## 7. 本地直连观看

适用于本地能直接访问 Jetson `8765` 端口的网络。

Win11 PowerShell:

```powershell
cd C:\openpi
py -3 examples\droid_demo\local_viewer.py --remote-ws ws://JETSON_IP:8765/ws --port 7860
```

macOS:

```bash
cd /path/to/openpi
python3 examples/droid_demo/local_viewer.py --remote-ws ws://JETSON_IP:8765/ws --port 7860
```

然后打开：

```text
http://127.0.0.1:7860
```

页面会实时显示传入模型的 exterior image、wrist image、当前 action、
roundtrip latency、server infer latency，以及最终 metrics JSON。

## 8. 本地通过 SSH tunnel 观看

适用于 Jetson 端口不方便直接暴露、公司网络拦截端口或 Windows 防火墙麻烦的情况。

Win11 PowerShell 先开 tunnel：

```powershell
ssh -L 8765:127.0.0.1:8765 hhy@JETSON_IP
```

保持这个 SSH 窗口不要关闭。再开一个 PowerShell：

```powershell
cd C:\openpi
py -3 examples\droid_demo\local_viewer.py --remote-ws ws://127.0.0.1:8765/ws --port 7860
```

macOS:

```bash
ssh -L 8765:127.0.0.1:8765 hhy@JETSON_IP
```

另开一个终端：

```bash
cd /path/to/openpi
python3 examples/droid_demo/local_viewer.py --remote-ws ws://127.0.0.1:8765/ws --port 7860
```

SSH tunnel 模式下，本地 viewer 必须使用 `ws://127.0.0.1:8765/ws`，不要使用
`ws://JETSON_IP:8765/ws`。

## 9. Connect、Stop、Fresh

- `Connect`: 连接远端 streaming server，并启动一个 DROID replay 任务。
- `Stop`: 断开当前浏览器连接。远端 run 最多会在当前 blocking policy query
  结束后感知断连。
- `Fresh`: 断开旧连接并请求远端启动新任务。新任务的 `episode_dir` 必须不同于
  当前或上一次任务；如果存在不同 prompt 的候选 episode，服务端会优先选择不同
  prompt。

Fresh 请求会在 WebSocket URL 上追加 `fresh=1`、`exclude_episode`、
`exclude_prompt` 和 `nonce`。这些参数由本地 viewer 自动处理，用户不需要手动填写。

Fresh 不可用时，页面会显示 `fresh_unavailable`，常见原因是：

- `--data-root` 下只有 1 个可用 raw episode。
- 远端 streaming server 使用了固定 `--episode-dir`。
- episode 目录缺少 `trajectory.h5` 或 `trajectory.hdf5`。
- episode 目录缺少 `recordings/MP4`。

## 10. 运行产物

远端 demo server 会写入：

```text
/home/openpi/demo_runs/<timestamp>/metrics.json
/home/openpi/demo_runs/<timestamp>/trajectory_preview.mp4
```

如果当前环境缺少 MP4 encoder，`metrics.json` 仍会写出，最终 JSON 里会包含
`video_error`。

## 11. 常见问题

### 本地网页打开但没有帧

先看本地页面顶部状态。如果是 `connecting` 或 `error`，说明本地浏览器没有连上
远端 streaming server。检查：

```powershell
py -3 examples\droid_demo\local_viewer.py --remote-ws ws://127.0.0.1:8765/ws --port 7860
```

在 SSH tunnel 模式下必须使用 `127.0.0.1`。

### WebSocket 连接失败

直连模式下确认 Jetson IP 和端口：

```bash
hostname -I
ss -ltnp | grep 8765
```

如果网络不通，改用 SSH tunnel。

### Jetson policy server 未启动

demo streaming server 会连接 `127.0.0.1:8000`。确认 shell 1 仍在运行：

```bash
ss -ltnp | grep 8000
```

如果没有监听，重新启动 `scripts/serve_policy.py`。

### 找不到 `trajectory.h5` 或 `recordings/MP4`

确认传给 `--data-root` 的目录包含 DROID raw episode：

```bash
find /home/openpi/droid_demo_data \( -name trajectory.h5 -o -name trajectory.hdf5 \) | head
find /home/openpi/droid_demo_data -path "*/recordings/MP4" | head
```

也可以显式指定一个 episode：

```bash
.pi0.5/bin/python3.10 examples/droid_demo/remote_demo_server.py \
  --data-root /home/openpi/droid_demo_data \
  --episode-dir /home/openpi/droid_demo_data/2023-12-04/<episode_dir> \
  --dry-run
```

### Fresh 显示 `fresh_unavailable`

先确认 episode 数量：

```bash
find /home/openpi/droid_demo_data \( -name trajectory.h5 -o -name trajectory.hdf5 \) \
  -exec dirname {} \; | sort -u | wc -l
```

如果数量小于 2，需要再放入至少一个 DROID raw episode。也要确认 streaming server
不是用 `--episode-dir` 固定启动的。

### Windows 防火墙或公司网络拦截端口

优先使用 SSH tunnel。这样浏览器和本地 viewer 只访问 `127.0.0.1`，远端只需要 SSH
可达。

### 页面显示结束但 `policy_task_success` 是 null

这是预期行为。这个 demo 使用 DROID raw episode 离线回放，模型输出不会真的控制
机器人改变下一帧 observation，所以不能判断真实任务是否成功。真实 DROID/RoboArena
成功率需要接入真实机器人环境和人工/benchmark 评估。
