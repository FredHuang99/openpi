# Jetson AGX Orin pi0.5 Setup

These commands are intended to run on the Jetson AGX Orin 64GB host, not on a
desktop clone. They assume Ubuntu 22.04, L4T 36.4.x, CUDA 12.6, cuDNN 9.3, and
the current OpenPI checkout/branch.

Do not use `uv run` for the final inference commands after installing the
Jetson PyTorch wheels. Use `.pi0.5/bin/python3.10` directly.

## 1. Preflight

```bash
cd ~/openpi
git status --short --branch
uname -a
lsb_release -a
python3.10 --version
jetson_release || true
nvcc --version || true
nvidia-smi || true
```

## 2. System packages

If apt reports unmet dependencies, repair the package state first:

```bash
sudo apt-get update
sudo apt --fix-broken install
sudo apt-get dist-upgrade -y
sudo apt-get autoremove -y
sudo apt-get update
```

If `apt --fix-broken install` proposes removing NVIDIA JetPack packages, stop
and inspect with `apt-cache policy` before accepting.

```bash
sudo apt-get update
sudo apt-get install -y \
  ca-certificates \
  curl \
  wget \
  git \
  git-lfs \
  build-essential \
  python3.10 \
  python3.10-dev \
  python3-pip \
  python3-venv \
  libopenblas-dev \
  libomp-dev \
  libjpeg-dev \
  zlib1g-dev \
  libpng-dev \
  libgl1 \
  libgl1-mesa-dev \
  libglew-dev \
  libosmesa6-dev \
  libglib2.0-0

git lfs install
```

Run the remaining project, uv, checkpoint, and inference steps as the normal
repository owner, for example `hhy`, not as `root`. If you already entered a
root shell for apt, switch back before continuing:

```bash
exit
cd /home/hhy/openpi
git status --short --branch
git lfs install
```

If root-created files now block the normal user, fix ownership once:

```bash
sudo chown -R hhy:hhy /home/hhy/openpi
```

## 3. Install uv and create a clean Python 3.10 environment

```bash
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

export PATH="$HOME/.local/bin:$PATH"
uv --version

cd ~/openpi
rm -rf .pi0.5
uv venv --python 3.10 .pi0.5
. .pi0.5/bin/activate
python --version
```

If `uv venv --python 3.10` cannot find Python 3.10, install the system
`python3.10` packages above first. Avoid `uv python install 3.10` on Jetson
unless the system Python is unavailable.

## 4. Install OpenPI dependencies

```bash
cd ~/openpi
export PATH="$HOME/.local/bin:$PATH"
. .pi0.5/bin/activate

GIT_LFS_SKIP_SMUDGE=1 uv lock --python 3.10
GIT_LFS_SKIP_SMUDGE=1 uv sync --python 3.10 --no-dev
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

uv pip install gsutil
```

It is fine if `uv.lock` changes locally during this Jetson install pass. Do not
commit that lockfile change unless you intentionally want the branch to become a
Python 3.10 Jetson deployment branch.

## 5. Replace desktop PyTorch with Jetson PyTorch wheels

```bash
cd ~/openpi
export PATH="$HOME/.local/bin:$PATH"
. .pi0.5/bin/activate

mkdir -p /tmp/openpi-jetson-wheels
cd /tmp/openpi-jetson-wheels

curl -L -o torch-2.3.0-cp310-cp310-linux_aarch64.whl \
  https://nvidia.box.com/shared/static/zvultzsmd4iuheykxy17s4l2n91ylpl8.whl
curl -L -o torchaudio-2.3.0-cp310-cp310-linux_aarch64.whl \
  https://nvidia.box.com/shared/static/9si945yrzesspmg9up4ys380lqxjylc3.whl
curl -L -o torchvision-0.18.0-cp310-cp310-linux_aarch64.whl \
  https://nvidia.box.com/shared/static/u0ziu01c0kyji4zz3gxam79181nebylf.whl

cd ~/openpi
uv pip uninstall -y torch torchvision torchaudio || true
UV_SKIP_WHEEL_FILENAME_CHECK=1 uv pip install --force-reinstall --no-deps \
  /tmp/openpi-jetson-wheels/torch-2.3.0-cp310-cp310-linux_aarch64.whl \
  /tmp/openpi-jetson-wheels/torchvision-0.18.0-cp310-cp310-linux_aarch64.whl \
  /tmp/openpi-jetson-wheels/torchaudio-2.3.0-cp310-cp310-linux_aarch64.whl
```

Verify CUDA before continuing:

```bash
.pi0.5/bin/python3.10 - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("cudnn:", torch.backends.cudnn.version())
x = torch.ones(4, device="cuda")
print("cuda tensor:", (x + x).cpu().tolist())
PY
```

If this fails to import CUDA, use the Jetson AI Lab fallback:

On headless Jetson sessions, importing `torchvision` may print X11/EGL warnings
such as `X11 connection rejected` or `nvbufsurftransform: Could not get EGL
display connection`. Continue if `torch.cuda.is_available()` is `True` and the
CUDA tensor test succeeds.

```bash
cd ~/openpi
. .pi0.5/bin/activate
uv pip uninstall -y torch torchvision torchaudio triton || true
uv pip install --force-reinstall --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
  torch==2.11.0 torchvision==0.26.0 torchaudio==2.10.0 triton==3.6.0

.pi0.5/bin/python3.10 - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("cudnn:", torch.backends.cudnn.version())
x = torch.ones(4, device="cuda")
print("cuda tensor:", (x + x).cpu().tolist())
PY
```

## 6. Patch transformers in the virtual environment

```bash
cd ~/openpi
. .pi0.5/bin/activate
uv pip show transformers

TRANSFORMERS_DIR=$(.pi0.5/bin/python3.10 - <<'PY'
import pathlib
import transformers
print(pathlib.Path(transformers.__file__).parent)
PY
)

PATCH_DIR="$PWD/src/openpi/models_pytorch/transformers_replace"
test -f "$PATCH_DIR/models/siglip/check.py"
cp -av "$PATCH_DIR"/models/gemma/* "$TRANSFORMERS_DIR"/models/gemma/
cp -av "$PATCH_DIR"/models/paligemma/* "$TRANSFORMERS_DIR"/models/paligemma/
cp -av "$PATCH_DIR"/models/siglip/* "$TRANSFORMERS_DIR"/models/siglip/

test -f "$TRANSFORMERS_DIR/models/siglip/check.py"

.pi0.5/bin/python3.10 - <<'PY'
from transformers.models.siglip import check
print("transformers_replace ok:", check.check_whether_transformers_replace_is_installed_correctly())
PY
```

## 7. Download pi05_droid

```bash
cd ~/openpi
. .pi0.5/bin/activate

.pi0.5/bin/python3.10 - <<'PY'
from openpi.shared import download
path = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")
print(path)
PY
```

If the download was interrupted, remove the partial cache and retry:

```bash
rm -rf "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_droid"*
```

## 8. Convert pi05_droid to a PyTorch checkpoint

```bash
cd ~/openpi
. .pi0.5/bin/activate

CKPT="$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_droid"
OUT="$PWD/torch_pi05_droid"

rm -rf "$OUT"
.pi0.5/bin/python3.10 examples/convert_jax_model_to_pytorch.py \
  --checkpoint_dir "$CKPT" \
  --config_name pi05_droid \
  --output_path "$OUT" \
  --precision float32

rm -rf "$OUT/assets"
cp -a "$CKPT/assets" "$OUT/assets"

test -f "$OUT/model.safetensors"
test -f "$OUT/config.json"
test -f "$OUT/assets/droid/norm_stats.json"
```

## 9. Local pi0.5 smoke test

```bash
cd ~/openpi
.pi0.5/bin/python3.10 scripts/pi05_droid_smoke.py \
  --checkpoint-dir ./torch_pi05_droid \
  --num-steps 10 \
  --warmup-steps 2 \
  --denoise-steps 10
```

The script should print `actions_shape`, action dtype, and timing statistics.
The expected DROID action shape is `(15, 8)`.

## 10. Run the policy server

Run this in shell 1:

```bash
cd ~/openpi
.pi0.5/bin/python3.10 scripts/serve_policy.py \
  --port=8000 \
  policy:checkpoint \
  --policy.config=pi05_droid \
  --policy.dir=./torch_pi05_droid
```

After the server finishes loading, run this in shell 2:

```bash
cd ~/openpi
.pi0.5/bin/python3.10 examples/simple_client/main.py \
  --env DROID \
  --host 127.0.0.1 \
  --port 8000 \
  --num-steps 20
```

If the first client request times out while the model warms up, leave shell 1
running and repeat the shell 2 client command.
