## Training Benchmark

Local playground for benchmarking GPT-2 fine-tuning throughput with PyTorch on WikiText-103.

### Prerequisites
- Install [`uv`](https://docs.astral.sh/uv/) and ensure it is on your `PATH`.
- A working PyTorch backend (CPU or GPU/Rocm/CUDA) supported by your hardware.

### Setup
```bash
uv sync
```

### Run GPT-2 Finetuning
Launch the training entrypoint (see `uv run train-gpt2 --help` for options):
```bash
uv run train-gpt2 --num-train-epochs 1 --train-batch-size 2 --fp16
```

Artifacts (checkpoints, logs, metrics) are written to `runs/gpt2-wikitext` by default. Adjust arguments to match your hardware capacity.
Add `--subset-size 1024` (or any positive integer) to quickly iterate on a shuffled subset before scaling up.

### Framework 13 (Ryzen AI) One-Epoch Recipe
1. Make sure the ROCm-enabled build of PyTorch is active. A quick check:
   ```bash
   uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
   If this reports `False`, reinstall PyTorch from the ROCm wheelhouse:
   ```bash
   uv pip install --index-url https://download.pytorch.org/whl/rocm6.0 torch torchvision --upgrade
   ```
2. Kick off a one-epoch benchmark tuned for the Framework 13’s VRAM limits:
   ```bash
  uv run train-gpt2 \
    --num-train-epochs 1 \
    --train-batch-size 1 \
    --gradient-accumulation-steps 16 \
    --learning-rate 5e-5 \
    --warmup-steps 200 \
    --bf16 \
    --logging-steps 25 \
    --save-total-limit 1 \
    --subset-size 4096
   ```
   Use `--fp16` instead of `--bf16` if bfloat16 kernels are unavailable.
3. Monitor throughput via the training logs; adjust `--gradient-accumulation-steps` upward if you hit out-of-memory errors or downward if utilization is low.

### Apple M4 Mac One-Epoch Recipe
1. Ensure Metal acceleration is active:
   ```bash
   PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python -c "import torch; print(torch.backends.mps.is_available())"
   ```
   If `False`, reinstall the current-nightly build with MPS kernels:
   ```bash
   uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu --upgrade
   ```
2. Launch a balanced one-epoch run for the M4’s unified memory:
   ```bash
  PYTORCH_ENABLE_MPS_FALLBACK=1 uv run train-gpt2 \
    --num-train-epochs 1 \
    --train-batch-size 2 \
    --gradient-accumulation-steps 8 \
    --learning-rate 5e-5 \
    --warmup-steps 200 \
    --logging-steps 25 \
    --save-total-limit 1 \
    --subset-size 4096
   ```
   Mixed precision is not yet stable on MPS—leave both `--fp16` and `--bf16` unset for best results.
3. Watch `Activity Monitor` (GPU history) alongside training logs. If memory pressure rises, lower `--train-batch-size` or increase `--gradient-accumulation-steps`.



# Setup Linux on Framework to work

1. AMD requires the 6.14 OEM kernel for Ryzen+ROCm on Linux:
```bash
sudo apt update
sudo apt install -y linux-oem-24.04
sudo reboot
# after reboot:
uname -r  # should show 6.14.x
```
2. Now install the Ryzen software/ROCm stack the AMD way:
```bash
# add AMD installer for Ubuntu 24.04 (adjust if AMD bumps the minor)
sudo apt update
wget https://repo.radeon.com/amdgpu-install/7.0.2/ubuntu/noble/amdgpu-install_7.0.2.70002-1_all.deb
sudo apt install -y ./amdgpu-install_7.0.2.70002-1_all.deb

# install ROCm userspace for Ryzen (use inbox kernel drivers → --no-dkms)
sudo amdgpu-install -y --usecase=rocm --no-dkms

# add your user to the right groups and reboot
sudo usermod -a -G render,video $USER
sudo reboot

# Sanity check
groups | grep -E 'render|video'
rocminfo | grep -E 'Name:|Marketing Name:|gfx11'
```

3. AMD’s Ryzen PyTorch page lists the exact ROCm 7.0.2 wheels for Python 3.12 and shows the verification commands:
```bash
mkdir -p ~/pytorch-rocm && cd ~/pytorch-rocm
uv venv --python 3.12 .venv
source .venv/bin/activate
python -V   # should show 3.12.x from .venv

# Replace the URLs below with the exact ones from AMD’s page if newer.
uv pip install \
  https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0.2/torch-2.8.0%2Bgitc497508-cp312-cp312-linux_x86_64.whl \
  https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0.2/torchvision-0.23.0%2Brocm7.0.2.git824e8c87-cp312-cp312-linux_x86_64.whl \
  https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0.2/torchaudio-2.8.0%2Brocm7.0.2.git6e1c7fe9-cp312-cp312-linux_x86_64.whl \
  https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0.2/triton-3.4.0%2Brocm7.0.2.gitf9e5bf54-cp312-cp312-linux_x86_64.whl


  # OPTIONAL Enable experimental AOTriton kernels:  
  # enable for this shell session
   export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
```

3. Small test
```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("HIP available?", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x.t()
    print("ok:", y.shape, y.device)
PY
```