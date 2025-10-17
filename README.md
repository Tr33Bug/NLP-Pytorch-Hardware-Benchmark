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
    --save-total-limit 1
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
    --save-total-limit 1
   ```
   Mixed precision is not yet stable on MPS—leave both `--fp16` and `--bf16` unset for best results.
3. Watch `Activity Monitor` (GPU history) alongside training logs. If memory pressure rises, lower `--train-batch-size` or increase `--gradient-accumulation-steps`.
