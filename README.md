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
Launch the training entrypoint (see `uv run train-gpt2 -- --help` for options):
```bash
uv run train-gpt2 -- --num-train-epochs 1 --train-batch-size 2 --fp16
```

Artifacts (checkpoints, logs, metrics) are written to `runs/gpt2-wikitext` by default. Adjust arguments to match your hardware capacity.

### Framework 13 (Ryzen AI) One-Epoch Recipe
1. Make sure the ROCm-enabled build of PyTorch is active. A quick check:
   ```bash
   uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
   If this reports `False`, install a ROCm wheel from the [PyTorch ROCm index](https://pytorch.org/get-started/locally/).
2. Kick off a one-epoch benchmark tuned for the Framework 13’s VRAM limits:
   ```bash
   uv run train-gpt2 -- \
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
