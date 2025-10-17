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
