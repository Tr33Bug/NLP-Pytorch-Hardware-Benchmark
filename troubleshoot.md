Got it — since ROCm and devices look fine but torch.cuda.is_available() is still False, this almost always boils down to either:
	1.	You’re accidentally importing a CPU-only torch inside the venv, or
	2.	PyTorch’s HIP plugin can’t find ROCm libs (e.g., libamdhip64.so, rocblas, miopen) at runtime.

Here’s a focused “prove it, then fix it” flow. Copy/paste the probes first; based on what they show, apply the matching fix.

⸻

🔎 Probes (run inside your uv venv)

Probe 1 — Are you loading a ROCm build of torch?

python - <<'PY'
import sys, torch, inspect, pkgutil
print("Python:", sys.executable)
print("Torch file:", torch.__file__)
print("Torch version:", torch.__version__)
print("torch.version.hip:", getattr(torch.version, "hip", None))
print("Compiled w/ CUDA (HIP)?:", torch.backends.cuda.is_built())
PY

# Also check pip metadata of what's actually installed in this venv
uv pip show torch | sed -n '1,80p'

Expect: torch.version.hip is not None, and is_built() is True.
If hip=None or is_built()=False, you’ve got a CPU wheel → jump to Fix A.

⸻

Probe 2 — Can PyTorch’s HIP plugin find ROCm libs?

python - <<'PY'
import os, glob, subprocess, torch
lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
cands = ["libtorch_hip.so", "libamdhip64.so"]
for name in cands:
    paths = glob.glob(os.path.join(lib_dir, name)) or glob.glob(f"/opt/rocm/lib*/{name}")
    print(f"{name} candidates:", paths)
    for p in paths:
        print(f"ldd {p}:")
        out = subprocess.run(["ldd", p], text=True, capture_output=True)
        print(out.stdout or out.stderr)
PY

Scan the ldd output for “not found” lines (commonly libamdhip64.so, librocblas.so, libMIOpen.so, librccl.so).
If anything is missing → Fix B.

⸻

Probe 3 — Do you actually have the ROCm core libs?

ls -l /opt/rocm/lib*/libamdhip64.so* 2>/dev/null || echo "libamdhip64.so not present"
ls -l /opt/rocm/lib*/librocblas.so*  2>/dev/null || echo "rocblas missing"
ls -l /opt/rocm/lib*/libMIOpen.so*   2>/dev/null || echo "miopen missing"
ls -l /opt/rocm/lib*/librccl.so*     2>/dev/null || echo "rccl missing"

If any are missing → Fix B.

⸻

Probe 4 — Is the dynamic loader seeing ROCm?

# one-shot check (current shell only)
env LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm/lib64:$LD_LIBRARY_PATH" \
    python - <<'PY'
import torch
print("HIP available?", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY

If this flips to True, it’s an environment / loader-path issue → Fix C.

⸻

✅ Fixes (apply what matches your probe results)

Fix A — Ensure you have the ROCm torch wheels (not CPU)

Sometimes pip/uv silently pulled the CPU wheel from PyPI.

# Still in the venv
uv pip uninstall -y torch torchvision torchaudio triton

# Reinstall ONLY the AMD wheels (adjust URLs if you used newer ones)
uv pip install \
  https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0.2/torch-2.8.0%2Bgitc497508-cp312-cp312-linux_x86_64.whl \
  https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0.2/torchvision-0.23.0%2Brocm7.0.2.git824e8c87-cp312-cp312-linux_x86_64.whl \
  https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0.2/torchaudio-2.8.0%2Brocm7.0.2.git6e1c7fe9-cp312-cp312-linux_x86_64.whl \
  https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0.2/triton-3.4.0%2Brocm7.0.2.gitf9e5bf54-cp312-cp312-linux_x86_64.whl

# Re-check
python - <<'PY'
import torch; print("hip:", getattr(torch.version, "hip", None)); print("HIP avail:", torch.cuda.is_available())
PY

Tip: keep a requirements.txt with the exact URLs so nothing gets “upgraded” from PyPI later.

⸻

Fix B — Install any missing ROCm userland bits

If Probe 2/3 showed missing libs:

# Core runtime + math + DNN + comms (names match ROCm 7 on Ubuntu 24.04)
sudo apt update
sudo apt install -y \
  hip-runtime-amd \
  rocm-hip-libraries \
  rocblas \
  hipblaslt \
  miopen-hip \
  rccl

# re-check the libs and ldd again

If ldd still can’t find them, it’s a loader path issue → Fix C.

⸻

Fix C — Make ROCm libs visible to your Python

Two robust options; pick one:

Option 1: LD_LIBRARY_PATH (quick & explicit)

# add to your shell init (e.g., ~/.bashrc)
echo 'export PATH="/opt/rocm/bin:$PATH"' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm/lib64:$LD_LIBRARY_PATH"' >> ~/.bashrc
source ~/.bashrc

# verify
python - <<'PY'
import torch
print("HIP avail?", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY

Option 2: ld.so conf (system-wide, no env var needed)

printf "/opt/rocm/lib\n/opt/rocm/lib64\n" | sudo tee /etc/ld.so.conf.d/rocm.conf
sudo ldconfig

# verify
python - <<'PY'
import torch
print("HIP avail?", torch.cuda.is_available())
PY


⸻

If it’s still False
	•	Make sure your venv is Python 3.12 (matches cp312 wheels): python -V.
	•	Nuke & recreate a clean venv, then only install the AMD wheels (Fix A).
	•	Cross-check with Docker to confirm the host stack is fine:

docker run -it --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 8G \
  rocm/pytorch:latest python - <<'PY'
import torch
print("HIP?", torch.cuda.is_available())
if torch.cuda.is_available(): print(torch.cuda.get_device_name(0))
PY

If Docker returns True, the issue is 100% your Python/loader path; Fix A/C will solve it.

⸻

If you paste the outputs of Probe 1 (especially torch.version.hip and is_built()), plus any “not found” lines from Probe 2, I’ll pinpoint the exact missing piece and give you the minimal command to fix it.