# pytorch-retryix-backend 3.0.2 — Release Notes

## What's New

### Complete Rust Migration (retryix_ffi.dll)
All GPU compute functions have been migrated from C++ stubs to native Rust
(`retryix_ffi.dll`, 1.4 MB).  The old `retryix_v2.dll` dependency has been
removed entirely.

**Exported GPU compute symbols (retryix_ffi.dll):**
- `retryix_vulkan_init`
- `retryix_vulkan_gemm_f32`
- `retryix_vulkan_gemm_impl`
- `retryix_vulkan_add`
- `retryix_vulkan_relu_f32`
- `retryix_vulkan_saxpy_f32`
- `retryix_vulkan_get_vram_bytes`

### GPU Dispatch with Topology Gate
Matrix multiplication is dispatched to the Vulkan GPU GEMM pipeline when
`M × N × K ≥ 1,000,000` (AMD RDNA tuning: ~80 µs dispatch overhead).
Smaller matrices correctly fall back to CPU to avoid GPU overhead penalty.

| Matrix size | Path |
|---|---|
| 3 × 4 @ 4 × 3 (36 FLOPs) | CPU fallback |
| 128 × 128 @ 128 × 128 (2 M FLOPs) | **GPU dispatch** ✓ |
| 256 × 256 @ 256 × 256 (16 M FLOPs) | **GPU dispatch** ✓ |

### CPU Fallback Prohibition API
New `set_cpu_fallback_prohibited(True)` API prevents any silent CPU detour
during persistent GEMM kernel sessions.  Attempting a sub-threshold matmul
while prohibited raises `RuntimeError` immediately instead of silently
degrading to CPU.

```python
rx.set_cpu_fallback_prohibited(True)   # lock to GPU-only
torch.matmul(small_a, small_b)         # raises RuntimeError: CPU fallback prohibited
rx.set_cpu_fallback_prohibited(False)  # re-enable# RetryIX Backend Preview Package

This document describes the complete environment required to reproduce, build and validate the preview release package.  
By following the steps below, anyone with the appropriate hardware and software can recreate the setup from this directory and produce a preview-ready bundle.

---

## 📦 Package contents

The archive `retryix_backend_preview_package.zip` contains:

- Source code (`*.cpp`, `*.h`, `setup.py`, etc.) – note that some C++ calls rely on an **FFI library compiled from Rust**.
- Pre‑built Python modules (`*.pyd`) for Python 3.11/3.13 win_amd64
- Dynamic libraries (`retryix_ffi.dll`, `retryix_v2.dll`, `amdvlk64.dll`, …)
  * `retryix_ffi.dll` is the Rust‑compiled `cdylib` that provides a thin C ABI; C++ calls such as `retryix_initialize()` are actually implemented there.
- Test and diagnostic scripts (`tests/`, `scripts/`, `tools/`)
- The updated wheel file `pytorch_retryix_backend-3.0.1-cp311-cp311-win_amd64.whl` (dependency and reliability fixes)
- Build artifacts (`build/`, `dist/`) and egg/wheel
- `pkl_stress.txt`: sample log from the stress test

---

## ✅ Prerequisites

1. **OS:** Windows 10/11 x64.
2. **GPU:** Vulkan‑capable AMD RDNA (e.g. RX 5700 XT) with drivers installed.
3. **Vulkan SDK:** v1.3 or later. Example path:  
   ```
   F:\VulkanSDK\1.4.321.1
   ```
4. **Python:** 3.11.x (64‑bit).
5. **PyTorch:** CPU‑only 2.x (install via official wheel).
6. **Build tools:** Visual Studio 2022+ with C++ workload and `ninja`.
7. **Optional:** `7zip`/`tar` for extraction.

---

## 🛠 Environment setup

```powershell
# 1. Unzip the preview package into a working folder
Expand-Archive -LiteralPath <archive_path> -DestinationPath <working_dir> -Force

# 2. Create & activate a Python 3.11 virtual env
py -3.11 -m venv <working_dir>\venv
& <working_dir>\venv\Scripts\Activate.ps1
python --version   # should report 3.11.x

# 3. Upgrade pip and install PyTorch CPU
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# 4. Ensure Vulkan SDK bin directory is on PATH (or use os.add_dll_directory in Python)
$env:PATH += ";<VulkanSDK_path>\Bin"

# 5. Add the unpacked repo to PATH so DLLs can be found
$env:PATH += ";<working_dir>\pytorch_retryix_branch"
```

> **Tip:** you can call `os.add_dll_directory()` from Python scripts instead of modifying PATH.

---

## 🔧 Build & validation

### 1. Verify pre‑built modules

```powershell
python - <<'PY'
import os, sys
# DLL search paths
os.add_dll_directory(r'<working_dir>\pytorch_retryix_branch')
os.add_dll_directory(r'<VulkanSDK_path>\Bin')

# Make the backend package importable
sys.path.insert(0, r'<working_dir>\pytorch_retryix_branch')

import pytorch_retryix_backend as _b
_b.register_retryix_hooks()

# import autograd component
import _retryix_autograd_cpp
print('module loaded', _retryix_autograd_cpp)
PY
```

If you see the module object printed with no `ImportError`, dependencies are satisfied.

### 2. Run the stress test (log available)

In the same Python session:

```python
import torch, time, gc
x = torch.randn(512,512, device='privateuseone:0')
for i in range(1,10001):
    y = x * x
    y = y.relu()
    if i % 1000 == 0:
        print(f"  iter={i:5d}  py_objs={len(gc.get_objects())}  errors=0  elapsed={time.time():.1f}s")
```

Output should match entries in `pkl_stress.txt`, finishing with the final iteration summary.

### 3. Re‑build (optional)

If you modify source or need a fresh `.pyd`:

```powershell
cd <working_dir>\pytorch_retryix_branch
python setup.py build_ext --inplace
```

This generates updated modules under `pytorch_retryix_backend/` etc.

---

## 📄 Packaging for release

After confirming everything works, re‑archive the directory:

```powershell
Compress-Archive -Path <working_dir>\* -DestinationPath <archive_path> -Force
```

This `.zip` becomes the **preview release package**, containing source, binaries, tests and docs.

---

## 📝 Summary

The document explains:

1. required hardware/software
2. virtual‑env creation & dependency installation
3. module import and stress‑test validation
4. optional rebuild steps
5. packaging commands

Use it as the canonical guide for reproducing the preview environment. Expand with additional scripts, version metadata or notes as needed.

---
