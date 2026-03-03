# RetryIX Backend 3.1.0 — Major Optimization Highlights

---

## 1. Runtime Optimization Suite Now Fully Operational

In all releases prior to 3.1.0, the three optimizer subsystems — MFE, Bus, and SVM — existed in the codebase but triggered an immediate `0xC0000005` Access Violation the moment any of their functions were called, making them completely unusable in practice. 3.1.0 resolves every blocking issue; all three subsystems now return real values from the Rust runtime.

```python
import pytorch_retryix_backend as rxb
rxb.init()

# MFE — online EMA performance tracker
hid = rxb.mfe_create()
rxb.mfe_step(hid, [0.87, 12.4, 0.3])   # feed GPU util / bandwidth / latency
metrics = rxb.mfe_read_metrics(hid)     # [mean, variance, std, decay, ...]

# Bus — PCIe / NVMe controller analysis
rxb.bus_init()
cfg  = rxb.bus_get_optimal_config()              # recommended controller + theoretical BW
bw   = rxb.bus_benchmark_bandwidth(0, 64)        # measured GB/s
sugg = rxb.bus_get_optimization_suggestions(0)   # scheduling hints (JSON)

# SVM — NUMA-aware memory placement advice
hint = rxb.svm_suggest_optimization()   # returns placement hint (JSON)
rxb.svm_optimize_placement()            # trigger in-place page migration to optimal NUMA node
```

---

## 2. Bus Subsystem — Full Rust ABI Alignment

### Root Cause

`retryix_ffi.dll` never exported C-struct-pointer variants of the Bus functions. The C header declared:

```cpp
// old declaration (wrong)
retryix_bus_result_t  retryix_bus_get_optimal_config(retryix_bus_info_t* out);
retryix_bus_result_t  retryix_bus_monitor_status(int ctrl_id, retryix_bus_status_t* out);
retryix_bus_result_t  retryix_bus_get_optimization_suggestions(int ctrl_id, retryix_bus_optimization_t* out);
```

The actual Rust DLL exports are **scalar and flat-buffer** outputs:

```cpp
// correct signatures (aligned with ffi_bus.rs)
int  retryix_bus_get_optimal_config(float* out_theoretical_gbps, int* out_lanes);
int  retryix_bus_monitor_status(int ctrl_id, float* out_peak, float* out_util, uint64_t* out_bytes);
int  retryix_bus_get_optimization_suggestions(int ctrl_id, int* out_flags, char* out_buf, size_t buf_len);
```

### Crash Mechanism

A Rust `String` is a **fat pointer triple** in memory: data pointer + length + capacity (24 bytes total). When the Rust runtime wrote this triple into what the C side expected to be a `char[128]` field, it placed an arbitrary heap address into a local stack array. Any subsequent read or free of that "string" touched memory the C process did not own — manifesting as a deferred `0xC0000005` access violation.

### What Was Fixed

- All three Bus `pfn` type definitions replaced with scalar/buffer variants
- Five previously unmapped Bus symbols added:

| New Symbol | Purpose |
|-----------|---------|
| `retryix_bus_optimize_bandwidth` | Actively optimize bandwidth, returns measured GB/s |
| `retryix_bus_enable_fallback_mode` | Toggle conservative fallback scheduling mode |
| `retryix_bus_get_controller_count` | Enumerate NVMe controllers in the system |
| `retryix_bus_get_controller_bandwidth` | Query bandwidth of a specific controller |
| `retryix_bus_get_theoretical_bandwidth` | Compute PCIe-gen theoretical peak bandwidth |

- The three JSON builder functions (`bus_get_optimal_config_json`, `bus_monitor_status_json`, `bus_get_optimization_suggestions_json`) were rewritten to assemble results from individual scalar queries rather than reading from a struct layout.

---

## 3. Python Wrappers — Dynamic Import Fix

### Root Cause

All 15 optimizer Python wrapper functions assumed a module-level `_C` variable was always in scope. In reality `_C` is only bound to the namespace after `rxb.init()` completes. Calling any optimizer wrapper before that — or via an alternate import path — raised a `NameError` which the old `except AttributeError` silently swallowed, causing the call to return a default/empty value with no warning to the caller.

### Fix

Each function's `try` block now issues an explicit dynamic import:

```python
# Before (silently failed)
def bus_init() -> bool:
    try:
        return _C.bus_init()
    except AttributeError:
        return False

# After (correct)
def bus_init() -> bool:
    try:
        from . import _C
        return _C.bus_init()
    except (AttributeError, ImportError):
        return False
```

This pattern was applied to all 15 functions:

`mfe_create` · `mfe_destroy` · `mfe_step` · `mfe_set_decay` · `mfe_read_metrics` · `mfe_run_experiment`  
`bus_init` · `bus_cleanup` · `bus_get_optimal_config` · `bus_set_performance_mode` · `bus_get_optimization_suggestions` · `bus_benchmark_bandwidth` · `bus_monitor_status`  
`svm_suggest_optimization` · `svm_optimize_placement`

---

## 4. MFE (Metric Feedback Engine) — Design Overview

MFE is a lightweight online EMA (Exponential Moving Average) tracker designed to let the application layer feed GPU runtime performance values in and read back statistics to assist scheduling decisions.

```
Input  (mfe_step)
  └─ features[] — arbitrary-length float vector
         │  e.g. [GPU util %, bandwidth GB/s, latency ms, temperature °C, ...]
         ↓
   EMA update  (decay = 0.99 default, configurable)
         │
         ↓
Output (mfe_read_metrics)
  └─ [mean, variance, std_dev, decay_factor, ...]   up to 8 slots
```

**Typical usage pattern:** periodically feed `bus_benchmark_bandwidth()` return values into MFE; after a few steps read `variance` — high variance indicates unstable bandwidth and can be used to trigger `bus_enable_fallback_mode(1)` for conservative scheduling.

---

## 5. SVM Optimizer — NUMA Placement Hints

`svm_suggest_optimization()` execution flow:

1. Construct a temporary `SvmContext` on the Rust side
2. Query system NUMA topology (node count, distance matrix, memory bandwidth)
3. Compute optimal placement suggestion based on current working-set access hot spots
4. Return a JSON string — Python side can call `json.loads()` directly

`svm_optimize_placement()` is the zero-argument variant: executes page migration directly inside the Rust runtime without requiring JSON parsing on the Python side.

---

## 6. Lazy DLL Symbol Resolution

All DLL symbols in `retryix_optimizer.cpp` are resolved via `GetModuleHandleA("retryix_ffi.dll")` + `GetProcAddress` **on first call**, not at module load time.

Benefits:
- If the DLL is missing or version-mismatched, the Python module still imports cleanly — only the actual optimizer call returns an error code
- No need to statically bind every Bus/SVM/MFE symbol at link time
- Enables hot-swapping `retryix_ffi.dll` without recompiling the `.pyd`

```cpp
// Typical lazy resolution pattern
static auto* pfn = reinterpret_cast<pfn_bus_init_t>(
    GetProcAddress(GetModuleHandleA("retryix_ffi.dll"), "retryix_bus_init"));
if (!pfn) return false;
return pfn() == 0;
```

---

## 7. 12-Section Matrix Stability Suite — All Pass

| # | Test | Special Consideration |
|---|------|-----------------------|
| A | Basic arithmetic + division | Roundtrip `(a/d)*d ≈ a` to tolerate RDNA1 `rcp_f32` ~2.5 ULP error vs CPU |
| B | 2048×2048 GEMM | Cross-checked against CPU result |
| C | Recurrent stability, 500 iters | Uses `tanh(W @ x)` to prevent spectral blowup from random matrix |
| D | Memory layout | Contiguous / non-contiguous / transpose |
| E | Batched BMM | Batched matrix multiply + broadcasting |
| F | Mixed precision | float32 ↔ float16 round-trip |
| G | VMA allocator | `get_budget()` + large tensor alloc/free cycle |
| H | Activation functions | relu / sigmoid / tanh / gelu / silu |
| I | Three stacked Conv2d | Forward-only (PrivateUse1 has no autograd kernel for conv inputs) |
| J | LayerNorm + Attention | Multi-head self-attention block |
| K | SGD optimizer step | Parameter update on GPU tensor |
| L | VRAM budget cycle | 10× large tensor alloc/dealloc — verifies no VRAM leak |

**Result: 12 / 12 PASS (~10 s total, AMD RX 5700 XT)**
