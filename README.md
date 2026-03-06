# RetryIX Backend — 3.1.0 → 3.1.3 Release Notes

> Released: 2026-03-06

---

## Background

3.1.0 solved the critical C-ABI mismatch that caused `0xC0000005` crashes when calling
bus optimizer functions. With that crash fixed and all 15 FFI functions validated,
3.1.3 focuses entirely on **SVM memory management depth** — the key pain point for
large model workloads on AMD RDNA where VRAM pressure causes silent allocation failures
instead of graceful fallback to host memory.

---

## Breaking Changes

**None** — all public PyTorch operator APIs (`.to("privateuseone:0")`, `torch.Tensor`
methods, autograd) are fully backward-compatible with 3.1.0.

---

## What Changed

### 1. Vulkan Layer: 3-Tier SVM Fallback (`retryix_svm_layer.c`)

The Vulkan implicit layer was completely rewritten. Key additions:

#### `FallbackTier` enum
```c
typedef enum FallbackTier {
    FALLBACK_NONE     = 0,  /* DEVICE_LOCAL — VRAM */
    FALLBACK_CACHED   = 1,  /* HOST_VISIBLE | HOST_CACHED */
    FALLBACK_COHERENT = 2,  /* HOST_VISIBLE | HOST_COHERENT */
    FALLBACK_VISIBLE  = 3,  /* HOST_VISIBLE only */
} FallbackTier;
```

#### `MemRecord` hash table
Open-addressing hash table (default 4096 slots, configurable via
`RETRYIX_SVM_MEM_TABLE_SZ`) maps each `VkDeviceMemory` handle to its
`(size, heap_index, tier)` for accurate per-heap pressure accounting.

#### `our_vkFreeMemory` (NEW intercept)
Previous versions only intercepted `vkAllocateMemory`. 3.1.3 also intercepts
`vkFreeMemory`, which removes the `MemRecord` entry and decrements the heap
pressure counter on release. Without this, preemptive fallback would trigger
permanently after the first VRAM spike even after tensors were freed.

#### Preemptive routing
When `heap_pressure_pct(DEVICE_LOCAL) >= RETRYIX_SVM_VRAM_THRESHOLD_PCT`, new
allocations are routed directly to Tier 1 without attempting DEVICE_LOCAL first.
Disabled via `RETRYIX_SVM_DISABLE_PREEMPTIVE=1`.

#### Per-tier statistics on `vkDestroyDevice`
```
[RetryIX-SVM-Layer] Device destroyed: total_alloc=1234  hard_fail=0
  tier0(device_local):  1200 allocs  7680 MB
  tier1(host_cached):     28 allocs   896 MB
  tier2(host_coherent):    6 allocs   192 MB
  tier3(host_visible):     0 allocs     0 MB
  preemptive_routes:      22
```

#### Environment variables

| Variable | Default | Description |
|---|---|---|
| `RETRYIX_SVM_VRAM_THRESHOLD_PCT` | `90` | VRAM % triggering preemptive fallback |
| `RETRYIX_SVM_VERBOSE` | `0` | Per-allocation tier log |
| `RETRYIX_SVM_DISABLE_PREEMPTIVE` | `0` | Disable preemptive routing |
| `RETRYIX_SVM_MEM_TABLE_SZ` | `4096` | MemRecord hash table slots |

---

### 2. New Rust Module: `fallback.rs`

A new `retryix_svm::fallback` module provides the Rust-side policy and statistics
engine that mirrors the C layer's decision logic.

#### `FallbackPolicy`
```rust
pub enum FallbackPolicy {
    /// Always allocate regardless of pressure; no eviction.
    Aggressive,
    /// Evict oldest N allocations when budget exhausted, then retry.
    Conservative { max_retries: u32 },
    /// EMA-based: evict 25% of pool when smoothed pressure exceeds threshold.
    Adaptive { ema_alpha_x100: u32, threshold_pct: u32 },
}
```

#### `FallbackStats`
Thread-safe counters using `AtomicU64` with EMA failure-rate tracking.
Provides `FallbackReport` snapshots with `fallback_rate_pct()` and
`fallback_bytes_pct()` helpers.

#### `decide_fallback()`
Pressure-aware decision function that returns a `FallbackDecision` containing
`{target_tier, reason, is_preemptive}`.

---

### 3. `allocator.rs`: SmallSlabAllocator + TieredAllocator

#### `SmallSlabAllocator`
- 64 slots × 4096 bytes (256 KiB total slab area)
- `AtomicBool` per slot — no mutex on the fast path for ≤4 KB allocations
- Delegates larger allocations to the backing `HostAllocator`

#### `TieredAllocator`
- `Vec<Box<dyn SvmAllocator>>` tier list — default: SmallSlab → Host
- `HashMap<usize, (tier_index, size, align)>` ownership map
- `default_tiers()` constructor wires standard two-tier config

---

### 4. `context.rs`: Eviction Support

Added to `AllocationRecord`:
```rust
pub evictable: bool,   // marks allocation as eligible for pool_shrink eviction
```

Added to `SvmContextInner`:
```rust
pub pool_budget_bytes: usize,   // soft cap for estimate_pressure()
pub pressure_hint_pct: u32,     // last pressure reading, set by monitor
```

New methods: `mark_evictable()`, `set_pool_budget()`, `pool_budget()`,
`set_pressure_hint()`, `pressure_hint()`.

---

### 5. `ops.rs`: Real Pool Management

Previous `pool_clear()` and `pool_shrink()` were no-op stubs. Now:

#### `pool_clear()`
```rust
// Takes ownership of all AllocationRecords, deallocates every one.
let records = std::mem::take(&mut inner.allocations);
for rec in records { allocator.deallocate(rec.ptr, rec.size, rec.alignment); }
```

#### `pool_shrink()`
Partitions allocations into `evictable` and `keep` using `partition()`.
Deallocates only the evictable set; returns `SvmResult::NotSupported` if none
are marked evictable.

#### `svm_alloc_with_policy()`
New public function. Dispatches to one of three strategies:
- `Aggressive` → passthrough to `svm_alloc`
- `Conservative` → evicts oldest N, then retries
- `Adaptive` → computes current pressure; if above `threshold_pct`, evicts 25%

---

### 6. `monitor.rs`: Pressure Estimation + FallbackStats Integration

The monitoring module was extended with pressure-aware functions:

#### `estimate_pressure()`
Returns `PressureReport { current_pct, is_high, budget_bytes, allocated_bytes }`.
Updates `context.pressure_hint_pct` as a side effect so the allocator can
read it without holding the context lock.

New test coverage: `test_pressure_no_budget`, `test_pressure_with_budget`,
`test_pressure_updates_hint`.

---

### 7. Version Bump: 3.0.0/3.1.0 → 3.1.3

All 10 version string locations unified to `3.1.3`:

| File | Old | New |
|---|---|---|
| `retryix_rs/Cargo.toml` (workspace) | 3.0.0 | 3.1.3 |
| `retryix_types/lib.rs` | 3.0.0 | 3.1.3 |
| `retryix_api/lib.rs` (×2) | 3.0.0 | 3.1.3 |
| `ffi_core.rs` | 3.0.0 | 3.1.3 |
| `ffi_device.rs` | 3.0.0 | 3.1.3 |
| `version.txt` | 3.1.0 | 3.1.3 |
| `pytorch_retryix_backend/__init__.py` | 3.1.0 | 3.1.3 |
| `setup.cfg` | 3.0.2 | 3.1.3 |
| `setup_package.py` | 3.0.2 | 3.1.3 |

---

### 8. `retryix_ffi.dll` Size

| Version | Size |
|---|---|
| 3.1.0 | 614,912 bytes |
| **3.1.3** | **1,479,168 bytes** |

The increase (+864 KiB) is attributable to the new SVM fallback Rust code compiled
into the FFI binary: `fallback.rs`, expanded `allocator.rs`, new `ops.rs` functions,
and the Vulkan layer C code now compiled as a DLL.

---

## Test Results

All workspace tests pass on AMD RX 5700 XT, RDNA1:

```
retryix_types:   17 passed   0 failed
retryix_svm:    151 passed   0 failed
retryix_api:     11 passed   0 failed
```

Vulkan layer loaded and active during integration tests:
```
[RetryIX-SVM-Layer] Device created: 16 memory types, 3 heaps
[RetryIX-SVM-Layer] SVM fallback pool: type[1] heap[1] = 32721 MB available
[RetryIX-SVM-Layer] Device destroyed: total_alloc=6  svm_fallbacks=0
```

---

## Wheel Artifact

```
dist/pytorch_retryix_backend-3.1.3-cp311-cp311-win_amd64.whl  (1,665,850 bytes)
```

Contents:
- `pytorch_retryix_backend/retryix_ffi.dll`  — 1,479,168 bytes (v3.1.3 Rust)
- `pytorch_retryix_backend/_C.cp311-win_amd64.pyd`
- `pytorch_retryix_backend/_retryix_autograd_cpp.cp311-win_amd64.pyd`
- `pytorch_retryix_backend/__init__.py`  — version `"3.1.3"`
- `pytorch_retryix_backend/shaders/`  — 10 shader files (.comp + .spv)
