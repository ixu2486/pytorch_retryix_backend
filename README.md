# RetryIX Backend — 3.0.2 → 3.0.3 Release Notes

> Released: 2026-03-02

---

## Background

After 3.0.2 shipped, we noticed that `TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)` only had **7 operators registered**, even though `retryix_ops.cpp` already contained implementations for **200+**. This meant that multimodal models (Vision Transformers, CLIP, LLaVA, Stable Diffusion, etc.) were silently falling back to CPU for the vast majority of ops. 3.0.3 is a pure **registration-layer** hotfix — no changes to any underlying kernel implementations were needed.

---

## What Changed

### 1. Added `#include "retryix_ops.hpp"`

`pytorch_retryix_backend_vulkan.cpp` was missing the include for the full operator declaration header. Without it, `retryix_ops::*` symbols were invisible inside the registration block, so nothing could be wired up.

---

### 2. RX_IMPL — Safe Registration Macro

A try-catch wrapper macro was added around every `m.impl()` call. This prevents a single schema mismatch from throwing during DLL init and taking down the entire module load:

```cpp
#define RX_IMPL(op, fn) \
    do { try { \
        m.impl(op, TORCH_FN(fn)); \
    } catch(const std::exception& _rx_e) { \
        std::cerr << "[RetryIX] WARN skip " op ": " << _rx_e.what() << "\n"; \
    } } while(0)
```

---

### 3. `sub.Tensor` Alpha Wrapper

The ATen schema for `sub.Tensor` is `(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor`, but `retryix_ops::sub` only takes two arguments. A thin wrapper handles the alpha path:

```cpp
static at::Tensor sub_with_alpha_retryix(
        const at::Tensor& a, const at::Tensor& b, const at::Scalar& alpha) {
    if (alpha.toDouble() == 1.0) return retryix_ops::sub(a, b);
    // alpha != 1 is rare (e.g. residual scaling); scale on CPU side then subtract
    auto b_cpu = b.cpu().mul_(alpha);
    return retryix_ops::sub(a, b_cpu.to(a.device()));
}
```

---

### 4. Full Multimodal Operator Registration (150+ ops)

| Category | Operators (selection) |
|----------|-----------------------|
| **Memory creation** | `empty_strided`, `zeros_like`, `ones_like`, `full`, `arange`, `eye`, `fill_`, `normal_`, `bernoulli_`, `uniform_` |
| **Device transfer** | `_to_copy`, `copy_`, `clone`, `detach`, `_local_scalar_dense` |
| **Arithmetic** | `sub.Tensor`, `div.Tensor` |
| **Matrix multiply** | `mm`, `bmm`, `addmm`, `mv`, `addmv`, `addr`, `baddbmm`, `dot`, `ger`, `outer` |
| **Activations** | `relu`, `relu_`, `sigmoid`, `tanh`, `gelu`, `silu`, `elu`, `elu_`, `leaky_relu`, `leaky_relu_`, `hardsigmoid`, `hardswish`, `hardtanh`, `softplus`, `selu`, `mish`, `prelu` + all backward variants |
| **Softmax** | `_softmax`, `_log_softmax` |
| **Shape / view** | `view`, `reshape`, `transpose`, `permute`, `unsqueeze`, `squeeze`, `cat`, `stack`, `chunk`, `split`, `unbind`, `flip`, `roll`, `pixel_shuffle`, `pixel_unshuffle`, `constant_pad_nd`, … |
| **Indexing / slicing** | `slice`, `select`, `index`, `index_put_`, `index_select`, `gather`, `scatter`, `scatter_add`, `masked_fill`, `masked_select`, `where`, `nonzero`, … |
| **Elementwise unary** | `abs`, `neg`, `sqrt`, `rsqrt`, `exp`, `log`, `pow`, `reciprocal`, `sign`, `ceil`, `floor`, `round`, `sin`, `cos`, `tan`, `erf`, `erfinv`, `isnan`, `isinf`, `isfinite`, `nan_to_num`, … |
| **Clamp** | `clamp`, `clamp_min`, `clamp_max`, `clamp_` |
| **Comparison** | `eq`, `ne`, `gt`, `lt`, `ge`, `le` |
| **Logical** | `logical_and`, `logical_or`, `logical_not` |
| **Reduction** | `sum`, `sum.dim_IntList`, `mean`, `max`, `min`, `argmax`, `argmin`, `prod`, `all`, `any`, `amax`, `amin`, `logsumexp`, `topk`, `sort`, `argsort`, `cumsum`, `std`, `var`, `norm`, `trace` |
| **Normalization** | `layer_norm`, `group_norm`, `batch_norm`, `instance_norm`, `native_group_norm_backward` |
| **Pooling** | `max_pool2d`, `max_pool2d_with_indices`, `avg_pool2d`, `adaptive_avg_pool2d` + backward ops |
| **Upsample** | `upsample_nearest2d`, `upsample_bilinear2d`, `upsample_bicubic2d` + backward ops |
| **Loss functions** | `mse_loss`, `l1_loss`, `smooth_l1_loss`, `binary_cross_entropy`, `binary_cross_entropy_with_logits`, `nll_loss.nd` |
| **Dropout** | `native_dropout`, `native_dropout_backward` |
| **Embedding** | `embedding` |
| **Matrix helpers** | `diag`, `tril`, `triu` |
| **Foreach / optimizer** | `_foreach_add_`, `_foreach_mul_`, `_foreach_addcdiv_`, `_foreach_addcmul_` |

---

### 5. `torch.amp` Environment Fix

Discovered that the `torch 2.10.0+cpu` installation was missing the `torch/amp/` subdirectory entirely, causing `import torch` to fail with a circular-import error. Restored `autocast_mode.py`, `grad_scaler.py`, and `__init__.py` by extracting them from the pip HTTP cache. This is an environment-level fix and does not affect the package itself.

---

## Test Results

Smoke-tested on **AMD Radeon RX 5700 XT** (RDNA1, Vulkan compute):

```
PASS: add         PASS: sub         PASS: mul         PASS: div
PASS: relu        PASS: sigmoid     PASS: gelu        PASS: softmax
PASS: mm          PASS: bmm
PASS: layer_norm
PASS: max_pool2d
PASS: embedding
PASS: sum         PASS: argmax

Result: 15/15 passed
```

---

## Package Info

| Field | Value |
|-------|-------|
| Version | `3.0.3` |
| Wheel | `pytorch_retryix_backend-3.0.3-cp311-cp311-win_amd64.whl` |
| Python | 3.11.x |
| PyTorch | 2.10.0+cpu |
| Files changed | `pytorch_retryix_backend_vulkan.cpp`, `version.txt` |

---

## Bug Reports

This is an **early preview release**. While the smoke test suite passes 15/15 and standard model architectures (ResNet, ViT, BERT-style transformers, basic diffusion pipelines) are expected to work, you may still hit:

- Operators that are registered but whose GPU kernel path isn't fully validated for all input shapes / dtypes
- Edge cases in backward passes for less common activation functions
- Memory layout issues when mixing strided and contiguous tensors across ops

If you run into a failure, please open an issue and include:

1. The full Python traceback
2. The operator name (e.g. `aten::scatter.src`)
3. Input tensor shapes, dtypes, and device
4. GPU model and driver version

Your reports directly drive which ops get hardened in 3.0.4.

---

## Breaking Changes

None. All 3.0.2 public APIs remain fully backward-compatible.
