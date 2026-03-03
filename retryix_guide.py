"""
retryix_ffi 底層庫 — 新用戶快速入門指南
========================================

本檔案整合了「不看原始碼就不會知道」的呼叫陷阱，以及每個 API 群組的
正確使用範式。可直接執行驗證，也可作為 ctypes 呼叫的參考手冊。

架構總覽（由上到下）
─────────────────────────────────────────────────────────────────────
  PyTorch 張量 (torch.Tensor)
      ↓  aten 運算子 (PrivateUse1 dispatch)
  pytorch_retryix_backend（Python wrapper，import 為 r）
      ↓  內部 ctypes 呼叫
  retryix_ffi.dll （Rust CDylib，本指南的主題）
      ↓  Crate 依賴
  retryix_ai / retryix_bus / retryix_svm / retryix_memory / retryix_mfe …

三種 FFI 呼叫範式
─────────────────────────────────────────────────────────────────────
  #1  直接回傳值     fn foo() -> c_int
  #2  Output pointer fn foo(size, *mut *mut T) -> c_int   ← 最容易搞錯
  #3  不透明 handle  fn create(*mut *mut c_void) → 拿 handle → destroy
─────────────────────────────────────────────────────────────────────
"""

import ctypes
import sys

import pytorch_retryix_backend as r
import torch

# ─── 載入 DLL 單例 ─────────────────────────────────────────────────────────────
_dll_path = r.pkg_dir + r"\retryix_ffi.dll"
ffi = ctypes.CDLL(_dll_path)

SEP  = "=" * 65
PASS = "\033[32m  ✓ PASS\033[0m"
FAIL = "\033[31m  ✗ FAIL\033[0m"

def section(title: str):
    print(f"\n{SEP}\n  {title}\n{SEP}")

def check(cond: bool, msg: str):
    tag = PASS if cond else FAIL
    print(f"{tag}  {msg}")
    if not cond:
        sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# §1  初始化流程
# ══════════════════════════════════════════════════════════════════════════════
section("§1  系統初始化流程")

# ── 正確步驟 ────────────────────────────────────────────────────────────────

ffi.retryix_initialize.restype  = ctypes.c_int
ffi.retryix_initialize.argtypes = []
rc = ffi.retryix_initialize()
check(rc == 0, f"retryix_initialize() → {rc}  (0=成功)")

# ── 版本查詢（三個 output-pointer 都可為 NULL） ──────────────────────────────

ffi.retryix_get_version.restype  = ctypes.c_char_p   # 回傳靜態 C 字串
ffi.retryix_get_version.argtypes = [
    ctypes.POINTER(ctypes.c_int),   # *major
    ctypes.POINTER(ctypes.c_int),   # *minor
    ctypes.POINTER(ctypes.c_int),   # *patch
]
major, minor, patch = ctypes.c_int(0), ctypes.c_int(0), ctypes.c_int(0)
ver_str = ffi.retryix_get_version(
    ctypes.byref(major), ctypes.byref(minor), ctypes.byref(patch))
print(f"  版本: {ver_str.decode()}  ({major.value}.{minor.value}.{patch.value})")

# ── 錯誤碼解讀（所有 rc < 0 都可用此函數翻譯） ────────────────────────────────

ffi.retryix_get_error_string.restype  = ctypes.c_char_p
ffi.retryix_get_error_string.argtypes = [ctypes.c_int]
for code in [0, -1, -9, -10]:
    desc = ffi.retryix_get_error_string(code)
    print(f"  rc={code:3d} → {desc.decode()}")


# ══════════════════════════════════════════════════════════════════════════════
# §2  AI 張量層（retryix_ai_*）
#
# ┌─ 最常見陷阱 #1：tensor_create 的 shape 參數 ─────────────────────────────┐
# │  函數簽名：                                                              │
# │    fn retryix_ai_tensor_create(                                          │
# │        ndim:      c_int,           ← 維度數，最大 8                      │
# │        shape_ptr: *const i64,      ← ★ 必須是 c_int64 陣列指標           │  
# │        dtype:     u32,             ← 0=Fp32 1=Fp16 2=Int8 3=Int32 4=Bool │
# │    ) -> *mut Tensor                ← NULL 表示失敗                       │
# │                                                                          │
# │  ❌ 錯誤（access violation at 0x2 或 立即崩潰）：                        │
# │     shape_ptr 傳入 c_int(2)  → Rust 把整數值 2 當指標解參考 0x2          │
# │     shape_ptr 傳入 c_void_p  → 指標位址值 >> 8，ndim 超限 → NULL         │
# │                                                                          │
# │  ✅ 正確：先建 c_int64 陣列，再取其指標                                   │
# └──────────────────────────────────────────────────────────────────────────┘
# ══════════════════════════════════════════════════════════════════════════════
section("§2  AI 張量層：tensor_create / fill / add / matmul / relu / softmax / copy / destroy")

# ── 初始化 AI 子系統 ─────────────────────────────────────────────────────────
ffi.retryix_ai_core_cleanup.restype  = ctypes.c_int
ffi.retryix_ai_core_cleanup.argtypes = []
ffi.retryix_ai_core_init.restype     = ctypes.c_int
ffi.retryix_ai_core_init.argtypes    = []
ffi.retryix_ai_core_cleanup()
rc = ffi.retryix_ai_core_init()
check(rc == 0, f"retryix_ai_core_init() → {rc}")

# ── tensor_create 設定 argtypes（一次設定即可） ───────────────────────────────
ffi.retryix_ai_tensor_create.restype  = ctypes.c_void_p
ffi.retryix_ai_tensor_create.argtypes = [
    ctypes.c_int,                     # ndim
    ctypes.POINTER(ctypes.c_int64),   # shape_ptr  ← 一定是 int64，不是 int32
    ctypes.c_uint32,                  # dtype (0=Fp32)
]

# ── 建立張量的正確範式 ────────────────────────────────────────────────────────
# 建立一個 2-D Fp32 張量 [2, 3]
shape_2x3 = (ctypes.c_int64 * 2)(2, 3)           # ← 長度要與 ndim 一致
h_a = ffi.retryix_ai_tensor_create(2, shape_2x3, 0)
check(bool(h_a), f"tensor_create([2,3], Fp32) → {'0x%X' % h_a if h_a else 'NULL'}")

# 建立相同形狀的第二個張量（b）與結果張量（r）
h_b = ffi.retryix_ai_tensor_create(2, (ctypes.c_int64 * 2)(2, 3), 0)
h_r = ffi.retryix_ai_tensor_create(2, (ctypes.c_int64 * 2)(2, 3), 0)
check(bool(h_b) and bool(h_r), "tensor_create h_b / h_r")

# ── tensor_fill ──────────────────────────────────────────────────────────────
ffi.retryix_ai_tensor_fill.restype  = ctypes.c_int
ffi.retryix_ai_tensor_fill.argtypes = [ctypes.c_void_p, ctypes.c_float]
check(ffi.retryix_ai_tensor_fill(h_a, 2.0) == 0, "tensor_fill(h_a, 2.0)")
check(ffi.retryix_ai_tensor_fill(h_b, 3.0) == 0, "tensor_fill(h_b, 3.0)")

# ── 邊界：NULL → -1（NullPtr） ────────────────────────────────────────────────
rc_null = ffi.retryix_ai_tensor_fill(None, 1.0)
check(rc_null == -1, f"tensor_fill(NULL) → {rc_null}  (期望 -1=NullPtr)")

# ── ai_add ───────────────────────────────────────────────────────────────────
ffi.retryix_ai_add.restype  = ctypes.c_int
ffi.retryix_ai_add.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
check(ffi.retryix_ai_add(h_a, h_b, h_r) == 0,
      "ai_add(2.0, 3.0 → r)  r[i] 應=5.0")

# ── ai_relu ───────────────────────────────────────────────────────────────────
# ★ 輸出張量的 shape 和 dtype 必須與輸入完全相同
h_relu = ffi.retryix_ai_tensor_create(2, (ctypes.c_int64 * 2)(2, 3), 0)
ffi.retryix_ai_relu.restype  = ctypes.c_int
ffi.retryix_ai_relu.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
check(ffi.retryix_ai_relu(h_r, h_relu) == 0, "ai_relu  (全正 → 不變)")

# ── ai_matmul ────────────────────────────────────────────────────────────────
# ★ 僅支援 2D Fp32；result shape 必須精確等於 [m, n]，否則 -9=InvalidParameter
# A=[2,3] × B=[3,2] → R=[2,2]
h_bt = ffi.retryix_ai_tensor_create(2, (ctypes.c_int64 * 2)(3, 2), 0)
h_mm = ffi.retryix_ai_tensor_create(2, (ctypes.c_int64 * 2)(2, 2), 0)
ffi.retryix_ai_tensor_fill(h_bt, 1.0)

ffi.retryix_ai_matmul.restype  = ctypes.c_int
ffi.retryix_ai_matmul.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int,    ctypes.c_int,     # transpose_a, transpose_b (0=No 1=Yes)
]
check(ffi.retryix_ai_matmul(h_a, h_bt, h_mm, 0, 0) == 0,
      "ai_matmul [2,3]×[3,2]→[2,2]")

# shape 不匹配 → 應回傳 -9
rc_bad = ffi.retryix_ai_matmul(h_a, h_bt, h_r, 0, 0)  # h_r 是 [2,3] 不是 [2,2]
check(rc_bad == -9, f"ai_matmul shape 不合 → {rc_bad}  (期望 -9=InvalidParameter)")

# ── ai_softmax ───────────────────────────────────────────────────────────────
# softmax 的 numel 必須與 output 相等；dtype 必須是 Fp32
h_sm_in  = ffi.retryix_ai_tensor_create(1, (ctypes.c_int64 * 1)(6), 0)
h_sm_out = ffi.retryix_ai_tensor_create(1, (ctypes.c_int64 * 1)(6), 0)
ffi.retryix_ai_tensor_fill(h_sm_in, 1.0)
ffi.retryix_ai_softmax.restype  = ctypes.c_int
ffi.retryix_ai_softmax.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
check(ffi.retryix_ai_softmax(h_sm_in, h_sm_out) == 0,
      "ai_softmax [1.0×6]  總和應=1.0")

# ── tensor_copy ───────────────────────────────────────────────────────────────
# ★ dst 和 src 的 dtype + size_bytes 必須完全一致
h_cp = ffi.retryix_ai_tensor_create(1, (ctypes.c_int64 * 1)(6), 0)
ffi.retryix_ai_tensor_copy.restype  = ctypes.c_int
ffi.retryix_ai_tensor_copy.argtypes = [ctypes.c_void_p, ctypes.c_void_p]  # (dst, src)
check(ffi.retryix_ai_tensor_copy(h_cp, h_sm_out) == 0, "tensor_copy(dst, src)")

# ── tensor_destroy（每個 handle 只能 destroy 一次） ───────────────────────────
ffi.retryix_ai_tensor_destroy.restype  = None
ffi.retryix_ai_tensor_destroy.argtypes = [ctypes.c_void_p]
for h in [h_a, h_b, h_r, h_relu, h_bt, h_mm, h_sm_in, h_sm_out, h_cp]:
    ffi.retryix_ai_tensor_destroy(h)
print("  tensor_destroy × 9 → ok")

ffi.retryix_ai_core_cleanup()
print("  ai_core_cleanup → ok")


# ══════════════════════════════════════════════════════════════════════════════
# §3  記憶體管理層
#
# ★ 兩套 API，行為完全不同，名稱前綴也不同：
#
#   retryix_mem_*     ← 低階直接分配（類似 malloc/free），「直接回傳指標」
#     retryix_mem_alloc(size)        -> *mut u8   (直接回傳，NULL=失敗)
#     retryix_mem_free(ptr, size)    -> void
#
#   retryix_memory_*  ← 高階追蹤型分配，「output-pointer 模式」
#     retryix_memory_alloc(size, *mut *mut u8)  -> c_int  ← ★ 不回傳指標！
#     retryix_memory_free(ptr, size)            -> c_int
#     retryix_memory_validate()                 -> c_int  ← ★ 無參數！
#     retryix_memory_print_stats()              -> void
#     retryix_memory_cleanup()                  -> c_int
# ══════════════════════════════════════════════════════════════════════════════
section("§3  記憶體管理層：retryix_mem_* vs retryix_memory_*")

# ── 低階：直接回傳指標 ─────────────────────────────────────────────────────────
ffi.retryix_mem_alloc.restype  = ctypes.c_void_p
ffi.retryix_mem_alloc.argtypes = [ctypes.c_size_t]
ffi.retryix_mem_free.restype   = None
ffi.retryix_mem_free.argtypes  = [ctypes.c_void_p, ctypes.c_size_t]

ptr_low = ffi.retryix_mem_alloc(128)
check(bool(ptr_low), f"retryix_mem_alloc(128) → {'0x%X' % ptr_low if ptr_low else 'NULL'}")
ffi.retryix_mem_free(ptr_low, 128)
print("  retryix_mem_free(ptr, 128) → ok")

# ── 高階：output-pointer 模式 ──────────────────────────────────────────────────
ffi.retryix_memory_alloc.restype  = ctypes.c_int
ffi.retryix_memory_alloc.argtypes = [ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)]
ffi.retryix_memory_free.restype   = ctypes.c_int
ffi.retryix_memory_free.argtypes  = [ctypes.c_void_p, ctypes.c_size_t]

out_ptr = ctypes.c_void_p(0)
rc = ffi.retryix_memory_alloc(256, ctypes.byref(out_ptr))
ptr_hi = out_ptr.value
check(rc == 0 and bool(ptr_hi),
      f"retryix_memory_alloc(256, &out) rc={rc}  ptr={'0x%X' % ptr_hi if ptr_hi else 'NULL'}")

rc = ffi.retryix_memory_free(ptr_hi, 256)
check(rc == 0, f"retryix_memory_free(ptr, 256) → {rc}")

# ── validate / print_stats / cleanup（無參數） ────────────────────────────────
ffi.retryix_memory_validate.restype  = ctypes.c_int
ffi.retryix_memory_validate.argtypes = []          # ★ 無參數！
rc = ffi.retryix_memory_validate()
check(rc == 0, f"retryix_memory_validate() → {rc}  (0=健康)")

ffi.retryix_memory_print_stats.restype  = None
ffi.retryix_memory_print_stats.argtypes = []
ffi.retryix_memory_print_stats()
print("  retryix_memory_print_stats() → ok")

ffi.retryix_memory_cleanup.restype  = ctypes.c_int
ffi.retryix_memory_cleanup.argtypes = []
rc = ffi.retryix_memory_cleanup()
check(rc == 0, f"retryix_memory_cleanup() → {rc}")


# ══════════════════════════════════════════════════════════════════════════════
# §4  SVM 上下文（retryix_svm_*）
#
# ★ SvmContext 是不透明 handle（範式 #3）：
#     create_context(..., **out_context) → 取得 *SvmContext
#     svm_alloc(ctx, size, flags, **out_ptr)  ← 雙層 output-pointer
#     svm_free(ctx, ptr)
#     destroy_context(ctx)
#
# ★ flags 為位元組合（SvmFlags）：
#     READ_WRITE = 0x01   WRITE_ONLY = 0x02   READ_ONLY = 0x04
#     FINE_GRAIN = 0x08   ATOMICS    = 0x10
# ══════════════════════════════════════════════════════════════════════════════
section("§4  SVM 上下文：create / alloc / free / destroy")

ffi.retryix_svm_create_context.restype  = ctypes.c_int
ffi.retryix_svm_create_context.argtypes = [
    ctypes.c_void_p,                    # cl_context  (可傳 NULL)
    ctypes.c_void_p,                    # device      (可傳 NULL)
    ctypes.c_void_p,                    # config      (可傳 NULL)
    ctypes.POINTER(ctypes.c_void_p),    # **out_context  ← output-pointer
]
out_ctx = ctypes.c_void_p(0)
rc = ffi.retryix_svm_create_context(None, None, None, ctypes.byref(out_ctx))
svm_ctx = out_ctx.value
check(rc == 0 and bool(svm_ctx),
      f"svm_create_context() rc={rc}  ctx={'0x%X' % svm_ctx if svm_ctx else 'NULL'}")

# svm_alloc：ctx + size + flags + **out_ptr
ffi.retryix_svm_alloc.restype  = ctypes.c_int
ffi.retryix_svm_alloc.argtypes = [
    ctypes.c_void_p,                    # context
    ctypes.c_size_t,                    # size
    ctypes.c_uint32,                    # flags  (0x01 = READ_WRITE)
    ctypes.POINTER(ctypes.c_void_p),    # **out_ptr
]
out_buf = ctypes.c_void_p(0)
rc = ffi.retryix_svm_alloc(svm_ctx, 4096, 0x01, ctypes.byref(out_buf))
buf = out_buf.value
check(rc == 0 and bool(buf),
      f"svm_alloc(4096, READ_WRITE) rc={rc}  ptr={'0x%X' % buf if buf else 'NULL'}")

ffi.retryix_svm_free.restype  = ctypes.c_int
ffi.retryix_svm_free.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
rc = ffi.retryix_svm_free(svm_ctx, buf)
check(rc == 0, f"svm_free() → {rc}")

ffi.retryix_svm_destroy_context.restype  = ctypes.c_int
ffi.retryix_svm_destroy_context.argtypes = [ctypes.c_void_p]
rc = ffi.retryix_svm_destroy_context(svm_ctx)
check(rc == 0, f"svm_destroy_context() → {rc}")


# ══════════════════════════════════════════════════════════════════════════════
# §5  Bus PCIe 控制器（retryix_bus_*）
#
# ★ 「取得頻寬」的函數都是 output-pointer 模式，不是直接回傳 float：
#     retryix_bus_get_controller_count()                    → c_int（直接回傳）
#     retryix_bus_get_controller_bandwidth(id, *out_gbps)   → c_int
#     retryix_bus_benchmark_bandwidth(id, mb, *out_gbps)    → c_int
#     retryix_bus_monitor_status(id, *peak, *util, *bytes)  → c_int
#
# ★ retryix_bus_get_optimal_bandwidth(*out_gbps) 也是 output-pointer：
#     不要試圖從回傳值讀頻寬，它只是 rc！
# ══════════════════════════════════════════════════════════════════════════════
section("§5  Bus 控制器：enumerate / bandwidth / benchmark")

ffi.retryix_bus_scheduler_init.restype  = ctypes.c_int
ffi.retryix_bus_scheduler_init.argtypes = []
ffi.retryix_bus_scheduler_init()

# controller_count：直接回傳整數（範式 #1）
ffi.retryix_bus_get_controller_count.restype  = ctypes.c_int
ffi.retryix_bus_get_controller_count.argtypes = []
n = ffi.retryix_bus_get_controller_count()
print(f"  controller_count = {n}")

# bandwidth：output-pointer（範式 #2）
ffi.retryix_bus_get_controller_bandwidth.restype  = ctypes.c_int
ffi.retryix_bus_get_controller_bandwidth.argtypes = [
    ctypes.c_int, ctypes.POINTER(ctypes.c_float)]    # (id, *out_gbps)
bw = ctypes.c_float(0.0)
rc = ffi.retryix_bus_get_controller_bandwidth(0, ctypes.byref(bw))
check(rc == 0, f"get_controller_bandwidth(0) → {bw.value:.2f} GB/s  rc={rc}")

# benchmark：output-pointer（範式 #2）
ffi.retryix_bus_benchmark_bandwidth.restype  = ctypes.c_int
ffi.retryix_bus_benchmark_bandwidth.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]  # (id, mb, *out_gbps)
bench = ctypes.c_float(0.0)
rc = ffi.retryix_bus_benchmark_bandwidth(0, 16, ctypes.byref(bench))
check(rc == 0, f"benchmark_bandwidth(0, 16 MB) → {bench.value:.2f} GB/s  rc={rc}")

# monitor_status：多個 output-pointer，可為 NULL
ffi.retryix_bus_monitor_status.restype  = ctypes.c_int
ffi.retryix_bus_monitor_status.argtypes = [
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),   # *peak_gbps
    ctypes.POINTER(ctypes.c_float),   # *util_pct
    ctypes.POINTER(ctypes.c_uint64),  # *bytes  (可傳 NULL)
]
peak, util = ctypes.c_float(0.0), ctypes.c_float(0.0)
rc = ffi.retryix_bus_monitor_status(0, ctypes.byref(peak), ctypes.byref(util), None)
check(rc == 0, f"monitor_status(0)  peak={peak.value:.2f} GB/s  util={util.value:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# §6  MFE 句柄（retryix_mfe_*）——不透明 handle 完整生命週期
#
# 生命週期：
#   create(*out_handle)  → handle 非 0
#   step(handle, *data, n)
#   set_decay(handle, decay)
#   read_metrics(handle, *buf, max_n, *written)
#   run_experiment(handle)
#   destroy(handle)
#
# ★ 所有操作都透過 c_void_p handle，不要試圖解參考它
# ══════════════════════════════════════════════════════════════════════════════
section("§6  MFE 不透明句柄：create / step / read_metrics / destroy")

ffi.retryix_mfe_create.restype  = ctypes.c_int
ffi.retryix_mfe_create.argtypes = [ctypes.POINTER(ctypes.c_void_p)]   # **out_handle
out_h = ctypes.c_void_p(0)
rc = ffi.retryix_mfe_create(ctypes.byref(out_h))
mfe_h = out_h.value
check(rc == 0 and bool(mfe_h),
      f"mfe_create() rc={rc}  handle={'0x%X' % mfe_h if mfe_h else 'NULL'}")

# step：傳入 float32 特徵向量
ffi.retryix_mfe_step.restype  = ctypes.c_int
ffi.retryix_mfe_step.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
features = (ctypes.c_float * 4)(0.1, 0.2, 0.3, 0.4)
rc = ffi.retryix_mfe_step(mfe_h, features, 4)
check(rc == 0, f"mfe_step(4 features) → {rc}")

# set_decay（c_double，不是 c_float）
ffi.retryix_mfe_set_decay.restype  = ctypes.c_int
ffi.retryix_mfe_set_decay.argtypes = [ctypes.c_void_p, ctypes.c_double]
rc = ffi.retryix_mfe_set_decay(mfe_h, 0.95)
check(rc == 0, f"mfe_set_decay(0.95) → {rc}")

# read_metrics：output array + written 計數
ffi.retryix_mfe_read_metrics.restype  = ctypes.c_int
ffi.retryix_mfe_read_metrics.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),  # *metrics buf
    ctypes.c_int,                     # max_metrics
]
metrics_buf = (ctypes.c_double * 8)()
rc = ffi.retryix_mfe_read_metrics(mfe_h, metrics_buf, 8)
check(rc == 0, f"mfe_read_metrics() → rc={rc}  metrics[0]={metrics_buf[0]:.4f}")

# destroy
ffi.retryix_mfe_destroy.restype  = ctypes.c_int
ffi.retryix_mfe_destroy.argtypes = [ctypes.c_void_p]
rc = ffi.retryix_mfe_destroy(mfe_h)
check(rc == 0, f"mfe_destroy() → {rc}")


# ══════════════════════════════════════════════════════════════════════════════
# §7  Python wrapper 層的已知限制
#
# pytorch_retryix_backend（import r）有提供 r.add / r.matmul / r.randn …
# 但這些 Python wrappers 內部用的是 _*_impl 函數指標，
# 在不帶完整 GPU compute 初始化時這些指標是 None。
#
# ★ 正確用法：讓 PyTorch PrivateUse1 dispatch 處理，而非直接呼叫 r.* 算子
# ══════════════════════════════════════════════════════════════════════════════
section("§7  Python wrapper 層：r.* 算子的使用限制")

r.register_retryix_hooks()
if r.is_retryix_available():
    device = torch.device("privateuseone:0")
else:
    device = torch.device("cpu")
print(f"  裝置：{device}")

# ❌ 不要這樣：r.add(a, b)、r.matmul(a, b) — _impl 為 None 時會 TypeError
# ✅ 正確做法：用標準 PyTorch，指定 device= 讓 dispatch 自動走 RetryIX 路徑

a = torch.randn(4, 4, device=device)
b = torch.randn(4, 4, device=device)

c_add    = a + b                          # aten::add → PrivateUse1 dispatch
c_mm     = torch.matmul(a, b)            # aten::mm  → PrivateUse1 dispatch
c_relu   = torch.relu(a)                 # aten::relu
randn_ok = torch.randn(8, 8, device=device)  # 不是 r.randn(8, 8)

check(c_add.shape == (4, 4),    f"a + b → {c_add.shape}")
check(c_mm.shape  == (4, 4),    f"matmul → {c_mm.shape}")
check(c_relu.shape == (4, 4),   f"relu → {c_relu.shape}")
check(randn_ok.shape == (8, 8), f"randn(device=) → {randn_ok.shape}")

# ── r.* 函數安全呼叫的幾個例外（確實可用） ───────────────────────────────────
print(f"\n  可正常使用的 r.* 函數：")
print(f"    r.memory_allocated()  = {r.memory_allocated()}")
print(f"    r.memory_reserved()   = {r.memory_reserved()}")
budget = r.get_budget()
print(f"    r.get_budget()        = dict（total_vram={budget.get('total_vram', '?')} bytes）")
print(f"    r.device_count()      = {r.device_count()}")
print(f"    r.is_retryix_available() = {r.is_retryix_available()}")


# ══════════════════════════════════════════════════════════════════════════════
# §8  錯誤碼速查表
# ══════════════════════════════════════════════════════════════════════════════
section("§8  RetryixResult 錯誤碼速查")

error_table = {
    0: "Success",
    -1: "NullPtr        — 傳入 NULL 指標（最常見）",
    -2: "NoDevice       — 找不到 GPU 裝置",
    -3: "NoPlatform     — 找不到 Vulkan/OpenCL 平台",
    -4: "OpenCl         — OpenCL 運行期錯誤",
    -5: "BufferTooSmall — 輸出緩衝區不夠大",
    -6: "FileIo         — 檔案讀寫錯誤",
    -7: "SvmNotSupported — 此裝置不支援 SVM",
    -8: "InvalidDevice  — 裝置 ID 超出範圍",
    -9: "InvalidParameter — shape 不符 / dtype 不符 / ndim>8",
   -10: "OutOfMemory    — 記憶體不足",
   -11: "KernelCompilation — SPIR-V / 著色器編譯失敗",
   -12: "AtomicNotSupported — 此裝置不支援 atomic 操作",
   -99: "Unknown        — 未歸類錯誤",
}
for code, desc in error_table.items():
    print(f"  {code:4d}  {desc}")


# ══════════════════════════════════════════════════════════════════════════════
# §9  三種 FFI 範式的 ctypes 模板（複製即用）
# ══════════════════════════════════════════════════════════════════════════════
section("§9  ctypes 呼叫模板（複製即用）")

print("""
  ── 範式 #1：直接回傳值 ─────────────────────────────────────────────
  ffi.foo.restype  = ctypes.c_int
  ffi.foo.argtypes = [ctypes.c_int]
  rc = ffi.foo(42)                   # rc < 0 表示失敗

  ── 範式 #2：output-pointer ─────────────────────────────────────────
  ffi.bar.restype  = ctypes.c_int
  ffi.bar.argtypes = [ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)]
  out = ctypes.c_void_p(0)
  rc  = ffi.bar(256, ctypes.byref(out))
  ptr = out.value                    # 實際指標在 out.value，不是 rc！

  ── 範式 #3：不透明 handle ────────────────────────────────────────────
  # create
  ffi.thing_create.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
  out_h = ctypes.c_void_p(0)
  ffi.thing_create(ctypes.byref(out_h))
  handle = out_h.value

  # 使用
  ffi.thing_do.argtypes = [ctypes.c_void_p, ...]
  ffi.thing_do(handle, ...)

  # destroy（必須配對呼叫，否則記憶體洩漏）
  ffi.thing_destroy.argtypes = [ctypes.c_void_p]
  ffi.thing_destroy(handle)

  ── shape 陣列（tensor_create 專用） ─────────────────────────────────
  # ✅ 正確：c_int64 陣列，不是 c_int
  shape = (ctypes.c_int64 * ndim)(dim0, dim1, ...)
  ffi.retryix_ai_tensor_create(ndim, shape, dtype_u32)

  # dtype 對應：Fp32=0  Fp16=1  Int8=2  Int32=3  Bool=4
""")

# ── 系統清理 ────────────────────────────────────────────────────────────────
ffi.retryix_cleanup.restype  = None
ffi.retryix_cleanup.argtypes = []
ffi.retryix_cleanup()

print(f"\n{SEP}")
print("  retryix_guide.py 全部通過")
print(
    "  索引：§1 初始化  §2 AI張量  §3 記憶體  §4 SVM\n"
    "        §5 Bus    §6 MFE    §7 Python層限制\n"
    "        §8 錯誤碼  §9 ctypes模板"
)
print(SEP)
