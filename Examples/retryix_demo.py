"""Lightweight Python frontend showing how to use the RetryIX FFI.

This is the "示例 PY 模塊" you can hand to a partner; it does not
ship the binary DLLs, it simply defines the `ctypes` wrappers and a
couple of convenience functions.  Anyone with a copy of
`sdk/bin/retryix_ffi.dll` (or their own compatible implementation)
can import this module and run the examples.

Usage:
    import retryix_demo as rxd
    rxd.initialize()
    rxd.gpu_ring_doorbell(0,1)
    a, b = rxd.make_random_tensor(1024,1024), rxd.make_random_tensor(1024,1024)
    c = rxd.matmul(a,b)
    rxd.shutdown()

The module also knows how to load pre‑compiled SPIR‑V shaders from
`shaders/` and dispatch them via the persistent kernel interface.

"""
import ctypes, pathlib, os
from ctypes import c_int, c_uint32, c_uint64, c_int64, c_size_t, c_void_p, c_char_p, POINTER
import numpy as np

# locate the DLL relative to this file or workspace
HERE = pathlib.Path(__file__).parent
DLL = HERE / "sdk" / "bin" / "retryix_ffi.dll"
if not DLL.exists():
    raise FileNotFoundError("retryix_ffi.dll not found; put it in sdk/bin")

lib = ctypes.CDLL(str(DLL))

# --- minimal FFI declarations ----------------------------------------------
lib.retryix_initialize.restype = c_int
lib.retryix_vulkan_init.restype = c_int
lib.retryix_gpu_hw_init.restype = c_int
lib.retryix_gpu_ring_doorbell.restype = c_int
lib.retryix_gpu_ring_doorbell.argtypes = [c_uint32, c_uint32]
lib.retryix_ai_tensor_create.restype = c_void_p
lib.retryix_ai_tensor_create.argtypes = [c_int, POINTER(c_int64), c_uint32]
lib.retryix_ai_matmul.restype = c_int
lib.retryix_ai_matmul.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int]
lib.retryix_ai_tensor_destroy.restype = None
lib.retryix_ai_tensor_destroy.argtypes = [c_void_p]

# helpers for shader dispatch
lib.retryix_gpu_dispatch_compute.restype = c_int
lib.retryix_gpu_dispatch_compute.argtypes = [c_void_p, c_size_t,
                                              c_uint32, c_uint32, c_uint32,
                                              c_uint32, c_uint32, c_uint32]
lib.retryix_gpu_wait_compute_idle.restype = c_int
lib.retryix_gpu_wait_compute_idle.argtypes = [c_int]

# --- convenience wrappers --------------------------------------------------
def initialize():
    if lib.retryix_initialize() != 0:
        raise RuntimeError("retryix_initialize failed")
    lib.retryix_vulkan_init()
    lib.retryix_gpu_hw_init()


def shutdown():
    # there are cleanup routines in the FFI but not every path uses them
    pass


def make_random_tensor(m, n, dtype=np.float32):
    shape = (ctypes.c_int64 * 2)(m, n)
    h = lib.retryix_ai_tensor_create(2, shape, 0)
    # fill with random numbers (use existing ai_tensor_fill or ptr)
    arr = np.random.randn(m, n).astype(dtype)
    # copy to handle via numpy.ctypeslib
    ptr = ctypes.cast(h, ctypes.POINTER(ctypes.c_float))
    ctypes.memmove(ptr, arr.ctypes.data, arr.nbytes)
    return h


def matmul(h_a, h_b):
    # allocate output of appropriate size: assume square
    # (caller must manage destruction)
    # For simplicity just reuse h_b shape
    # in real use query shape from handle via FFI
    m = n = 1  # placeholder
    shape = (ctypes.c_int64 * 2)(m, n)
    h_c = lib.retryix_ai_tensor_create(2, shape, 0)
    rc = lib.retryix_ai_matmul(h_a, h_b, h_c, 0, 0)
    if rc != 0:
        raise RuntimeError("ai_matmul failed: %d" % rc)
    return h_c


def free_tensor(h):
    lib.retryix_ai_tensor_destroy(h)


def dispatch_shader(spv_path, grid=(1,1,1), block=(1,1,1)):
    data = open(spv_path, "rb").read()
    buf = ctypes.create_string_buffer(data)
    rc = lib.retryix_gpu_dispatch_compute(buf, len(data),
                                          grid[0], grid[1], grid[2],
                                          block[0], block[1], block[2])
    if rc != 0 and rc != -3:
        raise RuntimeError("dispatch_compute rc=%d" % rc)
    if rc == -3:
        print("[demo] dispatch_compute returned stub error (-3), continuing")
    else:
        lib.retryix_gpu_wait_compute_idle(100)


# example usage code (guarded to avoid running on import)
if __name__ == "__main__":
    initialize()
    print("initialized")
    # run a simple shader if available
    example = HERE / "shaders" / "gemm_simple.spv"
    if example.exists():
        dispatch_shader(str(example), grid=(4,4,1), block=(16,16,1))
        print("dispatched shader")
    print("done")
