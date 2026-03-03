"""
retryix_ffi_safe.py
====================
Safe Python wrappers for retryix_ffi.dll.
安全 Python 包裝層，對應 retryix_ffi.dll。

每個 wrapper 在呼叫 C ABI **之前** 先驗證引數；
誤用時拋出明確的 Python 例外，不會直接 crash。

Usage / 使用方式:
    from retryix_ffi_safe import RetryixAI, RetryixMemory, bus_controller_count

    with RetryixAI() as ai:
        h = ai.tensor_create([4, 8])          # Fp32 by default
        ai.tensor_fill(h, 1.0)
        h2 = ai.tensor_create([4, 8])
        ai.tensor_fill(h2, 2.0)
        out = ai.tensor_create([4, 8])
        ai.add(h, h2, out)
        ai.tensor_destroy(h, h2, out)

    with RetryixMemory() as mem:
        ptr = mem.alloc(256)
        mem.free(ptr, 256)

    n = bus_controller_count()  # returns int directly
"""

import ctypes
import pathlib
import glob as _glob

# ──────────────────────────────────────────────────────────────
# 錯誤碼對照  Error code mapping
# ──────────────────────────────────────────────────────────────
_ERRCODES = {
    0:   "Success",
    -1:  "NullPtr — a required pointer argument is NULL",
    -2:  "InsufficientBuffer",
    -3:  "HardwareAccessError — invalid controller_id or hardware fault",
    -4:  "PowerLimitationActive",
    -5:  "ThermalThrottling",
    -9:  "InvalidParameter — shape/dtype/numel mismatch, or ndim out of range",
    -10: "OutOfMemory",
}

# dtype 常數  dtype constants
DTYPE_FP32  = 0
DTYPE_FP16  = 1
DTYPE_INT8  = 2
DTYPE_INT32 = 3
DTYPE_BOOL  = 4
_DTYPE_NAMES = {0: "Fp32", 1: "Fp16", 2: "Int8", 3: "Int32", 4: "Bool"}


def _describe(rc: int) -> str:
    return _ERRCODES.get(rc, f"Unknown error code {rc}")


class RetryixFFIError(RuntimeError):
    """Raised when the FFI function returns a non-zero error code.
    FFI 函數回傳非零錯誤碼時拋出。"""
    def __init__(self, fn_name: str, rc: int):
        super().__init__(
            f"{fn_name}() returned {rc}: {_describe(rc)}"
        )
        self.rc = rc


# ──────────────────────────────────────────────────────────────
# DLL 載入  DLL loading
# ──────────────────────────────────────────────────────────────

def _load_dll() -> ctypes.CDLL:
    """Locate and load retryix_ffi.dll from the pytorch_retryix_backend package."""
    try:
        import pytorch_retryix_backend as _r
        pkg_dir = pathlib.Path(_r.__file__).parent
    except ImportError:
        pkg_dir = pathlib.Path(".")

    candidates = list(pkg_dir.glob("retryix_ffi.dll"))
    if not candidates:
        candidates = list(pathlib.Path(".").glob("**/retryix_ffi.dll"))
    if not candidates:
        raise FileNotFoundError(
            "retryix_ffi.dll not found.  "
            "Make sure pytorch_retryix_backend is installed."
        )
    return ctypes.CDLL(str(candidates[0]))


# ──────────────────────────────────────────────────────────────
# 引數驗證輔助  Argument validation helpers
# ──────────────────────────────────────────────────────────────

def _require_handle(handle, name: str = "handle"):
    """Raise TypeError if handle is None/0 (NULL)."""
    if not handle:
        raise TypeError(
            f"{name} is NULL.  "
            "Did you forget to call tensor_create(), or was it already destroyed?"
        )


def _require_shape(shape, name: str = "shape"):
    """Validate that shape is a non-empty sequence of positive integers."""
    if not hasattr(shape, "__len__"):
        raise TypeError(
            f"{name} must be a list or tuple of ints, got {type(shape).__name__}.\n"
            "  ❌ tensor_create(2, 2, 0)      — plain int, Rust treats it as pointer → crash\n"
            "  ✅ tensor_create([2, 3])        — list of ints, safe"
        )
    if len(shape) == 0:
        raise ValueError(f"{name} must not be empty.")
    if len(shape) > 8:
        raise ValueError(
            f"{name} has {len(shape)} dimensions, max allowed is 8."
        )
    for i, d in enumerate(shape):
        if not isinstance(d, int) or d <= 0:
            raise ValueError(
                f"{name}[{i}] = {d!r} is not a positive integer."
            )


def _require_dtype(dtype: int, name: str = "dtype"):
    if dtype not in _DTYPE_NAMES:
        raise ValueError(
            f"{name} = {dtype} is not a valid dtype.  "
            f"Valid values: {list(_DTYPE_NAMES.items())}"
        )


def _check_rc(fn_name: str, rc: int):
    """Raise RetryixFFIError if rc != 0."""
    if rc != 0:
        raise RetryixFFIError(fn_name, rc)


# ──────────────────────────────────────────────────────────────
# RetryixAI  —  AI tensor API
# ──────────────────────────────────────────────────────────────

class RetryixAI:
    """
    Safe wrapper for retryix_ai_* FFI functions.
    retryix_ai_* FFI 函數的安全包裝。

    Use as a context manager to ensure core_init / core_cleanup are paired:
    建議用 with 敘述確保 init/cleanup 成對呼叫：

        with RetryixAI() as ai:
            h = ai.tensor_create([4, 4])
            ai.tensor_fill(h, 1.0)
            ai.tensor_destroy(h)
    """

    def __init__(self, dll: ctypes.CDLL | None = None):
        self._lib = dll if dll is not None else _load_dll()
        self._setup_sigs()

    def _setup_sigs(self):
        L = self._lib

        L.retryix_ai_core_init.restype  = ctypes.c_int
        L.retryix_ai_core_init.argtypes = []

        L.retryix_ai_core_cleanup.restype  = ctypes.c_int
        L.retryix_ai_core_cleanup.argtypes = []

        L.retryix_ai_tensor_create.restype  = ctypes.c_void_p
        L.retryix_ai_tensor_create.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64),  # shape_ptr — MUST be c_int64 array!
            ctypes.c_uint32,
        ]

        L.retryix_ai_tensor_destroy.restype  = None
        L.retryix_ai_tensor_destroy.argtypes = [ctypes.c_void_p]

        L.retryix_ai_tensor_fill.restype  = ctypes.c_int
        L.retryix_ai_tensor_fill.argtypes = [ctypes.c_void_p, ctypes.c_float]

        L.retryix_ai_tensor_copy.restype  = ctypes.c_int
        L.retryix_ai_tensor_copy.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

        L.retryix_ai_add.restype  = ctypes.c_int
        L.retryix_ai_add.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

        L.retryix_ai_matmul.restype  = ctypes.c_int
        L.retryix_ai_matmul.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int,
        ]

        L.retryix_ai_relu.restype  = ctypes.c_int
        L.retryix_ai_relu.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

        L.retryix_ai_softmax.restype  = ctypes.c_int
        L.retryix_ai_softmax.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

    # ── lifecycle ────────────────────────────────────────────

    def init(self):
        """Call retryix_ai_core_init().  Raises on failure."""
        _check_rc("retryix_ai_core_init", self._lib.retryix_ai_core_init())

    def cleanup(self):
        """Call retryix_ai_core_cleanup()."""
        _check_rc("retryix_ai_core_cleanup", self._lib.retryix_ai_core_cleanup())

    def __enter__(self):
        self.init()
        return self

    def __exit__(self, *_):
        try:
            self.cleanup()
        except RetryixFFIError:
            pass
        return False

    # ── tensor create / destroy ──────────────────────────────

    def tensor_create(self, shape: list[int], dtype: int = DTYPE_FP32) -> int:
        """
        Allocate a tensor and return its opaque handle (int address).

        Parameters
        ----------
        shape : list[int]
            Dimensions, e.g. [4, 8].  Max 8 dims, each > 0.
            ⚠  Must be a list/tuple — do NOT pass a plain int.
        dtype : int
            0=Fp32 (default), 1=Fp16, 2=Int8, 3=Int32, 4=Bool.

        Returns
        -------
        int  Opaque tensor handle (non-zero on success).

        Raises
        ------
        TypeError   if shape is not a sequence
        ValueError  if shape is empty, ndim > 8, or dtype invalid
        MemoryError if the underlying allocator returns NULL
        """
        _require_shape(shape)
        _require_dtype(dtype)

        ndim = len(shape)
        arr  = (ctypes.c_int64 * ndim)(*shape)
        h    = self._lib.retryix_ai_tensor_create(ndim, arr, ctypes.c_uint32(dtype))
        if not h:
            raise MemoryError(
                f"retryix_ai_tensor_create(ndim={ndim}, shape={shape}, "
                f"dtype={_DTYPE_NAMES[dtype]}) returned NULL.  "
                "Possible causes: OOM, ndim > 8, or null shape_ptr."
            )
        return h

    def tensor_destroy(self, *handles: int):
        """
        Destroy one or more tensor handles.

        Passing None or 0 is safe (no-op per the Rust implementation).
        ⚠  Never call ctypes.free() on a handle directly.
        """
        for h in handles:
            self._lib.retryix_ai_tensor_destroy(h)

    # ── data operations ──────────────────────────────────────

    def tensor_fill(self, handle: int, value: float):
        """
        Fill all elements with a constant float value.

        Raises
        ------
        TypeError          if handle is NULL
        RetryixFFIError    if the underlying call fails
        """
        _require_handle(handle, "handle")
        _check_rc("retryix_ai_tensor_fill",
                  self._lib.retryix_ai_tensor_fill(handle, ctypes.c_float(value)))

    def tensor_copy(self, dst: int, src: int):
        """
        Copy data from src into dst.

        Both tensors must have identical dtype and size_bytes.
        ⚠  dst comes FIRST (same order as memcpy).

        Raises
        ------
        TypeError          if either handle is NULL
        RetryixFFIError    rc=-9 if dtype or size_bytes mismatch
        """
        _require_handle(dst, "dst")
        _require_handle(src, "src")
        _check_rc("retryix_ai_tensor_copy",
                  self._lib.retryix_ai_tensor_copy(dst, src))

    def add(self, a: int, b: int, result: int):
        """
        Element-wise addition: result[i] = a[i] + b[i]  (Fp32 only).

        All three handles must refer to tensors with identical dtype and numel.

        Raises
        ------
        TypeError          if any handle is NULL
        RetryixFFIError    rc=-9 if dtype or numel mismatch
        """
        _require_handle(a, "a")
        _require_handle(b, "b")
        _require_handle(result, "result")
        _check_rc("retryix_ai_add",
                  self._lib.retryix_ai_add(a, b, result))

    def matmul(self, a: int, b: int, result: int,
               transpose_a: bool = False, transpose_b: bool = False):
        """
        2-D matrix multiplication: result = A @ B  (Fp32, 2-D only).

        ⚠  result must be pre-allocated with the EXACT output shape [m, n].
           If the shape does not match, rc=-9 (InvalidParameter) is raised.
        ⚠  Only 2-D Fp32 tensors are supported.

        Raises
        ------
        TypeError          if any handle is NULL
        RetryixFFIError    rc=-9 for 3D+ tensors, non-Fp32, or shape mismatch
        """
        _require_handle(a, "a")
        _require_handle(b, "b")
        _require_handle(result, "result")
        _check_rc("retryix_ai_matmul",
                  self._lib.retryix_ai_matmul(
                      a, b, result,
                      ctypes.c_int(int(transpose_a)),
                      ctypes.c_int(int(transpose_b)),
                  ))

    def relu(self, inp: int, out: int):
        """
        ReLU: out[i] = max(0, inp[i])  (Fp32).

        Raises
        ------
        TypeError          if either handle is NULL
        RetryixFFIError    rc=-9 if dtype or numel mismatch
        """
        _require_handle(inp, "inp")
        _require_handle(out, "out")
        _check_rc("retryix_ai_relu",
                  self._lib.retryix_ai_relu(inp, out))

    def softmax(self, inp: int, out: int):
        """
        Softmax over the flat buffer (treats tensor as 1-D, Fp32 only).

        inp.numel() must equal out.numel().

        Raises
        ------
        TypeError          if either handle is NULL
        RetryixFFIError    rc=-9 if dtype != Fp32 or numel mismatch
        """
        _require_handle(inp, "inp")
        _require_handle(out, "out")
        _check_rc("retryix_ai_softmax",
                  self._lib.retryix_ai_softmax(inp, out))


# ──────────────────────────────────────────────────────────────
# RetryixMemory  —  high-level allocator API
# ──────────────────────────────────────────────────────────────

class RetryixMemory:
    """
    Safe wrapper for retryix_memory_* FFI functions.
    retryix_memory_* FFI 函數的安全包裝。

    Use as a context manager:
    建議用 with 敘述：

        with RetryixMemory() as mem:
            ptr = mem.alloc(256)
            mem.free(ptr, 256)
    """

    def __init__(self, dll: ctypes.CDLL | None = None):
        self._lib = dll if dll is not None else _load_dll()
        self._setup_sigs()

    def _setup_sigs(self):
        L = self._lib

        #  alloc(size, *out_ptr) -> c_int   ← output-pointer pattern
        L.retryix_memory_alloc.restype  = ctypes.c_int
        L.retryix_memory_alloc.argtypes = [
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p),  # out_ptr
        ]

        L.retryix_memory_free.restype  = ctypes.c_int
        L.retryix_memory_free.argtypes = [ctypes.c_void_p, ctypes.c_size_t]

        #  validate() -> c_int   ← ZERO arguments!
        L.retryix_memory_validate.restype  = ctypes.c_int
        L.retryix_memory_validate.argtypes = []

        L.retryix_memory_print_stats.restype  = None
        L.retryix_memory_print_stats.argtypes = []

        L.retryix_memory_cleanup.restype  = ctypes.c_int
        L.retryix_memory_cleanup.argtypes = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        try:
            self.cleanup()
        except RetryixFFIError:
            pass
        return False

    def alloc(self, size: int) -> int:
        """
        Allocate `size` bytes.  Returns the allocated address as int.

        ⚠  Uses the output-pointer pattern internally — the address is NOT
           the return value of the C function.  This wrapper hides that detail.
        ⚠  內部使用 output-pointer 模式，此包裝已隱藏細節，直接回傳位址。

        Parameters
        ----------
        size : int   Number of bytes to allocate.  Must be > 0.

        Returns
        -------
        int  Allocated address (non-zero).

        Raises
        ------
        ValueError          if size <= 0
        RetryixFFIError     rc=-10 if OOM, rc=-1 if internal null ptr
        MemoryError         if the returned pointer is NULL
        """
        if not isinstance(size, int) or size <= 0:
            raise ValueError(
                f"size must be a positive integer, got {size!r}.\n"
                "  ❌ mem.alloc(256, 64)   — two args; 64 treated as write target → crash\n"
                "  ✅ mem.alloc(256)       — one arg, wrapper handles output-pointer"
            )
        out = ctypes.c_void_p(0)
        rc  = self._lib.retryix_memory_alloc(
            ctypes.c_size_t(size), ctypes.byref(out)
        )
        _check_rc("retryix_memory_alloc", rc)
        if not out.value:
            raise MemoryError(
                f"retryix_memory_alloc({size}) succeeded (rc=0) but returned NULL pointer."
            )
        return out.value

    def free(self, ptr: int, size: int):
        """
        Free `size` bytes at `ptr`.

        `size` must match the value passed to alloc().

        Raises
        ------
        TypeError           if ptr is 0/None
        ValueError          if size <= 0
        RetryixFFIError     on non-zero return code
        """
        if not ptr:
            raise TypeError("ptr is NULL — cannot free a NULL pointer.")
        if not isinstance(size, int) or size <= 0:
            raise ValueError(f"size must be a positive integer, got {size!r}.")
        _check_rc("retryix_memory_free",
                  self._lib.retryix_memory_free(
                      ctypes.c_void_p(ptr), ctypes.c_size_t(size)
                  ))

    def validate(self) -> int:
        """
        Validate the allocator state.  Returns 0 if healthy.

        ⚠  The underlying C function takes ZERO arguments.
           This wrapper enforces that — no argument should be passed.
        ⚠  底層 C 函數無參數；此包裝已強制如此，不接受任何引數。

        Raises
        ------
        RetryixFFIError  if allocator is in an inconsistent state
        """
        rc = self._lib.retryix_memory_validate()
        _check_rc("retryix_memory_validate", rc)
        return rc

    def print_stats(self):
        """Print allocator statistics to stdout."""
        self._lib.retryix_memory_print_stats()

    def cleanup(self):
        """Release all resources held by the high-level allocator."""
        _check_rc("retryix_memory_cleanup",
                  self._lib.retryix_memory_cleanup())


# ──────────────────────────────────────────────────────────────
# Bus helpers
# ──────────────────────────────────────────────────────────────

def bus_controller_count(dll: ctypes.CDLL | None = None) -> int:
    """
    Return the number of detected bus controllers.

    ⚠  The C function retryix_bus_get_controller_count() returns the count
       DIRECTLY as its return value, NOT via an output pointer.
       Do not pass any argument to it.
    ⚠  C 函數直接回傳 count，不是 output-pointer 模式，不需傳入任何引數。

    Returns
    -------
    int  Number of controllers (>= 0).
    """
    lib = dll if dll is not None else _load_dll()
    lib.retryix_bus_get_controller_count.restype  = ctypes.c_int
    lib.retryix_bus_get_controller_count.argtypes = []
    return lib.retryix_bus_get_controller_count()


def bus_controller_bandwidth(controller_id: int = 0,
                              dll: ctypes.CDLL | None = None) -> float:
    """
    Return actual bandwidth (GB/s) for `controller_id`.

    Raises
    ------
    RetryixFFIError  rc=-3 if controller_id is invalid
    """
    lib = dll if dll is not None else _load_dll()
    lib.retryix_bus_get_controller_bandwidth.restype  = ctypes.c_int
    lib.retryix_bus_get_controller_bandwidth.argtypes = [
        ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
    out = ctypes.c_float(0.0)
    _check_rc("retryix_bus_get_controller_bandwidth",
              lib.retryix_bus_get_controller_bandwidth(controller_id, ctypes.byref(out)))
    return out.value
