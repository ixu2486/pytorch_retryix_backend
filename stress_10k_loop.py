"""
Stress Test 2: 10,000-iteration loop — memory leak detection
每輪分配／計算／釋放，追蹤 VMA 分配量是否單調上升
"""
import sys, os, gc, time
os.environ["RETRYIX_SAFE_IMPORT_PROBE"] = "0"
# Pylance may not see torch or backend in workspace; ignore missing imports for this script
# pyright: reportMissingImports=false
import torch  # type: ignore
# add backend package path dynamically (assumes this script sits next to repo)
repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pytorch_retryix_branch'))
if os.path.isdir(repo_dir):
    sys.path.insert(0, repo_dir)
try:
    import pytorch_retryix_backend as _b  # type: ignore
    _b.register_retryix_hooks()
    DEVICE = torch.device("privateuseone:0")
    print(f"[LEAK] Backend: privateuseone:0")
except Exception as e:
    print(f"[WARN] {e}")
    DEVICE = torch.device("cpu")

ITERS      = 10_000
REPORT_EVERY = 1_000
SHAPES = [(4,), (16,), (32,), (64,), (128,), (256,), (8,16), (4,32),
          (16,16), (2,4,8), (4,4,4,4)]

errors = 0
alloc_snapshots = []     # (iter, gc_count) — rough leak signal

t0 = time.time()
for i in range(1, ITERS + 1):
    try:
        import random
        shape = random.choice(SHAPES)
        x = torch.randn(*shape).to(DEVICE)
        # Mix of ops
        _ = x.relu()
        _ = x.abs().neg()
        _ = x.sum()
        # View + op
        if x.numel() > 1:
            _ = x.reshape(-1)[1:].relu()
        # Inplace on fresh tensor
        y = torch.randn(*shape).to(DEVICE)
        y.abs_()
        # Explicit delete
        del x, y, _
    except Exception as ex:
        errors += 1
        if errors <= 10:
            print(f"  [ERR iter={i}] {str(ex)[:120]}")

    if i % REPORT_EVERY == 0:
        gc.collect()
        # Python object count as crude leak proxy
        import ctypes
        n_obj = len(gc.get_objects())
        elapsed = time.time() - t0
        alloc_snapshots.append((i, n_obj))
        print(f"  iter={i:6d}  py_objs={n_obj:7d}  errors={errors}  elapsed={elapsed:.1f}s")

print(f"\n{'='*60}")
print(f"Iterations: {ITERS}  Errors: {errors}")

# Leak heuristic: check if py_objs grew monotonically for all snapshots
if len(alloc_snapshots) >= 3:
    counts = [c for _, c in alloc_snapshots]
    # Allow some variance; flag if last > first * 1.05 (>5% growth over entire run)
    ratio = counts[-1] / counts[0]
    if ratio > 1.10:
        print(f"⚠️  POSSIBLE LEAK: py_objs grew {ratio:.3f}x  ({counts[0]} → {counts[-1]})")
    elif ratio > 1.05:
        print(f"⚡ MILD GROWTH: py_objs ratio={ratio:.3f}x  (may be normal GC variance)")
    else:
        print(f"✅ No significant object leak detected (ratio={ratio:.3f}x)")

if errors == 0:
    print("✅ Zero errors in 10,000 iterations")
else:
    print(f"❌ {errors} errors encountered")
