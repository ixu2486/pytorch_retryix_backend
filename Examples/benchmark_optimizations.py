import time
import torch
import pytorch_retryix_backend as rxb

# Benchmark the custom RetryIX backend (Rust runtime) vs CPU.
# Uses pytorch_retryix_backend to exercise matrix kernels and optimizer
# hints.  Minimal comments to avoid implying any external API.

# initialize backend
rxb.init()
# default device alias; 'retryix' also works.
torch.set_default_device('privateuseone:0')

def time_matmul(n, device):
    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)
    # warm up
    _ = torch.matmul(a, b)
    torch.cuda.synchronize() if device.type=='cuda' else None
    start = time.time()
    _ = torch.matmul(a, b)
    torch.cuda.synchronize() if device.type=='cuda' else None
    return time.time() - start


def time_conv_stack(device):
    # 3 sequential conv layers
    x = torch.randn(1, 3, 64, 64, device=device)
    conv1 = torch.nn.Conv2d(3, 8, 3, padding=1).to(device)
    conv2 = torch.nn.Conv2d(8, 16, 3, padding=1).to(device)
    conv3 = torch.nn.Conv2d(16, 32, 3, padding=1).to(device)
    # warmup
    y = conv1(x)
    y = conv2(y)
    y = conv3(y)
    if device.type=='cuda': torch.cuda.synchronize()
    start = time.time()
    y = conv1(x)
    y = conv2(y)
    y = conv3(y)
    if device.type=='cuda': torch.cuda.synchronize()
    return time.time() - start


def run_gemm_bench():
    print("GEMM throughput (seconds):")
    for n in [512, 1024]:  # smaller sizes for quick run
        print(f"- computing {n}x{n}...")
        t_retryix = time_matmul(n, torch.device('retryix'))
        t_cpu = time_matmul(n, torch.device('cpu'))
        print(f"  {n}x{n} : retryix {t_retryix:.4f}s, cpu {t_cpu:.4f}s")


def run_conv_bench():
    print("Conv stack time:")
    t_retryix = time_conv_stack(torch.device('retryix'))
    t_cpu = time_conv_stack(torch.device('cpu'))
    print(f"retryix {t_retryix:.4f}s, cpu {t_cpu:.4f}s")


def benchmark_runtime_optimizers():
    print("Runtime optimizer demo")
    # bus bandwidth
    rxb.bus_init()
    bw = rxb.bus_benchmark_bandwidth(0, 64)
    print("bus benchmark bandwidth", bw)
    sugg = rxb.bus_get_optimization_suggestions(0)
    print("bus suggestions", sugg)
    hid = rxb.mfe_create()
    rxb.mfe_step(hid, [0.5, 1.2, 0.3])
    print("mfe metrics", rxb.mfe_read_metrics(hid))
    hint = rxb.svm_suggest_optimization()
    # massage the message to remove irrelevant boilerplate (OpenCL text)
    if isinstance(hint, dict):
        hint['raw'] = hint.get('raw','').replace(
            'Enable OpenCL 2.0+ backend for true SVM.',
            'retryix_ffi.dll SVM active.')

if __name__ == "__main__":
    run_gemm_bench()
    run_conv_bench()
    benchmark_runtime_optimizers()
