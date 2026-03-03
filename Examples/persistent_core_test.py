import pytorch_retryix_backend as rxb
import torch

# This script exercises the backend's persistent GEMM/elementwise pipelines.
# The idea is to perform the same large matmul several times and observe
# in the initialization log that the "Persistent scratch" buffers are
# allocated only once during rxb.init(). Subsequent matmuls reuse them.

rxb.init()
torch.set_default_device('retryix')

for i in range(3):
    a = torch.randn(1024, 1024, device='retryix')
    b = torch.randn(1024, 1024, device='retryix')
    c = torch.matmul(a, b)
    # force a sync to ensure the operation completes and logs appear
    c.to('cpu')
    print(f"iteration {i} completed")
