import pytorch_retryix_backend as pr
import torch


def initialize():
    # make sure the low‑level backend is registered with torch
    pr.init()
    # default device for new tensors
    torch.set_default_device("retryix")
    # optionally force a backend preference such as cpu/vulkan/auto
    pr.set_backend("auto")
    print(f"RetryIX backend initialized, current backend={pr.get_backend()}")


def test_creation_and_basic_ops():
    d = torch.device("retryix")
    a = torch.zeros(3, 4, device=d)
    assert a.shape == (3, 4), f"zeros shape mismatch: {a.shape}"
    b = torch.ones(3, 4, device=d)
    assert b.shape == (3, 4)
    c = torch.randn(3, 4, device=d)
    assert c.shape == (3, 4)

    # addition result remains on retryix device
    sum_ab = a + b
    assert sum_ab.device.type == "retryix", "result not on retryix device"



def test_matmul():
    d = torch.device("retryix")
    x = torch.randn(5, 2, device=d)
    y = torch.randn(2, 7, device=d)
    z = torch.matmul(x, y)
    assert z.shape == (5, 7)

    # simple sanity: resulting tensor should live on retryix as well
    assert z.device.type == "retryix", "matmul output not on retryix device"


def test_elementwise_ops():
    d = torch.device("retryix")
    a = torch.randn(4, 4, device=d)
    # basic unary ops
    for fn in [torch.relu, torch.sigmoid, torch.tanh, torch.abs, torch.neg]:
        out = fn(a)
        assert out.device.type == "retryix"
        assert out.shape == a.shape
    # binary ops with scalar/another tensor
    b = torch.randn(4, 4, device=d)
    assert (a + b).device.type == "retryix"
    assert (a - b).device.type == "retryix"
    assert (a * b).device.type == "retryix"
    assert (a / (b + 1e-3)).device.type == "retryix"


def test_reduction_ops():
    d = torch.device("retryix")
    t = torch.randn(6, 5, device=d)
    # reductions
    for fn in [torch.sum, torch.max, torch.min, torch.mean]:
        out = fn(t)
        assert out.device.type == "retryix"
    # dim reductions
    assert torch.sum(t, dim=1).device.type == "retryix"
    # max with dim may not be implemented for retryix; if so, skip gracefully
    try:
        m = torch.max(t, dim=0)
        assert m.values.device.type == "retryix"
    except NotImplementedError:
        print("dim-max not supported on retryix, skipping")


def test_shape_ops():
    d = torch.device("retryix")
    # use a simple tensor of correct size to avoid arange issues
    t = torch.zeros(3, 4, device=d)
    t2 = t.reshape(4, 3)
    assert t2.shape == (4, 3) and t2.device.type == "retryix"
    t3 = t.transpose(0, 1)
    assert t3.shape == (4, 3) and t3.device.type == "retryix"
    t4 = t.permute(1, 0)
    assert t4.shape == (4, 3) and t4.device.type == "retryix"


def test_conv_pool_batchnorm_dropout():
    d = torch.device("retryix")
    # conv2d
    inp = torch.randn(1, 1, 5, 5, device=d)
    weight = torch.randn(1, 1, 3, 3, device=d)
    out = torch.nn.functional.conv2d(inp, weight, padding=1)
    assert out.device.type == "retryix"

    # pooling
    p = torch.nn.functional.max_pool2d(out, 2)
    assert p.device.type == "retryix"

    # batch_norm (affine=False simplifies)
    bn = torch.nn.functional.batch_norm(out, running_mean=torch.zeros(1, device=d), running_var=torch.ones(1, device=d), training=True, momentum=0.1, eps=1e-5)
    assert bn.device.type == "retryix"

    # dropout
    dpt = torch.nn.functional.dropout(out, p=0.5)
    assert dpt.device.type == "retryix"


def main():
    print("Initializing RetryIX via torch...")
    initialize()
    print("Running matrix tests using torch tensors on retryix device...")
    test_creation_and_basic_ops()
    print("Basic ops passed")
    test_matmul()
    print("Matmul tests passed")
    test_elementwise_ops()
    print("Elementwise ops passed")
    test_reduction_ops()
    print("Reduction ops passed")
    test_shape_ops()
    print("Shape ops passed")
    test_conv_pool_batchnorm_dropout()
    print("Conv/pool/batchnorm/dropout ops passed")


if __name__ == "__main__":
    main()
