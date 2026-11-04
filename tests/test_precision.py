import math, torch, pytest
from certiprec import precision as P

def test_bf16_slices_representable_and_ordered():
    x = torch.randn(1000)
    s = P.split_bf16(x, 3)
    for t in s:
        assert torch.equal(P.round_bf16(t), t)
    assert (s[1].abs() <= x.abs() * 2 ** -8 * 1.01 + 1e-30).all()

def test_tf32_rounding_mantissa():
    x = torch.randn(1000)
    y = P.round_tf32(x)
    assert ((y.view(torch.int32) & 0x1FFF) == 0).all()
    assert ((y - x).abs() <= x.abs() * 2 ** -11 * 1.0001).all()

@pytest.mark.parametrize("s", ["fp32", "tf32", "bf16x1", "bf16x3", "bf16x6", "bf16x9"])
def test_error_bound_holds(s):
    g = torch.Generator().manual_seed(1)
    X = torch.randn(64, 128, generator=g); W = torch.randn(128, 32, generator=g)
    err = (P.emu_matmul_raw(X, W, s).double() - X.double() @ W.double()).abs()
    assert (err <= P.error_bound(X, W, s)).all()

def test_accuracy_ordering():
    g = torch.Generator().manual_seed(2)
    X = torch.randn(128, 256, generator=g); W = torch.randn(256, 128, generator=g)
    ref = X.double() @ W.double()
    e = {s: ((P.emu_matmul_raw(X, W, s).double() - ref).norm() / ref.norm()).item()
         for s in ["bf16x1", "tf32", "bf16x3", "fp32"]}
    assert e["bf16x1"] > e["tf32"] > e["bf16x3"]
    assert e["bf16x3"] < 1e-5 and e["bf16x1"] > 1e-3

def test_products():
    assert [P.products(s) for s in ["fp32", "bf16x1", "bf16x3", "bf16x6", "bf16x9"]] == [1, 1, 3, 6, 9]

def test_autograd_backward_emulated():
    x = torch.randn(8, 16, requires_grad=True); w = torch.randn(16, 4)
    y = P.emu_matmul(x, w, "bf16x3"); y.sum().backward()
    ref = torch.ones(8, 4) @ w.t()
    assert torch.allclose(x.grad, ref, rtol=1e-5, atol=1e-5)


def test_three_slice_exact_reconstruction():
    g = torch.Generator().manual_seed(9)
    x = torch.randn(4096, generator=g) * 10
    parts = P.split_bf16(x, 3)
    # Float64 sum avoids hiding a missing residual behind FP32 addition rounding.
    assert torch.equal(sum(s.double() for s in parts), x.double())


@pytest.mark.parametrize('scheme', ['bf16x1', 'bf16x2', 'bf16x3', 'bf16x6'])
def test_batched_matmul_backward_and_bound(scheme):
    g = torch.Generator().manual_seed(91)
    x = torch.randn(3, 4, 12, generator=g, requires_grad=True)
    w = torch.randn(1, 12, 5, generator=g, requires_grad=True)
    y = P.emu_matmul(x, w, scheme)
    y.sum().backward()
    assert w.grad.shape == w.shape and x.grad.shape == x.shape
    exact = x.detach().double() @ w.detach().double()
    assert ((y.detach().double() - exact).abs() <= P.error_bound(x.detach(), w.detach(), scheme)).all()
