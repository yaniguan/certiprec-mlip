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
