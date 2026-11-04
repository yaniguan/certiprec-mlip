"""Emulated low-precision matmul schemes (CPU-exact emulation of tensor-core numerics).

Each scheme approximates X @ W by a sum of products of BF16-representable slices,
accumulated in FP32. Products of two BF16 numbers are exact in FP32 (8+8 <= 24
significand bits), so a FP32 matmul over BF16-valued tensors reproduces the
tensor-core contract "BF16 inputs, FP32 accumulate" up to accumulation order.
"""
from __future__ import annotations
import math
import torch

U_BF16 = 2.0 ** -8    # unit roundoff of bfloat16 (8 significand bits incl. implicit)
U_TF32 = 2.0 ** -11   # unit roundoff of TF32 (11 significand bits)
U_FP32 = 2.0 ** -24

# scheme -> (number of slices, list of (i, j) slice products)
SCHEMES = {
    "bf16x1": (1, [(0, 0)]),
    "bf16x3": (2, [(0, 0), (0, 1), (1, 0)]),                    # FACTO / Henry et al.
    "bf16x6": (3, [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (2, 0)]),
    "bf16x9": (3, [(i, j) for i in range(3) for j in range(3)]),
}
NATIVE = ("fp64", "fp32", "tf32")


def products(scheme: str) -> int:
    """Number of tensor-core products a scheme issues (1 for native fp32/tf32)."""
    if scheme in SCHEMES:
        return len(SCHEMES[scheme][1])
    return 1


def round_bf16(x: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest-even to bfloat16, returned in x.dtype (fp32)."""
    return x.to(torch.bfloat16).to(x.dtype)


def round_tf32(x: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest-even of fp32 to TF32 (10 explicit mantissa bits)."""
    assert x.dtype == torch.float32
    b = x.contiguous().view(torch.int32)
    lsb = (b >> 13) & 1
    b = (b + 0x0FFF + lsb) & ~0x1FFF
    return b.view(torch.float32)


def split_bf16(x: torch.Tensor, n: int) -> list[torch.Tensor]:
    """x ~= sum of n BF16-representable slices, residuals computed in fp32."""
    out, r = [], x
    for _ in range(n):
        s = round_bf16(r)
        out.append(s)
        r = r - s
    return out


def emu_matmul_raw(x: torch.Tensor, w: torch.Tensor, scheme: str) -> torch.Tensor:
    if scheme == "fp64":
        return x.double() @ w.double()
    if scheme == "fp32":
        return x.float() @ w.float()
    x, w = x.float(), w.float()
    if scheme == "tf32":
        return round_tf32(x) @ round_tf32(w)
    n, pairs = SCHEMES[scheme]
    xs, ws = split_bf16(x, n), split_bf16(w, n)
    # accumulate smallest terms first (as an FP32 accumulator would do best)
    acc = None
    for i, j in sorted(pairs, key=lambda p: -(p[0] + p[1])):
        t = xs[i] @ ws[j]
        acc = t if acc is None else acc + t
    return acc


class _EmuMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, scheme):
        ctx.save_for_backward(x, w)
        ctx.scheme = scheme
        return emu_matmul_raw(x, w, scheme)

    @staticmethod
    def backward(ctx, g):
        x, w = ctx.saved_tensors
        gx = gw = None
        if ctx.needs_input_grad[0]:   # force path: also emulated (as in FACTO)
            gx = emu_matmul_raw(g, w.t(), ctx.scheme).to(x.dtype)
        if ctx.needs_input_grad[1]:
            gw = emu_matmul_raw(x.reshape(-1, x.shape[-1]).t(),
                                g.reshape(-1, g.shape[-1]), ctx.scheme).to(w.dtype)
        return gx, gw, None


def emu_matmul(x, w, scheme: str):
    if scheme in ("fp64", "fp32"):
        return x @ w  # fully differentiable (incl. double backward) for training
    return _EmuMatmul.apply(x, w, scheme)


def gamma(n: int, u: float) -> float:
    return n * u / (1 - n * u)


def error_bound(x: torch.Tensor, w: torch.Tensor, scheme: str) -> torch.Tensor:
    """A-priori elementwise bound |fl_s(XW) - XW| <= c_s(K) * (|X||W|).

    Splitting truncation: slice k of x satisfies |x_k| <= u^k |x| (1+u)^k (RNE).
    Omitted product (i,j) contributes <= u^(i+j)|X||W|; dropped tails of the
    n-slice expansion contribute <= u^n. FP32 accumulation over K terms adds
    gamma_K(u_fp32) (deterministic, worst case; probabilistic ~sqrt(K)u is
    typically much tighter, see Higham & Mary 2019).
    """
    K = x.shape[-1]
    absxw = x.abs().double() @ w.abs().double()
    acc = gamma(K + 2, U_FP32)
    if scheme == "fp64":
        c = gamma(K, 2.0 ** -53)
    elif scheme == "fp32":
        c = acc
    elif scheme == "tf32":
        c = 2 * U_TF32 + U_TF32 ** 2 + acc
    else:
        n, pairs = SCHEMES[scheme]
        u = U_BF16 * (1 + U_BF16)
        kept = set(pairs)
        c = 0.0
        for i in range(n + 1):
            for j in range(n + 1):
                if (i, j) in kept:
                    continue
                c += u ** (i + j) if (i < n and j < n) else 0.0
        c += 2 * u ** n + u ** (2 * n)     # tails beyond the n slices
        c += len(pairs) * acc
    return c * absxw
