"""Colab/A100 micro-benchmark (NOT yet run; v1 task).

Usage on a Colab A100 runtime:
    !git clone https://github.com/yaniguan/certiprec-mlip && cd certiprec-mlip && pip install -e .
    !python scripts/colab_a100_bench.py
Measures real tensor-core latency of FP32 (SIMT), TF32, BF16x1/x3/x6 split GEMMs
for FACTO-like widths (D = 256..2048) and N atoms, plus achieved accuracy vs FP64.
"""
import json, torch, time
from certiprec.precision import split_bf16, SCHEMES

assert torch.cuda.is_available(), "needs a GPU runtime"
dev = "cuda"

def gemm(x, w, scheme):
    if scheme == "fp32":
        torch.backends.cuda.matmul.allow_tf32 = False; return x @ w
    if scheme == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True; return x @ w
    n, pairs = SCHEMES[scheme]
    xs = [s.bfloat16() for s in split_bf16(x, n)]; ws = [s.bfloat16() for s in split_bf16(w, n)]
    acc = None
    for i, j in pairs:  # NOTE: bf16 GEMM output rounded to bf16 by default -> request fp32 out
        t = torch.matmul(xs[i], ws[j], out_dtype=torch.float32) if "out_dtype" in torch.matmul.__doc__ else (xs[i].float() @ ws[j].float())
        acc = t if acc is None else acc + t
    return acc

def bench(f, it=50):
    for _ in range(5): f()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(it): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / it * 1e3

out = {"gpu": torch.cuda.get_device_name()}
for N in (4096, 21952):
    for D in (256, 1024, 2048):
        x = torch.randn(N, D, device=dev); w = torch.randn(D, D, device=dev) / D ** 0.5
        ref = x.double() @ w.double()
        for s in ("fp32", "tf32", "bf16x1", "bf16x3", "bf16x6"):
            ms = bench(lambda: gemm(x, w, s))
            err = ((gemm(x, w, s).double() - ref).norm() / ref.norm()).item()
            out[f"N{N}_D{D}_{s}"] = {"ms": ms, "rel_err": err}
            print(N, D, s, f"{ms:.3f} ms", f"{err:.2e}")
json.dump(out, open("results_a100.json", "w"), indent=1)
