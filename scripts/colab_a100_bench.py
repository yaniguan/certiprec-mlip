"""A100 benchmark runner. GPU evidence is PENDING until this is run on CUDA.

Install in Colab with notebooks/colab_a100.ipynb. Dense BF16 GEMMs request FP32
outputs explicitly. cuEq split passes use real fused_tp FP32 CUDA kernels on
BF16-rounded operands; they are not BF16 tensor-core kernels. No CPU fallback.
"""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from certiprec.precision import SCHEMES, split_bf16


def timed(fn, iterations):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def dense_gemm(x, w, scheme):
    if scheme in ('fp32', 'tf32'):
        torch.backends.cuda.matmul.allow_tf32 = scheme == 'tf32'
        return x @ w
    torch.backends.cuda.matmul.allow_tf32 = False
    n, pairs = SCHEMES[scheme]
    xs = [s.bfloat16() for s in split_bf16(x, n)]
    ws = [s.bfloat16() for s in split_bf16(w, n)]
    acc = None
    for i, j in sorted(pairs, key=lambda p: -(p[0] + p[1])):
        # Fail if this API is unavailable. Casting operands to FP32 is not a substitute.
        term = torch.mm(xs[i], ws[j], out_dtype=torch.float32)
        acc = term if acc is None else acc + term
    return acc


def cue_split(op, w, x, scheme):
    if scheme == 'fp32':
        return op([w, x])[0]
    n, pairs = SCHEMES[scheme]
    # Inputs to fused_tp remain FP32; rounded slices carry BF16 values.
    xs, ws = split_bf16(x, n), split_bf16(w, n)
    acc = None
    for i, j in sorted(pairs, key=lambda p: -(p[0] + p[1])):
        term = op([ws[j], xs[i]])[0]
        acc = term if acc is None else acc + term
    return acc


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default='results/gpu')
    ap.add_argument('--atoms', type=int, default=4096)
    ap.add_argument('--widths', nargs='+', type=int, default=[64, 256])
    ap.add_argument('--iterations', type=int, default=30)
    ap.add_argument('--allow-other-gpu', action='store_true')
    args = ap.parse_args()
    if not torch.cuda.is_available():
        ap.error('CUDA is required. GPU results remain PENDING.')
    gpu = torch.cuda.get_device_name()
    if 'A100' not in gpu and not args.allow_other_gpu:
        ap.error('Select a Colab A100 or pass --allow-other-gpu; device is ' + gpu)
    import cuequivariance as cue
    import cuequivariance_torch as cuet
    import cuequivariance_ops_torch  # Require the actual CUDA extension.
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    meta = {'status': 'MEASURED', 'gpu': gpu, 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'cuequivariance': cue.__version__, 'cuequivariance_torch': version('cuequivariance-torch'),
            'iterations': args.iterations, 'timing': 'CUDA events, includes splitting and launches',
            'scope': 'forward microbenchmarks only; model force/MD GPU results PENDING'}
    schemes = ['fp32', 'tf32', 'bf16x1', 'bf16x2', 'bf16x3', 'bf16x6']
    with torch.no_grad():
        for width in args.widths:
            x = torch.randn(args.atoms, width, device='cuda')
            w = torch.randn(width, width, device='cuda') / width ** .5
            reference = x.double() @ w.double()
            rows = {}
            for scheme in schemes:
                fn = lambda: dense_gemm(x, w, scheme)
                y = fn()
                assert y.dtype == torch.float32
                rows[scheme] = {'milliseconds': timed(fn, args.iterations),
                                'rel_fro_err_vs_fp64': float((y.double() - reference).norm() / reference.norm())}
            save(out / f'dense_N{args.atoms}_D{width}.json', {'meta': meta, 'data': rows})

            # Bilinear equivariant channel mixing can be expanded into k exact kernel passes.
            irreps = cue.Irreps('O3', f'{width}x0e + {width}x1o')
            polynomial = cue.descriptors.linear(irreps, irreps).polynomial
            op = cuet.SegmentedPolynomial(polynomial, method='fused_tp', math_dtype=torch.float32).cuda()
            if op.method != 'fused_tp':
                raise RuntimeError('cuEq fell back to ' + op.method + '; refusing a fake CUDA benchmark')
            exact_op = cuet.SegmentedPolynomial(polynomial, method='fused_tp', math_dtype=torch.float64).cuda()
            weights = torch.randn(1, polynomial.inputs[0].size, device='cuda') / width ** .5
            features = torch.randn(args.atoms, polynomial.inputs[1].size, device='cuda')
            reference = exact_op([weights.double(), features.double()])[0]
            rows = {}
            torch.backends.cuda.matmul.allow_tf32 = False
            for scheme in (s for s in schemes if s != 'tf32'):
                fn = lambda: cue_split(op, weights, features, scheme)
                y = fn()
                rows[scheme] = {'milliseconds': timed(fn, args.iterations),
                                'rel_fro_err_vs_fp64': float((y.double() - reference).norm() / reference.norm())}
            save(out / f'cue_fused_tp_N{args.atoms}_D{width}.json', {
                'meta': meta, 'backend': 'real cuEq fused_tp FP32 kernels, k BF16-rounded slice passes',
                'tensor_core_bf16': False, 'tf32': 'UNSUPPORTED for this fused_tp API; dense TF32 is measured separately',
                'native_bf16xk_fused_kernel': 'PENDING', 'data': rows})
            print('Saved dense and real cuEq kernel results for width', width, flush=True)


if __name__ == '__main__':
    main()
