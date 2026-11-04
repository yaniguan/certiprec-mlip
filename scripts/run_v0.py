"""Bounded, resumable CPU evidence. Each completed experiment is written atomically."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import cuequivariance as cue

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from certiprec import certify as C, precision as P, profiling
from certiprec.layers import prec_layers, set_schemes
from certiprec.lj import cluster_configs
from certiprec.md import drift_and_rms, run_nve
from certiprec.train import fit_equivariant


def write(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default='results/v0')
    ap.add_argument('--steps', type=int, default=600)
    ap.add_argument('--dt', type=float, default=.002)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--configs', type=int, default=60)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--force-budget', type=float)
    ap.add_argument('--quick', action='store_true', help='120 steps, 30 fit configurations, still 3 seeds')
    ap.add_argument('--no-resume', action='store_true')
    ap.add_argument('--profile', action='store_true')
    ap.add_argument('--torch-profiler', action='store_true', help='profile one force call; save small text table')
    args = ap.parse_args()
    if args.quick:
        args.steps, args.configs = 120, 30
    if args.steps < 10 or args.seeds < 1 or args.configs < 20 or args.dt <= 0:
        ap.error('need steps >=10, seeds >=1, configs >=20 and dt >0')
    torch.set_num_threads(args.threads)
    profiling.enable(args.profile or os.getenv('CERTIPREC_PROFILE') == '1')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sources = sorted((Path(__file__).resolve().parents[1] / 'src/certiprec').glob('*.py')) + [Path(__file__)]
    digest = hashlib.sha256(b''.join(p.read_bytes() for p in sources)).hexdigest()[:16]
    config = {k: v for k, v in vars(args).items() if k not in ('out', 'no_resume', 'profile', 'torch_profiler')}
    versions = {'torch': str(torch.__version__), 'numpy': np.__version__, 'cuequivariance': cue.__version__}
    run_id = hashlib.sha256((digest + json.dumps(config, sort_keys=True) + json.dumps(versions, sort_keys=True)).encode()).hexdigest()[:16]
    started = time.perf_counter()
    meta = {'run_id': run_id, 'source_hash': digest, 'config': config, **versions,
            'device': 'cpu', 'gpu_status': 'PENDING', 'date_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}

    def stage(name, fn):
        path = out / (name + '.json')
        if path.exists() and not args.no_resume:
            saved = json.loads(path.read_text())
            if saved.get('meta', {}).get('run_id') == run_id:
                print('resume', name, flush=True)
                return saved['data']
        tick = time.perf_counter()
        data = fn()
        write(path, {'meta': meta, 'seconds': time.perf_counter() - tick, 'data': data})
        print('done', name, f'{time.perf_counter() - tick:.2f}s', flush=True)
        return data

    def matmuls():
        g = torch.Generator().manual_seed(0)
        result = {}
        for k in (16, 64, 256, 1024):
            x = torch.randn(64, k, generator=g)
            w = torch.randn(k, 64, generator=g) / math.sqrt(k)
            ref = x.double() @ w.double()
            result[k] = {}
            for s in ('fp32', 'tf32', 'bf16x1', 'bf16x2', 'bf16x3', 'bf16x6'):
                err = (P.emu_matmul_raw(x, w, s).double() - ref).abs()
                bound = P.error_bound(x, w, s)
                result[k][s] = {'rel_fro_err': float(err.norm() / ref.norm()),
                                'max_abs_err': float(err.max()), 'bound_holds': bool((err <= bound).all()),
                                'max_error_over_bound': float((err / bound).max())}
        return result
    mm = stage('matmul', matmuls)
    x, e, f, box = cluster_configs(args.configs)
    tick = time.perf_counter()
    model, fit = fit_equivariant(x, e, f, box)
    fit['seconds'] = time.perf_counter() - tick
    write(out / 'fit.json', {'meta': meta, 'data': fit})
    print('fit', fit, flush=True)
    test = list(x[int(.8 * len(x)):])
    calibration = test[:3]
    count = len(prec_layers(model))
    names = [getattr(m, 'name', 'readout') for m in prec_layers(model)]

    def measure(policy, reference):
        set_schemes(model, policy)
        got = [model.energy_forces(p, box) for p in test]
        return {'energy_rmse_per_atom': float(np.sqrt(np.mean([((float(a[0]) - float(b[0])) / len(x[0])) ** 2 for a, b in zip(got, reference)]))),
                'force_rmse': float(np.sqrt(np.mean([(a[1].double() - b[1].double()).square().mean().item() for a, b in zip(got, reference)])))}

    def layer_errors():
        set_schemes(model, 'fp32')
        reference = [model.energy_forces(p, box) for p in test]
        uniform = {s: measure(s, reference) for s in ('bf16x1', 'bf16x2', 'bf16x3', 'bf16x6')}
        per_layer = {}
        for layer, name in enumerate(names):
            per_layer[name] = {}
            for s in ('bf16x1', 'bf16x2', 'bf16x3'):
                policy = ['fp32'] * count
                policy[layer] = s
                per_layer[name][s] = measure(policy, reference)
        q, _ = np.linalg.qr(np.random.default_rng(31).normal(size=(3, 3)))
        q[:, 0] *= np.linalg.det(q)
        rotation = {}
        for s in ('fp32', 'bf16x1', 'bf16x3'):
            set_schemes(model, s)
            en, fo = model.energy_forces(test[0], box)
            er, fr = model.energy_forces(test[0] @ q, box)
            rotation[s] = {'energy_abs_error': abs(float(en - er)),
                           'force_rms_error': float(np.sqrt(np.mean((fr.numpy() - fo.numpy() @ q) ** 2)))}
        import cuequivariance_torch as cuet
        generator = torch.Generator().manual_seed(7)
        check = {}
        for op in (model.sh, model.message, model.mix, model.square):
            xs = [torch.randn(1 if i == 0 and op is not model.sh else 9, o.size, generator=generator)
                  for i, o in enumerate(op.polynomial.inputs)]
            op.scheme = 'fp32'
            reference_op = cuet.SegmentedPolynomial(op.polynomial, method='naive', math_dtype=torch.float32)
            with torch.no_grad():
                check[op.name] = float((op(*xs) - reference_op(xs)[0]).abs().max())
        return {'reference': 'FP32 same model', 'n_configs': len(test), 'layer_names': names,
                'uniform': uniform, 'per_layer': per_layer, 'rotation': rotation,
                'cue_torch_naive_max_abs_errors': check}
    layers = stage('layer_errors', layer_errors)

    def allocation():
        floor, table, _ = C.sensitivity_table(model, calibration, box)
        flops = C.layer_flops(model, len(x[0]))
        high_score = C.predict_sigma(floor, table, ['bf16x3'] * count)
        low_score = C.predict_sigma(floor, table, ['bf16x1'] * count)
        budget = args.force_budget if args.force_budget is not None else max(2 * high_score, .03 * low_score)
        proposal = C.allocate(floor, table, flops, budget)
        certificate = C.certify_policy(model, calibration, box, proposal['policy'], budget)
        if not certificate['budget_met']:
            # Prefix interactions can invalidate the proposal score: actually check a fallback.
            for fallback in ('bf16x3', 'bf16x6', 'fp32'):
                policy = [fallback] * count
                certificate = C.certify_policy(model, calibration, box, policy, budget)
                if certificate['budget_met']:
                    proposal = {'policy': policy, 'cost': C.policy_cost(flops, policy), 'fallback': True}
                    break
        if not certificate['budget_met']:
            raise ValueError('Even the FP32 floor exceeds the requested budget.')
        heldout = C.certify_policy(model, test[3:], box, proposal['policy'], budget)
        return {**proposal, 'layer_names': names, 'fp32_floor_vs_fp64': floor,
                'force_budget': budget, 'budget_rule': 'max(2*BF16x3 proposal, 0.03*BF16x1 proposal), unless specified',
                'layer_flops_estimate': flops, 'per_layer_force_errors': {f'{l}:{s}': v for (l, s), v in table.items()},
                'calibration_certificate': certificate, 'heldout_audit': heldout,
                'cost_relative_to_uniform_bf16x3': proposal['cost'] / C.policy_cost(flops, ['bf16x3'] * count)}
    alloc = stage('allocation', allocation)
    policies = {'fp32': 'fp32', 'bf16x1': 'bf16x1', 'bf16x3': 'bf16x3', 'certiprec': alloc['policy']}
    runs = {name: [] for name in policies}
    for seed in range(args.seeds):
        pos = test[seed % len(test)]
        v = np.random.default_rng(100 + seed).normal(0, .05, pos.shape)
        v -= v.mean(0)
        for name, policy in policies.items():
            def nve():
                ts, hs, _, _ = run_nve(model, pos, v, box, policy, args.dt, args.steps, every=max(1, args.steps // 100))
                drift, rms = drift_and_rms(ts, hs, len(pos))
                return {'seed': seed, 'policy': policy, 'signed_drift_per_atom_time': drift,
                        'detrended_rms_per_atom': rms, 'final_abs_dH_per_atom': abs(float(hs[-1] - hs[0])) / len(pos),
                        'max_abs_dH_per_atom': float(np.max(np.abs(hs - hs[0]))) / len(pos),
                        'time': ts.tolist(), 'dH_per_atom': ((hs - hs[0]) / len(pos)).tolist()}
            run = stage(f'nve_seed{seed}_{name}', nve)
            runs[name].append(run)
            print(f'  seed {seed} {name}: drift={run["signed_drift_per_atom_time"]:.6g}', flush=True)
    summary = {name: {'mean_abs_drift': float(np.mean([abs(r['signed_drift_per_atom_time']) for r in rr])),
                      'mean_detrended_rms': float(np.mean([r['detrended_rms_per_atom'] for r in rr])),
                      'rms_final_dH': float(np.sqrt(np.mean([r['final_abs_dH_per_atom'] ** 2 for r in rr])))}
               for name, rr in runs.items()}
    md = {'steps': args.steps, 'dt': args.dt, 'seeds': args.seeds, 'n_atoms': len(x[0]), 'units': 'LJ reduced',
          'hamiltonian': 'FP64 evaluation of the same fitted model, not analytic LJ',
          'initial_velocity_std': .05, 'summary': summary}
    write(out / 'nve.json', {'meta': meta, 'data': md})
    if args.torch_profiler:
        set_schemes(model, 'bf16x3')
        with profiling.torch_profile(True) as prof:
            model.energy_forces(test[0], box)
        (out / 'torch_profile.txt').write_text(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=20))
    write(out / 'profiling.json', {'meta': meta, 'data': profiling.snapshot(),
                                   'note': 'Inclusive CPU wall timers; nested matmul FLOPs are estimates; excludes backward FLOPs.'})
    total = time.perf_counter() - started
    write(out / 'results.json', {'meta': meta, 'matmul': mm, 'fit': fit, 'layers': layers,
                                 'allocation': alloc, 'nve': md, 'invocation_seconds': total})
    print('NVE', json.dumps(summary), flush=True)
    print(f'Complete in {total:.1f}s. GPU results: PENDING.', flush=True)


if __name__ == '__main__':
    main()
