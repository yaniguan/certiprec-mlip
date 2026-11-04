"""Plot measured CPU JSON only. GPU results remain PENDING."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('out', nargs='?', default='results/v0')
    args = ap.parse_args()
    out = Path(args.out)
    figs = out / 'figs'
    figs.mkdir(exist_ok=True)
    def read(name):
        return json.loads((out / (name + '.json')).read_text())['data']
    mm, layers, alloc, md = [read(n) for n in ('matmul', 'layer_errors', 'allocation', 'nve')]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    ks = sorted(mm, key=int)
    for s in mm[ks[0]]:
        axes[0].loglog(list(map(int, ks)), [mm[k][s]['rel_fro_err'] for k in ks], 'o-', label=s)
    axes[0].set(xlabel='Inner dimension K', ylabel='Relative Frobenius error vs FP64', title='CPU matmul emulation')
    axes[0].legend(fontsize=7)
    uniform = layers['uniform']
    axes[1].bar(list(uniform), [r['force_rmse'] for r in uniform.values()])
    axes[1].set(yscale='log', ylabel='Force RMSE vs FP32', title='Equivariant model, held-out inputs')
    for name in md['summary']:
        run = read('nve_seed0_' + name)
        axes[2].plot(run['time'], run['dH_per_atom'], label=name, lw=.9)
    axes[2].set(xlabel='Time (LJ units)', ylabel='FP64 model ΔH / atom', title='NVE seed 0')
    axes[2].set_yscale('symlog', linthresh=1e-5)
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figs / 'evidence.png', dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    names = list(md['summary'])
    values = [[abs(read(f'nve_seed{s}_{name}')['signed_drift_per_atom_time']) for s in range(md['seeds'])] for name in names]
    axes[0].bar(names, [np.mean(v) for v in values], alpha=.6)
    for i, v in enumerate(values):
        axes[0].scatter([i] * len(v), v, color='k', s=14)
    axes[0].set(yscale='log', ylabel='|drift| / atom / time (LJ units)', title=f'{md["seeds"]} short NVE runs')
    c, h = alloc['calibration_certificate'], alloc['heldout_audit']
    positions = np.arange(2)
    axes[1].bar(positions - .18, [d['actual_force_rms'] for d in (c, h)], .36, label='Actual force error')
    axes[1].bar(positions + .18, [d['force_rms_bound'] for d in (c, h)], .36, label='Triangle bound')
    axes[1].axhline(c['budget'], color='k', ls='--', lw=1, label='Budget')
    axes[1].set(xticks=positions, xticklabels=[f'Calibration ({c["n_configs"]})', f'Held-out ({h["n_configs"]})'],
                ylabel='Force RMS vs FP64', title='Finite-set certificate and separate audit')
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figs / 'certificate_and_drift.png', dpi=130)
    plt.close(fig)
    print('Wrote', figs)


if __name__ == '__main__':
    main()
