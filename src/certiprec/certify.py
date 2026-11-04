"""The CertiPrec mechanism: sensitivity -> certified per-layer BF16xk allocation.

1. Sensitivity: for each PrecLinear layer l and scheme s, measure the RMS force
   error sigma_{l,s} (vs. the fp64 model) with ONLY layer l in scheme s.
2. Composition: rounding errors of different layers are ~independent, so
   sigma_F^2(policy) ~= sigma_F^2(fp32 floor) + sum_l [sigma_{l,s_l}^2 - floor_l^2].
3. NVE certificate: spurious force dF does work dt * dF.v per step. If dF is
   decorrelated step-to-step (rounding noise), the fp64-Hamiltonian performs a
   random walk with
        RMS[H(t) - H(0)] ~= sigma_F * dt * sqrt(3 N kT / m) * sqrt(t / dt)
   (+ the bounded shadow-Hamiltonian oscillation of Verlet). Inverting gives a
   force-error budget for a target energy-drift tolerance over a horizon.
4. Allocation: minimise tensor-core cost sum_l flops_l * products(s_l) subject
   to sigma_F(policy) <= budget (exact search for few layers, greedy otherwise).
"""
from __future__ import annotations
import itertools
import math
import numpy as np
import torch
from .layers import prec_layers, set_schemes
from .precision import products

CANDIDATES = ("bf16x1", "bf16x3", "bf16x6")


def force_rms_error(model, configs, box, schemes, ref_forces):
    set_schemes(model, schemes)
    errs = []
    for p, fr in zip(configs, ref_forces):
        _, f = model.energy_forces(p, box, torch.float32)
        errs.append(((f.double() - fr) ** 2).mean().item())
    return math.sqrt(float(np.mean(errs)))


def reference_forces(model, configs, box):
    set_schemes(model, "fp64")
    return [model.energy_forces(p, box, torch.float64)[1] for p in configs]


def sensitivity_table(model, configs, box, candidates=CANDIDATES):
    ref = reference_forces(model, configs, box)
    L = len(prec_layers(model))
    floor = force_rms_error(model, configs, box, ["fp32"] * L, ref)
    table = {}
    for l in range(L):
        for s in candidates:
            pol = ["fp32"] * L; pol[l] = s
            table[(l, s)] = force_rms_error(model, configs, box, pol, ref)
    return floor, table, ref


def predict_sigma(floor, table, policy):
    v = floor ** 2 + sum(max(table[(l, s)] ** 2 - floor ** 2, 0.0) for l, s in enumerate(policy) if s != "fp32")
    return math.sqrt(v)


def layer_flops(model, n_atoms):
    return [2 * n_atoms * m.in_features * m.out_features for m in prec_layers(model)]


def policy_cost(flops, policy):
    """Cost in 'BF16 tensor-core FLOP units' (forward; backward scales identically)."""
    return sum(f * products(s) for f, s in zip(flops, policy))


def force_budget(target_rms_per_atom, horizon, dt, n_atoms, kT=1.0, m=1.0):
    """Max sigma_F (per component) so that predicted RMS random-walk of H/N over horizon <= target."""
    n_steps = horizon / dt
    return target_rms_per_atom * n_atoms / (dt * math.sqrt(3 * n_atoms * kT / m) * math.sqrt(n_steps))


def predict_energy_walk(sigma_f, horizon, dt, n_atoms, kT=1.0, m=1.0):
    """Predicted RMS of [H(t)-H(0)]/N at t=horizon from force noise alone."""
    return sigma_f * dt * math.sqrt(3 * n_atoms * kT / m) * math.sqrt(horizon / dt) / n_atoms


def allocate(floor, table, flops, budget, candidates=CANDIDATES):
    L = len(flops)
    best = None
    for pol in itertools.product(candidates, repeat=L):
        sig = predict_sigma(floor, table, pol)
        if sig <= budget:
            c = policy_cost(flops, pol)
            if best is None or c < best[0]:
                best = (c, list(pol), sig)
    if best is None:  # infeasible: fall back to the most accurate candidate everywhere
        pol = [candidates[-1]] * L
        best = (policy_cost(flops, pol), pol, predict_sigma(floor, table, pol))
    return {"policy": best[1], "cost": best[0], "predicted_sigma_f": best[2]}
