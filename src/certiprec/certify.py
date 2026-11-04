"""Deterministic local bounds and finite-set force certificates.

The triangle inequality certifies a policy on the configurations actually
checked. No independence assumption, unseen-configuration guarantee, or global
NVE drift guarantee is made. The random-walk helpers are conditional estimates.
"""
import itertools
import math

import numpy as np
import torch

from .layers import prec_layers, get_schemes, set_schemes
from .precision import products

CANDIDATES = ('bf16x1', 'bf16x2', 'bf16x3', 'bf16x6')


def force_rms_error(model, configs, box, schemes, ref_forces):
    old = get_schemes(model)
    try:
        set_schemes(model, schemes)
        errs = [((model.energy_forces(p, box)[1].double() - fr) ** 2).mean().item()
                for p, fr in zip(configs, ref_forces)]
        return math.sqrt(float(np.mean(errs)))
    finally:
        set_schemes(model, old)


def reference_forces(model, configs, box, dtype=torch.float64):
    old = get_schemes(model)
    try:
        set_schemes(model, 'fp64' if dtype == torch.float64 else 'fp32')
        return [model.energy_forces(p, box, dtype)[1] for p in configs]
    finally:
        set_schemes(model, old)


def sensitivity_table(model, configs, box, candidates=CANDIDATES):
    ref = reference_forces(model, configs, box)
    count = len(prec_layers(model))
    floor = force_rms_error(model, configs, box, 'fp32', ref)
    table = {}
    for layer in range(count):
        for scheme in candidates:
            policy = ['fp32'] * count
            policy[layer] = scheme
            table[layer, scheme] = force_rms_error(model, configs, box, policy, ref)
    return floor, table, ref


def predict_sigma(floor, table, policy):
    """Conservative proposal score; final certification must check the prefixes."""
    return floor + sum(table[layer, scheme] + floor for layer, scheme in enumerate(policy)
                       if scheme != 'fp32')


def layer_flops(model, n_atoms):
    # All directed pairs; includes tiny coefficient contractions as an estimate.
    result = []
    for layer in prec_layers(model):
        if hasattr(layer, 'flops_per_item'):
            batch = n_atoms * (n_atoms - 1) if layer.name in ('spherical_harmonics', 'edge_tensor_product') else n_atoms
            result.append(batch * layer.flops_per_item)
        else:
            result.append(2 * n_atoms * layer.in_features * layer.out_features)
    return result


def policy_cost(flops, policy):
    """Forward BF16 product count model, not measured latency or native FP32 cost."""
    return sum(f * products(s) for f, s in zip(flops, policy))


def allocate(floor, table, flops, budget, candidates=CANDIDATES):
    best = None
    for policy in itertools.product(candidates, repeat=len(flops)):
        score = predict_sigma(floor, table, policy)
        cost = policy_cost(flops, policy)
        if score <= budget and (best is None or cost < best[0]):
            best = cost, list(policy), score
    if best is None:
        raise ValueError('No candidate meets the force budget; increase budget or use FP32.')
    return {'policy': best[1], 'cost': best[0], 'proposal_force_rms': best[2]}


def certify_policy(model, configs, box, policy, budget):
    """Exact telescoping triangle bound on a finite set, including FP32 floor.

    Norm is RMS over all checked force components. Prefix l changes one layer
    with all preceding changes in place, so interactions are included.
    """
    old = get_schemes(model)
    def forces(schemes, dtype=torch.float32):
        set_schemes(model, schemes)
        return torch.stack([model.energy_forces(p, box, dtype)[1].double() for p in configs])
    def rms(x):
        return float(x.square().mean().sqrt())
    try:
        ref = forces('fp64', torch.float64)
        prev = forces('fp32')
        terms = [rms(prev - ref)]
        prefix = ['fp32'] * len(policy)
        for layer, scheme in enumerate(policy):
            prefix[layer] = scheme
            cur = forces(prefix)
            terms.append(rms(cur - prev))
            prev = cur
        bound = sum(terms)
        actual = rms(prev - ref)
        return {'scope': 'checked configurations only; emulated backward forces',
                'n_configs': len(configs), 'telescoping_terms': terms,
                'force_rms_bound': bound, 'actual_force_rms': actual,
                'max_force_l2': float((prev - ref).flatten(1).norm(dim=1).max()),
                'budget': budget, 'budget_met': bound <= budget,
                'bound_holds': actual <= bound + 1e-14}
    finally:
        set_schemes(model, old)


def force_budget(target_rms_per_atom, horizon, dt, n_atoms, kT=1.0, m=1.0):
    return target_rms_per_atom * n_atoms / (dt * math.sqrt(3 * n_atoms * kT / m) * math.sqrt(horizon / dt))


def predict_energy_walk(sigma_f, horizon, dt, n_atoms, kT=1.0, m=1.0):
    """Heuristic, requiring decorrelated zero-mean noise. Not a certificate."""
    return sigma_f * dt * math.sqrt(3 * n_atoms * kT / m) * math.sqrt(horizon / dt) / n_atoms
