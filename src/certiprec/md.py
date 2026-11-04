"""Velocity-Verlet NVE in fp64 state; forces from model under a precision policy.

The conserved quantity is always evaluated with the fp64 model (the 'true'
model Hamiltonian), isolating dynamical error from measurement error.
"""
import numpy as np
import torch
from .layers import set_schemes


def run_nve(model, pos, vel, box, schemes, dt, n_steps, every=10):
    if dt <= 0 or n_steps < 1 or every < 1:
        raise ValueError("Need positive dt, n_steps and sampling interval.")
    ref_dtype = torch.float64
    def force(p):
        set_schemes(model, schemes)
        dtype = torch.float64 if schemes == "fp64" else torch.float32
        _, f = model.energy_forces(p, box, dtype)
        return f.double().numpy()
    def h_true(p, v):
        set_schemes(model, "fp64")
        e, _ = model.energy_forces(p, box, ref_dtype)
        return float(e) + 0.5 * float((v ** 2).sum())
    pos, vel = pos.copy(), vel.copy()
    f = force(pos)
    ts, hs = [0.0], [h_true(pos, vel)]
    for step in range(1, n_steps + 1):
        vel += 0.5 * dt * f
        pos += dt * vel
        f = force(pos)
        vel += 0.5 * dt * f
        if step % every == 0 or step == n_steps:
            ts.append(step * dt); hs.append(h_true(pos, vel))
    return np.array(ts), np.array(hs), pos, vel


def drift_and_rms(ts, hs, n_atoms):
    e = (hs - hs[0]) / n_atoms
    a, b = np.polyfit(ts, e, 1)
    return float(a), float(np.sqrt(np.mean((e - (a * ts + b)) ** 2)))
