import math, numpy as np, torch
from certiprec.model import RadialMLIP
from certiprec.layers import set_schemes, get_schemes
from certiprec.lj import fcc, lj_energy_forces
from certiprec.md import run_nve, drift_and_rms
from certiprec import certify as C

def _model():
    torch.manual_seed(0); return RadialMLIP(width=16, depth=2)

def test_lj_forces_finite_difference():
    pos, box = fcc(); pos = pos + 0.05 * np.random.default_rng(0).standard_normal(pos.shape)
    e, f = lj_energy_forces(pos, box); h = 1e-6; p2 = pos.copy(); p2[3, 1] += h
    assert abs(-(lj_energy_forces(p2, box)[0] - e) / h - f[3, 1]) < 1e-3

def test_model_invariance_and_force_equivariance():
    m = _model().double(); set_schemes(m, "fp64")
    pos, box = fcc(); pos = pos + 0.05 * np.random.default_rng(1).standard_normal(pos.shape)
    e1, f1 = m.energy_forces(pos, box, torch.float64)
    e2, _ = m.energy_forces(pos + np.array([0.3, -0.2, 0.1]), box, torch.float64)
    assert abs(e1 - e2) < 1e-9
    perm = np.random.default_rng(2).permutation(len(pos))
    e3, f3 = m.energy_forces(pos[perm], box, torch.float64)
    assert abs(e1 - e3) < 1e-9 and torch.allclose(f1[perm], f3)
    assert f1.sum(0).abs().max() < 1e-9  # Newton's 3rd law

def test_schemes_settable_and_force_error_ordering():
    m = _model(); pos, box = fcc(); pos = pos + 0.05 * np.random.default_rng(3).standard_normal(pos.shape)
    set_schemes(m, ["bf16x1", "bf16x3", "fp32"]); assert get_schemes(m) == ["bf16x1", "bf16x3", "fp32"]
    ref = C.reference_forces(m, [pos], box)
    e1 = C.force_rms_error(m, [pos], box, "bf16x1", ref)
    e3 = C.force_rms_error(m, [pos], box, "bf16x3", ref)
    assert e1 > 10 * e3

def test_nve_short_runs_conserve():
    m = _model(); pos, box = fcc()
    v = np.random.default_rng(4).standard_normal(pos.shape) * 0.3
    ts, hs, _, _ = run_nve(m, pos, v, box, "fp32", 0.002, 50, every=5)
    d, r = drift_and_rms(ts, hs, len(pos))
    assert np.isfinite(hs).all() and abs(hs[-1] - hs[0]) / len(pos) < 1e-3

def test_certificate_roundtrip_and_allocation():
    b = C.force_budget(1e-4, 10.0, 0.004, 108)
    assert math.isclose(C.predict_energy_walk(b, 10.0, 0.004, 108), 1e-4, rel_tol=1e-12)
    floor = 1e-6; table = {(l, s): v for l in range(2) for s, v in
                           [("bf16x1", 1e-2), ("bf16x3", 1e-5), ("bf16x6", 2e-6)]}
    table[(1, "bf16x1")] = 1e-6 * 1.5  # layer 1 insensitive
    out = C.allocate(floor, table, [100, 100], budget=5e-5)
    assert out["policy"] == ["bf16x3", "bf16x1"]
