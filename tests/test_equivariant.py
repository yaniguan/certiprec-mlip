import numpy as np
import pytest
import torch
import cuequivariance as cue

from certiprec.equivariant import EquivariantMLIP
from certiprec.layers import set_schemes, get_schemes
from certiprec.lj import cluster_configs
from certiprec.md import run_nve
from certiprec.segmented import SegmentedTP
from certiprec import certify as C


@pytest.fixture
def system():
    torch.manual_seed(4)
    model = EquivariantMLIP()
    pos = cluster_configs(2)[0][0]
    return model, pos


def test_rotation_translation_permutation_and_vector_features(system):
    m, p = system
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    e, f = m.energy_forces(p)
    er, fr = m.energy_forces(p @ q.numpy() + .2)
    assert abs(e - er) < 2e-6
    assert torch.allclose(fr, f @ q, atol=3e-6, rtol=1e-4)
    perm = np.arange(len(p))[::-1].copy()
    ep, fp = m.energy_forces(p[perm])
    assert abs(e - ep) < 2e-6
    assert torch.allclose(f[perm], fp, atol=3e-6)
    assert f.sum(0).abs().max() < 1e-6
    # Hidden l=1 channels are present and feed the quadratic invariants.
    assert m.mix.polynomial.outputs[0].size == 4 * m.width


def test_bf16x3_energy_and_forces_close(system):
    m, p = system
    e, f = m.energy_forces(p)
    set_schemes(m, 'bf16x3')
    eq, fq = m.energy_forces(p)
    assert abs(e - eq) < 2e-5
    assert torch.allclose(fq, f, atol=2e-5, rtol=1e-3)


@pytest.mark.parametrize('kind', ['sh', 'message', 'mix', 'square'])
def test_cue_torch_naive_reference(system, kind):
    cuet = pytest.importorskip('cuequivariance_torch')
    m, _ = system
    op = getattr(m, kind)
    xs = [torch.randn(1 if k == 0 and kind != 'sh' else 7, o.size)
          for k, o in enumerate(op.polynomial.inputs)]
    reference = cuet.SegmentedPolynomial(op.polynomial, method='naive', math_dtype=torch.float32)
    expected = reference(xs)[0]
    actual = op(*xs)
    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-5)
    # Check input derivatives too, including repeated inputs in spherical harmonics.
    for x in xs:
        x.requires_grad_(True)
    g = torch.autograd.grad(op(*xs).square().sum(), xs)
    gr = torch.autograd.grad(reference(xs)[0].square().sum(), xs)
    assert all(torch.allclose(a, b, atol=2e-5, rtol=3e-5) for a, b in zip(g, gr))


def test_finite_set_certificate_and_infeasibility(system):
    m, p = system
    old = get_schemes(m)
    policy = ['bf16x3'] * len(old)
    result = C.certify_policy(m, [p], None, policy, budget=1e-3)
    assert result['budget_met'] and result['bound_holds']
    assert get_schemes(m) == old
    with pytest.raises(ValueError, match='No candidate'):
        C.allocate(1., {(0, 'bf16x1'): 2.}, [10], .1, candidates=['bf16x1'])


def test_equivariant_nve_short(system):
    m, p = system
    v = np.random.default_rng(1).normal(0, .05, p.shape)
    for policy in ['fp32', 'bf16x3']:
        ts, h, _, _ = run_nve(m, p, v, None, policy, .002, 12, every=3)
        assert np.isfinite(h).all() and abs(h[-1] - h[0]) / len(p) < 1e-4
