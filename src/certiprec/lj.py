"""Shifted-force Lennard-Jones reference system (reduced units, fp64, numpy)."""
import numpy as np

RC = 2.5


def fcc(ncell=3, density=0.8):
    a = (4 / density) ** (1 / 3)
    base = np.array([[0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]])
    pts = [(base + np.array([i, j, k])) for i in range(ncell) for j in range(ncell) for k in range(ncell)]
    return np.concatenate(pts) * a, a * ncell


def lj_energy_forces(pos, box, rc=RC):
    d = pos[:, None, :] - pos[None, :, :]
    d -= box * np.round(d / box)
    r2 = (d ** 2).sum(-1)
    n = len(pos)
    iu = np.triu_indices(n, 1)
    r = np.sqrt(r2[iu]); dv = d[iu]
    m = r < rc
    r, dv = r[m], dv[m]
    def u(x): return 4 * (x ** -12 - x ** -6)
    def du(x): return 4 * (-12 * x ** -13 + 6 * x ** -7)
    e = (u(r) - u(rc) - (r - rc) * du(rc)).sum()
    fmag = -(du(r) - du(rc))  # -dU/dr
    fv = (fmag / r)[:, None] * dv
    f = np.zeros_like(pos)
    i, j = iu[0][m], iu[1][m]
    np.add.at(f, i, fv); np.add.at(f, j, -fv)
    return e, f


def sample_configs(n_cfg=400, temps=(0.6, 0.9, 1.2, 1.6), dt=0.004, every=25, seed=0):
    rng = np.random.default_rng(seed)
    pos0, box = fcc()
    X, E, F = [], [], []
    per_t = n_cfg // len(temps)
    for T in temps:
        pos = pos0 + 0.02 * rng.standard_normal(pos0.shape)
        v = rng.standard_normal(pos.shape) * np.sqrt(T); v -= v.mean(0)
        e, f = lj_energy_forces(pos, box)
        for step in range(500 + per_t * every):
            v += 0.5 * dt * f; pos += dt * v; pos %= box
            e, f = lj_energy_forces(pos, box); v += 0.5 * dt * f
            if step % 10 == 0:  # Berendsen-like rescale toward T
                Tc = (v ** 2).sum() / (3 * len(pos) - 3)
                v *= np.sqrt(1 + 0.1 * (T / Tc - 1))
            if step >= 500 and step % every == 0:
                X.append(pos.copy()); E.append(e); F.append(f.copy())
    return np.array(X), np.array(E), np.array(F), box


def cluster_configs(n_cfg=60, seed=0):
    """Eight-atom free cluster, small distortions of a cube, LJ reduced units."""
    rng = np.random.default_rng(seed)
    base = np.array(list(__import__('itertools').product((0., 1.), repeat=3)))
    configs, energies, forces = [], [], []
    for _ in range(n_cfg):
        pos = (base - .5) * rng.uniform(1.18, 1.38) + rng.normal(0, .025, base.shape)
        e, f = lj_energy_forces(pos, 100.)
        configs.append(pos); energies.append(e); forces.append(f)
    return np.array(configs), np.array(energies), np.array(forces), None
