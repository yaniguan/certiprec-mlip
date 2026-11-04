import numpy as np
import torch
from .model import RadialMLIP
from .layers import set_schemes


def train(X, E, F, box, epochs=1500, lr=3e-3, w_f=10.0, seed=0, log=print):
    torch.manual_seed(seed)
    model = RadialMLIP().double()
    set_schemes(model, "fp64")
    Xt = torch.tensor(X); Et = torch.tensor(E); Ft = torch.tensor(F)
    with torch.no_grad():
        G = torch.cat([model.descriptors(x, box) for x in Xt[:50]])
        model.g_mean.copy_(G.mean(0)); model.g_std.copy_(G.std(0) + 1e-6)
        model.e_shift.copy_(Et.mean() / X.shape[1])
    n = len(X); idx = np.random.default_rng(seed).permutation(n)
    tr, va = idx[: int(0.9 * n)], idx[int(0.9 * n):]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    rng = np.random.default_rng(seed + 1)
    def loss_on(ids):
        le = lf = 0.0
        for i in ids:
            p = Xt[i].clone().requires_grad_(True)
            e = model(p, box)
            (g,) = torch.autograd.grad(e, p, create_graph=True)
            le = le + ((e - Et[i]) / X.shape[1]) ** 2
            lf = lf + ((-g - Ft[i]) ** 2).mean()
        return le / len(ids), lf / len(ids)
    for ep in range(epochs):
        b = rng.choice(tr, 8, replace=False)
        le, lf = loss_on(b)
        loss = le + w_f * lf
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if ep % 250 == 0 or ep == epochs - 1:
            log(f"epoch {ep} E_rmse/atom={le.sqrt().item():.4f} F_rmse={lf.sqrt().item():.4f}")
    le, lf = loss_on(va)
    return model.float(), {"val_E_rmse_per_atom": le.sqrt().item(), "val_F_rmse": lf.sqrt().item()}


def fit_equivariant(X, E, F, box=None, seed=0, ridge=1e-8):
    """Fit only a linear readout of fixed equivariant features by least squares.

    Energy/atom and force components have equal weight. The final fifth of
    configurations is held out. No checkpoint is needed: fitting is deterministic.
    """
    from .equivariant import EquivariantMLIP
    torch.manual_seed(seed)
    model = EquivariantMLIP().double()
    set_schemes(model, 'fp64')
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    n_train = int(.8 * len(X))
    rows, targets = [], []
    for pos, e, force in zip(X[:n_train], E[:n_train], F[:n_train]):
        p = torch.tensor(pos, dtype=torch.float64, requires_grad=True)
        features = model.features(p, box).sum(0)
        jac = torch.stack([torch.autograd.grad(h, p, retain_graph=True)[0].flatten() for h in features])
        rows.append(torch.cat([features / len(pos), features.new_ones(1)]).unsqueeze(0))
        targets.append(torch.tensor([e / len(pos)], dtype=torch.float64))
        rows.append(torch.cat([-jac.t(), jac.new_zeros(jac.shape[1], 1)], dim=1))
        targets.append(torch.tensor(force.flatten(), dtype=torch.float64))
    a, b = torch.cat(rows), torch.cat(targets)
    scales = a.square().mean(0).sqrt().clamp_min(1e-8)
    scaled = a / scales
    solution = torch.linalg.solve(scaled.t() @ scaled + ridge * torch.eye(a.shape[1], dtype=a.dtype), scaled.t() @ b) / scales
    with torch.no_grad():
        model.readout.weight.copy_(solution[:-1][None])
        model.readout.bias.copy_(solution[-1:])
    model = model.float()
    set_schemes(model, 'fp32')
    errors = [model.energy_forces(p, box) for p in X[n_train:]]
    return model, {'method': 'fixed random equivariant features; energy+force linear least squares',
                   'n_train': n_train, 'n_validation': len(errors), 'n_atoms': len(X[0]),
                   'parameters_fitted': len(solution),
                   'val_E_rmse_per_atom': float(np.sqrt(np.mean([((float(e) - t) / len(X[0])) ** 2 for (e, _), t in zip(errors, E[n_train:])]))),
                   'val_F_rmse': float(np.sqrt(np.mean([(f.numpy() - t) ** 2 for (_, f), t in zip(errors, F[n_train:])])))}
