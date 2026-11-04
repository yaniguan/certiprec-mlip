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
