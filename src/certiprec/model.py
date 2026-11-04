"""Minimal atom-centered MLIP: smooth radial symmetry functions + per-atom MLP.

The descriptor is rotation/translation/permutation invariant, so energies are
invariant and autograd forces are exactly equivariant. Dense channel mixing
(the part FACTO accelerates with BF16x3) lives in PrecLinear layers.
v0 limitation: invariant (L=0) features only; L>0 equivariant mixing is v1.
"""
import math
import torch
from torch import nn
from .layers import PrecLinear


class RadialMLIP(nn.Module):
    def __init__(self, n_rbf=16, width=64, depth=3, rc=2.5, r_min=0.8):
        super().__init__()
        self.rc = rc
        self.register_buffer("mu", torch.linspace(r_min, rc, n_rbf))
        self.eta = (n_rbf / (rc - r_min)) ** 2
        self.register_buffer("g_mean", torch.zeros(n_rbf))
        self.register_buffer("g_std", torch.ones(n_rbf))
        self.register_buffer("e_shift", torch.zeros(()))
        dims = [n_rbf] + [width] * depth + [1]
        self.layers = nn.ModuleList(PrecLinear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        for i, layer in enumerate(self.layers):
            layer.profile_name = f'radial_linear_{i}'

    def descriptors(self, pos, box):
        d = pos[:, None, :] - pos[None, :, :]
        d = d - box * torch.round(d / box)
        n = pos.shape[0]
        eye = torch.eye(n, dtype=torch.bool, device=pos.device)
        r2 = (d ** 2).sum(-1).masked_fill(eye, 4 * self.rc ** 2)
        r = torch.sqrt(r2)
        fc = torch.where(r < self.rc, 0.5 * (torch.cos(math.pi * r / self.rc) + 1), torch.zeros_like(r))
        g = torch.exp(-self.eta * (r[..., None] - self.mu) ** 2) * fc[..., None]
        return g.sum(1)

    def forward(self, pos, box):
        g = (self.descriptors(pos, box) - self.g_mean.to(pos.dtype)) / self.g_std.to(pos.dtype)
        h = g
        for i, lin in enumerate(self.layers):
            h = lin(h)
            if i < len(self.layers) - 1:
                h = torch.nn.functional.silu(h)
        return h.sum() + self.e_shift.to(pos.dtype) * pos.shape[0]

    def energy_forces(self, pos, box, dtype=torch.float32):
        p = torch.as_tensor(pos, dtype=dtype).clone().requires_grad_(True)
        e = self(p, box)
        (g,) = torch.autograd.grad(e, p)
        return e.detach(), -g.detach()
