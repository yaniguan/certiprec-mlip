"""Small O(3) model: radial × spherical harmonics, atom mixing, quadratic invariants.

One message pass with l=0,1 features and correlation order two. This is a toy
MACE/e3nn-style model, not a MACE checkpoint or a FACTO implementation.
"""
import math

import cuequivariance as cue
import torch
from torch import nn

from .layers import PrecLinear
from .segmented import SegmentedTP


class EquivariantMLIP(nn.Module):
    def __init__(self, n_rbf=12, width=4, rc=2.5):
        super().__init__()
        self.n_rbf, self.width, self.rc = n_rbf, width, rc
        self.register_buffer('centers', torch.linspace(0.85, rc, n_rbf))
        self.sh = SegmentedTP(cue.descriptors.spherical_harmonics(cue.O3(1, -1), [0, 1]), 'spherical_harmonics')
        radial = cue.Irreps('O3', f'{n_rbf}x0e')
        angular = cue.Irreps('O3', '0e + 1o')
        msg = cue.descriptors.channelwise_tensor_product(radial, angular)
        self.message = SegmentedTP(msg, 'edge_tensor_product')
        hidden = cue.Irreps('O3', f'{width}x0e + {width}x1o')
        mix = cue.descriptors.linear(msg.outputs[0].irreps, hidden)
        self.mix = SegmentedTP(mix, 'atom_linear')
        square = cue.descriptors.channelwise_tensor_product(hidden, hidden, [cue.O3(0, 1)])
        self.square = SegmentedTP(square, 'quadratic_invariants')
        self.message_weights = nn.Parameter(torch.ones(1, msg.polynomial.inputs[0].size))
        self.mix_weights = nn.Parameter(torch.randn(1, mix.polynomial.inputs[0].size) / math.sqrt(n_rbf))
        self.square_weights = nn.Parameter(torch.ones(1, square.polynomial.inputs[0].size))
        self.readout = PrecLinear(n_rbf + square.polynomial.outputs[0].size, 1)
        self.readout.profile_name = 'readout'

    def features(self, pos, box=None):
        n = len(pos)
        i, j = torch.where(~torch.eye(n, device=pos.device, dtype=torch.bool))
        d = pos[j] - pos[i]
        if box is not None:
            d = d - box * torch.round(d / box)
        r = d.norm(dim=-1).clamp_min(1e-6)
        cutoff = torch.where(r < self.rc, (1 - r / self.rc) ** 3, torch.zeros_like(r))
        radial = torch.exp(-((r[:, None] - self.centers.to(pos.dtype)) * 5) ** 2) * cutoff[:, None]
        y = self.sh(d / r[:, None])
        edge = self.message(self.message_weights.to(pos.dtype), radial, y)
        atom = pos.new_zeros(n, edge.shape[-1]).index_add(0, i, edge)
        hidden = self.mix(self.mix_weights.to(pos.dtype), atom)
        invariants = self.square(self.square_weights.to(pos.dtype), hidden, hidden)
        radial_atom = pos.new_zeros(n, self.n_rbf).index_add(0, i, radial)
        return torch.cat([radial_atom, invariants], dim=-1)

    def forward(self, pos, box=None):
        return self.readout(self.features(pos, box)).sum()

    def energy_forces(self, pos, box=None, dtype=torch.float32):
        p = torch.as_tensor(pos, dtype=dtype, device=self.centers.device).clone().requires_grad_(True)
        e = self(p, box)
        (g,) = torch.autograd.grad(e, p)
        return e.detach(), -g.detach()
