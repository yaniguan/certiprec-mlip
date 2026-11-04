import torch
from torch import nn
from .precision import emu_matmul


class PrecLinear(nn.Linear):
    """nn.Linear whose matmul (forward AND force backward) uses a selectable scheme."""
    scheme: str = "fp32"

    def forward(self, x):
        y = emu_matmul(x, self.weight.t().to(x.dtype) if self.scheme in ("fp32", "fp64")
                       else self.weight.t(), self.scheme)
        return y.to(x.dtype) + self.bias.to(x.dtype)


def prec_layers(model):
    return [m for m in model.modules() if isinstance(m, PrecLinear)]


def set_schemes(model, schemes):
    layers = prec_layers(model)
    if isinstance(schemes, str):
        schemes = [schemes] * len(layers)
    assert len(schemes) == len(layers)
    for m, s in zip(layers, schemes):
        m.scheme = s


def get_schemes(model):
    return [m.scheme for m in prec_layers(model)]
