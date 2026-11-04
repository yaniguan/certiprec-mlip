import torch
from torch import nn
from .precision import emu_matmul, products
from .profiling import layer_timer


class PrecLinear(nn.Linear):
    """nn.Linear whose matmul (forward AND force backward) uses a selectable scheme."""
    scheme: str = "fp32"

    def forward(self, x):
        with layer_timer(getattr(self, 'profile_name', 'linear'),
                         2 * x.numel() * self.out_features * products(self.scheme)):
            y = emu_matmul(x, self.weight.t().to(x.dtype), self.scheme)
        return y.to(x.dtype) + (self.bias.to(x.dtype) if self.bias is not None else 0)


def prec_layers(model):
    return [m for m in model.modules() if isinstance(m, PrecLinear) or getattr(m, "precision_layer", False)]


def set_schemes(model, schemes):
    layers = prec_layers(model)
    if isinstance(schemes, str):
        schemes = [schemes] * len(layers)
    assert len(schemes) == len(layers)
    for m, s in zip(layers, schemes):
        m.scheme = s


def get_schemes(model):
    return [m.scheme for m in prec_layers(model)]
