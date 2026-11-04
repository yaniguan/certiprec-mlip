"""Adaptive BF16xk precision with local and finite-set error certificates."""
from .precision import (round_bf16, round_tf32, split_bf16, emu_matmul, SCHEMES,
                        products, error_bound)
from .layers import PrecLinear, set_schemes, get_schemes
__version__ = "0.1.0"
