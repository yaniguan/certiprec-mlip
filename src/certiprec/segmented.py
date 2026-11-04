"""CPU interpreter of cuEquivariance descriptors with selectable matmul arithmetic.

The descriptors and coefficient tables come from cuEquivariance (Apache-2.0).
This interpreter is original code. See docs/BORROWED.md.
"""
import math

import torch
from torch import nn

from .precision import emu_matmul, products
from .profiling import layer_timer


def binary_contract(a, sa, b, sb, out, scheme):
    """Lower a binary einsum (unique labels, no ellipsis) to a batched GEMM."""
    # Sum labels that occur in only one operand and are absent from the result.
    for side in (0, 1):
        t, labels, other = (a, sa, sb) if side == 0 else (b, sb, sa)
        for label in list(labels):
            if label not in other and label not in out:
                t = t.sum(labels.index(label))
                labels = labels.replace(label, '')
        if side == 0:
            a, sa = t, labels
        else:
            b, sb = t, labels
    batch = ''.join(c for c in out if c in sa and c in sb)
    summed = ''.join(c for c in sa if c in sb and c not in out)
    left = ''.join(c for c in out if c in sa and c not in sb)
    right = ''.join(c for c in out if c in sb and c not in sa)
    dims = {}
    for t, labels in ((a, sa), (b, sb)):
        for c, size in zip(labels, t.shape):
            dims[c] = max(dims.get(c, 1), size)
    def pack(t, labels, order):
        return t.permute([labels.index(c) for c in order]).expand([dims[c] for c in order])
    bs = [dims[c] for c in batch]
    m, k, n = [math.prod(dims[c] for c in s) for s in (left, summed, right)]
    aa = pack(a, sa, batch + left + summed).reshape(bs + [m, k])
    bb = pack(b, sb, batch + summed + right).reshape(bs + [k, n])
    y = emu_matmul(aa, bb, scheme).reshape([dims[c] for c in batch + left + right])
    return y.permute([(batch + left + right).index(c) for c in out])


class SegmentedTP(nn.Module):
    """Evaluate one output polynomial. Supports repeated inputs and constants."""
    scheme = 'fp32'
    precision_layer = True

    def __init__(self, descriptor, name):
        super().__init__()
        self.polynomial = descriptor.polynomial
        self.name = name
        self.plans = []
        self.flops_per_item = 0
        for op, d in self.polynomial.operations:
            slices = [o.segment_slices() for o in d.operands]
            for path in d.paths:
                key = f'coeff_{len(self.plans)}'
                self.register_buffer(key, torch.tensor(path.coefficients.copy(), dtype=torch.float64))
                inputs = [(bid, slices[i][path.indices[i]], d.get_segment_shape(i, path),
                           'Z' + str(d.subscripts.operands[i]))
                          for i, bid in enumerate(op.buffers[:-1])]
                target = slices[-1][path.indices[-1]]
                shape = d.get_segment_shape(-1, path)
                self.plans.append((key, str(d.coefficient_subscripts), inputs, target,
                                   shape, 'Z' + str(d.subscripts.operands[-1])))
                dims = d.get_path_dimensions_dict(path)
                self.flops_per_item += 2 * max(1, math.prod(dims.values())) * max(1, len(inputs))

    def forward(self, *xs):
        batch = max(x.shape[0] for x in xs)
        output = xs[0].new_zeros(batch, self.polynomial.outputs[0].size)
        with layer_timer(self.name, batch * self.flops_per_item * products(self.scheme)):
            for key, sc, inputs, target, shape, out in self.plans:
                a = getattr(self, key).to(xs[0].dtype)
                labels = sc
                terms = [(xs[bid][..., sl].reshape(xs[bid].shape[0], *sh), ss)
                         for bid, sl, sh, ss in inputs]
                for i, (b, sb) in enumerate(terms):
                    needed = out + ''.join(ss for _, ss in terms[i + 1:])
                    result = ''.join(dict.fromkeys(c for c in labels + sb if c in needed))
                    a = binary_contract(a, labels, b, sb, result, self.scheme)
                    labels = result
                if 'Z' not in labels:
                    a = a.unsqueeze(0).expand(batch, *a.shape)
                    labels = 'Z' + labels
                a = torch.einsum(labels + '->' + out, a)
                output[..., target] = output[..., target] + a.reshape(batch, math.prod(shape))
        return output
