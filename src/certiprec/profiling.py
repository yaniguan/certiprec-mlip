"""Small opt-in CPU timers and operation counts; counts cover forward matmuls."""
import os
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext

import torch

_ENABLED = os.getenv('CERTIPREC_PROFILE', '0') == '1'
_ROWS = defaultdict(lambda: {'calls': 0, 'seconds': 0.0, 'flops': 0})


def enable(value=True):
    global _ENABLED
    _ENABLED = value


@contextmanager
def layer_timer(name, flops=0):
    if not _ENABLED:
        yield
        return
    start = time.perf_counter()
    with torch.profiler.record_function(name):
        yield
    row = _ROWS[name]
    row['calls'] += 1
    row['seconds'] += time.perf_counter() - start
    row['flops'] += int(flops)


def snapshot(reset=False):
    rows = {k: dict(v) for k, v in _ROWS.items()}
    if reset:
        _ROWS.clear()
    return rows


def torch_profile(enabled=False):
    return torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                  with_flops=True) if enabled else nullcontext()
