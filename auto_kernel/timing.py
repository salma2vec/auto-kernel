"""Synchronized timing for pure forward kernels; input preparation is external."""
from time import perf_counter
import torch


def synchronize(device):
    if device.startswith('cuda'):
        torch.cuda.synchronize(device)
    elif device.startswith('mps'):
        torch.mps.synchronize()


def benchmark_callable(fn, device, warmup, iterations):
    if iterations < 1 or warmup < 0:
        raise ValueError('iterations must be positive and warmup nonnegative')
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        synchronize(device)
        timings = []
        for _ in range(iterations):
            synchronize(device)
            start = perf_counter()
            fn()
            synchronize(device)
            timings.append(perf_counter() - start)
    return sum(timings) / len(timings)
