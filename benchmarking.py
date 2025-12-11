# benchmarking utilities for different triton kernels

import torch
import time

def benchmark(kernel_1, kernel_2):

    """
    Benchmark two functions: kernel_1 is triton and kernel_2 is pytorch.
    """

    x = torch.randn(1024, 1024).cuda()
    y = torch.randn(1024, 1024).cuda()

    # warm up
    for _ in range(10):
        kernel_1(x, y)
        kernel_2(x, y)

    sizes = [2**i for i in range(5, 16)]

    kernel_1_times, kernel_2_times = [], []

    for size in sizes:

        x = torch.randn(size, size).cuda()
        y = torch.randn(size, size).cuda()

        # benchmark kernel_1
        torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in range(10):
            kernel_1(x, y)

        torch.cuda.synchronize()
        end = time.perf_counter()
        kernel_1_times.append((end - start) / 10)

        # benchmark kernel_2
        torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in range(10):
            kernel_2(x, y)

        torch.cuda.synchronize()
        end = time.perf_counter()
        kernel_2_times.append((end - start) / 10)

    return sizes, kernel_1_times, kernel_2_times
