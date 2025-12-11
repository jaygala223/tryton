import torch
import triton
import triton.language as tl

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@triton.jit
def softmax_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    output = tl.exp(x)
    output = output / tl.sum(output)

    tl.store(output_ptr + offsets, output, mask=mask)


def softmax(x: torch.Tensor):

    output = torch.empty_like(x).cuda()

    assert x.is_cuda and output.is_cuda, "All tensors must be on CUDA"

    n_elements = output.numel()

    BLOCK_SIZE = 1024

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    softmax_kernel[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return output


# regular softmax implementation
def regular_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:

    numerator = torch.exp(x)
    denominator = torch.sum(numerator, dim=dim, keepdim=True)

    return numerator / denominator

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(12, 28, 1)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='softmax-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))


def benchmark(size, provider, kernel_1=None, kernel_2=None):
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: kernel_1(x), quantiles=quantiles)
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: kernel_2(x), quantiles=quantiles)
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":

    answer = regular_softmax(torch.tensor([2.0, 1.0, 0.1]))
    print(answer)  # tensor([0.6590, 0.2424, 0.0986])

    benchmark.run(kernel_1=softmax, kernel_2=regular_softmax, print_data=True, save_path="./", show_plots=True)

    print("\n\nTesting Wall Clock Time Now...\n\n")

    for size in [2**i for i in range(12, 20, 1)]:
        x = torch.rand(size, device=DEVICE, dtype=torch.float32)

        # Warmup
        for _ in range(10):
            softmax(x)
            regular_softmax(x)

        # Timing Triton Softmax
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(100):
            softmax(x)
        end.record()

        torch.cuda.synchronize()
        triton_time = start.elapsed_time(end) / 100  # in milliseconds

        # Timing Torch Softmax
        torch.cuda.synchronize()
        start.record()
        for _ in range(100):
            regular_softmax(x)
        end.record()

        torch.cuda.synchronize()
        torch_time = start.elapsed_time(end) / 100  # in milliseconds

        print(f"Size: {size:>10}, Triton Time: {triton_time:.4f} ms, Torch Time: {torch_time:.4f} ms")