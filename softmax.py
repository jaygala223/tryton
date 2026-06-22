import torch
import triton
import triton.language as tl

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@triton.jit
def softmax_kernel(x_ptr, output_ptr, n_cols, BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(axis=0)
    row_start = pid * n_cols
    # block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    x = tl.load(x_ptr + row_start + offsets, mask=mask)
    x = x - tl.max(x)
    numerator = tl.exp(x)
    output = numerator / tl.sum(numerator)

    tl.store(output_ptr + row_start + offsets, output, mask=mask)


def softmax(x: torch.Tensor):

    output = torch.empty_like(x).cuda()

    assert x.is_cuda and output.is_cuda, "All tensors must be on CUDA"

    n_rows, n_cols = x.shape

    n_elements = output.numel()

    BLOCK_SIZE = triton.next_power_of_2(n_cols) # We will process one row per block, so block size is number of columns

    # grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    grid = (n_rows,)  # We launch one block per row, so grid size is number of rows

    softmax_kernel[grid](x, output, n_cols, BLOCK_SIZE=BLOCK_SIZE)

    return output


def test_softmax_kernel(size: tuple, atol=1e-3, rtol=1e-3, device=DEVICE):
    """
    Here is where we test the wrapper function and kernel that we wrote 
    above to ensure all our values are correct, using pytorch as the 
    correct answer to compare against

    we'll use an irregular number of rows & cols to verify that our padding mechanism works
    """
    torch.manual_seed(0)
    assert type(size) is tuple and len(size) == 2
    x = torch.randn(size[0], size[1], device=DEVICE)

    z_tri = softmax(x)
    z_ref = torch.softmax(x, axis=1)
    torch.testing.assert_close(z_tri, z_ref, atol=atol, rtol=rtol)
    print("PASSED")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[128 * i for i in range(2, 100)],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=["Triton", "Torch"],
        styles=[('blue', '-'), ('green', '-')],
        ylabel="GB/s",
        plot_name="triton-softmax-performance",
        args={'M': 4096} # values for function arguments not in x_names
    ))
def benchmark(M, N, provider):
    # making the input data
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)

    # these two lines ensure more accurate benchmarks; i usually forget to use them but it's not a big deal
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)

    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: softmax(x))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        # 2 = number of memory operations (1 read + 1 write)
        # x.numel() = number of elements
        # x.element_size() = bytes per element (4 for float32)
        # 1e-9 converts bytes to GB
        # 1e-3 converts milliseconds to seconds
    return gbps(ms)


if __name__ == "__main__":

    test_softmax_kernel(size=(10215, 1021))

    benchmark.run(save_path='.', print_data=True)