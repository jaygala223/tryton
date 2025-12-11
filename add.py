import torch
import triton
import triton.language as tl
# from benchmarking import benchmark
from benchmarking_2 import benchmark


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    output = x + y

    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):

    output = torch.empty_like(x).cuda()

    assert x.is_cuda and y.is_cuda and output.is_cuda, "All tensors must be on CUDA"

    n_elements = output.numel()

    BLOCK_SIZE = 1024

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return output

if __name__ == "__main__":

    benchmark.run(kernel_1=add, kernel_2=lambda x, y: x + y, print_data=True, save_path="./")
