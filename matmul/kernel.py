from pathlib import Path

import torch
import triton
import triton.language as tl
import time


def naive_matmul(a, b):

    assert a.shape[1] == b.shape[0], "Incompatible shapes for matrix multiplication"

    M = a.shape[0]
    K = a.shape[1]
    N = b.shape[1]

    # empty output matrix
    c = torch.zeros((M, N), device=a.device, dtype=torch.float32)

    for i in range(M):
        for j in range(N):
            for k in range(K):
                c[i][j] += a[i][k] * b[k][j]
    
    return c


def python_blockwise_matmul(a, b):
    assert a.shape[1] == b.shape[0], "Incompatible shapes for matrix multiplication"

    M = a.shape[0]
    K = a.shape[1]
    N = b.shape[1]

    # empty output matrix
    c = torch.zeros((M, N), device=a.device, dtype=a.dtype)

    block_size = 4

    for i in range(0, M, block_size):
        for j in range(0, N, block_size):
            for k in range(0, K, block_size):
                # Compute the block of C
                for ii in range(i, min(i + block_size, M)):
                    for jj in range(j, min(j + block_size, N)):
                        sum = 0.0
                        for kk in range(k, min(k + block_size, K)):
                            sum += a[ii][kk] * b[kk][jj]
                        c[ii][jj] += sum
    
    return c


@triton.jit
def triton_blockwise_matmul_kernel(a_ptr, b_ptr, c_ptr, M, K, N, BLOCK_SIZE: tl.constexpr):
    row_start = tl.program_id(axis=0)
    col_start = tl.program_id(axis=1)

    row_offsets = row_start * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col_offsets = col_start * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    c_block = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE, num_stage=4):
        k_offsets = k + tl.arange(0, BLOCK_SIZE)
        
        a_block = tl.load(
            a_ptr + row_offsets[:, None] * K + k_offsets[None, :],
            mask=(row_offsets[:, None] < M) & (k_offsets[None, :] < K),
            other=0.0
        )
        b_block = tl.load(
            b_ptr + k_offsets[:, None] * N + col_offsets[None, :],
            mask=(k_offsets[:, None] < K) & (col_offsets[None, :] < N),
            other=0.0
        )

        c_block += tl.dot(a_block, b_block)

    tl.store(
        c_ptr + row_offsets[:, None] * N + col_offsets[None, :],
        c_block,
        mask=(row_offsets[:, None] < M) & (col_offsets[None, :] < N)
    )


def triton_blockwise_matmul(a, b):
    assert a.shape[1] == b.shape[0], "Matrix A's columns must match Matrix B's rows"

    M, K, N  = a.shape[0], a.shape[1], b.shape[1]

    c = torch.zeros((M,N), device=a.device, dtype=a.dtype)

    BLOCK_SIZE = 64

    grid = (triton.cdiv(M, BLOCK_SIZE), triton.cdiv(N, BLOCK_SIZE))

    triton_blockwise_matmul_kernel[grid](a, b, c, M, K, N, BLOCK_SIZE=BLOCK_SIZE)

    return c


@triton.jit
def triton_grouped_matmul_kernel(a_ptr, b_ptr, c_ptr, M, K, N,
                                  BLOCK_SIZE: tl.constexpr, GROUP_SIZE: tl.constexpr):
    # 1D PID → group-major (row, col) mapping
    PID = tl.program_id(axis=0)
    num_blocks_m = tl.cdiv(M, BLOCK_SIZE)
    num_blocks_n = tl.cdiv(N, BLOCK_SIZE)
    num_pid_in_group = GROUP_SIZE * num_blocks_n

    group_id = PID // num_pid_in_group
    first_row = group_id * GROUP_SIZE
    group_size_adj = min(num_blocks_m - first_row, GROUP_SIZE)

    row_start = first_row + ((PID % num_pid_in_group) % group_size_adj)
    col_start = (PID % num_pid_in_group) // group_size_adj

    # same blockwise matmul logic from here
    row_offsets = row_start * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col_offsets = col_start * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    c_block = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE):
        k_offsets = k + tl.arange(0, BLOCK_SIZE)

        a_block = tl.load(
            a_ptr + row_offsets[:, None] * K + k_offsets[None, :],
            mask=(row_offsets[:, None] < M) & (k_offsets[None, :] < K),
            other=0.0
        )
        b_block = tl.load(
            b_ptr + k_offsets[:, None] * N + col_offsets[None, :],
            mask=(k_offsets[:, None] < K) & (col_offsets[None, :] < N),
            other=0.0
        )

        c_block += tl.dot(a_block, b_block)

    tl.store(
        c_ptr + row_offsets[:, None] * N + col_offsets[None, :],
        c_block,
        mask=(row_offsets[:, None] < M) & (col_offsets[None, :] < N)
    )


def triton_grouped_matmul(a, b):
    assert a.shape[1] == b.shape[0], "Matrix A's columns must match Matrix B's rows"

    M, K, N = a.shape[0], a.shape[1], b.shape[1]

    c = torch.zeros((M, N), device=a.device, dtype=a.dtype)

    BLOCK_SIZE = 64
    GROUP_SIZE = 8

    grid = (triton.cdiv(M, BLOCK_SIZE) * triton.cdiv(N, BLOCK_SIZE), )

    triton_grouped_matmul_kernel[grid](a, b, c, M, K, N, BLOCK_SIZE=BLOCK_SIZE, GROUP_SIZE=GROUP_SIZE)

    return c


# ─── Autotuned grouped matmul ───

autotune_configs = [
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE': 8}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE': 8}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE': 8}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 64, 'GROUP_SIZE': 8}, num_stages=3, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=5, num_warps=2),
    triton.Config({'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=5, num_warps=2),
    triton.Config({'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=5, num_warps=2),
]

@triton.autotune(configs=autotune_configs, key=['M', 'N', 'K'])
@triton.jit
def triton_autotuned_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, K, N,
    stride_a_m, stride_a_k,
    stride_b_k, stride_b_n,
    stride_c_m, stride_c_n,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    # 1D PID → group-major mapping (same idea as grouped kernel)
    PID = tl.program_id(axis=0)
    num_blocks_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_blocks_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE * num_blocks_n

    group_id = PID // num_pid_in_group
    first_row = group_id * GROUP_SIZE
    group_size_adj = min(num_blocks_m - first_row, GROUP_SIZE)

    PID_M = first_row + ((PID % num_pid_in_group) % group_size_adj)
    PID_N = (PID % num_pid_in_group) // group_size_adj

    # compute offsets for this block
    offsets_m = PID_M * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = PID_N * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)

    # 2D pointer grids for A and B (using strides for non-contiguous support)
    a_offsets = offsets_m[:, None] * stride_a_m + offsets_k[None, :] * stride_a_k
    b_offsets = offsets_k[:, None] * stride_b_k + offsets_n[None, :] * stride_b_n

    # accumulate in float32 for precision
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # k-loop: walk through K dimension in chunks of BLOCK_SIZE_K
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        mask_k = offsets_k < K - k * BLOCK_SIZE_K

        a_block = tl.load(a_ptr + a_offsets, mask=mask_k[None, :], other=0.0)
        b_block = tl.load(b_ptr + b_offsets, mask=mask_k[:, None], other=0.0)

        accumulator = tl.dot(a_block, b_block, acc=accumulator)

        # slide pointers forward along K
        a_offsets += BLOCK_SIZE_K * stride_a_k
        b_offsets += BLOCK_SIZE_K * stride_b_k

    # store result
    c_offsets = offsets_m[:, None] * stride_c_m + offsets_n[None, :] * stride_c_n
    c_mask = (offsets_m[:, None] < M) & (offsets_n[None, :] < N)
    tl.store(c_ptr + c_offsets, accumulator, mask=c_mask)


def triton_grouped_autotuned_matmul(a, b):
    assert a.shape[1] == b.shape[0], "Matrix A's columns must match Matrix B's rows"

    M, K, N = a.shape[0], a.shape[1], b.shape[1]

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # 1D grid — size depends on which config autotune picks, so use a lambda
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']), )

    triton_autotuned_matmul_kernel[grid](
        a, b, c,
        M, K, N,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )

    return c


def test_matmul():
    M, K, N = 1023, 1023, 1023
    a = torch.randn((M, K), device='cuda')
    b = torch.randn((K, N), device='cuda')

    c_triton = triton_blockwise_matmul(a, b)
    c_grouped = triton_grouped_matmul(a, b)
    c_grouped_autotuned = triton_grouped_autotuned_matmul(a, b)
    c_torch = torch.matmul(a, b)

    assert torch.allclose(c_triton, c_torch, atol=1e-3, rtol=1e-3), "Triton blockwise matmul does not match PyTorch's matmul"
    assert torch.allclose(c_grouped, c_torch, atol=1e-3, rtol=1e-3), "Triton grouped matmul does not match PyTorch's matmul"
    assert torch.allclose(c_grouped_autotuned, c_torch, atol=1e-3, rtol=1e-3), "Triton grouped autotuned matmul does not match PyTorch's matmul"
    print("PASSED!")


DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],
        x_vals=[128 * i for i in range(2, 33)],
        line_arg='provider',
        line_vals=['torch', 'triton_blockwise', 'triton_grouped', 'triton_grouped_autotuned'],
        line_names=['PyTorch', 'Triton Blockwise', 'Triton Grouped', 'Triton Grouped Autotuned'],
        styles=[('green', '-'), ('blue', '-'), ('red', '-'), ('orange', '-')],
        ylabel='TFLOPS',
        plot_name='matmul-performance',
        args={},
    ))
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float32)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float32)

    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.matmul(a, b))
    elif provider == 'triton_blockwise':
        ms = triton.testing.do_bench(lambda: triton_blockwise_matmul(a, b))
    elif provider == 'triton_grouped':
        ms = triton.testing.do_bench(lambda: triton_grouped_matmul(a, b))
    elif provider == 'triton_grouped_autotuned':
        ms = triton.testing.do_bench(lambda: triton_grouped_autotuned_matmul(a, b))

    tflops = 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return tflops


if __name__ == "__main__":
    test_matmul()

    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark.run(save_path=str(Path(__file__).parent / 'results'), print_data=True)