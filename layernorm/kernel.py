"""
LayerNorm, where we finally connect our kernels to PyTorch's backpropagation graph.
This kernel is fast but only works when a full row fits in SRAM, so we trade generality
for speed.

What you'll learn:
- Writing a backward pass kernel
- Re-using intermediate values from the forward pass in the backward pass
- Hooking into PyTorch's autograd graph with torch.autograd.Function
- Locks and atomic operations
- Splitting a calculation across sequential kernels with intermediate tensors

Recommended order to read the code in:
Step 1 - unit test
Step 2 - wrapper
Step 3 - forward pass kernel
Step 4 - backward pass kernels
Step 5 - benchmark

see original triton documentation:
https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
"""
from pathlib import Path

import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')


######### Step 3 #########
@triton.jit
def _layernorm_forward(
    x_ptr, y_ptr,           # tensors of shape (M, N)
    w_ptr, b_ptr,           # tensors of shape (N)
    mean_ptr, rstd_ptr,     # tensors of shape (M)
    stride_M,               # elements to skip when moving to the next row of x
    N,                      # embedding dimension
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # one program per row
    row = tl.program_id(0)
    x_ptr += row * stride_M
    y_ptr += row * stride_M

    # mean
    sum_accumulator = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        # x may be fp16 but we accumulate in fp32 for accuracy, and other=0. because
        #  zeros don't affect a summation
        x = tl.load(x_ptr + cols, mask=cols < N, other=0.).to(tl.float32)
        sum_accumulator += x
    mean = tl.sum(sum_accumulator, axis=0) / N

    # variance & reciprocal standard deviation
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + cols, mask=cols < N, other=0.).to(tl.float32)
        # mask here to avoid the out-of-bounds entries contributing (0. - mean)
        diff = tl.where(cols < N, x - mean, 0.)
        acc += diff * diff
    var = tl.sum(acc, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    # stash mean & rstd so the backward pass doesn't have to recompute them
    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)

    # normalize and apply the linear transformation
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(w_ptr + cols, mask=mask)
        b = tl.load(b_ptr + cols, mask=mask)
        x = tl.load(x_ptr + cols, mask=mask)

        x_hat = (x - mean) * rstd
        y = x_hat * w + b

        tl.store(y_ptr + cols, y, mask=mask)


######### Step 4 #########
@triton.jit
def _layernorm_backward_dLdx(
    x_ptr, dLdx_ptr, dLdy_ptr,                              # tensors of shape (M, N)
    w_ptr,                                                  # tensor of shape (N)
    dLdw_intermediate_ptr, dLdb_intermediate_ptr,           # tensors of shape (GROUP_SIZE, N)
    mean_ptr, rstd_ptr,                                     # tensors of shape (M)
    locks_ptr,                                              # tensor of shape (2 * GROUP_SIZE)
    stride, N,                                              # run-time values
    GROUP_SIZE: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,   # compile-time values
):
    """
    Each PID owns one row: it writes that row of dLdx outright, but its contribution to
    dLdw/dLdb is only a partial sum since w and b receive gradients from all M rows.
    PIDs are assigned to GROUP_SIZE interleaved groups, and each group accumulates into
    its own row of dLdw_intermediate/dLdb_intermediate. The next kernel reduces those
    (GROUP_SIZE, N) buffers down to the final (N,) gradients.
    """
    PID = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N  # a whole row lives in one block, so this is the only mask we need
    x_ptr += PID * stride
    dLdx_ptr += PID * stride
    dLdy_ptr += PID * stride

    # it's generally faster to batch loads together rather than alternating loads and flops
    x = tl.load(x_ptr + cols, mask=mask, other=0).to(tl.float32)
    dLdy = tl.load(dLdy_ptr + cols, mask=mask, other=0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask).to(tl.float32)
    mean = tl.load(mean_ptr + PID)
    rstd = tl.load(rstd_ptr + PID)

    x_normalized = tl.where(mask, (x - mean) * rstd, 0.)
    dydx_normed = tl.where(mask, w * dLdy, 0.)
    # c1 and c2 are just intermediary labels; the names don't have any real meaning
    c1 = tl.sum(x_normalized * dydx_normed, axis=0) / N
    c2 = tl.sum(dydx_normed, axis=0) / N
    dLdx = (dydx_normed - (x_normalized * c1 + c2)) * rstd

    tl.store(dLdx_ptr + cols, dLdx, mask=mask)

    # this PID's partial contribution to the (N,) shaped weight & bias gradients
    dLdw_contribution = (dLdy * x_normalized).to(w.dtype)
    dLdb_contribution = (dLdy).to(w.dtype)

    """
    We can't naively load-add-store into the shared buffer, since every PID in a group would
    do so at unpredictable times and clobber each other's writes. So we use a lock: a tensor
    of shape (2 * GROUP_SIZE) and dtype int32 initialized to zeros.
    - The first GROUP_SIZE entries hold the state of each lock (0 = free, 1 = held).
    - The next GROUP_SIZE entries record whether a lock has ever been used, because the very
        first writer can skip the read-and-add and just store its own values.
    Only M // GROUP_SIZE PIDs ever queue on any one lock, so the groups still run in parallel
    with each other.
    """
    lock_id = PID % GROUP_SIZE
    locks_ptr += lock_id
    count_ptr = locks_ptr + GROUP_SIZE
    # we can use N in place of a .stride() here since these tensors are allocated specifically
    #  for this purpose and are therefore guaranteed to be contiguous
    dLdw_intermediate_ptrs = dLdw_intermediate_ptr + lock_id * N + cols
    dLdb_intermediate_ptrs = dLdb_intermediate_ptr + lock_id * N + cols

    # atomic_cas() compares the value at a memory location against a given value and, only if
    #  they match, swaps in a new value. Here: if it's 0 (free) set it to 1 (held) and return 0
    #  to exit the loop; if it's already 1, leave it and return 1 so we keep spinning.
    while tl.atomic_cas(locks_ptr, 0, 1) == 1:
        pass

    count = tl.load(count_ptr)
    if count == 0:
        # first PID to reach this lock: skip the accumulation and flag the lock as used
        tl.atomic_xchg(count_ptr, 1)
    else:
        # load and add in one step (+=) so as not to consume unnecessary SRAM
        dLdw_contribution += tl.load(dLdw_intermediate_ptrs, mask=mask)
        dLdb_contribution += tl.load(dLdb_intermediate_ptrs, mask=mask)

    tl.store(dLdw_intermediate_ptrs, dLdw_contribution, mask=mask)
    tl.store(dLdb_intermediate_ptrs, dLdb_contribution, mask=mask)

    # release the lock; whichever waiting PID's atomic_cas() sees the 0 first goes next
    tl.atomic_xchg(locks_ptr, 0)


@triton.jit
def _layernorm_backward_dLdw_dLdb(
    dLdw_intermediate_ptr, dLdb_intermediate_ptr,   # tensors of shape (GROUP_SIZE, N)
    dLdw_ptr, dLdb_ptr,                             # tensors of shape (N)
    GROUP_SIZE, N,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    # here PIDs are split along the N dimension rather than across rows
    PID = tl.program_id(0)
    col_ptrs = PID * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    dLdw_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    dLdb_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for i in range(0, GROUP_SIZE, BLOCK_SIZE_M):
        row_ptrs = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (row_ptrs[:, None] < GROUP_SIZE) & (col_ptrs[None, :] < N)
        offsets = row_ptrs[:, None] * N + col_ptrs[None, :]

        # other=0. so masked-out entries don't affect the sum
        dLdw_acc += tl.load(dLdw_intermediate_ptr + offsets, mask=mask, other=0.)
        dLdb_acc += tl.load(dLdb_intermediate_ptr + offsets, mask=mask, other=0.)

    sum_dLdw = tl.sum(dLdw_acc, axis=0)
    sum_dLdb = tl.sum(dLdb_acc, axis=0)

    tl.store(dLdw_ptr + col_ptrs, sum_dLdw, mask=col_ptrs < N)
    tl.store(dLdb_ptr + col_ptrs, sum_dLdb, mask=col_ptrs < N)


######### Step 2 #########
class LayerNorm(torch.autograd.Function):
    """
    Subclassing torch.autograd.Function with static forward() and backward() methods is how
    we make a custom op that plays nice with PyTorch's autograd graph.
    """

    @staticmethod
    def forward(
        ctx,                # supplied by the parent class, not by the caller
        x,
        normalized_shape,   # unused, but kept so the signature matches PyTorch's
        weight,
        bias,
        eps,
    ):
        M, N = x.reshape(-1, x.shape[-1]).shape
        mean = torch.empty((M, ), dtype=torch.float32, device=x.device)
        rstd = torch.empty((M, ), dtype=torch.float32, device=x.device)
        y = torch.empty_like(x)

        # 64KB is a conservative guess at the smallest SRAM our GPU is likely to have, and
        #  .element_size() is the bytes per entry, so this is how many entries fit in SRAM
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            # supporting this would require parallelizing within the feature dimension
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)

        _layernorm_forward[(M, )](  # one program per row
            x, y, weight, bias,
            mean, rstd,
            x.stride(0),
            N,
            eps,
            BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
        )

        # ctx stashes anything the backward pass will need; tensors go through
        #  save_for_backward while meta-parameters are set as plain attributes
        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps

        return y

    @staticmethod
    def backward(
        ctx,
        dLdy,  # gradient of the loss with respect to y
    ):
        x, w, b, mean, rstd = ctx.saved_tensors
        M, N = x.reshape(-1, x.shape[-1]).shape

        dLdw = torch.empty((N, ), dtype=w.dtype, device=w.device)
        dLdb = torch.empty((N, ), dtype=w.dtype, device=w.device)
        dLdx = torch.empty_like(dLdy)

        # heuristics for how many parallel reduction streams to use for dLdw & dLdb
        GROUP_SIZE = 64
        if N <= 8192: GROUP_SIZE = 96
        if N <= 4096: GROUP_SIZE = 128
        if N <= 1024: GROUP_SIZE = 256

        # partial sums that the first kernel accumulates into and the second kernel reduces
        dLdw_intermediate = torch.zeros((GROUP_SIZE, N), dtype=x.dtype, device=w.device)
        dLdb_intermediate = torch.zeros((GROUP_SIZE, N), dtype=x.dtype, device=w.device)

        # first GROUP_SIZE entries track whether each lock is held; the rest track whether a
        #  lock has been used before, since the first use is handled differently in the kernel
        locks = torch.zeros(2 * GROUP_SIZE, dtype=torch.int32, device=w.device)

        _layernorm_backward_dLdx[(M, )](  # parallelize across rows
            x, dLdx, dLdy,
            w, dLdw_intermediate, dLdb_intermediate,
            mean, rstd,
            locks,
            x.stride(0), N,
            GROUP_SIZE=GROUP_SIZE, BLOCK_SIZE_N=ctx.BLOCK_SIZE, num_warps=ctx.num_warps,
        )

        # a separate kernel because this reduction needs far fewer PIDs than the one above and
        #  can't start until every PID there has finished writing its partial sums
        grid = lambda meta: [triton.cdiv(N, meta['BLOCK_SIZE_N'])]  # parallelize within rows
        _layernorm_backward_dLdw_dLdb[grid](
            dLdw_intermediate, dLdb_intermediate, dLdw, dLdb,
            min(GROUP_SIZE, M), N,
            BLOCK_SIZE_M=32, BLOCK_SIZE_N=128,
        )

        # backward() must return one value per input of forward() (excluding ctx) and in the
        #  same order; None marks the inputs that don't need a gradient
        return dLdx, None, dLdw, dLdb, None


# a reference to .apply so we can call this like a function
layernorm = LayerNorm.apply


######### Step 1 #########
def test_layernorm_kernel(M, N, dtype, eps=1e-5, device=DEVICE):
    x = -2.3 + 0.5 * torch.randn((M, N), dtype=dtype, device=device)
    weight = torch.rand((N, ), dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand((N, ), dtype=dtype, device=device, requires_grad=True)
    dLdy = .1 * torch.randn_like(x)
    # flipping requires_grad on here rather than at x's definition keeps the -2.3 and 0.5
    #  ops out of the graph, which matters in the benchmark below where they'd otherwise
    #  confound our timings with PyTorch's elementwise kernels
    x.requires_grad_(True)

    y_tri = layernorm(x, (N,), weight, bias, eps)
    y_ref = torch.nn.functional.layer_norm(x, (N,), weight, bias, eps).to(dtype)
    torch.testing.assert_close(y_tri, y_ref, atol=1e-2, rtol=0)  # rtol=0 -> absolute only
    print("Passed fwd")

    # retain_graph lets us run backward twice on the same graph (once for triton, once for
    #  torch); it costs memory so only use it when you actually need it
    y_tri.backward(dLdy, retain_graph=True)  # writes into x.grad, weight.grad, bias.grad
    # clone so the gradients survive the reset below
    dLdx_tri, dLdw_tri, dLdb_tri = [_.grad.clone() for _ in [x, weight, bias]]
    x.grad, weight.grad, bias.grad = None, None, None

    y_ref.backward(dLdy, retain_graph=True)
    dLdx_ref, dLdw_ref, dLdb_ref = [_.grad.clone() for _ in [x, weight, bias]]

    torch.testing.assert_close(dLdx_tri, dLdx_ref, atol=1e-2, rtol=0)
    torch.testing.assert_close(dLdb_tri, dLdb_ref, atol=1e-2, rtol=0)
    torch.testing.assert_close(dLdw_tri, dLdw_ref, atol=1e-2, rtol=0)
    print("Passed bwd")


######### Step 5 #########
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        # past 32 the kernel breaks because a row no longer fits in 64KB
        x_vals=[512 * i for i in range(2, 32)],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='layer-norm-backward',
        args={'M': 4096, 'dtype': torch.float16, 'mode': 'backward'},
    ))
def benchmark(M, N, dtype, provider, mode='backward', eps=1e-5, device=DEVICE):
    x_shape = (M, N)
    w_shape = (N, )
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    dLdy = .1 * torch.randn_like(x)
    # see the note in the unit test about why requires_grad_ is set here
    x.requires_grad_(True)
    quantiles = [0.5, 0.05, 0.95]

    def y_fwd():
        if provider == "triton":
            return layernorm(x, w_shape, weight, bias, eps)
        if provider == "torch":
            return torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps)

    if mode == 'forward':
        gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=500)
    if mode == 'backward':
        y = y_fwd()
        gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)  # noqa: F811, E704
        # grad_to_none clears x.grad between runs so gradient accumulation isn't timed
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: y.backward(dLdy, retain_graph=True),
                                                     quantiles=quantiles, grad_to_none=[x], rep=500)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    test_layernorm_kernel(1151, 8192, torch.float16)

    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark.run(save_path=str(Path(__file__).parent / 'results'), print_data=True)
