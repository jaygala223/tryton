from pathlib import Path

import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

@triton.jit
def _seeded_dropout(
    x_ptr,
    output_ptr,
    n_elements,
    p, # a float32 probability, so range [0,1]
    seed, # a single int32
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed, offsets)
    print(random[0], random.shape)
    x_keep = random > p
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)

def seeded_dropout(x, p, seed):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    _seeded_dropout[grid](x, output, n_elements, p, seed, BLOCK_SIZE=1024)
    return output

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[2**i for i in range(12, 26)],
        x_log=True,
        line_arg='provider',
        line_vals=['torch', 'triton'],
        line_names=['PyTorch', 'Triton'],
        styles=[('green', '-'), ('blue', '-')],
        ylabel='GB/s',
        plot_name='dropout-performance',
        args={'p': 0.5},
    ))

def benchmark(N, p, provider):
    x = torch.randn(N, device=DEVICE, dtype=torch.float32)
    torch_dropout = torch.nn.Dropout(p=p)
    seed = 123

    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch_dropout(x))
    elif provider == 'triton':
        ms = triton.testing.do_bench(lambda: seeded_dropout(x, p, seed))

    gbps = 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps


if __name__ == "__main__":
    # quick sanity check
    x = torch.randn(size=(8, ), device=DEVICE)
    output1 = seeded_dropout(x, p=0.5, seed=123)
    output2 = seeded_dropout(x, p=0.5, seed=123)
    output3 = seeded_dropout(x, p=0.5, seed=512)
    print(x, output1, output2, output3, sep="\n")
    assert torch.equal(output1, output2), "Same seed should produce same output"
    assert not torch.equal(output1, output3), "Different seed should produce different output"
    print("PASSED")

    # benchmark
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark.run(save_path=str(Path(__file__).parent / 'results'), print_data=True)