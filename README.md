# Tryton: Learning Triton Lang

**Author:** Jay Gala

A collection of Triton kernels for learning GPU programming in Python.

## What's Triton?

Triton is a language for writing efficient CUDA kernels in Python without dealing with raw CUDA code.

## Repository Layout

Each kernel has its own directory. Its implementation is in `kernel.py`, and
benchmark CSV files and plots remain in that kernel's `results/` subdirectory.

- `vector_add/` - vector addition (starter example).
- `softmax/` - row-wise softmax.
- `fused_softmax/` - fused softmax tutorial implementation.
- `dropout/` - seeded dropout.
- `matmul/` - block-wise matrix multiplication.

## Quick Start

```bash
# Install Triton
pip install triton numpy pandas matplotlib torch

# Run the vector addition example
python3 vector_add/kernel.py

# Run a benchmark; output is saved under vector_add/results/
python3 vector_add/kernel.py --benchmark
```

Run the other examples by replacing `vector_add` with the corresponding kernel
directory, for example `python3 softmax/kernel.py` or
`python3 fused_softmax/kernel.py --benchmark`.

## Resources

- [Triton Documentation](https://triton-lang.org)
- [Vector Add Tutorial](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)
