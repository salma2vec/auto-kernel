# auto-kernel

Autonomous GPU kernel optimization workflow.

## Overview

`auto-kernel` helps you profile, analyze, and optimize GPU kernels through iterative experimentation. Given a kernel, the workflow:

1. **Profiles** the kernel to understand whether it is compute-bound or memory-bound
2. **Proposes** an optimization hypothesis based on the optimization guide
3. **Modifies** the kernel code
4. **Benchmarks** the modified kernel against the reference implementation
5. **Decides** whether to keep or discard the change
6. **Logs** the result to `MEMORY.md`, per-kernel memory, and `results.tsv`
7. **Repeats** until the results are satisfactory

## Project Structure

```text
auto-kernel/
├── program.md              # Optimization workflow protocol
├── CUDA_OPTIMIZATION.md    # Optimization knowledge base
├── MEMORY.md               # Global optimization log
├── results.tsv             # Experiment results tracking
├── prepare.py              # Environment preparation and validation
├── profile.py              # Macro profiling and roofline analysis
├── ncu_profile.py          # Nsight Compute analysis and diffing
├── bench.py                # Benchmark harness and correctness checking
├── kernel.py               # The active kernel under optimization
├── reference.py            # Reference implementations for correctness
├── kernels/                # Baseline kernels
├── kernels_optimized/      # Saved optimized kernels
├── memory/                 # Per-kernel experiment logs
├── ncu_reports/            # Profiling artifacts
└── pyproject.toml
```

## Quick Start

```bash
# Install dependencies
uv sync

# Prepare the environment
uv run prepare.py

# Run a benchmark
uv run bench.py

# Profile the current kernel
uv run profile.py

# Run deep analysis
uv run ncu_profile.py
```

The repository ships with `rms_norm` as the default kernel in `kernel.py`, so the loop works immediately after setup. By default, `bench.py` and `profile.py` now auto-select CUDA when it is available and fall back to CPU otherwise.

## How It Works

`program.md` defines the iteration protocol. A typical loop looks like this:

1. Inspect profiling data to identify the bottleneck
2. Use `profile.py` to understand the macro roofline picture
3. Use `ncu_profile.py` to understand the specific bottleneck
4. Consult `CUDA_OPTIMIZATION.md` for possible strategies
5. Make one focused change to `kernel.py`
6. Run `bench.py`
7. Keep the change if correctness holds and the result improves
8. Record the outcome in `MEMORY.md`, `memory/<kernel>.md`, and `results.tsv`

## Adding Your Own Kernels

To add a kernel for optimization:

1. **Create the kernel module** at `kernels/your_kernel.py` exporting:
   - `KERNEL_TYPE: str`
   - `kernel_fn(**inputs) -> torch.Tensor`
   - `get_inputs() -> dict`
   - `get_flops(inputs) -> int`
   - `get_bytes(inputs) -> int`
2. **Add a reference implementation** in `reference.py`
3. **Copy the kernel into `kernel.py`** and start iterating:

```bash
cp kernels/your_kernel.py kernel.py
uv run bench.py
uv run profile.py
uv run ncu_profile.py
```

## Requirements

- Python >= 3.9
- PyTorch
- [uv](https://github.com/astral-sh/uv) package manager
- CUDA-capable GPU and Nsight Compute for full GPU profiling

## License

MIT
