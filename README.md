# auto-kernel

Autonomous GPU kernel optimization workflow.

## Overview

`auto-kernel` helps you profile, extract, and optimize GPU kernels through iterative experimentation. The workflow is built around a single editable file, `kernel.py`, so experiments stay reviewable and easy to revert.

1. **Profiles** either the active kernel, a workload file, or a model adapter into a ranked manifest
2. **Extracts** the selected bottleneck into `kernel.py`
3. **Benchmarks** the active kernel against the reference implementation
4. **Decides** whether to keep or discard the change
5. **Logs** the result to `MEMORY.md`, per-kernel memory, and `results.tsv`
6. **Orchestrates** the next target using Amdahl-style time-share reasoning
7. **Verifies** the projected end-to-end speedup from recorded kernel improvements

## Project Structure

```text
auto-kernel/
├── program.md              # Optimization workflow protocol
├── CUDA_OPTIMIZATION.md    # Optimization knowledge base
├── MEMORY.md               # Global optimization log
├── results.tsv             # Experiment results tracking
├── prepare.py              # Environment preparation and validation
├── profile.py              # Kernel profiling and workload manifest generation
├── extract.py              # Activates a profiled bottleneck in kernel.py
├── orchestrate.py          # Recommends the next kernel to optimize
├── verify.py               # Reports aggregate end-to-end speedup
├── ncu_profile.py          # Nsight Compute analysis and diffing
├── bench.py                # Benchmark harness and correctness checking
├── kernel.py               # The active kernel under optimization
├── reference.py            # Reference implementations for correctness
├── kernels/                # Baseline kernels
├── kernels_optimized/      # Saved optimized kernels
├── models/                 # Example workload definitions
├── auto_kernel/            # Shared helpers for manifests and orchestration
├── memory/                 # Per-kernel experiment logs
├── ncu_reports/            # Profiling artifacts
├── workspace/              # Runtime manifests and active selections (gitignored)
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

# Generate a workload manifest
uv run profile.py --workload models/rms_norm_stack.py --manifest-out workspace/profile.json

# Or profile a live model adapter
uv run profile.py --model models/rms_norm_model.py --input-shape 4,128,512 --manifest-out workspace/profile.json

# Activate the top bottleneck in kernel.py
uv run extract.py --profile-path workspace/profile.json

# Resume from the best saved variant for that target if one exists
uv run extract.py --profile-path workspace/profile.json --prefer-optimized

# Run deep analysis
uv run ncu_profile.py

# Recommend the next kernel and report aggregate impact
uv run orchestrate.py --profile-path workspace/profile.json --results-path results.tsv
uv run verify.py --profile-path workspace/profile.json --results-path results.tsv

# Measure the workload against saved optimized artifacts
uv run verify.py --profile-path workspace/profile.json --workspace-root workspace --measure-workload

# Or verify the optimized kernels back inside the model
uv run verify.py --profile-path workspace/profile.json --model models/rms_norm_model.py --input-shape 4,128,512
```

The repository ships with `rms_norm` as the default kernel in `kernel.py`, so the loop works immediately after setup. By default, `bench.py` and `profile.py` now auto-select CUDA when it is available and fall back to CPU otherwise.

## How It Works

`program.md` defines the iteration protocol. A typical loop looks like this:

1. Use `profile.py --workload ... --manifest-out workspace/profile.json` to rank the workload bottlenecks
2. Use `extract.py` to make the top bottleneck the active `kernel.py`
3. Use `profile.py` and `ncu_profile.py` to understand the bottleneck
4. Consult `CUDA_OPTIMIZATION.md` for strategies
5. Make one focused change to `kernel.py`
6. Run `bench.py`
7. Keep the change if correctness holds and the result improves
8. Use `orchestrate.py` and `verify.py` to track the next target and total impact
9. Record important notes in `MEMORY.md` and `memory/<kernel>.md`

## Workload-Driven Loop

The workload path is the bridge from a model or workload definition to the single-kernel hot loop:

```bash
uv run profile.py --workload models/rms_norm_stack.py --manifest-out workspace/profile.json
uv run extract.py --profile-path workspace/profile.json
uv run bench.py --hypothesis "baseline"
uv run orchestrate.py --profile-path workspace/profile.json --results-path results.tsv
uv run verify.py --profile-path workspace/profile.json --results-path results.tsv
```

If your environment does not support `uv run`, the scripts can also be invoked directly with `python3`.

When you keep an experiment with `uv run bench.py --kept yes`, the active `kernel.py` snapshot is saved under `workspace/optimized/`. You can resume that best-known variant later with `extract.py --prefer-optimized`.

## Model-Driven Loop

The model path is the new bridge toward live-model parity. You can point `profile.py` and `verify.py` at a model module instead of hand-authoring a workload manifest:

```bash
uv run profile.py --model models/rms_norm_model.py --input-shape 4,128,512 --manifest-out workspace/profile.json
uv run extract.py --profile-path workspace/profile.json
uv run bench.py --hypothesis "baseline"
uv run verify.py --profile-path workspace/profile.json --model models/rms_norm_model.py --input-shape 4,128,512
```

Model modules can use one of two contracts:

1. Explicit adapters:
   - export `trace_model(...)` to produce ranked bottlenecks
   - export `verify_model(...)` to run end-to-end correctness and timing
2. Generic adapters:
   - export `build_model(...)`
   - export `build_inputs(...)`
   - export `MODEL_TARGETS`

Each `MODEL_TARGETS` entry should include:

- `name`
- `module_name`
- `kernel_type`
- optional `kernel_path`
- optional `weight_attr` and `eps_attr` for `rms_norm`

The generic tracing path times the declared target modules inside the running model. The generic reintegration path currently ships built-in replacement logic for `rms_norm`; other kernel types should still provide a custom `verify_model(...)` adapter.

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

To add a workload entry point, create a module under `models/` that exports either `get_workload()` or `WORKLOAD`. Each entry should include:

- `name`
- `kernel_type`
- `kernel_path`
- either `baseline_time_ms` for static manifests or `input_kwargs` for real kernel timing

See [models/custom.py](/Users/salma/auto-kernel/models/custom.py) for a starting template.

To add a model adapter, create a module under `models/` that exports either the explicit adapter hooks or the generic contract described above. See [models/rms_norm_model.py](/Users/salma/auto-kernel/models/rms_norm_model.py) for a concrete example.

## Requirements

- Python >= 3.9
- PyTorch
- [uv](https://github.com/astral-sh/uv) package manager
- CUDA-capable GPU and Nsight Compute for full GPU profiling

## License

MIT
