# auto-kernel

Autonomous GPU kernel optimization, centered on a single editable file.

`auto-kernel` is a workload-first loop for finding a hot kernel, pulling it into `kernel.py`, iterating on it, and measuring whether the change actually improves end-to-end runtime.

It is built for the common optimization problem that high-signal systems repos solve well:

- find the real bottleneck instead of guessing
- optimize one thing at a time
- keep the active change small and reviewable
- preserve the best variant as an artifact
- measure local speedups against workload-level impact

## Why This Repo Exists

Most kernel tuning workflows are either too manual or too magical.

Manual workflows make it hard to keep experiments organized.

Magical workflows hide the active change behind generated code and large orchestration layers.

`auto-kernel` takes a simpler stance:

- one active kernel lives in [`kernel.py`](kernel.py)
- one ranked manifest tells you what matters next
- one benchmark loop decides whether a change stays
- one verification path reports projected or measured workload impact

That makes the repo easy to audit, easy to diff, and easy to resume.

## What It Does

`auto-kernel` supports three optimization entry points:

| Mode | Input | Output |
|---|---|---|
| Direct kernel | active [`kernel.py`](kernel.py) | correctness + timing + saved optimized variant |
| Workload-driven | workload module under [`models/`](models) | ranked bottleneck manifest |
| Model-driven | live PyTorch model adapter | bottleneck discovery + reintegration verification |

The default loop is:

1. Profile a workload, model, or active kernel.
2. Extract the highest-value bottleneck into [`kernel.py`](kernel.py).
3. Make one focused optimization change.
4. Benchmark against the reference implementation.
5. Keep or discard the result.
6. Track projected or measured workload-level gains.

## Highlights

- Single active edit surface: optimize in [`kernel.py`](kernel.py), not across generated files.
- Ranked bottleneck manifests: profile by workload share before optimizing.
- Saved best variants: accepted kernels are persisted under `workspace/optimized/`.
- Model reintegration path: verify optimized kernels back inside a live PyTorch model.
- Generic model support: auto-discovers common leaf modules when `MODEL_TARGETS` is absent.
- CPU fallback: [`bench.py`](bench.py) and [`profile.py`](profile.py) auto-select CUDA when available and fall back to CPU otherwise.

## Quick Start

### Option A: `uv` workflow

```bash
uv sync
uv run prepare.py
uv run bench.py --iterations 1 --warmup 0
```

### Option B: local virtualenv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python prepare.py
python bench.py --iterations 1 --warmup 0
```

The repository ships with `rms_norm` as the default active kernel, so the loop works immediately after setup.

## The 5-Minute Tour

### 1. Check the environment

```bash
uv run prepare.py
```

This reports Python, Torch, CUDA, Triton, and Nsight Compute availability.

### 2. Benchmark the active kernel

```bash
uv run bench.py
```

This compares the active implementation in [`kernel.py`](kernel.py) against the reference in [`reference.py`](reference.py), then records the result.

### 3. Profile a workload

```bash
uv run profile.py --workload models/rms_norm_stack.py --manifest-out workspace/profile.json
```

This produces a ranked bottleneck manifest that tells you where optimization time is most valuable.

### 4. Extract the top bottleneck

```bash
uv run extract.py --profile-path workspace/profile.json
```

That copies the selected baseline kernel into [`kernel.py`](kernel.py) so the next edit is focused and reviewable.

### 5. Verify workload impact

```bash
uv run orchestrate.py --profile-path workspace/profile.json --results-path results.tsv
uv run verify.py --profile-path workspace/profile.json --results-path results.tsv
```

That gives you the next target plus the projected end-to-end speedup implied by your accepted results.

## Common Workflows

### Workload-driven optimization

Use this when you already know the workload shape and want to optimize by bottleneck share.

```bash
uv run profile.py --workload models/rms_norm_stack.py --manifest-out workspace/profile.json
uv run extract.py --profile-path workspace/profile.json
uv run bench.py --hypothesis "baseline"
uv run orchestrate.py --profile-path workspace/profile.json --results-path results.tsv
uv run verify.py --profile-path workspace/profile.json --results-path results.tsv
```

If you keep an experiment with `--kept yes`, the active kernel snapshot is saved under `workspace/optimized/`.

You can later resume the best known variant for a target:

```bash
uv run extract.py --profile-path workspace/profile.json --prefer-optimized
```

### Model-driven optimization

Use this when you want the bottleneck ranking and reintegration check to come from a live PyTorch model.

```bash
uv run profile.py --model models/rms_norm_model.py --input-shape 4,128,512 --manifest-out workspace/profile.json
uv run extract.py --profile-path workspace/profile.json
uv run bench.py --hypothesis "baseline"
uv run verify.py --profile-path workspace/profile.json --model models/rms_norm_model.py --input-shape 4,128,512
```

Model modules can use either:

1. Explicit adapters
2. Generic adapters

### Explicit model adapters

Export:

- `trace_model(...)`
- `verify_model(...)`

Use this when the model needs custom tracing or reintegration logic.

### Generic model adapters

Export:

- `build_model(...)`
- `build_inputs(...)`
- optional `MODEL_TARGETS`

Each `MODEL_TARGETS` entry may include:

- `name`
- `module_name`
- `kernel_type`
- optional `kernel_path`
- optional `input_param`
- optional `input_arg`
- optional `attribute_bindings`
- optional `parameter_bindings`
- optional `constant_bindings`
- backward-compatible `weight_attr`, `bias_attr`, and `eps_attr`

If `MODEL_TARGETS` is omitted, the generic path auto-discovers supported leaf modules from `named_modules()` for common PyTorch-style classes such as:

- `RMSNorm`
- `LayerNorm`
- `Linear`
- `Softmax`

The generic reintegration path is binding-driven, so built-in replacements are not limited to `rms_norm`.

## Repository Layout

```text
auto-kernel/
├── program.md              # optimization protocol
├── CUDA_OPTIMIZATION.md    # optimization notes and heuristics
├── MEMORY.md               # global experiment log
├── results.tsv             # experiment history
├── prepare.py              # environment validation
├── profile.py              # workload and model profiling
├── extract.py              # activate a bottleneck in kernel.py
├── bench.py                # correctness and timing harness
├── orchestrate.py          # choose the next kernel to optimize
├── verify.py               # report projected or measured workload impact
├── ncu_profile.py          # Nsight Compute analysis
├── kernel.py               # active kernel under optimization
├── reference.py            # reference implementations
├── kernels/                # baseline kernel inventory
├── models/                 # workload and model examples
├── auto_kernel/            # shared manifests and orchestration helpers
├── memory/                 # per-kernel notes
├── ncu_reports/            # profiler artifacts
└── workspace/              # manifests, active selections, and optimized artifacts (gitignored)
```

## Adding Your Own Kernel

Create a kernel module under [`kernels/`](kernels) that exports:

- `KERNEL_TYPE`
- `kernel_fn(**inputs)`
- `get_inputs(...)`
- `get_flops(inputs)`
- `get_bytes(inputs)`

Then add the matching reference implementation in [`reference.py`](reference.py).

To start iterating:

```bash
cp kernels/your_kernel.py kernel.py
uv run bench.py
uv run profile.py
uv run ncu_profile.py
```

## Adding Your Own Workload

Create a module under [`models/`](models) that exports either `get_workload()` or `WORKLOAD`.

Each entry should include:

- `name`
- `kernel_type`
- `kernel_path`
- either `baseline_time_ms` or `input_kwargs`

See [models/custom.py](models/custom.py) for a starter template.

## Current Scope

This repo now supports generic reintegration for more than `rms_norm`, and the built-in auto-discovery path covers common module families such as `RMSNorm`, `LayerNorm`, `Linear`, and `Softmax`.

What it does not claim yet is full profiler-driven extraction from completely untouched arbitrary PyTorch models.

That last slice still benefits from model-specific adapter logic when the boundary between model module and kernel contract is ambiguous.

## Requirements

- Python 3.9+
- PyTorch
- [`uv`](https://github.com/astral-sh/uv) for the default workflow
- CUDA-capable GPU and Nsight Compute for full GPU profiling

## Testing

Run the full suite with:

```bash
python -m unittest -v
```

If you are using the local venv created above:

```bash
./.venv/bin/python -m unittest -v
```

## License

[MIT](LICENSE)
