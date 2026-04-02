# Auto-Kernel Program

Follow this optimization protocol strictly.

## Available Kernels

The `kernels/` directory contains baseline kernels. Never modify files in `kernels/`.

Optimized kernels belong in `kernels_optimized/`, which mirrors the structure of `kernels/`.

```text
kernels/                        # Baseline, read-only
└── <kernel>.py

kernels_optimized/              # Optimized versions
└── <kernel>.py
```

Each kernel module must export:

- `KERNEL_TYPE: str`
- `kernel_fn(**inputs) -> torch.Tensor` (or tuple)
- `get_inputs() -> dict`
- `get_flops() -> int`
- `get_bytes() -> int`

Each kernel type must also have a benchmark config in `bench.py` under `KERNEL_CONFIGS`.

## Setup Phase

1. Run `uv run prepare.py` to validate the environment.
2. Read `CUDA_OPTIMIZATION.md` to review known optimization patterns and anti-patterns.
3. Read `MEMORY.md` for the global optimization summary.
4. If you are starting from a workload, generate a ranked profile manifest:

   ```bash
   uv run profile.py --workload models/rms_norm_stack.py --manifest-out workspace/profile.json
   ```

   If you are starting from a live model adapter instead:

   ```bash
   uv run profile.py --model models/rms_norm_model.py --input-shape 4,128,512 --manifest-out workspace/profile.json
   ```

5. Select a kernel to optimize:

   From a workload manifest:

   ```bash
   uv run extract.py --profile-path workspace/profile.json
   ```

   To resume from the best saved artifact for a target instead of the baseline source:

   ```bash
   uv run extract.py --profile-path workspace/profile.json --prefer-optimized
   ```

   Or manually:

   ```bash
   cp kernels/<kernel>.py kernel.py
   ```

   If resuming from a previous optimized version:

   ```bash
   cp kernels_optimized/<kernel>.py kernel.py
   ```

6. Read `memory/<kernel_type>.md` if it exists.
7. Read `kernel.py` to understand the current implementation.
8. Read `reference.py` to understand the correctness target.

## Experiment Loop

### Step 1: Benchmark

Run the benchmark harness:

```bash
uv run bench.py > run.log 2>&1
```

For quick iteration:

```bash
uv run bench.py --quick > run.log 2>&1
```

Extract the key metrics from `run.log`:

```bash
grep "kernel=\|experiment_id\|correctness\|throughput_tflops\|bandwidth_gb_s\|pct_peak_compute\|pct_peak_bandwidth\|bottleneck\|speedup_vs_pytorch\|peak_vram_mb" run.log
```

### Step 2: Macro Performance Analysis

Use the benchmark output to understand the macro performance picture:

1. Compute throughput
2. Memory bandwidth
3. Peak-relative utilization
4. Bottleneck classification
5. Speedup versus the reference implementation

### Step 3: Deep Profiling

Run Nsight Compute analysis:

```bash
uv run ncu_profile.py > ncu.log 2>&1
```

For targeted analysis:

```bash
uv run ncu_profile.py --skills roofline,memory,warp_stall > ncu.log 2>&1
```

To compare two NCU CSV reports:

```bash
uv run ncu_profile.py --diff before.csv after.csv > ncu_diff.log 2>&1
```

If Nsight Compute is unavailable, `ncu_profile.py` falls back to analytical mode and still reports a bottleneck, findings, and next actions.

### Step 4: Hypothesize

Form a single focused hypothesis:

> Hypothesis: what you will change and why it should help
> Macro evidence: which benchmark metrics point to the bottleneck direction
> Deep evidence: which profiler findings point to the specific cause

Rules:

- One change per experiment
- Ground hypotheses in measured evidence
- Check `memory/<kernel_type>.md` before retrying an old idea

### Step 5: Modify

Edit `kernel.py` to implement the hypothesis.

### Step 6: Benchmark Again

Run:

```bash
uv run bench.py --experiment-id <id> --hypothesis "<hypothesis>" > run.log 2>&1
```

### Step 7: Decide

| Condition | Action |
|-----------|--------|
| correctness = FAIL | discard the change |
| correctness = PASS and performance improves materially | keep the change |
| correctness = PASS and performance is flat or worse | discard the change |

If you keep a result, mark it explicitly:

```bash
uv run bench.py --experiment-id <id> --hypothesis "<hypothesis>" --kept yes
```

This saves the current `kernel.py` as the best-known artifact for the active target under `workspace/optimized/`.

### Step 8: Record

Update:

- `results.tsv`
- `memory/<kernel_type>.md`
- `MEMORY.md`
- `CUDA_OPTIMIZATION.md` when you discover a reusable pattern or anti-pattern
- `workspace/profile.json` as you refresh workload rankings

### Step 9: Repeat

Stop when:

- gains have plateaued over several experiments
- known optimization patterns are exhausted
- deeper profiling no longer yields a credible next hypothesis

## Switching Kernels

Save the current best kernel:

```bash
cp kernel.py kernels_optimized/<kernel>.py
```

Switch to the next baseline:

```bash
cp kernels/<next_kernel>.py kernel.py
```

Save optimized kernel results before leaving a kernel, and review its per-kernel memory before resuming later.

## End-to-End Tracking

Use the recorded results and workload manifest together:

```bash
uv run orchestrate.py --profile-path workspace/profile.json --results-path results.tsv
uv run verify.py --profile-path workspace/profile.json --results-path results.tsv
```

`orchestrate.py` recommends the next kernel based on recorded best speedups and time share. `verify.py` reports the current projected end-to-end speedup from all recorded improvements.

To measure the declared workload against saved optimized artifacts instead of only using projected speedups:

```bash
uv run verify.py --profile-path workspace/profile.json --workspace-root workspace --measure-workload
```

To verify optimized kernels back inside a running model adapter:

```bash
uv run verify.py --profile-path workspace/profile.json --model models/rms_norm_model.py --input-shape 4,128,512
```

Model adapters can be explicit or generic:

- Explicit adapters export `trace_model(...)` and `verify_model(...)`.
- Generic adapters export `build_model(...)`, `build_inputs(...)`, and `MODEL_TARGETS`.

The generic model path now times declared target modules directly from the running model. Built-in reintegration support currently targets `rms_norm`; other kernel types should continue to provide a custom `verify_model(...)` implementation until more replacement modules are added.

## Important Rules

1. Never break correctness.
2. Never modify files in `kernels/`.
3. One experiment should correspond to one hypothesis.
4. Always benchmark before deciding whether to keep a change.
5. Keep per-kernel and global memory updated.
6. Use benchmark output and deep profiling together.
7. Treat excessive VRAM growth as a regression.
