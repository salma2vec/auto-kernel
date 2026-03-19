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
4. Select a kernel to optimize:

   ```bash
   cp kernels/<kernel>.py kernel.py
   ```

   If resuming from a previous optimized version:

   ```bash
   cp kernels_optimized/<kernel>.py kernel.py
   ```

5. Read `memory/<kernel_type>.md` if it exists.
6. Read `kernel.py` to understand the current implementation.
7. Read `reference.py` to understand the correctness target.

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
grep "experiment_id\|correctness\|throughput_tflops\|bandwidth_gb_s\|pct_peak_compute\|pct_peak_bandwidth\|bottleneck\|speedup_vs_pytorch\|peak_vram_mb" run.log
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

### Step 8: Record

Update:

- `results.tsv`
- `memory/<kernel_type>.md`
- `MEMORY.md`
- `CUDA_OPTIMIZATION.md` when you discover a reusable pattern or anti-pattern

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

## Important Rules

1. Never break correctness.
2. Never modify files in `kernels/`.
3. One experiment should correspond to one hypothesis.
4. Always benchmark before deciding whether to keep a change.
5. Keep per-kernel and global memory updated.
6. Use benchmark output and deep profiling together.
7. Treat excessive VRAM growth as a regression.
