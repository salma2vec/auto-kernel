# Search-decision calibration protocol

Version 1, 2026-10-14. Freeze and commit the runner, protocol, tests and decision
thresholds before any calibration measurements. A failed gate remains a result;
do not tune thresholds or discard trials after execution.

This experiment tests the existing promotion rule under controlled local timing
noise. It does not generate a kernel or measure a kernel speedup. The null control
compares the exact same native RMSNorm function as incumbent and candidate. The
positive control compares native RMSNorm with an output-identical version that
performs eight extra clones. The positive control is deliberately artificial: it
checks that a rule conservative enough to reject nulls can still detect an obvious
slowdown. It is not an optimization claim.

Run 100 fixed null trials (seeds 3000–3099) and 20 fixed positive controls (seeds
4000–4019), cycling evenly through shapes `(1,32,512)`, `(4,128,512)`,
`(4,256,512)` and `(8,128,512)`. Use FP32, one host thread, five randomized-order
timing repetitions, three warmups and 25 calls per recorded sample. Input creation
is outside timing. Randomize candidate/incumbent evaluation order and the tested
implementation/reference order within each repetition. Synchronize the accelerator
around each timed call, matching the current search harness. Retain every order and
raw timing. MPS runs require CPU fallback to be disabled.

Use the unchanged search decision: at least 1.05× median improvement, no overlap
between the observed candidate and incumbent timing ranges, and at least 1.05×
improvement in speedup relative to the common explicit RMSNorm reference. Do not
replace this descriptive heuristic with a statistical test inside this protocol.

The frozen gate passes only with zero false promotions in 100 null trials, at
least 19 positive detections in 20 controls, and at least four detections among
the five positive controls for every shape. Any missing, duplicated, nonfinite or
nonpositive timing makes the report incomplete and unable to pass. CPU and MPS
are separate experiments; neither validates CUDA.

This is a machine-local calibration across four shapes on one session. Even a
passing result is not a universal false-promotion rate, an independence-based
confidence bound or evidence of search value beyond native operators. PyTorch's
[benchmark documentation](https://docs.pytorch.org/docs/stable/benchmark_utils)
motivates warmup, repeated measurements, fixed thread count and accelerator
synchronization; the runner intentionally preserves this project's more
conservative per-call synchronization so it exercises the deployed decision rule.
[MPS environment documentation](https://docs.pytorch.org/docs/stable/mps_environment_variables.html)
defines `PYTORCH_ENABLE_MPS_FALLBACK=1` as CPU fallback, which this protocol rejects.

After committing the frozen source, run serially:

```bash
python -m experiments.search_calibration --device cpu --output workspace/calibration-cpu
PYTORCH_ENABLE_MPS_FALLBACK=0 python -m experiments.search_calibration \
  --device mps --output workspace/calibration-mps
```
