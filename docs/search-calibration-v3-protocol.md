# Family-wise search-decision calibration protocol

Version 3, 2026-10-14. Freeze and commit source, tests, seeds and thresholds before
measurement. Run CPU and MPS once each, serially. Preserve every outcome and stop
after these two runs; do not tune or rerun failures.

Versions 1 and 2 made 100 null decisions while applying a 0.05 per-decision sign
threshold. Version 2 therefore improved sensitivity but still admitted one CPU
and two MPS same-code promotions. Version 3 controls that declared family of 100
null comparisons with a Bonferroni threshold of `0.05 / 100 = 0.0005`. NIST's
[Bonferroni description](https://itl.nist.gov/div898/handbook/prc/section4/prc473.htm)
explains how dividing alpha across a finite, preselected family maintains an
overall confidence bound. This is an analytical correction, not a threshold fit
to the observed false-promotion magnitudes.

Each trial has 11 paired blocks. Promotion requires median paired speedup at least
1.05 and a one-sided exact sign p-value at most 0.0005. Eleven favorable non-tied
pairs give `p = 1/2048 = 0.00048828125`; ten give `p = 12/2048 = 0.005859375` and
cannot pass. Candidate, incumbent and reference use deterministic randomized
cyclic orders. Across 11 blocks, each implementation occupies every timing
position either three or four times. Input creation stays outside timing and
accelerator work is synchronized around calls.

Use fresh, disjoint seeds: 100 null seeds 5000–5099 and 20 positive seeds
6000–6019. Cycle the same four shapes and keep FP32, one host thread, three
warmups, 25 calls per recorded timing, the same identical-code null, and the same
output-identical eight-clone slowdown control. The aggregate gate remains zero
false promotions, at least 19/20 positive detections, and at least 4/5 detections
per shape. The gate reconstructs block orders, paired ratios, medians, p-values
and decisions from raw timings; malformed or incomplete evidence cannot pass.

CPU and MPS are separate machine-local calibrations. MPS fallback must be disabled.
A pass does not validate CUDA, establish a universal error rate, or demonstrate
added kernel-search value.

After committing the frozen revision:

```bash
python -m experiments.search_calibration_bonferroni --device cpu \
  --output workspace/search-calibration-cpu-v3
PYTORCH_ENABLE_MPS_FALLBACK=0 python -m experiments.search_calibration_bonferroni \
  --device mps --output workspace/search-calibration-mps-v3
```
