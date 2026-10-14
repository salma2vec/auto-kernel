# Paired search-decision calibration protocol

Version 2, 2026-10-14. Freeze and commit this protocol, implementation and tests
before measuring. Run each CPU and MPS experiment once. Preserve failures without
changing thresholds or rerunning selected trials.

Version 1 found zero false promotions but only 8/20 deliberately slowed controls
on CPU, and one false promotion plus 17/20 detections on MPS. Its global range
gate compared candidate and incumbent blocks measured at different times. The MPS
false promotion occurred when identical code measured materially faster in the
later block; most missed positive controls had overlapping global ranges despite
large median effects. Those complete negative results remain in
[the v1 report](search-calibration.md).

Version 2 interleaves candidate, incumbent and the common explicit RMSNorm
reference inside every repetition. Their order is randomized per block. Promotion
requires both a median paired speedup of at least 1.05 and a one-sided exact sign
test p-value no greater than 0.05. With five non-tied pairs this requires the
candidate to be faster in all five blocks (`p = 1/32`). Ties are excluded when
computing the exact binomial tail and therefore cannot help a five-pair trial
pass. The [NIST sign-test description](https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/signtest.htm)
supports the paired, distribution-light direction check and notes its lower power.
The 1.05 effect threshold remains a separate practical requirement.

Everything else is held fixed from v1: 100 null trials at seeds 3000–3099, 20
positive controls at seeds 4000–4019, four evenly cycled shapes, FP32, one host
thread, five repetitions, three warmups and 25 calls per timing. Nulls compare
identical native RMSNorm code. Positive controls compare native RMSNorm against
an output-identical implementation with eight extra clones. The frozen aggregate
gate still requires zero false promotions, at least 19/20 positive detections and
at least 4/5 detections at every shape. Any incomplete, duplicate, nonfinite or
nonpositive record cannot pass.

Input creation remains outside timing. Accelerator work is synchronized, and MPS
CPU fallback must be disabled. CPU and MPS are distinct machine-local experiments.
A pass is not a universal error rate, CUDA evidence or a kernel-speedup result.

After committing the frozen revision, run serially:

```bash
python -m experiments.search_calibration_paired --device cpu \
  --output workspace/search-calibration-cpu-v2
PYTORCH_ENABLE_MPS_FALLBACK=0 python -m experiments.search_calibration_paired \
  --device mps --output workspace/search-calibration-mps-v2
```
