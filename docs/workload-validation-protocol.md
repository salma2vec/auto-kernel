# Whole-workload and native-baseline protocol

Version 1, 2026-10-14. Freeze source before execution. This is an implementation
and local-reliability test on the existing two-RMSNorm synthetic model, not a
pretrained-model or general GPU result. No measurements are claimed here.

Five fixed seeds 142–146. Discovery shapes `(1,32,512)`, `(4,128,512)`; confirmation
shapes `(4,256,512)`, `(8,128,512)`. FP32, existing model atol 1e-5 / rtol 1e-4.
Correctness additionally covers strided inputs and scales 0, 1e-4, 1, 1e3 against the
CPU model, checking finite outputs, exact shape/dtype and unchanged inputs.

Compare original explicit operations, multiplication-based squaring, and native
RMSNorm replacing both model layers. Input creation is outside synchronized
timing. Each case uses five randomized-order repetitions, three warmups and 25
calls per timing. Select only on discovery data, using the existing 1.05 material
speedup/nonoverlapping-range/paired-reference rule, applied to equal-weight
geometric timing aggregates. Freeze and save the selected source decision before
measuring confirmation inputs. No re-selection on confirmation losses/timings.

Report all policies and fallback cases. For each run, calculate geometric mean
confirmation speedup versus the original model and separately versus always-native.
Run six same-native-code comparison blocks per seed (30 total) with the same
promotion heuristic, retaining all observed false promotions. They do not
establish a general false-promotion rate. No kernel-type Amdahl projection.

Engineering usefulness requires speedup ≥ 1.05 versus original in at least 4/5 runs,
no confirmation case median regression > 5%, and zero promoted identical-code
controls. Added search value separately requires speedup ≥ 1.03 versus always-native
in at least 4/5 runs, no confirmation case median regression > 5% against
always-native as well as against original, and zero false promotions. These are
practical descriptive thresholds, not significance tests. If the first passes
and the second fails, report successful native integration without added search
value. If both fail, retain the negative result. Any correctness, timing or
execution failure leaves the report incomplete and cannot pass either gate.

Use CPU or an explicitly available MPS device with CPU fallback disabled; each is
a distinct experiment. No CUDA, training/backward, mixed-dtype or custom generated
kernel claim follows. Native substitution is an established technique. Reusing
the chosen native implementation as its own comparator gives speedup exactly 1,
which correctly prevents an added-search-value claim from that selection alone.
