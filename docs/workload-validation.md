# Native integration succeeds; added search value is unproven

Measured on 2026-10-14 on Apple M1 MPS with CPU fallback disabled. The runner,
tests and [prospective protocol](workload-validation-protocol.md) were frozen at
`ea94b31` before execution. All 50 tests passed before the run.

The workload contains two RMSNorm layers of width 512. It is a synthetic
micro-workload, not a pretrained transformer. Search sees two discovery shapes,
then freezes its selection before measuring two new confirmation shapes. Each
timing uses five randomized-order repetitions and synchronized execution.

| Seed | Chosen policy | Confirmation speedup vs original | Vs always-native |
| --- | --- | ---: | ---: |
| 142 | Native RMSNorm | 5.262891× | 1.000000× |
| 143 | Native RMSNorm | 4.804880× | 1.000000× |
| 144 | Native RMSNorm | 4.954127× | 1.000000× |
| 145 | Native RMSNorm | 4.959754× | 1.000000× |
| 146 | Native RMSNorm | 4.937313× | 1.000000× |

Speedups are equal-weight geometric means across the two confirmation shapes,
computed from each method's median timing. All 480 CPU-reference correctness
comparisons passed, including zero, small and large values and strided inputs.
All 30 identical-native-code controls were rejected. No confirmation case
regressed against either comparison baseline.

The frozen **engineering gate passed in 5/5 runs**. The **added-search-value gate
failed, with 0/5 runs passing**: the selected policy is the always-native comparator
itself. This demonstrates correct native integration and repeatable workload
improvement against explicit operations. It provides no evidence that this
bounded search improves on simply choosing native RMSNorm. Native substitution
is an established optimization, not a novel kernel result.

The [compressed raw report](../experiments/results/mps-m1-workload-seed142-146.json.gz)
contains sources, hashes, every timing order/block, correctness records,
selection decisions, control trials and machine-evaluated gates. Uncompressed
SHA256: `83c549d3fdd42fe725ad2248c78d2811b5c042f8d06eabdcb89fdd6d77e89c3c`.
The aggregates were separately recomputed from the raw timing blocks.

To reproduce from a clean committed source checkout:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 python -m experiments.workload_validation \
  --device mps --output workspace/workload-confirmation
```

Use `--device cpu` for a separate CPU experiment. No CUDA, backward/training,
mixed-precision, pretrained-model latency or broad workload advantage follows
from these MPS measurements. Shared shapes and a single machine limit the
scope; five seeds and 30 controls are descriptive checks, not significance
tests or a measured general false-promotion rate.
