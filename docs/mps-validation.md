# Apple M1 accelerator validation

Measured on 2026-10-14 with an Apple M1 (7 GPU cores, 8 GiB unified memory),
macOS 14.6, Python 3.11.15 and PyTorch 2.14.1. CUDA, `nvcc` and Nsight Compute
were unavailable. These are actual Apple MPS GPU runs with CPU fallback disabled;
they do not validate CUDA execution or NVIDIA roofline assumptions.

The [complete raw archive](../experiments/results/mps-m1-seed42-44.json.gz)
retains the baseline/candidate sources, hashes, repeat timings, decisions and
CPU-reference comparisons for all 12 searches. Its source revision is recorded
inside the archive. SHA-256:
`b5a082ba627e22f1546ddca715a9b3aad89ccb40f0baea4adfad99a657c55f59`.

| Operation | Completed searches | CPU-reference checks | Accepted candidates |
| --- | ---: | ---: | --- |
| RMSNorm | 3 | 3 / 3 | Native RMSNorm in seeds 43 and 44 |
| Linear | 3 | 3 / 3 | Native linear in seed 42 |
| Softmax | 3 | 3 / 3 | None; all no-change controls rejected |
| LayerNorm | 3 | 3 / 3 | None; all no-change controls rejected |

All searches used FP32, seeds 42–44, five timing repetitions of 25 synchronized
calls, three warmup calls and the existing 1.05 promotion threshold. Timing shape
was `(16, 64, 128)`; correctness also exercised other shapes, numerical scales,
edge cases and three internal seeds per search. The additional CPU comparison
uses the timing shape. All candidates that reached timing passed the configured
correctness gate. No tolerance was relaxed for this run.

The acceptance decisions varied across seeds. These observations establish that
the search and acceptance/rejection paths ran on MPS; they do not establish a
stable model-level speedup. This was a shared workstation, not an isolated
performance lab. The six no-change controls were rejected, but a small control
set cannot establish a false-promotion rate. Measured timings include dispatch
and synchronization overhead. No training/backward, mixed-dtype, large-model or
CUDA validation is implied.

From a source checkout:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 python -m experiments.accelerator_validation \
  --device mps --seeds 42 43 44 --repeats 5 --iterations 25 \
  --output workspace/mps-validation
```

Use a new output directory for each run. `--device cuda` requires an actual CUDA
device; it will not silently run on another backend. The runner now stops an
individual search if CPU-reference parity fails, preserves the failure record,
and returns a nonzero exit status if any search fails. The recorded run predates
that reporting/guard refinement; every CPU check already passed, so its measured
trials are unaffected. Default `auto` selection elsewhere remains CUDA-or-CPU;
Apple MPS use is explicit.
