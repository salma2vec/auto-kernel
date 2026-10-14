# Experiment protocol and related work

Updated 2026-10-14.

## Protocol

1. Profile the intended workload or model on the intended hardware. Rank targets by measured time share; distinguish supplied analytical durations from measurements.
2. Preserve the baseline and reference. Select one operation and state one falsifiable hypothesis about compute, memory traffic, dispatch cost or fusion.
3. Generate a candidate using a local transformation or a trusted implementation. Store its source and hash before executing it.
4. Validate against independently maintained reference functions using several seeds, shapes, numerical scales and edge cases. Reject dtype changes and mutations of inputs. Reuse trusted input generation rather than candidate-provided fixtures.
5. Measure candidate and reference repeatedly with warmup, accelerator synchronization and input allocation outside the timed region. Retain all samples, including negative results.
6. Remeasure the incumbent near the decision. Promote only when the candidate meets the configured improvement threshold and timing-stability gate. Otherwise preserve the incumbent.
7. Reintegrate into the declared workload/model and verify correctness and runtime again. A local speedup or an Amdahl projection is not measured model-level improvement.

`optimize.py` implements steps 2–6 for its bounded candidate family after the operation has been selected. Profiling, extraction and reintegration are separate explicit commands. Search does not automatically publish changes or make git commits. There is no LLM service integration or arbitrary compiled-model extraction.

## Measurement limits

The harness currently evaluates forward operations in FP32. It is not a backward-pass, multi-dtype, all-layout or adversarial kernel verifier. Finite shape/seed checks cannot establish universal correctness. Built-in native candidates may fail strict tolerances because of floating-point operation ordering; those failures are retained rather than silently relaxing the gate.

Synchronized wall-clock measurements include dispatch costs. Timing ranges and paired-reference speedups reduce common noise but are not a hypothesis test and cannot eliminate thermal or scheduler drift. Benchmark on idle dedicated hardware, fix thread counts/power settings, repeat full runs, and compare appropriate vendor/compiler baselines. Peak-memory accounting and model-level memory validation remain limited. Candidate modules are trusted executable code; process and OS isolation are required for adversarial programs.

The promotion rule has a separate prospective [null and sensitivity calibration
protocol](search-calibration-protocol.md). It exercises the deployed heuristic
without treating a same-code comparison as an optimization result.

## Primary research context

- [KernelBench (Ouyang et al., 2025)](https://arxiv.org/abs/2502.10517) evaluates kernel correctness together with speedup. This workbench follows that separation but does not reproduce KernelBench's problem suite or results.
- [KernelBench-Verified (Zhang et al., 2026)](https://arxiv.org/abs/2607.16241) discusses benchmark exploitation, varied input distributions and realistic accelerated baselines. Extra input shapes/seeds and trusted input generation here address a small subset of those concerns; no adversarial-security guarantee is made.
- [KernelOPT (Poddar et al., September 2026; revised October 2026)](https://arxiv.org/abs/2609.30059) combines dispatch-aware search with multi-stage correctness and performance gates. Our implementation uses bounded local candidates and preserves the incumbent on failure; it does not implement compiler graph surgery or reproduce KernelOPT's experiments.
- [PyTorch KernelAgent (March 2026)](https://pytorch.org/blog/kernelagent-hardware-guided-gpu-kernel-optimization-via-multi-agent-orchestration/) connects hardware signals with iterative optimization. The available roofline/Nsight commands can inform manual selection here; automatic hardware-guided multi-agent hypothesis generation remains outside scope.
- [PyTorch benchmark documentation](https://docs.pytorch.org/docs/stable/benchmark_utils) explains warmups, synchronization and repeated measurements. The present CLI retains raw repetitions; publication-grade studies should also account for timer overhead and variance at their measurement scale.

## Reproduction record

Keep the source revision, Python/PyTorch versions, device model, thread count, TF32 configuration, input shapes, seeds, candidate/reference sources, raw timings and decisions. `optimize.py` writes these fields and source snapshots into a fresh directory. CPU runs verify the workflow but cannot substantiate GPU speedups. A rejected candidate is a valid experiment result.
