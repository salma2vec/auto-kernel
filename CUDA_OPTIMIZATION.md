# CUDA Optimization Notes

Record reusable optimization patterns here.

## General Rules

- Verify correctness before trusting performance numbers.
- Use one hypothesis per experiment.
- Let benchmark output identify the macro bottleneck direction.
- Let deep profiling identify the specific cause.

## Memory-Bound Patterns

- Improve coalescing before changing arithmetic structure.
- Prefer vectorized loads when they reduce memory transactions cleanly.
- Improve locality before increasing pipeline depth.

## Compute-Bound Patterns

- Reduce redundant math before retuning launch geometry.
- Revisit tile shape when utilization is low but occupancy is healthy.

## Anti-Patterns

- Combining multiple unrelated changes into one experiment
- Keeping a faster kernel that fails correctness checks
- Retrying the same failed idea without new profiler evidence
