# End-to-End AutoKernel Design

## Goal

Turn `auto-kernel` from a single-kernel optimization prototype into an end-to-end workflow that can:

1. profile a model,
2. rank candidate bottleneck kernels,
3. extract one kernel into the agent-editable `kernel.py` surface,
4. iterate with a fixed evaluation loop,
5. track best variants and experiment history, and
6. verify aggregate end-to-end impact after reintegration.

## Design Summary

The core design keeps `kernel.py` as the only file an optimization agent edits during the hot loop. This follows the strongest idea from `karpathy/autoresearch`: one editable surface plus a fixed evaluation makes experiments comparable, reviewable, and easy to revert.

Everything else becomes orchestration around that surface:

- `prepare.py` remains environment and runtime setup.
- `profile.py` profiles a model or active kernel and produces structured bottleneck data.
- `extract.py` prepares a workspace for a selected kernel and updates `kernel.py`.
- `bench.py` remains the fixed experiment evaluator for the active kernel.
- `orchestrate.py` schedules which kernel to optimize next using recorded measurements.
- `verify.py` runs reintegration checks and reports end-to-end speedup.

## Architecture

### 1. Stable single-kernel core

The current repo already has the right conceptual center:

- a kernel contract,
- a reference implementation,
- one active editable kernel file,
- experiment logs,
- profiling and benchmarking scripts.

We will factor these shared responsibilities into small library modules under `auto_kernel/` so the scripts share one implementation instead of drifting.

### 2. Workspace-driven model pipeline

The end-to-end workflow will use a lightweight `workspace/` directory for runtime artifacts:

- active run metadata,
- profiled bottleneck summaries,
- extracted kernel manifests,
- optimized kernel snapshots,
- verification reports.

This keeps the repo source clean while allowing the workflow to resume safely.

### 3. Fixed evaluation loop

Every optimization experiment will continue to operate on the active `kernel.py` and use the same evaluator:

- correctness stages,
- timing and throughput,
- roofline-derived bottleneck hints,
- optional Nsight Compute analysis,
- append-only results and memory logs.

This preserves the autoresearch-like loop of edit, run, keep/discard.

### 4. Reintegration boundary

Extraction and verification will use structured metadata rather than implicit conventions. Each extracted kernel will have:

- kernel type,
- source model operator name,
- input shape and dtype,
- reference source,
- active status,
- best-known metrics,
- output path for optimized variants.

This is the boundary between per-kernel experimentation and whole-model verification.

## Key Decisions

### Preserve `kernel.py` as the edit target

This keeps the human/agent optimization surface tiny and avoids diff noise across many files.

### Prefer pure-Python, metadata-first pipeline over heavyweight integrations

The repo should remain hackable without requiring a full compiler or runtime stack beyond PyTorch and optional GPU tooling.

### Add reusable modules before adding more top-level scripts

The current top-level scripts already have overlapping responsibilities. Shared logic should move into `auto_kernel/` first so the new scripts are thin wrappers.

## Non-Goals For The First End-to-End Pass

- full CUDA/Triton code generation,
- HuggingFace export,
- multi-agent orchestration inside the repo,
- production-grade operator tracing across arbitrary models,
- direct parity with `RightNow-AI/autokernel` in one pass.

The first complete pass should be technically sound, resumable, and testable on CPU-first environments, with GPU acceleration and deep profiling as optional upgrades.

## Testing Strategy

- Unit tests for shared contracts, device resolution, logging, manifests, and orchestration decisions.
- Script-level tests for `prepare.py`, `bench.py`, `profile.py`, `extract.py`, `orchestrate.py`, and `verify.py`.
- End-to-end fixture tests that exercise the full profile -> extract -> bench -> verify loop with a small built-in kernel/model.

## Risks

- The environment here does not currently have PyTorch or pytest installed, so some runtime verification must remain limited to syntax and pure-Python/unit-level coverage until dependencies are available.
- The working tree already contains partial, uncommitted improvements and merge-conflict breakage. Recovery must land before any feature growth.

## Approved Direction

Proceed with the single editable `kernel.py` workflow and build the end-to-end pipeline around it.
