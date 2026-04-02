# End-to-End AutoKernel Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover the broken prototype and extend it into a complete profile -> extract -> optimize -> verify pipeline while preserving the single editable `kernel.py` experiment loop.

**Architecture:** Shared logic moves into `auto_kernel/` modules first so the top-level scripts become thin entry points. The end-to-end flow stores runtime manifests under `workspace/` and keeps the optimization hot loop fixed around `kernel.py`, `bench.py`, and append-only experiment logs.

**Tech Stack:** Python 3.9+, PyTorch, unittest/pytest-compatible tests, optional Nsight Compute

---

## Chunk 1: Recovery And Baseline Stabilization

### Task 1: Write tests for the intended CLI defaults and output shape

**Files:**
- Modify: `tests/test_device_resolution.py`
- Modify: `tests/test_end_to_end.py`

- [ ] **Step 1: Write failing tests for current intended behavior**
- [ ] **Step 2: Run the relevant targeted tests and verify failure**
- [ ] **Step 3: Resolve merge conflicts in `bench.py` and `profile.py` with the minimal behavior that satisfies the tests**
- [ ] **Step 4: Run the targeted tests again and verify they pass**
- [ ] **Step 5: Commit**

### Task 2: Make runtime verification use local temp/cache paths

**Files:**
- Modify: `tests/test_end_to_end.py`
- Create: `auto_kernel/test_env.py`

- [ ] **Step 1: Add a failing test that runs subprocesses with repo-local temp/cache directories**
- [ ] **Step 2: Run it to verify failure**
- [ ] **Step 3: Implement minimal helpers and wire the fixture to use them**
- [ ] **Step 4: Re-run tests**
- [ ] **Step 5: Commit**

## Chunk 2: Shared Library Extraction

### Task 3: Move benchmark/profile shared logic into `auto_kernel/`

**Files:**
- Create: `auto_kernel/contracts.py`
- Create: `auto_kernel/kernel_loader.py`
- Create: `auto_kernel/logging.py`
- Create: `auto_kernel/metrics.py`
- Modify: `bench.py`
- Modify: `profile.py`
- Modify: `ncu_profile.py`

- [ ] **Step 1: Add unit tests for the extracted helpers**
- [ ] **Step 2: Run tests and verify failure**
- [ ] **Step 3: Extract modules with no intended behavior change**
- [ ] **Step 4: Re-run tests**
- [ ] **Step 5: Commit**

## Chunk 3: End-to-End Pipeline

### Task 4: Add workspace manifest and profiling output model

**Files:**
- Create: `auto_kernel/workspace.py`
- Create: `auto_kernel/models.py`
- Modify: `prepare.py`
- Modify: `profile.py`
- Create: `extract.py`
- Create: `tests/test_workspace.py`

- [ ] **Step 1: Add failing tests for workspace manifest creation and profile output persistence**
- [ ] **Step 2: Run tests and verify failure**
- [ ] **Step 3: Implement the workspace and extraction primitives**
- [ ] **Step 4: Re-run tests**
- [ ] **Step 5: Commit**

### Task 5: Add orchestrator and verification flow

**Files:**
- Create: `orchestrate.py`
- Create: `verify.py`
- Create: `auto_kernel/orchestration.py`
- Create: `auto_kernel/verification.py`
- Create: `tests/test_orchestration.py`
- Modify: `tests/test_end_to_end.py`

- [ ] **Step 1: Add failing tests for scheduler decisions and profile -> extract -> verify fixture flow**
- [ ] **Step 2: Run tests and verify failure**
- [ ] **Step 3: Implement minimal orchestration and verification logic**
- [ ] **Step 4: Re-run tests**
- [ ] **Step 5: Commit**

## Chunk 4: Documentation And Finish

### Task 6: Update docs and usage examples to match the shipped workflow

**Files:**
- Modify: `README.md`
- Modify: `program.md`
- Modify: `CUDA_OPTIMIZATION.md`

- [ ] **Step 1: Add doc assertions to end-to-end tests where useful**
- [ ] **Step 2: Update docs to match implemented commands and flow**
- [ ] **Step 3: Run the final verification set**
- [ ] **Step 4: Commit**
