# auto-kernel

A workload-first kernel optimization workbench with a bounded automatic search loop. Profile a workload, select a hot operation, generate a concrete candidate, check it against the reference, and retain it only when repeated timings support an improvement.

The default search is offline and uses a small, explicit family of transformations. It is useful for studying the optimization/evaluation loop without an API key. It is not an unrestricted LLM kernel author or a claim of state-of-the-art GPU performance.

## Quick start

Python 3.10+ and PyTorch 2.13+ are required.

```bash
git clone https://github.com/salma2vec/auto-kernel.git
cd auto-kernel
python -m venv .venv
source .venv/bin/activate
pip install -e .
python prepare.py
python bench.py --iterations 10 --warmup 3
python optimize.py --output workspace/search-001
python -m unittest -q
```

The default active implementation is RMSNorm. CUDA is selected when available; CPU is a functional fallback. GPU timings and Nsight Compute analysis require suitable hardware and a local CUDA/Nsight installation.

[Apple M1 validation](docs/mps-validation.md) records 12 actual MPS searches with CPU-reference checks and retained raw trials. MPS is an explicit alternative backend; these measurements do not establish CUDA performance.

[Whole-workload confirmation](docs/workload-validation.md) replaces both layers
of the synthetic RMSNorm model and evaluates new input shapes after selection.
Native RMSNorm achieved 4.80–5.26× speedup over explicit operations across five
runs, with 480 correctness checks and zero promotions in 30 same-code controls.
It matched the always-native baseline exactly: integration succeeded, while the
separate added-search-value gate failed. This is not a pretrained-model result.

## Automatic search

`optimize.py` generates candidates with explicit hypotheses (for example, replacing a separate matrix multiplication and bias addition with a native linear operation), validates them, and makes accept/reject decisions. For RMSNorm it tests multiplication-based squaring and native normalization where supported. LayerNorm and Softmax currently have a native no-change control. Unsupported operators fail visibly as rejected trials.

```bash
python optimize.py --device cpu --seed 42 --repeats 5 --iterations 25 \
  --minimum-speedup 1.05 --output workspace/search-002
```

Each new output directory contains the original source, candidate sources, selected source, source hashes, environment information, raw repeat timings, correctness results, hypotheses and rejection reasons. The active kernel stays untouched during trials. The final selected source replaces it only after the search finishes; failed or slower candidates leave the incumbent intact. Existing output directories are rejected so experiments cannot silently overwrite one another.

Checks use trusted baseline input generators, three random seeds, multiple shapes, numerical scales, edge cases, additional odd-sized shapes and input-mutation detection. Timing excludes input construction/cloning and synchronizes CUDA work. Candidate and reference order is shuffled per repeat. Promotion requires a material improvement against a freshly remeasured incumbent, nonoverlapping observed timing ranges, and improved speedup relative to the paired reference. This is a conservative heuristic, not a statistical confidence interval.

An additional implementation can be evaluated with `--candidate path/to/candidate.py`. It must satisfy the same contract as [kernel.py](kernel.py). **Candidate Python modules execute with the current user's privileges.** This runner is for trusted local code; it is not a sandbox for arbitrary downloaded or model-generated programs.

## Workload-driven workflow

```bash
python profile.py --workload models/rms_norm_stack.py --manifest-out workspace/profile.json
python extract.py --profile-path workspace/profile.json
python optimize.py --output workspace/search-003
python bench.py --hypothesis 'validated search result' --kept yes
python orchestrate.py --profile-path workspace/profile.json --results-path results.tsv
python verify.py --profile-path workspace/profile.json --workspace-root workspace --measure-workload
```

Accepted snapshots from `bench.py --kept yes` are stored under `workspace/optimized/`. `extract.py --prefer-optimized` resumes them. Ordinary benchmark results remain pending until explicitly accepted. Projected Amdahl estimates use only accepted correct finite results; they are separate from measured workload verification. Kernel-type aggregation assumes a measured speedup transfers across that operation's workload shapes, so confirm with workload/model reintegration before reporting an end-to-end improvement.

## Model adapters

```bash
python profile.py --model models/rms_norm_model.py --input-shape 4,128,512 --manifest-out workspace/profile.json
python extract.py --profile-path workspace/profile.json
python verify.py --profile-path workspace/profile.json --model models/rms_norm_model.py --input-shape 4,128,512
```

Explicit adapters export `trace_model` and `verify_model`. Generic adapters export `build_model`, `build_inputs`, and optionally `MODEL_TARGETS`. Without explicit targets, supported RMSNorm, LayerNorm, Linear and Softmax leaf modules can be discovered. The binding-driven replacement path supports attributes, parameters and constants. Arbitrary PyTorch model extraction still requires a suitable adapter; finding a module does not prove its replacement is valid for every model.

Model verification exits unsuccessfully when correctness fails or its measurements are invalid. Generic tensor comparison requires matching shapes and dtypes and finite outputs. Custom adapters remain trusted Python and must implement their own correct checks. `--measure-workload` times individual kernels; it does not independently validate their outputs or measure a complete model. Profile manifests can reference executable local kernel paths, so treat them and the workspace as trusted inputs too.

## Profiling and extension

`profile.py` reports analytical arithmetic intensity and estimated roofline classification. `ncu_profile.py` can use Nsight Compute for hardware counter analysis. These signals guide experiment selection; the built-in candidate family is currently selected by operation type. Analytical FLOP/byte counts and nominal device peaks are approximations, not measured utilization.

Add a trusted module to `kernels/` with `KERNEL_TYPE`, `kernel_fn`, `get_inputs`, `get_flops` and `get_bytes`, and register its reference and benchmark configuration. A workload exports `get_workload()` or `WORKLOAD`; each record declares a name, operation type, kernel path, and either input arguments or a supplied baseline duration. See [models/custom.py](models/custom.py).

See [methodology and related work](docs/methodology.md) for the experiment protocol, current research context and limitations. Baselines live in `kernels/`; active work lives in `kernel.py`; generated experiment notes and artifacts live in the ignored `workspace/` directory.

[MIT license](LICENSE).
