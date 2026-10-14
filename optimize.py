"""Bounded, offline hypothesis search over trusted local kernel candidates."""
import argparse
import hashlib
import json
import math
import platform
import random
import statistics
from dataclasses import asdict, replace
from pathlib import Path

import torch

from bench import (
    benchmark_callable,
    get_benchmark_config,
    load_kernel_module,
    run_correctness_checks,
    run_correctness_pass,
    validate_kernel_module,
)
from prepare import resolve_device
from reference import REFERENCE_FUNCTIONS
from auto_kernel.workspace import write_bytes_atomic


def candidates(kernel_type):
    """Small transparent search space; hypotheses are falsifiable, not promises."""
    prefix = f"from kernels.{kernel_type} import *\nimport torch\n"
    if kernel_type == "rms_norm":
        return [("multiply_square", "Elementwise multiplication may lower exponentiation overhead.", prefix +
                 "def kernel_fn(x, weight, eps=1e-6):\n    return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps) * weight\n"),
                ("native_rms", "A native normalization may fuse memory traffic where supported.", prefix +
                 "def kernel_fn(x, weight, eps=1e-6):\n    return torch.nn.functional.rms_norm(x, (x.shape[-1],), weight, eps)\n")]
    if kernel_type == "linear":
        return [("fused_linear", "A native linear operator may fuse bias addition with matrix multiplication.", prefix +
                 "def kernel_fn(x, weight, bias=None):\n    return torch.nn.functional.linear(x, weight, bias)\n")]
    return [("native_control", "The native operator is a no-change control for timing noise.", prefix)]


def evaluate(contract, device, seed, repeats, iterations):
    """Use trusted baseline input generation, several seeds and unseen shapes."""
    cfg = get_benchmark_config(contract.kernel_type)
    baseline = validate_kernel_module(load_kernel_module(str(Path(__file__).parent / "kernels" / f"{contract.kernel_type}.py")))
    contract = replace(contract, get_inputs=baseline.get_inputs)
    with torch.inference_mode():
        for offset in range(3):
            torch.manual_seed(seed + offset)
            summary = run_correctness_checks(contract, cfg, device, quick=False)
            if summary.overall != "PASS":
                raise ValueError(f"correctness failed: {asdict(summary)}")
        for shape in [(1, 7, 31), (3, 9, 65)]:
            inputs = cfg.input_builder(contract, device, shape)
            ok, _, detail = run_correctness_pass(contract, REFERENCE_FUNCTIONS[contract.kernel_type], inputs, cfg.atol, cfg.rtol)
            if not ok:
                raise ValueError(detail)
        torch.manual_seed(seed + 101)
        inputs = cfg.input_builder(contract, device, cfg.timing_shape)
        reference = REFERENCE_FUNCTIONS[contract.kernel_type]
        samples = {"candidate_ms": [], "reference_ms": []}
        order_rng = random.Random(seed)
        for _ in range(repeats):
            order = [("candidate_ms", contract.kernel_fn), ("reference_ms", reference)]
            order_rng.shuffle(order)
            for name, fn in order:
                seconds = benchmark_callable(lambda: fn(**inputs), device, 3, iterations)
                samples[name].append(seconds * 1000)
        med_candidate = statistics.median(samples["candidate_ms"])
        med_reference = statistics.median(samples["reference_ms"])
    return {**samples, "candidate_median_ms": med_candidate,
            "reference_median_ms": med_reference, "speedup": med_reference / med_candidate,
            "shape": list(cfg.timing_shape), "correctness": "PASS"}


def exact_paired_sign_pvalue(paired_speedups):
    """One-sided exact sign-test p-value for candidate-faster pairs."""
    non_ties = [value for value in paired_speedups if value != 1]
    if not non_ties:
        return 1.0
    favorable = sum(value > 1 for value in non_ties)
    return sum(math.comb(len(non_ties), count)
               for count in range(favorable, len(non_ties) + 1)) / (2 ** len(non_ties))


def evaluate_pair(incumbent, candidate, device, seed, repeats, iterations):
    """Validate the candidate, then time candidate/incumbent/reference together."""
    if incumbent.kernel_type != candidate.kernel_type:
        raise ValueError("Candidate changed kernel type")
    cfg = get_benchmark_config(candidate.kernel_type)
    baseline = validate_kernel_module(load_kernel_module(
        str(Path(__file__).parent / "kernels" / f"{candidate.kernel_type}.py")
    ))
    incumbent = replace(incumbent, get_inputs=baseline.get_inputs)
    candidate = replace(candidate, get_inputs=baseline.get_inputs)
    with torch.inference_mode():
        for offset in range(3):
            torch.manual_seed(seed + offset)
            summary = run_correctness_checks(candidate, cfg, device, quick=False)
            if summary.overall != "PASS":
                raise ValueError(f"correctness failed: {asdict(summary)}")
        for shape in [(1, 7, 31), (3, 9, 65)]:
            inputs = cfg.input_builder(candidate, device, shape)
            ok, _, detail = run_correctness_pass(
                candidate, REFERENCE_FUNCTIONS[candidate.kernel_type],
                inputs, cfg.atol, cfg.rtol,
            )
            if not ok:
                raise ValueError(detail)

        torch.manual_seed(seed + 101)
        inputs = cfg.input_builder(candidate, device, cfg.timing_shape)
        implementations = {
            "candidate": candidate.kernel_fn,
            "incumbent": incumbent.kernel_fn,
            "reference": REFERENCE_FUNCTIONS[candidate.kernel_type],
        }
        samples = {name: [] for name in implementations}
        orders = []
        order_rng = random.Random(seed)
        for _ in range(repeats):
            order = list(implementations)
            order_rng.shuffle(order)
            orders.append(order)
            for name in order:
                seconds = benchmark_callable(
                    lambda fn=implementations[name]: fn(**inputs), device, 3, iterations
                )
                samples[name].append(seconds * 1000)

    reference_median = statistics.median(samples["reference"])
    incumbent_median = statistics.median(samples["incumbent"])
    candidate_median = statistics.median(samples["candidate"])
    paired_speedups = [incumbent_ms / candidate_ms for incumbent_ms, candidate_ms
                       in zip(samples["incumbent"], samples["candidate"])]
    common = {"reference_ms": samples["reference"],
              "reference_median_ms": reference_median,
              "shape": list(cfg.timing_shape), "correctness": "PASS", "orders": orders}
    incumbent_result = {**common, "candidate_ms": samples["incumbent"],
                        "candidate_median_ms": incumbent_median,
                        "speedup": reference_median / incumbent_median}
    candidate_result = {**common, "candidate_ms": samples["candidate"],
                        "candidate_median_ms": candidate_median,
                        "speedup": reference_median / candidate_median,
                        "paired_speedups": paired_speedups,
                        "paired_median_speedup": statistics.median(paired_speedups),
                        "paired_sign_pvalue": exact_paired_sign_pvalue(paired_speedups)}
    return incumbent_result, candidate_result


def select_candidate(incumbent, candidate, minimum_speedup):
    """Require a material paired effect and an exact one-sided sign-test gate."""
    if "paired_speedups" in candidate:
        paired = candidate["paired_speedups"]
        if (not isinstance(paired, list) or len(paired) < 3
                or any(not isinstance(value, (int, float))
                       or not math.isfinite(value) or value <= 0 for value in paired)):
            return False
        return (statistics.median(paired) >= minimum_speedup
                and exact_paired_sign_pvalue(paired) <= 0.05)
    # Compatibility for old reports and third-party callers without paired data.
    return (candidate["candidate_median_ms"] * minimum_speedup < incumbent["candidate_median_ms"]
            and max(candidate["candidate_ms"]) < min(incumbent["candidate_ms"])
            and candidate.get("speedup", 1.) > incumbent.get("speedup", 1.) * minimum_speedup)


def optimize(kernel_file, output, device="cpu", seed=42, repeats=5, iterations=25, minimum_speedup=1.05, extra_candidates=()):
    if repeats < 3 or iterations < 1 or not math.isfinite(minimum_speedup) or minimum_speedup <= 1:
        raise ValueError("Require repeats >= 3, iterations >= 1 and minimum_speedup > 1")
    kernel_file, output = Path(kernel_file), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    original = kernel_file.read_bytes()
    (output / "baseline.py").write_bytes(original)
    incumbent = validate_kernel_module(load_kernel_module(str(kernel_file)))
    initial = evaluate(incumbent, device, seed, repeats, iterations)
    report = {"schema_version": 1, "seed": seed, "device": device,
              "torch": torch.__version__, "python": platform.python_version(),
              "threads": torch.get_num_threads(), "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(device) if device.startswith("cuda") else None,
              "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
              "iterations": iterations, "repeats": repeats, "minimum_speedup": minimum_speedup,
              "baseline_sha256": hashlib.sha256(original).hexdigest(),
              "kernel_type": incumbent.kernel_type, "baseline": initial, "trials": []}
    best_source, best = original, initial
    proposals = candidates(incumbent.kernel_type)
    for path in extra_candidates:
        path = Path(path)
        proposals.append((path.stem, "User-supplied trusted local implementation.", path.read_text()))
    for index, (name, hypothesis, source) in enumerate(proposals):
        source_bytes = source.encode()
        snapshot = output / f"candidate-{index:03d}.py"
        snapshot.write_bytes(source_bytes)
        trial = {"name": name, "hypothesis": hypothesis, "sha256": hashlib.sha256(source_bytes).hexdigest(),
                 "source": snapshot.name, "accepted": False}
        try:
            contract = validate_kernel_module(load_kernel_module(str(snapshot)))
            if contract.kernel_type != incumbent.kernel_type:
                raise ValueError("Candidate changed kernel type")
            # Interleave candidate, incumbent and common reference in each
            # repeat so transient drift cannot masquerade as a separate-block win.
            incumbent_snapshot = output / f"incumbent-{index:03d}.py"
            incumbent_snapshot.write_bytes(best_source)
            incumbent_contract = validate_kernel_module(load_kernel_module(str(incumbent_snapshot)))
            best, result = evaluate_pair(
                incumbent_contract, contract, device, seed, repeats, iterations
            )
            trial.update(result)
            trial["incumbent_comparison"] = best
            if select_candidate(best, result, minimum_speedup):
                best_source, best = source_bytes, result
                trial["accepted"] = True
                trial["decision"] = "paired median exceeded threshold and one-sided sign gate passed"
            else:
                trial["decision"] = "rejected: paired effect or one-sided sign gate failed"
        except Exception as error:
            trial["decision"] = "rejected: evaluation failed"
            trial["error"] = f"{type(error).__name__}: {error}"
        report["trials"].append(trial)
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    # Keep the active file unchanged until every trial and report is complete.
    # Failed candidates therefore never need a destructive checkout/reset.
    if kernel_file.read_bytes() != original:
        raise RuntimeError("Active kernel changed during search; refusing to overwrite it")
    report["selected_sha256"] = hashlib.sha256(best_source).hexdigest()
    report["changed"] = best_source != original
    report["selected"] = best
    (output / "selected.py").write_bytes(best_source)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    write_bytes_atomic(kernel_file, best_source, mode=kernel_file.stat().st_mode & 0o777)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-file", default="kernel.py")
    parser.add_argument("--output", default="workspace/search")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--minimum-speedup", type=float, default=1.05)
    parser.add_argument("--candidate", action="append", default=[], help="Trusted local Python module; executes with your privileges")
    args = parser.parse_args()
    report = optimize(args.kernel_file, args.output, resolve_device(args.device), args.seed, args.repeats, args.iterations, args.minimum_speedup, args.candidate)
    print(json.dumps({"changed": report["changed"], "report": str(Path(args.output) / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
