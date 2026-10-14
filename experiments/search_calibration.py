"""Calibrate the search promotion rule with frozen null and slowdown controls.

The null controls compare the same native RMSNorm implementation under two
labels.  The positive controls compare that implementation with an exactly
equivalent version that performs additional clones.  These controls measure the
decision rule on this machine; they are not kernel-performance results.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

import torch

from auto_kernel.timing import benchmark_callable
from auto_kernel.workspace import write_bytes_atomic
from optimize import select_candidate
from prepare import collect_environment_report, resolve_device


PROTOCOL = {
    "version": 1,
    "null_seeds": list(range(3000, 3100)),
    "positive_seeds": list(range(4000, 4020)),
    "shapes": [[1, 32, 512], [4, 128, 512], [4, 256, 512], [8, 128, 512]],
    "repeats": 5,
    "iterations": 25,
    "warmup": 3,
    "minimum_speedup": 1.05,
    "slowdown_clones": 8,
    "maximum_false_promotions": 0,
    "minimum_positive_detections": 19,
    "minimum_positive_detections_per_shape": 4,
    "dtype": "float32",
}


def native_rms(inputs, weight, eps=1e-6):
    return torch.nn.functional.rms_norm(inputs, (inputs.shape[-1],), weight, eps)


def slowed_native_rms(inputs, weight, eps=1e-6):
    output = native_rms(inputs, weight, eps)
    for _ in range(PROTOCOL["slowdown_clones"]):
        output = output.clone()
    return output


def explicit_rms(inputs, weight, eps=1e-6):
    return inputs * torch.rsqrt(inputs.pow(2).mean(-1, keepdim=True) + eps) * weight


def make_inputs(shape, seed, device):
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(*shape, generator=generator, dtype=torch.float32).to(device)
    weight = torch.randn(shape[-1], generator=generator, dtype=torch.float32).to(device)
    return inputs, weight


def check_control_correctness(shape, seed, device):
    inputs, weight = make_inputs(shape, seed, device)
    before = inputs.clone()
    with torch.inference_mode():
        expected = native_rms(inputs, weight)
        actual = slowed_native_rms(inputs, weight)
    if (actual.shape != expected.shape or actual.dtype != expected.dtype
            or not bool(torch.isfinite(actual).all()) or not torch.equal(inputs, before)
            or not torch.equal(actual, expected)):
        raise ValueError(f"Slowdown control failed exact correctness for shape {shape}")
    return {"shape": list(shape), "seed": seed, "passed": True,
            "exact_match": True, "input_unchanged": True}


def measure_implementation(fn, reference, inputs, weight, device, seed):
    samples = {"candidate_ms": [], "reference_ms": []}
    rng = random.Random(seed)
    orders = []
    for _ in range(PROTOCOL["repeats"]):
        order = [("candidate_ms", fn), ("reference_ms", reference)]
        rng.shuffle(order)
        orders.append([name for name, _ in order])
        for name, implementation in order:
            seconds = benchmark_callable(
                lambda implementation=implementation: implementation(inputs, weight),
                device,
                PROTOCOL["warmup"],
                PROTOCOL["iterations"],
            )
            milliseconds = seconds * 1000
            if not math.isfinite(milliseconds) or milliseconds <= 0:
                raise ValueError("Invalid calibration timing")
            samples[name].append(milliseconds)
    candidate_median = statistics.median(samples["candidate_ms"])
    reference_median = statistics.median(samples["reference_ms"])
    return {
        **samples,
        "candidate_median_ms": candidate_median,
        "reference_median_ms": reference_median,
        "speedup": reference_median / candidate_median,
        "orders": orders,
    }


def run_trial(kind, shape, seed, device):
    inputs, weight = make_inputs(shape, seed, device)
    candidate_fn = native_rms
    incumbent_fn = native_rms if kind == "null" else slowed_native_rms
    evaluation_order = ["candidate", "incumbent"]
    random.Random(seed + 101).shuffle(evaluation_order)
    measurements = {}
    for offset, name in enumerate(evaluation_order):
        implementation = candidate_fn if name == "candidate" else incumbent_fn
        measurements[name] = measure_implementation(
            implementation, explicit_rms, inputs, weight, device, seed + 1000 * (offset + 1)
        )
    accepted = select_candidate(
        measurements["incumbent"], measurements["candidate"], PROTOCOL["minimum_speedup"]
    )
    return {
        "kind": kind,
        "seed": seed,
        "shape": list(shape),
        "evaluation_order": evaluation_order,
        "incumbent": measurements["incumbent"],
        "candidate": measurements["candidate"],
        "accepted": accepted,
    }


def acceptance_gate(trials):
    expected = {
        ("null", seed, tuple(PROTOCOL["shapes"][index % len(PROTOCOL["shapes"])]))
        for index, seed in enumerate(PROTOCOL["null_seeds"])
    }
    expected |= {
        ("positive", seed, tuple(PROTOCOL["shapes"][index % len(PROTOCOL["shapes"])]))
        for index, seed in enumerate(PROTOCOL["positive_seeds"])
    }
    observed = {(row.get("kind"), row.get("seed"), tuple(row.get("shape", []))) for row in trials}
    if len(trials) != len(expected) or observed != expected:
        raise ValueError("Acceptance requires every fixed trial exactly once")

    for row in trials:
        if type(row.get("accepted")) is not bool:
            raise ValueError("Every trial needs a boolean decision")
        for side in ("candidate", "incumbent"):
            measurement = row.get(side, {})
            for key in ("candidate_ms", "reference_ms"):
                values = measurement.get(key, [])
                if (len(values) != PROTOCOL["repeats"]
                        or any(not math.isfinite(value) or value <= 0 for value in values)):
                    raise ValueError("Acceptance requires complete finite positive timings")
            for key in ("candidate_median_ms", "reference_median_ms", "speedup"):
                value = measurement.get(key)
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                    raise ValueError("Acceptance requires finite positive timing summaries")

    null_trials = [row for row in trials if row["kind"] == "null"]
    positive_trials = [row for row in trials if row["kind"] == "positive"]
    false_promotions = sum(row["accepted"] for row in null_trials)
    positive_detections = sum(row["accepted"] for row in positive_trials)
    by_shape = defaultdict(lambda: {"null_trials": 0, "false_promotions": 0,
                                    "positive_trials": 0, "positive_detections": 0})
    for row in trials:
        bucket = by_shape[str(row["shape"])]
        bucket[f"{row['kind']}_trials"] += 1
        bucket["false_promotions" if row["kind"] == "null" else "positive_detections"] += row["accepted"]
    per_shape_positive_gate = all(
        row["positive_detections"] >= PROTOCOL["minimum_positive_detections_per_shape"]
        for row in by_shape.values()
    )
    return {
        "null_trials": len(null_trials),
        "false_promotions": false_promotions,
        "observed_false_promotion_rate": false_promotions / len(null_trials),
        "positive_trials": len(positive_trials),
        "positive_detections": positive_detections,
        "observed_positive_detection_rate": positive_detections / len(positive_trials),
        "by_shape": dict(by_shape),
        "no_false_promotions_gate": false_promotions <= PROTOCOL["maximum_false_promotions"],
        "positive_sensitivity_gate": (
            positive_detections >= PROTOCOL["minimum_positive_detections"]
            and per_shape_positive_gate
        ),
        "passed": (
            false_promotions <= PROTOCOL["maximum_false_promotions"]
            and positive_detections >= PROTOCOL["minimum_positive_detections"]
            and per_shape_positive_gate
        ),
        "interpretation": "Machine-local descriptive calibration, not a universal error rate",
    }


def run(output, device="cpu"):
    root = Path(__file__).resolve().parents[1]
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    if dirty:
        raise ValueError("Freeze and commit source before search calibration")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    device = resolve_device(device)
    if device == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise ValueError("Disable MPS CPU fallback")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    sources = [Path(__file__), root / "optimize.py", root / "auto_kernel/timing.py",
               root / "docs/search-calibration-protocol.md"]
    report = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "source_commit": revision,
        "source_clean": True,
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "device": device,
        "threads": 1,
        "environment": collect_environment_report(),
        "mps_cpu_fallback_enabled": False,
        "scope": "RMSNorm promotion-rule controls; no CUDA or kernel-speedup claim",
        "correctness": [],
        "trials": [],
        "complete": False,
    }

    def write_report():
        encoded = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()
        write_bytes_atomic(output / "report.json", encoded)

    try:
        for index, shape in enumerate(PROTOCOL["shapes"]):
            report["correctness"].append(check_control_correctness(shape, 2000 + index, device))
        schedule = [("null", seed) for seed in PROTOCOL["null_seeds"]]
        schedule += [("positive", seed) for seed in PROTOCOL["positive_seeds"]]
        for kind, seed in schedule:
            seeds = PROTOCOL[f"{kind}_seeds"]
            shape = PROTOCOL["shapes"][seeds.index(seed) % len(PROTOCOL["shapes"])]
            report["trials"].append(run_trial(kind, shape, seed, device))
            write_report()
        report["acceptance"] = acceptance_gate(report["trials"])
        report["complete"] = True
        report["passed"] = report["acceptance"]["passed"]
        write_report()
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        write_report()
        raise
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    args = parser.parse_args(argv)
    report = run(args.output, args.device)
    print(json.dumps(report["acceptance"], indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
