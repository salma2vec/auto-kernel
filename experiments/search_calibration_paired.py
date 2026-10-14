"""Prospectively calibrate the paired search promotion rule."""

import argparse
import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
from pathlib import Path

import torch

from auto_kernel.timing import benchmark_callable
from auto_kernel.workspace import write_bytes_atomic
from experiments.search_calibration import (
    acceptance_gate,
    check_control_correctness,
    explicit_rms,
    make_inputs,
    native_rms,
    slowed_native_rms,
)
from optimize import exact_paired_sign_pvalue, select_candidate
from prepare import collect_environment_report, resolve_device


PROTOCOL = {
    "version": 2,
    "null_seeds": list(range(3000, 3100)),
    "positive_seeds": list(range(4000, 4020)),
    "shapes": [[1, 32, 512], [4, 128, 512], [4, 256, 512], [8, 128, 512]],
    "repeats": 5,
    "iterations": 25,
    "warmup": 3,
    "minimum_paired_speedup": 1.05,
    "maximum_one_sided_sign_pvalue": 0.05,
    "slowdown_clones": 8,
    "maximum_false_promotions": 0,
    "minimum_positive_detections": 19,
    "minimum_positive_detections_per_shape": 4,
    "dtype": "float32",
}


def measure_paired(candidate_fn, incumbent_fn, inputs, weight, device, seed):
    implementations = {
        "candidate": candidate_fn,
        "incumbent": incumbent_fn,
        "reference": explicit_rms,
    }
    samples = {name: [] for name in implementations}
    orders = []
    rng = random.Random(seed)
    for _ in range(PROTOCOL["repeats"]):
        order = list(implementations)
        rng.shuffle(order)
        orders.append(order)
        for name in order:
            seconds = benchmark_callable(
                lambda fn=implementations[name]: fn(inputs, weight),
                device, PROTOCOL["warmup"], PROTOCOL["iterations"],
            )
            milliseconds = seconds * 1000
            if not math.isfinite(milliseconds) or milliseconds <= 0:
                raise ValueError("Invalid paired calibration timing")
            samples[name].append(milliseconds)
    reference_median = statistics.median(samples["reference"])
    paired_speedups = [incumbent / candidate for incumbent, candidate
                       in zip(samples["incumbent"], samples["candidate"])]
    common = {"reference_ms": samples["reference"],
              "reference_median_ms": reference_median, "orders": orders}
    incumbent_median = statistics.median(samples["incumbent"])
    candidate_median = statistics.median(samples["candidate"])
    incumbent = {**common, "candidate_ms": samples["incumbent"],
                 "candidate_median_ms": incumbent_median,
                 "speedup": reference_median / incumbent_median}
    candidate = {**common, "candidate_ms": samples["candidate"],
                 "candidate_median_ms": candidate_median,
                 "speedup": reference_median / candidate_median,
                 "paired_speedups": paired_speedups,
                 "paired_median_speedup": statistics.median(paired_speedups),
                 "paired_sign_pvalue": exact_paired_sign_pvalue(paired_speedups)}
    return incumbent, candidate


def run_trial(kind, shape, seed, device):
    inputs, weight = make_inputs(shape, seed, device)
    incumbent_fn = native_rms if kind == "null" else slowed_native_rms
    incumbent, candidate = measure_paired(native_rms, incumbent_fn, inputs, weight, device, seed)
    accepted = select_candidate(incumbent, candidate, PROTOCOL["minimum_paired_speedup"])
    return {"kind": kind, "seed": seed, "shape": list(shape),
            "incumbent": incumbent, "candidate": candidate, "accepted": accepted}


def run(output, device="cpu"):
    root = Path(__file__).resolve().parents[1]
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    if dirty:
        raise ValueError("Freeze and commit source before paired search calibration")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    device = resolve_device(device)
    if device == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise ValueError("Disable MPS CPU fallback")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    sources = [Path(__file__), root / "experiments/search_calibration.py",
               root / "optimize.py", root / "auto_kernel/timing.py",
               root / "docs/search-calibration-v2-protocol.md"]
    report = {
        "schema_version": 1, "protocol": PROTOCOL, "source_commit": revision,
        "source_clean": True,
        "source_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sources},
        "python": platform.python_version(), "torch": torch.__version__,
        "platform": platform.platform(), "device": device, "threads": 1,
        "environment": collect_environment_report(), "mps_cpu_fallback_enabled": False,
        "scope": "Paired RMSNorm decision controls; no CUDA or kernel-speedup claim",
        "correctness": [], "trials": [], "complete": False,
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
