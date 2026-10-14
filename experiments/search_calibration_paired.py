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
    check_control_correctness,
    explicit_rms,
    make_inputs,
    native_rms,
    slowed_native_rms,
)
from optimize import balanced_orders, exact_paired_sign_pvalue, select_candidate
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


def measure_paired(candidate_fn, incumbent_fn, inputs, weight, device, seed, protocol=PROTOCOL):
    implementations = {
        "candidate": candidate_fn,
        "incumbent": incumbent_fn,
        "reference": explicit_rms,
    }
    samples = {name: [] for name in implementations}
    if protocol.get("order_strategy") == "balanced_cyclic":
        orders = balanced_orders(seed, protocol["repeats"])
    else:
        orders = []
        rng = random.Random(seed)
        for _ in range(protocol["repeats"]):
            order = list(implementations)
            rng.shuffle(order)
            orders.append(order)
    for order in orders:
        for name in order:
            seconds = benchmark_callable(
                lambda fn=implementations[name]: fn(inputs, weight),
                device, protocol["warmup"], protocol["iterations"],
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


def run_trial(kind, shape, seed, device, protocol=PROTOCOL):
    inputs, weight = make_inputs(shape, seed, device)
    incumbent_fn = native_rms if kind == "null" else slowed_native_rms
    incumbent, candidate = measure_paired(
        native_rms, incumbent_fn, inputs, weight, device, seed, protocol
    )
    accepted = select_candidate(
        incumbent, candidate, protocol["minimum_paired_speedup"],
        protocol["maximum_one_sided_sign_pvalue"],
    )
    return {"kind": kind, "seed": seed, "shape": list(shape),
            "incumbent": incumbent, "candidate": candidate, "accepted": accepted}


def _matches(left, right):
    return (type(left) in (int, float) and type(right) in (int, float)
            and math.isfinite(left) and math.isfinite(right)
            and math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-15))


def acceptance_gate(trials, protocol=PROTOCOL):
    if 2 ** (-protocol["repeats"]) > protocol["maximum_one_sided_sign_pvalue"]:
        raise ValueError("Protocol has too few repeats for its sign threshold")
    if "familywise_alpha" in protocol:
        family_size = protocol.get("null_decision_family_size")
        if (family_size != len(protocol["null_seeds"])
                or not _matches(protocol["maximum_one_sided_sign_pvalue"],
                                protocol["familywise_alpha"] / family_size)):
            raise ValueError("Protocol family size and Bonferroni threshold disagree")
    expected = {
        ("null", seed, tuple(protocol["shapes"][index % len(protocol["shapes"])]))
        for index, seed in enumerate(protocol["null_seeds"])
    }
    expected |= {
        ("positive", seed, tuple(protocol["shapes"][index % len(protocol["shapes"])]))
        for index, seed in enumerate(protocol["positive_seeds"])
    }
    observed = {(row.get("kind"), row.get("seed"), tuple(row.get("shape", [])))
                for row in trials}
    if len(trials) != len(expected) or observed != expected:
        raise ValueError("Acceptance requires every fixed protocol trial exactly once")

    counts = {str(shape): {"null_trials": 0, "false_promotions": 0,
                           "positive_trials": 0, "positive_detections": 0}
              for shape in protocol["shapes"]}
    for row in trials:
        candidate = row.get("candidate", {})
        incumbent = row.get("incumbent", {})
        for record in (candidate, incumbent):
            for key in ("candidate_ms", "reference_ms"):
                values = record.get(key)
                if (not isinstance(values, list) or len(values) != protocol["repeats"]
                        or any(type(value) not in (int, float)
                               or not math.isfinite(value) or value <= 0 for value in values)):
                    raise ValueError("Acceptance requires complete finite positive raw timings")
            orders = record.get("orders")
            if (not isinstance(orders, list) or len(orders) != protocol["repeats"]
                    or any(not isinstance(order, list)
                           or sorted(order) != ["candidate", "incumbent", "reference"]
                           for order in orders)):
                raise ValueError("Acceptance requires complete randomized block orders")
        if (candidate["reference_ms"] != incumbent["reference_ms"]
                or candidate["orders"] != incumbent["orders"]):
            raise ValueError("Candidate and incumbent must share reference blocks and orders")
        if (protocol.get("order_strategy") == "balanced_cyclic"
                and candidate["orders"] != balanced_orders(row["seed"], protocol["repeats"])):
            raise ValueError("Stored orders do not match the frozen balanced schedule")

        candidate_median = statistics.median(candidate["candidate_ms"])
        incumbent_median = statistics.median(incumbent["candidate_ms"])
        reference_median = statistics.median(candidate["reference_ms"])
        summaries = (
            (candidate.get("candidate_median_ms"), candidate_median),
            (incumbent.get("candidate_median_ms"), incumbent_median),
            (candidate.get("reference_median_ms"), reference_median),
            (incumbent.get("reference_median_ms"), reference_median),
            (candidate.get("speedup"), reference_median / candidate_median),
            (incumbent.get("speedup"), reference_median / incumbent_median),
        )
        if any(not _matches(stored, recomputed) for stored, recomputed in summaries):
            raise ValueError("Stored timing summaries do not match raw paired timings")

        paired = [incumbent_ms / candidate_ms for incumbent_ms, candidate_ms
                  in zip(incumbent["candidate_ms"], candidate["candidate_ms"])]
        stored_paired = candidate.get("paired_speedups")
        if (not isinstance(stored_paired, list) or len(stored_paired) != len(paired)
                or any(not _matches(stored, recomputed)
                       for stored, recomputed in zip(stored_paired, paired))
                or not _matches(candidate.get("paired_median_speedup"), statistics.median(paired))
                or not _matches(candidate.get("paired_sign_pvalue"),
                                exact_paired_sign_pvalue(paired))):
            raise ValueError("Stored paired evidence does not match raw timings")

        recomputed_candidate = {**candidate, "paired_speedups": paired}
        accepted = select_candidate(
            incumbent, recomputed_candidate, protocol["minimum_paired_speedup"],
            protocol["maximum_one_sided_sign_pvalue"],
        )
        if type(row.get("accepted")) is not bool or row["accepted"] != accepted:
            raise ValueError("Stored decision does not match the frozen paired rule")
        shape_counts = counts[str(row["shape"])]
        shape_counts[f"{row['kind']}_trials"] += 1
        decision_key = "false_promotions" if row["kind"] == "null" else "positive_detections"
        shape_counts[decision_key] += accepted

    null_trials = sum(row["null_trials"] for row in counts.values())
    positive_trials = sum(row["positive_trials"] for row in counts.values())
    false_promotions = sum(row["false_promotions"] for row in counts.values())
    positive_detections = sum(row["positive_detections"] for row in counts.values())
    per_shape_positive_gate = all(
        row["positive_detections"] >= protocol["minimum_positive_detections_per_shape"]
        for row in counts.values()
    )
    no_false_promotions = false_promotions <= protocol["maximum_false_promotions"]
    positive_sensitivity = (
        positive_detections >= protocol["minimum_positive_detections"]
        and per_shape_positive_gate
    )
    return {
        "null_trials": null_trials,
        "false_promotions": false_promotions,
        "observed_false_promotion_rate": false_promotions / null_trials,
        "positive_trials": positive_trials,
        "positive_detections": positive_detections,
        "observed_positive_detection_rate": positive_detections / positive_trials,
        "by_shape": counts,
        "no_false_promotions_gate": no_false_promotions,
        "positive_sensitivity_gate": positive_sensitivity,
        "passed": no_false_promotions and positive_sensitivity,
        "interpretation": "Machine-local paired calibration, not a universal error rate",
    }


def run(output, device="cpu", *, protocol=PROTOCOL,
        protocol_path="docs/search-calibration-v2-protocol.md",
        scope="Paired RMSNorm decision controls; no CUDA or kernel-speedup claim",
        additional_sources=(), correctness_seed=2000):
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
               root / "optimize.py", root / "auto_kernel/timing.py", root / protocol_path]
    sources.extend(root / path for path in additional_sources)
    report = {
        "schema_version": 1, "protocol": protocol, "source_commit": revision,
        "source_clean": True,
        "source_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sources},
        "python": platform.python_version(), "torch": torch.__version__,
        "platform": platform.platform(), "device": device, "threads": 1,
        "environment": collect_environment_report(), "mps_cpu_fallback_enabled": False,
        "scope": scope,
        "correctness": [], "trials": [], "complete": False,
    }

    def write_report():
        encoded = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()
        write_bytes_atomic(output / "report.json", encoded)

    try:
        for index, shape in enumerate(protocol["shapes"]):
            report["correctness"].append(
                check_control_correctness(shape, correctness_seed + index, device)
            )
        schedule = [("null", seed) for seed in protocol["null_seeds"]]
        schedule += [("positive", seed) for seed in protocol["positive_seeds"]]
        for kind, seed in schedule:
            seeds = protocol[f"{kind}_seeds"]
            shape = protocol["shapes"][seeds.index(seed) % len(protocol["shapes"])]
            report["trials"].append(run_trial(kind, shape, seed, device, protocol))
            write_report()
        report["acceptance"] = acceptance_gate(report["trials"], protocol)
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
