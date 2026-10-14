"""Frozen whole-workload and always-native comparison for the RMSNorm search.

This is a synthetic two-normalization micro-workload, not a pretrained model.
Selection only sees discovery shapes; confirmation inputs are measured afterward.
"""

import argparse
import copy
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
from torch import nn

from auto_kernel.timing import benchmark_callable
from bench import load_kernel_module, validate_kernel_module
from models.rms_norm_model import MODEL_ATOL, MODEL_RTOL, build_model
from optimize import candidates, select_candidate
from prepare import resolve_device

PROTOCOL = {
    "version": 1, "seeds": [142, 143, 144, 145, 146],
    "discovery_shapes": [[1, 32, 512], [4, 128, 512]],
    "confirmation_shapes": [[4, 256, 512], [8, 128, 512]],
    "repeats": 5, "iterations": 25, "warmup": 3,
    "minimum_search_speedup": 1.05, "engineering_speedup": 1.05,
    "added_search_speedup": 1.03, "maximum_case_regression": 1.05,
    "minimum_passing_runs": 4, "controls_per_run": 6,
    "correctness_scales": [0.0, 1e-4, 1.0, 1e3],
    "correctness_layouts": ["contiguous", "strided"],
    "atol": MODEL_ATOL, "rtol": MODEL_RTOL, "dtype": "float32",
}


class KernelNorm(nn.Module):
    def __init__(self, original, kernel_fn):
        super().__init__()
        self.weight = original.weight
        self.eps = original.eps
        self.kernel_fn = kernel_fn

    def forward(self, inputs):
        return self.kernel_fn(inputs, self.weight, self.eps)


def replace_norms(original, kernel_fn):
    model = copy.deepcopy(original)
    for name in ("encoder_norm", "decoder_norm"):
        setattr(model, name, KernelNorm(getattr(model, name), kernel_fn))
    return model.eval()


def make_input(shape, seed, device, scale=1.0, layout="contiguous"):
    generator = torch.Generator().manual_seed(seed)
    width = shape[-1] * (2 if layout == "strided" else 1)
    value = torch.randn(*shape[:-1], width, generator=generator) * scale
    value = value.to(device)
    return value[..., ::2] if layout == "strided" else value


def check_correctness(models, shape, seed, device):
    reference = build_model(device="cpu").eval()
    records = []
    with torch.inference_mode():
        for scale in PROTOCOL["correctness_scales"]:
            for layout in PROTOCOL["correctness_layouts"]:
                expected = reference(make_input(shape, seed, "cpu", scale, layout))
                for name, model in models.items():
                    inputs = make_input(shape, seed, device, scale, layout)
                    before = inputs.clone()
                    actual = model(inputs).cpu()
                    if (actual.shape != expected.shape or actual.dtype != expected.dtype
                            or not bool(torch.isfinite(actual).all())
                            or not torch.equal(inputs, before)
                            or not torch.allclose(actual, expected, atol=MODEL_ATOL, rtol=MODEL_RTOL)):
                        raise ValueError(f"Whole-model correctness failed: {name}, {shape}, {scale}, {layout}")
                    records.append({
                        "method": name, "shape": list(shape), "scale": scale, "layout": layout,
                        "strides": list(inputs.stride()), "passed": True,
                        "max_absolute_error": float((actual - expected).abs().max()),
                    })
    return records


def measure_case(models, shape, seed, device, *, repeats=5, iterations=25, warmup=3):
    inputs = make_input(shape, seed, device)
    rng = random.Random(seed)
    blocks = []
    for _ in range(repeats):
        order = list(models)
        rng.shuffle(order)
        times = {}
        for name in order:
            value = 1000 * benchmark_callable(
                lambda model=models[name]: model(inputs), device, warmup, iterations
            )
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Invalid workload timing")
            times[name] = value
        blocks.append({"order": order, "milliseconds": times})
    return {"shape": list(shape), "seed": seed, "blocks": blocks}


def comparison(cases, method):
    """Equal-weight geometric aggregation; keep all underlying raw cases too."""
    candidate = [statistics.geometric_mean(case["blocks"][i]["milliseconds"][method]
                                          for case in cases)
                 for i in range(len(cases[0]["blocks"]))]
    reference = [statistics.geometric_mean(case["blocks"][i]["milliseconds"]["original"]
                                          for case in cases)
                 for i in range(len(cases[0]["blocks"]))]
    median = statistics.median(candidate)
    return {"candidate_ms": candidate, "reference_ms": reference,
            "candidate_median_ms": median, "reference_median_ms": statistics.median(reference),
            "speedup": statistics.median(reference) / median}


def choose_from_discovery(cases):
    selected, incumbent = "original", comparison(cases, "original")
    decisions = []
    for name in ("multiply_square", "native_rms"):
        measured = comparison(cases, name)
        accepted = select_candidate(incumbent, measured, PROTOCOL["minimum_search_speedup"])
        decisions.append({"method": name, "accepted": accepted,
                          "measured": measured, "incumbent": incumbent})
        if accepted:
            selected, incumbent = name, measured
    return selected, decisions


def summarize_run(seed, selected, cases, controls):
    shape_results = []
    for case in cases:
        medians = {name: statistics.median(block["milliseconds"][name] for block in case["blocks"])
                   for name in ("original", "multiply_square", "native_rms")}
        shape_results.append({
            "shape": case["shape"], "median_ms": medians,
            "speedup_vs_original": medians["original"] / medians[selected],
            "speedup_vs_always_native": medians["native_rms"] / medians[selected],
        })
    return {
        "seed": seed, "selected": selected, "confirmation_cases": shape_results,
        "geomean_speedup_vs_original": statistics.geometric_mean(
            row["speedup_vs_original"] for row in shape_results),
        "geomean_speedup_vs_always_native": statistics.geometric_mean(
            row["speedup_vs_always_native"] for row in shape_results),
        "false_promotions": sum(row["accepted"] for row in controls),
        "control_count": len(controls),
    }


def acceptance_gate(runs):
    if len(runs) != len(PROTOCOL["seeds"]) or {r["seed"] for r in runs} != set(PROTOCOL["seeds"]):
        raise ValueError("Acceptance requires all five fixed seeds")
    engineering, added, no_regressions, no_native_regressions, false_promotions = 0, 0, True, True, 0
    for run in runs:
        expected_shapes = {tuple(shape) for shape in PROTOCOL["confirmation_shapes"]}
        if (len(run["confirmation_cases"]) != len(expected_shapes)
                or {tuple(row["shape"]) for row in run["confirmation_cases"]} != expected_shapes
                or run["control_count"] != PROTOCOL["controls_per_run"]
                or type(run["false_promotions"]) is not int
                or not 0 <= run["false_promotions"] <= run["control_count"]):
            raise ValueError("Acceptance requires complete confirmation cases and controls")
        values = [run["geomean_speedup_vs_original"], run["geomean_speedup_vs_always_native"]]
        values.extend(row[key] for row in run["confirmation_cases"]
                      for key in ("speedup_vs_original", "speedup_vs_always_native"))
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("Invalid confirmation speedup")
        engineering += values[0] >= PROTOCOL["engineering_speedup"]
        added += values[1] >= PROTOCOL["added_search_speedup"]
        false_promotions += run["false_promotions"]
        no_regressions &= all(row["speedup_vs_original"] >= 1 / PROTOCOL["maximum_case_regression"]
                              for row in run["confirmation_cases"])
        no_native_regressions &= all(
            row["speedup_vs_always_native"] >= 1 / PROTOCOL["maximum_case_regression"]
            for row in run["confirmation_cases"])
    return {
        "engineering_passing_runs": engineering, "search_value_passing_runs": added,
        "false_promotions": false_promotions, "no_large_case_regression": no_regressions,
        "no_large_case_regression_vs_native": no_native_regressions,
        "engineering_gate": engineering >= PROTOCOL["minimum_passing_runs"] and
                            no_regressions and false_promotions == 0,
        "added_search_value_gate": added >= PROTOCOL["minimum_passing_runs"] and
                                  no_regressions and no_native_regressions and false_promotions == 0,
        "interpretation": "Descriptive practical gates, not significance tests or general GPU claims",
    }


def run(output, device="cpu"):
    root = Path(__file__).resolve().parents[1]
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True,
                           text=True, check=True).stdout.strip()
    if dirty:
        raise ValueError("Freeze and commit source before workload validation")
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                              text=True, check=True).stdout.strip()
    device = resolve_device(device)
    if device == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise ValueError("Disable MPS CPU fallback")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    report = {
        "protocol": PROTOCOL, "source_commit": revision, "source_clean": True,
        "python": platform.python_version(), "torch": torch.__version__,
        "platform": platform.platform(), "device": device, "threads": 1,
        "scope": "Synthetic two-RMSNorm workload; no pretrained-model or CUDA inference from MPS",
        "source_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (Path(__file__), root / "models/rms_norm_model.py",
                                       root / "docs/workload-validation-protocol.md")},
        "runs": [], "complete": False,
    }

    def write():
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    try:
        sources = {}
        functions = {}
        for name, hypothesis, source in candidates("rms_norm"):
            path = output / f"{name}.py"
            path.write_text(source)
            functions[name] = validate_kernel_module(load_kernel_module(str(path))).kernel_fn
            sources[name] = {"source": source, "hypothesis": hypothesis,
                             "sha256": hashlib.sha256(source.encode()).hexdigest()}
        report["candidates"] = sources
        timing = {name: PROTOCOL[name] for name in ("repeats", "iterations", "warmup")}
        for seed in PROTOCOL["seeds"]:
            torch.manual_seed(seed)
            original = build_model(device=device).eval()
            models = {"original": original, **{name: replace_norms(original, fn)
                                                for name, fn in functions.items()}}
            record = {"seed": seed, "correctness": [], "discovery": [], "confirmation": [],
                      "controls": [], "complete": False}
            report["runs"].append(record)
            for shape in PROTOCOL["discovery_shapes"]:
                record["correctness"].extend(check_correctness(models, shape, seed, device))
                record["discovery"].append(measure_case(models, shape, seed, device, **timing))
            selected, decisions = choose_from_discovery(record["discovery"])
            record["selected"] = selected
            record["selection_decisions"] = decisions
            # Persist the frozen decision before any confirmation measurements.
            write()
            for shape in PROTOCOL["confirmation_shapes"]:
                record["correctness"].extend(check_correctness(models, shape, seed + 10000, device))
                record["confirmation"].append(measure_case(
                    models, shape, seed + 10000, device, **timing))
            for index in range(PROTOCOL["controls_per_run"]):
                shape = PROTOCOL["confirmation_shapes"][index % 2]
                control = measure_case({"original": original,
                                        "native_a": models["native_rms"],
                                        "native_b": models["native_rms"]},
                                       shape, seed + 20000 + index, device, **timing)
                accepted = select_candidate(comparison([control], "native_a"),
                                            comparison([control], "native_b"),
                                            PROTOCOL["minimum_search_speedup"])
                record["controls"].append({"case": control, "accepted": accepted})
            record["summary"] = summarize_run(seed, selected, record["confirmation"], record["controls"])
            record["complete"] = True
            write()
            print(json.dumps(record["summary"]), flush=True)
        report["acceptance"] = acceptance_gate([record["summary"] for record in report["runs"]])
        report["complete"] = True
        write()
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        write()
        raise
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    args = parser.parse_args()
    run(args.output, args.device)
