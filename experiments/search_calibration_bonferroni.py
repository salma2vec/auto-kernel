"""Calibrate a family-wise paired promotion rule on fresh controls."""

import argparse
import json

from experiments.search_calibration_paired import run as run_paired_calibration
from optimize import MAX_PAIRED_SIGN_PVALUE


PROTOCOL = {
    "version": 3,
    "null_seeds": list(range(5000, 5100)),
    "positive_seeds": list(range(6000, 6020)),
    "shapes": [[1, 32, 512], [4, 128, 512], [4, 256, 512], [8, 128, 512]],
    "repeats": 11,
    "iterations": 25,
    "warmup": 3,
    "minimum_paired_speedup": 1.05,
    "maximum_one_sided_sign_pvalue": MAX_PAIRED_SIGN_PVALUE,
    "familywise_alpha": 0.05,
    "null_decision_family_size": 100,
    "order_strategy": "balanced_cyclic",
    "slowdown_clones": 8,
    "maximum_false_promotions": 0,
    "minimum_positive_detections": 19,
    "minimum_positive_detections_per_shape": 4,
    "dtype": "float32",
}


def run(output, device="cpu"):
    return run_paired_calibration(
        output,
        device,
        protocol=PROTOCOL,
        protocol_path="docs/search-calibration-v3-protocol.md",
        scope="Family-wise RMSNorm decision controls; no CUDA or kernel-speedup claim",
        additional_sources=("experiments/search_calibration_bonferroni.py",),
        correctness_seed=7000,
    )


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
