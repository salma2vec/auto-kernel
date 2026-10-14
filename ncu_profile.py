import argparse
import csv
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch

from bench import _format_optional, load_kernel_module, validate_kernel_module
from prepare import collect_environment_report, resolve_device
from profile import profile_kernel


DEFAULT_METRICS = {
    "roofline": [
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    ],
    "memory": [
        "l1tex__t_sector_hit_rate.pct",
        "lts__t_sector_op_read_hit_rate.pct",
        "dram__bytes.sum",
    ],
    "warp_stall": [
        "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
        "smsp__average_warps_issue_stalled_membar_per_issue_active.ratio",
        "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
        "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
    ],
}


def default_experiment_id() -> str:
    return datetime.now(timezone.utc).strftime("ncu-%Y%m%d-%H%M%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Nsight Compute analysis for the active kernel.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--skills", default="roofline,memory,warp_stall")
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--diff", nargs=2, metavar=("BEFORE", "AFTER"))
    return parser.parse_args()


def selected_metrics(skills: str) -> List[str]:
    names: List[str] = []
    for skill in [item.strip() for item in skills.split(",") if item.strip()]:
        names.extend(DEFAULT_METRICS.get(skill, []))
    if not names:
        for values in DEFAULT_METRICS.values():
            names.extend(values)
    seen = set()
    ordered = []
    for name in names:
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered


def parse_ncu_csv(path: Path) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header_map = None
        for row in reader:
            if not row:
                continue
            if header_map is None and "Metric Name" in row and "Metric Value" in row:
                header_map = {name: index for index, name in enumerate(row)}
                continue
            if header_map is None:
                continue
            metric_name = row[header_map["Metric Name"]].strip()
            metric_value = row[header_map["Metric Value"]].strip()
            if not metric_name or not metric_value:
                continue
            try:
                metrics[metric_name] = float(metric_value.replace(",", ""))
            except ValueError:
                continue
    return metrics


def summarize_metrics(metrics: Dict[str, float], fallback_bottleneck: str) -> Dict[str, str]:
    occupancy = metrics.get("sm__warps_active.avg.pct_of_peak_sustained_active")
    l1_hit = metrics.get("l1tex__t_sector_hit_rate.pct")
    l2_hit = metrics.get("lts__t_sector_op_read_hit_rate.pct")

    stall_candidates = {
        "long_scoreboard": metrics.get("smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio"),
        "membar": metrics.get("smsp__average_warps_issue_stalled_membar_per_issue_active.ratio"),
        "barrier": metrics.get("smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio"),
        "math_pipe_throttle": metrics.get("smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio"),
    }
    stall_candidates = {key: value for key, value in stall_candidates.items() if value is not None}
    top_stall = max(stall_candidates, key=stall_candidates.get) if stall_candidates else "unavailable"

    if top_stall == "long_scoreboard" or (l2_hit is not None and l2_hit < 60):
        finding = "Memory latency is the dominant microarchitectural limiter."
        action = "Improve coalescing, locality, or prefetching before chasing arithmetic changes."
        bottleneck = "memory_bound"
    elif occupancy is not None and occupancy < 35:
        finding = "Occupancy is low enough to limit latency hiding."
        action = "Reduce register pressure or retune block and tile sizes."
        bottleneck = fallback_bottleneck
    elif top_stall == "math_pipe_throttle":
        finding = "Math pipeline pressure is the most visible stall source."
        action = "Reduce redundant math, improve instruction mix, or retune tile size."
        bottleneck = "compute_bound"
    else:
        finding = "No single dominant stall source was extracted from the selected metrics."
        action = "Inspect the raw NCU CSV for a more targeted hypothesis."
        bottleneck = fallback_bottleneck

    return {
        "ncu_bottleneck": bottleneck,
        "ncu_top_stall": top_stall,
        "ncu_finding": finding,
        "ncu_action": action,
        "ncu_occupancy": _format_optional(occupancy),
        "ncu_l1_hit_rate": _format_optional(l1_hit),
        "ncu_l2_hit_rate": _format_optional(l2_hit),
    }


def diff_metrics(before: Dict[str, float], after: Dict[str, float]) -> Iterable[Tuple[str, float]]:
    common = sorted(set(before) & set(after))
    for name in common:
        yield name, after[name] - before[name]


def write_report(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analytical_fallback(args: argparse.Namespace) -> int:
    module = load_kernel_module("kernel.py")
    contract = validate_kernel_module(module)
    report = profile_kernel(contract, device=args.device, iterations=args.iterations, warmup=args.warmup)
    experiment_id = args.experiment_id or default_experiment_id()
    report_path = Path("ncu_reports") / f"{experiment_id}.txt"
    lines = [
        "mode=analytical",
        f"kernel={contract.kernel_type}",
        f"device={args.device}",
        f"ncu_bottleneck={report.bottleneck}",
        "ncu_top_stall=unavailable",
        "ncu_finding=Nsight Compute unavailable; using analytical roofline fallback.",
        "ncu_action=Install ncu and rerun on a CUDA device for microarchitectural analysis.",
        "ncu_occupancy=n/a",
        "ncu_l1_hit_rate=n/a",
        "ncu_l2_hit_rate=n/a",
        f"arithmetic_intensity={report.arithmetic_intensity:.4f}",
        f"throughput_tflops={report.throughput_tflops:.4f}",
        f"bandwidth_gb_s={report.bandwidth_gb_s:.4f}",
    ]
    write_report(report_path, lines)
    print("\n".join(lines + [f"ncu_raw_report={report_path}"]))
    return 0


def live_ncu_profile(args: argparse.Namespace) -> int:
    module = load_kernel_module("kernel.py")
    contract = validate_kernel_module(module)
    experiment_id = args.experiment_id or default_experiment_id()
    csv_path = Path("ncu_reports") / f"{experiment_id}.csv"

    metrics = selected_metrics(args.skills)
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = Path(tmpdir) / "ncu_runner.py"
        runner.write_text(
            "\n".join(
                [
                    "import torch",
                    "from bench import get_benchmark_config, load_kernel_module, validate_kernel_module, _clone_tree, _sync_if_needed",
                    "module = load_kernel_module('kernel.py')",
                    "contract = validate_kernel_module(module)",
                    f"config = get_benchmark_config('{contract.kernel_type}')",
                    f"inputs = config.input_builder(contract, '{args.device}', config.timing_shape)",
                    f"for _ in range({args.warmup}):",
                    "    contract.kernel_fn(**_clone_tree(inputs))",
                    f"    _sync_if_needed('{args.device}')",
                    f"for _ in range({args.iterations}):",
                    "    contract.kernel_fn(**_clone_tree(inputs))",
                    f"    _sync_if_needed('{args.device}')",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        command = [
            "ncu",
            "--target-processes",
            "all",
            "--csv",
            "--log-file",
            str(csv_path),
            "--metrics",
            ",".join(metrics),
            sys.executable,
            str(runner),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            return analytical_fallback(args)

    parsed = parse_ncu_csv(csv_path)
    macro = profile_kernel(contract, device=args.device, iterations=max(args.iterations, 1), warmup=args.warmup)
    summary = summarize_metrics(parsed, macro.bottleneck)
    lines = [
        "mode=ncu",
        f"kernel={contract.kernel_type}",
        f"device={args.device}",
        f"ncu_bottleneck={summary['ncu_bottleneck']}",
        f"ncu_top_stall={summary['ncu_top_stall']}",
        f"ncu_finding={summary['ncu_finding']}",
        f"ncu_action={summary['ncu_action']}",
        f"ncu_occupancy={summary['ncu_occupancy']}",
        f"ncu_l1_hit_rate={summary['ncu_l1_hit_rate']}",
        f"ncu_l2_hit_rate={summary['ncu_l2_hit_rate']}",
    ]
    write_report(csv_path.with_suffix(".summary.txt"), lines)
    print("\n".join(lines + [f"ncu_raw_report={csv_path}"]))
    return 0


def diff_mode(before_path: Path, after_path: Path) -> int:
    before = parse_ncu_csv(before_path)
    after = parse_ncu_csv(after_path)
    print("mode=diff")
    print(f"before={before_path}")
    print(f"after={after_path}")
    for name, delta in sorted(diff_metrics(before, after), key=lambda item: abs(item[1]), reverse=True)[:10]:
        print(f"diff_{name}={delta:.4f}")
    return 0


def main() -> int:
    args = parse_args()
    if args.diff:
        return diff_mode(Path(args.diff[0]), Path(args.diff[1]))

    args.device = resolve_device(args.device)
    environment = collect_environment_report()
    if args.device.startswith("cuda") and torch.cuda.is_available() and environment["ncu_available"]:
        return live_ncu_profile(args)
    return analytical_fallback(args)


if __name__ == "__main__":
    raise SystemExit(main())
