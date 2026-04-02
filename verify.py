import argparse
from pathlib import Path

from auto_kernel.measurement import measure_workload
from auto_kernel.orchestration import aggregate_speedup_by_kernel_type, project_end_to_end_speedup
from auto_kernel.results import best_speedups_by_kernel_type
from auto_kernel.workspace import read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report aggregate end-to-end speedup from recorded kernel results.")
    parser.add_argument("--profile-path", default="workspace/profile.json")
    parser.add_argument("--results-path", default="results.tsv")
    parser.add_argument("--workspace-root", default="workspace")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--measure-workload", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = read_json(Path(args.profile_path))
    grouped = aggregate_speedup_by_kernel_type(profile)
    optimized = best_speedups_by_kernel_type(Path(args.results_path))
    projected = project_end_to_end_speedup(grouped, optimized)

    print("mode=verify")
    print(f"workload_name={profile.get('workload_name', 'unknown')}")
    print(f"optimized_kernel_count={len([value for value in optimized.values() if value > 1.0])}")
    print(f"projected_end_to_end_speedup={projected:.4f}")
    if args.measure_workload:
        measured = measure_workload(
            profile=profile,
            workspace_root=Path(args.workspace_root),
            device=args.device,
            iterations=args.iterations,
        )
        print(f"measured_baseline_total_ms={measured['baseline_total_ms']:.4f}")
        print(f"measured_optimized_total_ms={measured['optimized_total_ms']:.4f}")
        print(f"measured_optimized_entry_count={int(measured['optimized_entries'])}")
        print(f"measured_end_to_end_speedup={measured['speedup']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
