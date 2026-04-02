import argparse
from pathlib import Path

from auto_kernel.orchestration import aggregate_speedup_by_kernel_type, project_end_to_end_speedup
from auto_kernel.results import best_speedups_by_kernel_type
from auto_kernel.workspace import read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report aggregate end-to-end speedup from recorded kernel results.")
    parser.add_argument("--profile-path", default="workspace/profile.json")
    parser.add_argument("--results-path", default="results.tsv")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
