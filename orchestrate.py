import argparse
from pathlib import Path

from auto_kernel.orchestration import (
    aggregate_speedup_by_kernel_type,
    choose_next_kernel,
    project_end_to_end_speedup,
)
from auto_kernel.results import best_speedups_by_kernel_type
from auto_kernel.workspace import read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Choose the next kernel to optimize.")
    parser.add_argument("--profile-path", default="workspace/profile.json")
    parser.add_argument("--results-path", default="results.tsv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = read_json(Path(args.profile_path))
    grouped = aggregate_speedup_by_kernel_type(profile)
    optimized = best_speedups_by_kernel_type(Path(args.results_path))
    choice = choose_next_kernel(grouped, optimized)
    projected = project_end_to_end_speedup(grouped, optimized)

    print("mode=orchestrate")
    print(f"workload_name={profile.get('workload_name', 'unknown')}")
    print(f"current_end_to_end_speedup={projected:.4f}")
    if choice:
        print(f"recommended_kernel_type={choice['kernel_type']}")
        print(f"recommended_kernel_share={float(choice.get('time_share', 0.0)):.4f}")
    else:
        print("recommended_kernel_type=none")
        print("recommended_kernel_share=0.0000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
