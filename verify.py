import argparse
from pathlib import Path

from auto_kernel.artifacts import resolve_optimized_kernel
from auto_kernel.measurement import measure_workload
from auto_kernel.model_adapters import verify_model_adapter
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
    parser.add_argument("--model", default=None)
    parser.add_argument("--class-name", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = read_json(Path(args.profile_path))
    workspace_root = Path(args.workspace_root)
    grouped = aggregate_speedup_by_kernel_type(profile)
    optimized = best_speedups_by_kernel_type(Path(args.results_path))
    projected = project_end_to_end_speedup(grouped, optimized)

    print("mode=verify")
    print(f"workload_name={profile.get('workload_name', 'unknown')}")
    print(f"optimized_kernel_count={len([value for value in optimized.values() if value > 1.0])}")
    print(f"projected_end_to_end_speedup={projected:.4f}")
    if args.model:
        optimized_kernels = {}
        for record in profile.get("bottlenecks", []):
            optimized_path = resolve_optimized_kernel(
                workspace_root=workspace_root,
                kernel_type=str(record["kernel_type"]),
                target_name=str(record["name"]),
            )
            if optimized_path.exists():
                optimized_kernels[str(record["name"])] = str(optimized_path)

        model_result = verify_model_adapter(
            model_path=Path(args.model).resolve(),
            optimized_kernels=optimized_kernels,
            profile=profile,
            device=args.device,
            iterations=args.iterations,
            class_name=args.class_name,
        )
        correctness = "PASS" if model_result.get("correct") else "FAIL"
        baseline_total_ms = float(model_result.get("baseline_total_ms", 0.0))
        optimized_total_ms = float(model_result.get("optimized_total_ms", baseline_total_ms))
        speedup = baseline_total_ms / optimized_total_ms if optimized_total_ms > 0 else 1.0
        print("mode=model_verify")
        print(f"model_correctness={correctness}")
        print(f"model_max_abs_diff={float(model_result.get('max_abs_diff', 0.0)):.6f}")
        print(f"model_baseline_total_ms={baseline_total_ms:.4f}")
        print(f"model_optimized_total_ms={optimized_total_ms:.4f}")
        print(f"model_end_to_end_speedup={speedup:.4f}")
        print(f"model_optimized_kernel_count={len(optimized_kernels)}")
    if args.measure_workload:
        measured = measure_workload(
            profile=profile,
            workspace_root=workspace_root,
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
