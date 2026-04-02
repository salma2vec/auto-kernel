from typing import Dict, Mapping


def aggregate_speedup_by_kernel_type(profile: Mapping[str, object]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, Dict[str, float]] = {}
    for record in profile.get("bottlenecks", []):
        kernel_type = str(record["kernel_type"])
        grouped.setdefault(
            kernel_type,
            {
                "kernel_type": kernel_type,
                "time_share": 0.0,
                "baseline_time_ms": 0.0,
            },
        )
        grouped[kernel_type]["time_share"] += float(record.get("time_share", 0.0))
        grouped[kernel_type]["baseline_time_ms"] += float(record.get("baseline_time_ms", 0.0))
    return grouped


def amdahl_speedup(time_share: float, kernel_speedup: float) -> float:
    if kernel_speedup <= 0:
        raise ValueError("kernel_speedup must be positive")
    return 1.0 / ((1.0 - time_share) + (time_share / kernel_speedup))


def project_end_to_end_speedup(
    grouped_profile: Mapping[str, Mapping[str, float]],
    optimized_speedups: Mapping[str, float],
) -> float:
    covered_share = 0.0
    effective_runtime = 0.0
    for kernel_type, record in grouped_profile.items():
        time_share = float(record.get("time_share", 0.0))
        covered_share += time_share
        speedup = float(optimized_speedups.get(kernel_type, 1.0))
        effective_runtime += time_share / max(speedup, 1e-9)

    uncovered_share = max(0.0, 1.0 - covered_share)
    effective_runtime += uncovered_share
    if effective_runtime <= 0:
        return 1.0
    return 1.0 / effective_runtime


def choose_next_kernel(
    grouped_profile: Mapping[str, Mapping[str, float]],
    optimized_speedups: Mapping[str, float],
) -> Dict[str, float]:
    remaining = [
        record
        for kernel_type, record in grouped_profile.items()
        if float(optimized_speedups.get(kernel_type, 1.0)) <= 1.0
    ]
    if not remaining:
        if not grouped_profile:
            return {}
        remaining = list(grouped_profile.values())
    return dict(max(remaining, key=lambda record: float(record.get("time_share", 0.0))))
