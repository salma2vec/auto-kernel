from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from auto_kernel.workloads import load_workload_module, workload_name


def parse_input_shape(raw: Optional[str]) -> Optional[Tuple[int, ...]]:
    if raw is None:
        return None
    parts = [item.strip() for item in raw.split(",") if item.strip()]
    if not parts:
        return None
    return tuple(int(item) for item in parts)


def normalize_bottlenecks(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    normalized = []
    total_time_ms = sum(float(record.get("baseline_time_ms", 0.0)) for record in records)
    ordered = sorted(records, key=lambda record: float(record.get("baseline_time_ms", 0.0)), reverse=True)
    for index, record in enumerate(ordered, start=1):
        time_share = 0.0 if total_time_ms <= 0 else float(record.get("baseline_time_ms", 0.0)) / total_time_ms
        normalized.append(
            {
                "rank": index,
                "name": record["name"],
                "kernel_type": record["kernel_type"],
                "kernel_path": str(record["kernel_path"]),
                "baseline_time_ms": float(record.get("baseline_time_ms", 0.0)),
                "time_share": time_share,
                "input_kwargs": dict(record.get("input_kwargs", {})),
            }
        )
    return {
        "total_time_ms": total_time_ms,
        "bottlenecks": normalized,
    }


def trace_model_adapter(
    model_path: Path,
    device: str,
    input_shape: Optional[Tuple[int, ...]],
    dtype: str,
    iterations: int,
    warmup: int,
    class_name: Optional[str],
) -> Dict[str, Any]:
    module = load_workload_module(str(model_path))
    if not hasattr(module, "trace_model"):
        raise AttributeError(
            "Model module must export trace_model(...) for adapter-driven model profiling in this environment."
        )

    traced = module.trace_model(
        device=device,
        input_shape=input_shape,
        dtype=dtype,
        iterations=iterations,
        warmup=warmup,
        class_name=class_name,
    )
    if not isinstance(traced, dict):
        raise TypeError("trace_model(...) must return a dict")

    normalized = normalize_bottlenecks(traced.get("bottlenecks", []))
    return {
        "workload_name": traced.get("workload_name", workload_name(module, model_path)),
        "device": traced.get("device", device),
        "total_time_ms": normalized["total_time_ms"],
        "bottlenecks": normalized["bottlenecks"],
    }


def verify_model_adapter(
    model_path: Path,
    optimized_kernels: Mapping[str, str],
    profile: Mapping[str, Any],
    device: str,
    iterations: int,
    class_name: Optional[str],
) -> Dict[str, Any]:
    module = load_workload_module(str(model_path))
    if not hasattr(module, "verify_model"):
        raise AttributeError(
            "Model module must export verify_model(...) for adapter-driven model verification in this environment."
        )

    result = module.verify_model(
        optimized_kernels=dict(optimized_kernels),
        profile=profile,
        device=device,
        iterations=iterations,
        class_name=class_name,
    )
    if not isinstance(result, dict):
        raise TypeError("verify_model(...) must return a dict")
    return result
