import copy
import importlib.util
import inspect
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Mapping

from auto_kernel.artifacts import resolve_optimized_kernel


def load_module(path: str) -> ModuleType:
    module_path = Path(path).resolve()
    if not module_path.exists():
        raise FileNotFoundError(f"Module not found: {module_path}")

    spec = importlib.util.spec_from_file_location(f"module_{module_path.stem}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create module spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clone_value(value: Any) -> Any:
    return copy.deepcopy(value)


def call_get_inputs(module: ModuleType, device: str, input_kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    get_inputs = getattr(module, "get_inputs")
    try:
        signature = inspect.signature(get_inputs)
    except (TypeError, ValueError):
        return get_inputs(device=device, **dict(input_kwargs))

    kwargs = dict(input_kwargs)
    if "device" in signature.parameters:
        kwargs["device"] = device
    return get_inputs(**kwargs)


def measure_kernel_module(path: str, input_kwargs: Mapping[str, Any], device: str, iterations: int) -> float:
    module = load_module(path)
    inputs = call_get_inputs(module, device=device, input_kwargs=input_kwargs)

    from auto_kernel.timing import benchmark_callable
    return benchmark_callable(lambda: module.kernel_fn(**inputs), device, 3, iterations) * 1000.0


def measure_workload(
    profile: Mapping[str, Any],
    workspace_root: Path,
    device: str,
    iterations: int,
) -> Dict[str, float]:
    baseline_total_ms = 0.0
    optimized_total_ms = 0.0
    optimized_entries = 0

    for record in profile.get("bottlenecks", []):
        kernel_path = str(record["kernel_path"])
        input_kwargs = dict(record.get("input_kwargs", {}))
        baseline_total_ms += measure_kernel_module(kernel_path, input_kwargs=input_kwargs, device=device, iterations=iterations)

        optimized_path = resolve_optimized_kernel(
            workspace_root,
            str(record["kernel_type"]),
            str(record["name"]),
        )
        selected_path = str(optimized_path if optimized_path.exists() else Path(kernel_path).resolve())
        optimized_total_ms += measure_kernel_module(selected_path, input_kwargs=input_kwargs, device=device, iterations=iterations)
        if optimized_path.exists():
            optimized_entries += 1

    speedup = baseline_total_ms / optimized_total_ms if optimized_total_ms > 0 else 1.0
    return {
        "baseline_total_ms": baseline_total_ms,
        "optimized_total_ms": optimized_total_ms,
        "optimized_entries": float(optimized_entries),
        "speedup": speedup,
    }
