import copy
import importlib.util
import inspect
import math
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

from auto_kernel.measurement import load_module
from auto_kernel.workloads import load_workload_module, workload_name


def parse_input_shape(raw: Optional[str]) -> Optional[Tuple[int, ...]]:
    if raw is None:
        return None
    parts = [item.strip() for item in raw.split(",") if item.strip()]
    if not parts:
        return None
    return tuple(int(item) for item in parts)


def _call_with_supported_kwargs(fn: Any, **kwargs: Any) -> Any:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(**kwargs)

    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return fn(**kwargs)

    supported = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }
    return fn(**supported)


def _maybe_import_torch() -> Any:
    if importlib.util.find_spec("torch") is None:
        return None

    import torch

    return torch


def _sync_device(device: str) -> None:
    torch = _maybe_import_torch()
    if torch is None:
        return
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _clone_value(value: Any) -> Any:
    return copy.deepcopy(value)


def _default_kernel_path(kernel_type: str) -> Path:
    return (Path(__file__).resolve().parent.parent / "kernels" / f"{kernel_type}.py").resolve()


def _resolve_kernel_path(kernel_path: Optional[str], model_path: Path, kernel_type: str) -> Path:
    if kernel_path is None:
        return _default_kernel_path(kernel_type)

    path = Path(str(kernel_path))
    if path.is_absolute():
        return path
    return (model_path.parent / path).resolve()


def _build_model_instance(module: Any, device: str, dtype: str, class_name: Optional[str]) -> Any:
    if hasattr(module, "build_model"):
        model = _call_with_supported_kwargs(
            module.build_model,
            device=device,
            dtype=dtype,
            class_name=class_name,
        )
    elif class_name and hasattr(module, class_name):
        model_cls = getattr(module, class_name)
        model = model_cls()
    else:
        raise AttributeError(
            "Model module must export trace_model(...) or the generic contract: "
            "build_model(...), build_inputs(...), and MODEL_TARGETS."
        )

    if hasattr(model, "eval"):
        model.eval()
    return model


def _build_model_inputs(module: Any, device: str, input_shape: Optional[Tuple[int, ...]], dtype: str) -> Any:
    if not hasattr(module, "build_inputs"):
        raise AttributeError(
            "Model module must export trace_model(...) or build_inputs(...) for generic model profiling."
        )

    return _call_with_supported_kwargs(
        module.build_inputs,
        device=device,
        input_shape=input_shape,
        dtype=dtype,
    )


def _resolve_targets(module: Any, model_path: Path) -> Sequence[Dict[str, Any]]:
    if hasattr(module, "get_model_targets"):
        targets = module.get_model_targets()
    elif hasattr(module, "MODEL_TARGETS"):
        targets = module.MODEL_TARGETS
    else:
        raise AttributeError(
            "Model module must export trace_model(...) or MODEL_TARGETS for generic model profiling."
        )

    if not isinstance(targets, list):
        raise TypeError("MODEL_TARGETS must be a list of target definitions")

    normalized = []
    for target in targets:
        if not isinstance(target, Mapping):
            raise TypeError("Each MODEL_TARGETS entry must be a mapping")

        module_name = str(target["module_name"])
        kernel_type = str(target["kernel_type"])
        normalized.append(
            {
                "name": str(target.get("name", module_name)),
                "module_name": module_name,
                "kernel_type": kernel_type,
                "kernel_path": str(_resolve_kernel_path(target.get("kernel_path"), model_path, kernel_type)),
                "weight_attr": str(target.get("weight_attr", "weight")),
                "eps_attr": str(target.get("eps_attr", "eps")),
                "input_arg": target.get("input_arg"),
            }
        )
    return normalized


def _walk_get(root: Any, path: str) -> Any:
    current = root
    for part in path.split("."):
        if isinstance(current, Mapping):
            current = current[part]
        elif part.isdigit() and isinstance(current, (list, tuple)):
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def _walk_set(root: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        if isinstance(parent, Mapping):
            parent = parent[part]
        elif part.isdigit() and isinstance(parent, (list, tuple)):
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)

    leaf = parts[-1]
    if isinstance(parent, MutableMapping):
        parent[leaf] = value
        return
    if leaf.isdigit() and isinstance(parent, list):
        parent[int(leaf)] = value
        return
    setattr(parent, leaf, value)


class TimedCallable:
    def __init__(self, wrapped: Any, device: str, sink: Dict[str, float]) -> None:
        self._wrapped = wrapped
        self._device = device
        self._sink = sink

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        _sync_device(self._device)
        start = perf_counter()
        result = self._wrapped(*args, **kwargs)
        _sync_device(self._device)
        self._sink["elapsed_seconds"] += perf_counter() - start
        return result


class RmsNormReplacement:
    def __init__(self, kernel_fn: Any, original: Any, spec: Mapping[str, Any]) -> None:
        self._kernel_fn = kernel_fn
        self._weight = _clone_value(getattr(original, str(spec.get("weight_attr", "weight"))))
        self._eps = getattr(original, str(spec.get("eps_attr", "eps")), 1e-6)
        self._input_arg = spec.get("input_arg")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._input_arg is not None and self._input_arg in kwargs:
            x = kwargs[self._input_arg]
        elif args:
            x = args[0]
        elif kwargs:
            x = next(iter(kwargs.values()))
        else:
            raise TypeError("Optimized rms_norm replacement expected an input tensor or value")
        return self._kernel_fn(x=x, weight=self._weight, eps=self._eps)


def _build_replacement(spec: Mapping[str, Any], original: Any, kernel_path: str) -> Any:
    kernel_module = load_module(kernel_path)
    if not hasattr(kernel_module, "kernel_fn"):
        raise AttributeError(f"Kernel module missing kernel_fn: {kernel_path}")

    kernel_type = str(spec["kernel_type"])
    if kernel_type == "rms_norm":
        return RmsNormReplacement(kernel_module.kernel_fn, original, spec)

    raise KeyError(
        f"Generic model reintegration does not yet support kernel type {kernel_type!r}. "
        "Export verify_model(...) for custom handling."
    )


def _run_model(module: Any, model: Any, inputs: Any) -> Any:
    if hasattr(module, "run_model"):
        return _call_with_supported_kwargs(module.run_model, model=model, inputs=inputs)
    if isinstance(inputs, Mapping):
        return model(**dict(inputs))
    if isinstance(inputs, tuple):
        return model(*inputs)
    if isinstance(inputs, list):
        return model(*inputs)
    return model(inputs)


def _measure_model(module: Any, model: Any, inputs: Any, device: str, iterations: int) -> float:
    total_seconds = 0.0
    count = max(iterations, 1)
    for _ in range(count):
        call_inputs = _clone_value(inputs)
        _sync_device(device)
        start = perf_counter()
        _run_model(module, model, call_inputs)
        _sync_device(device)
        total_seconds += perf_counter() - start
    return total_seconds / count * 1000.0


def _compare_values(actual: Any, expected: Any, atol: float, rtol: float) -> Tuple[bool, float]:
    torch = _maybe_import_torch()
    if torch is not None and isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
        max_abs_diff = float(torch.max(torch.abs(actual - expected)).item())
        return bool(torch.allclose(actual, expected, atol=atol, rtol=rtol)), max_abs_diff

    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        if set(actual.keys()) != set(expected.keys()):
            return False, float("inf")
        overall = True
        max_abs_diff = 0.0
        for key in actual:
            correct, diff = _compare_values(actual[key], expected[key], atol, rtol)
            overall = overall and correct
            max_abs_diff = max(max_abs_diff, diff)
        return overall, max_abs_diff

    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            return False, float("inf")
        overall = True
        max_abs_diff = 0.0
        for actual_item, expected_item in zip(actual, expected):
            correct, diff = _compare_values(actual_item, expected_item, atol, rtol)
            overall = overall and correct
            max_abs_diff = max(max_abs_diff, diff)
        return overall, max_abs_diff

    try:
        actual_value = float(actual)
        expected_value = float(expected)
    except (TypeError, ValueError):
        return actual == expected, 0.0 if actual == expected else float("inf")

    diff = abs(actual_value - expected_value)
    return math.isclose(actual_value, expected_value, rel_tol=rtol, abs_tol=atol), diff


def _trace_generic_model(
    module: Any,
    model_path: Path,
    device: str,
    input_shape: Optional[Tuple[int, ...]],
    dtype: str,
    iterations: int,
    warmup: int,
    class_name: Optional[str],
) -> Dict[str, Any]:
    model = _build_model_instance(module, device=device, dtype=dtype, class_name=class_name)
    inputs = _build_model_inputs(module, device=device, input_shape=input_shape, dtype=dtype)
    targets = _resolve_targets(module, model_path)
    sinks: Dict[str, Dict[str, float]] = {}

    for target in targets:
        original = _walk_get(model, str(target["module_name"]))
        sink = {"elapsed_seconds": 0.0}
        sinks[str(target["name"])] = sink
        _walk_set(model, str(target["module_name"]), TimedCallable(original, device=device, sink=sink))

    for _ in range(max(warmup, 0)):
        _run_model(module, model, _clone_value(inputs))

    for sink in sinks.values():
        sink["elapsed_seconds"] = 0.0

    count = max(iterations, 1)
    for _ in range(count):
        _run_model(module, model, _clone_value(inputs))

    records = []
    for target in targets:
        sink = sinks[str(target["name"])]
        records.append(
            {
                "name": target["name"],
                "kernel_type": target["kernel_type"],
                "kernel_path": target["kernel_path"],
                "baseline_time_ms": sink["elapsed_seconds"] / count * 1000.0,
            }
        )

    normalized = normalize_bottlenecks(records)
    return {
        "workload_name": workload_name(module, model_path),
        "device": device,
        "total_time_ms": normalized["total_time_ms"],
        "bottlenecks": normalized["bottlenecks"],
    }


def _verify_generic_model(
    module: Any,
    model_path: Path,
    optimized_kernels: Mapping[str, str],
    device: str,
    iterations: int,
    class_name: Optional[str],
    profile: Mapping[str, Any],
    input_shape: Optional[Tuple[int, ...]],
    dtype: str,
) -> Dict[str, Any]:
    baseline_model = _build_model_instance(module, device=device, dtype=dtype, class_name=class_name)
    try:
        optimized_model = copy.deepcopy(baseline_model)
    except Exception:
        optimized_model = _build_model_instance(module, device=device, dtype=dtype, class_name=class_name)

    inputs = _build_model_inputs(module, device=device, input_shape=input_shape, dtype=dtype)
    targets = _resolve_targets(module, model_path)

    for target in targets:
        optimized_path = optimized_kernels.get(str(target["name"]))
        if optimized_path is None:
            continue
        original = _walk_get(optimized_model, str(target["module_name"]))
        replacement = _build_replacement(target, original, optimized_path)
        _walk_set(optimized_model, str(target["module_name"]), replacement)

    baseline_output = _run_model(module, baseline_model, _clone_value(inputs))
    optimized_output = _run_model(module, optimized_model, _clone_value(inputs))
    atol = float(getattr(module, "MODEL_ATOL", 1e-5))
    rtol = float(getattr(module, "MODEL_RTOL", 1e-4))
    correct, max_abs_diff = _compare_values(optimized_output, baseline_output, atol=atol, rtol=rtol)

    baseline_total_ms = _measure_model(
        module,
        baseline_model,
        inputs=_clone_value(inputs),
        device=device,
        iterations=iterations,
    )
    optimized_total_ms = _measure_model(
        module,
        optimized_model,
        inputs=_clone_value(inputs),
        device=device,
        iterations=iterations,
    )
    return {
        "correct": correct,
        "max_abs_diff": max_abs_diff,
        "baseline_total_ms": baseline_total_ms,
        "optimized_total_ms": optimized_total_ms,
        "profile": profile,
    }


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
    if hasattr(module, "trace_model"):
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

    return _trace_generic_model(
        module=module,
        model_path=model_path,
        device=device,
        input_shape=input_shape,
        dtype=dtype,
        iterations=iterations,
        warmup=warmup,
        class_name=class_name,
    )


def verify_model_adapter(
    model_path: Path,
    optimized_kernels: Mapping[str, str],
    profile: Mapping[str, Any],
    device: str,
    iterations: int,
    class_name: Optional[str],
    input_shape: Optional[Tuple[int, ...]],
    dtype: str,
) -> Dict[str, Any]:
    module = load_workload_module(str(model_path))
    if hasattr(module, "verify_model"):
        result = _call_with_supported_kwargs(
            module.verify_model,
            optimized_kernels=dict(optimized_kernels),
            profile=profile,
            device=device,
            iterations=iterations,
            class_name=class_name,
            input_shape=input_shape,
            dtype=dtype,
        )
        if not isinstance(result, dict):
            raise TypeError("verify_model(...) must return a dict")
        return result

    return _verify_generic_model(
        module=module,
        model_path=model_path,
        optimized_kernels=optimized_kernels,
        device=device,
        iterations=iterations,
        class_name=class_name,
        profile=profile,
        input_shape=input_shape,
        dtype=dtype,
    )
