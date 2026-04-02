import argparse
import copy
import importlib.util
import inspect
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from types import ModuleType
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from auto_kernel.artifacts import persist_optimized_kernel
from auto_kernel.workspace import active_kernel_manifest_path, read_json
from prepare import resolve_device
from reference import REFERENCE_FUNCTIONS


Shape = Tuple[int, int, int]

RESULT_COLUMNS = [
    "experiment_id",
    "kernel_type",
    "target_name",
    "hypothesis",
    "correctness",
    "time_ms",
    "throughput",
    "peak_vram_mb",
    "kept",
    "speedup_vs_reference",
]

KNOWN_GPU_PEAKS = {
    "H100": {"peak_compute_tflops": 67.0, "peak_bandwidth_gb_s": 3350.0},
    "A100": {"peak_compute_tflops": 19.5, "peak_bandwidth_gb_s": 1555.0},
    "L40S": {"peak_compute_tflops": 18.7, "peak_bandwidth_gb_s": 864.0},
    "RTX 4090": {"peak_compute_tflops": 82.6, "peak_bandwidth_gb_s": 1008.0},
    "RTX 3090": {"peak_compute_tflops": 35.6, "peak_bandwidth_gb_s": 936.0},
}


@dataclass
class KernelContract:
    module: ModuleType
    kernel_type: str
    kernel_fn: Callable[..., Any]
    get_inputs: Callable[..., Dict[str, Any]]
    get_flops: Callable[..., int]
    get_bytes: Callable[..., int]


@dataclass
class BenchmarkConfig:
    quick_shapes: Sequence[Shape]
    full_shapes: Sequence[Shape]
    timing_shape: Shape
    atol: float
    rtol: float
    quick_iterations: int
    full_iterations: int
    default_warmup: int
    input_builder: Callable[[KernelContract, str, Shape], Dict[str, Any]]
    edge_case_builder: Callable[[KernelContract, str], Iterable[Tuple[str, Dict[str, Any]]]]


@dataclass
class StageResult:
    name: str
    status: str
    detail: str
    max_abs_diff: float


@dataclass
class CorrectnessSummary:
    overall: str
    stages: List[StageResult]
    max_abs_diff: float


@dataclass
class BenchmarkMetrics:
    time_ms: float
    reference_time_ms: float
    throughput_tflops: float
    bandwidth_gb_s: float
    pct_peak_compute: Optional[float]
    pct_peak_bandwidth: Optional[float]
    bottleneck: str
    speedup_vs_pytorch: float
    peak_vram_mb: float
    arithmetic_intensity: float
    peak_source: str


def _rms_norm_input_builder(contract: KernelContract, device: str, shape: Shape) -> Dict[str, Any]:
    batch_size, seq_len, hidden_size = shape
    return contract.get_inputs(
        device=device,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
    )


def _rms_norm_edge_cases(contract: KernelContract, device: str) -> Iterable[Tuple[str, Dict[str, Any]]]:
    base = contract.get_inputs(device=device, batch_size=2, seq_len=4, hidden_size=8)
    zeros = _clone_tree(base)
    zeros["x"] = torch.zeros_like(zeros["x"])
    ones = _clone_tree(base)
    ones["x"] = torch.ones_like(ones["x"])
    large = _clone_tree(base)
    large["x"] = torch.full_like(large["x"], 1e3)
    tiny = _clone_tree(base)
    tiny["x"] = torch.full_like(tiny["x"], 1e-6)
    return [
        ("zeros", zeros),
        ("ones", ones),
        ("large_values", large),
        ("tiny_values", tiny),
    ]


def _linear_input_builder(contract: KernelContract, device: str, shape: Shape) -> Dict[str, Any]:
    batch_size, seq_len, hidden_size = shape
    return contract.get_inputs(
        device=device,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        out_features=hidden_size,
    )


def _linear_edge_cases(contract: KernelContract, device: str) -> Iterable[Tuple[str, Dict[str, Any]]]:
    base = contract.get_inputs(device=device, batch_size=2, seq_len=4, hidden_size=8, out_features=8)
    zeros = _clone_tree(base)
    zeros["x"] = torch.zeros_like(zeros["x"])
    ones = _clone_tree(base)
    ones["x"] = torch.ones_like(ones["x"])
    return [
        ("zeros", zeros),
        ("ones", ones),
    ]


def _softmax_input_builder(contract: KernelContract, device: str, shape: Shape) -> Dict[str, Any]:
    batch_size, seq_len, hidden_size = shape
    return contract.get_inputs(
        device=device,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        dim=-1,
    )


def _softmax_edge_cases(contract: KernelContract, device: str) -> Iterable[Tuple[str, Dict[str, Any]]]:
    base = contract.get_inputs(device=device, batch_size=2, seq_len=4, hidden_size=8, dim=-1)
    zeros = _clone_tree(base)
    zeros["x"] = torch.zeros_like(zeros["x"])
    large = _clone_tree(base)
    large["x"] = torch.full_like(large["x"], 1e3)
    return [
        ("zeros", zeros),
        ("large_values", large),
    ]


def _layer_norm_input_builder(contract: KernelContract, device: str, shape: Shape) -> Dict[str, Any]:
    batch_size, seq_len, hidden_size = shape
    return contract.get_inputs(
        device=device,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
    )


def _layer_norm_edge_cases(contract: KernelContract, device: str) -> Iterable[Tuple[str, Dict[str, Any]]]:
    base = contract.get_inputs(device=device, batch_size=2, seq_len=4, hidden_size=8)
    zeros = _clone_tree(base)
    zeros["x"] = torch.zeros_like(zeros["x"])
    ones = _clone_tree(base)
    ones["x"] = torch.ones_like(ones["x"])
    return [
        ("zeros", zeros),
        ("ones", ones),
    ]


KERNEL_CONFIGS: Dict[str, BenchmarkConfig] = {
    "rms_norm": BenchmarkConfig(
        quick_shapes=((2, 4, 8), (4, 8, 16)),
        full_shapes=((2, 4, 8), (8, 32, 64), (16, 64, 128)),
        timing_shape=(16, 64, 128),
        atol=1e-5,
        rtol=1e-4,
        quick_iterations=5,
        full_iterations=25,
        default_warmup=5,
        input_builder=_rms_norm_input_builder,
        edge_case_builder=_rms_norm_edge_cases,
    ),
    "linear": BenchmarkConfig(
        quick_shapes=((2, 4, 8), (4, 8, 16)),
        full_shapes=((2, 4, 8), (8, 32, 64), (16, 64, 128)),
        timing_shape=(16, 64, 128),
        atol=1e-5,
        rtol=1e-4,
        quick_iterations=5,
        full_iterations=25,
        default_warmup=5,
        input_builder=_linear_input_builder,
        edge_case_builder=_linear_edge_cases,
    ),
    "softmax": BenchmarkConfig(
        quick_shapes=((2, 4, 8), (4, 8, 16)),
        full_shapes=((2, 4, 8), (8, 32, 64), (16, 64, 128)),
        timing_shape=(16, 64, 128),
        atol=1e-5,
        rtol=1e-4,
        quick_iterations=5,
        full_iterations=25,
        default_warmup=5,
        input_builder=_softmax_input_builder,
        edge_case_builder=_softmax_edge_cases,
    ),
    "layer_norm": BenchmarkConfig(
        quick_shapes=((2, 4, 8), (4, 8, 16)),
        full_shapes=((2, 4, 8), (8, 32, 64), (16, 64, 128)),
        timing_shape=(16, 64, 128),
        atol=1e-5,
        rtol=1e-4,
        quick_iterations=5,
        full_iterations=25,
        default_warmup=5,
        input_builder=_layer_norm_input_builder,
        edge_case_builder=_layer_norm_edge_cases,
    ),
}


def _clone_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: _clone_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_tree(item) for item in value)
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    return copy.deepcopy(value)


def _flatten_tensors(value: Any) -> List[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        tensors: List[torch.Tensor] = []
        for item in value:
            tensors.extend(_flatten_tensors(item))
        return tensors
    raise TypeError(f"Unsupported kernel output type: {type(value)!r}")


def _format_optional(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _sync_if_needed(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def load_kernel_module(path: str) -> ModuleType:
    kernel_path = Path(path).resolve()
    if not kernel_path.exists():
        raise FileNotFoundError(f"Kernel module not found: {kernel_path}")

    spec = importlib.util.spec_from_file_location("active_kernel", kernel_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create module spec for {kernel_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_kernel_module(module: ModuleType) -> KernelContract:
    required = ["KERNEL_TYPE", "kernel_fn", "get_inputs", "get_flops", "get_bytes"]
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise AttributeError(f"Kernel module missing required exports: {', '.join(missing)}")

    return KernelContract(
        module=module,
        kernel_type=module.KERNEL_TYPE,
        kernel_fn=module.kernel_fn,
        get_inputs=module.get_inputs,
        get_flops=module.get_flops,
        get_bytes=module.get_bytes,
    )


def get_benchmark_config(kernel_type: str) -> BenchmarkConfig:
    if kernel_type not in KERNEL_CONFIGS:
        raise KeyError(
            f"Missing benchmark config for kernel type: {kernel_type}. "
            "Add an entry to KERNEL_CONFIGS in bench.py."
        )
    return KERNEL_CONFIGS[kernel_type]


def call_metadata_fn(fn: Callable[..., int], inputs: Dict[str, Any]) -> int:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return int(fn(inputs))

    if len(signature.parameters) == 0:
        return int(fn())
    return int(fn(inputs))


def infer_peak_targets(device: str) -> Tuple[Optional[float], Optional[float], str]:
    env_compute = os.getenv("AUTO_KERNEL_PEAK_TFLOPS")
    env_bandwidth = os.getenv("AUTO_KERNEL_PEAK_BANDWIDTH_GB_S")
    if env_compute and env_bandwidth:
        return float(env_compute), float(env_bandwidth), "env"

    if device.startswith("cuda") and torch.cuda.is_available():
        device_name = torch.cuda.get_device_name()
        for key, values in KNOWN_GPU_PEAKS.items():
            if key in device_name:
                return values["peak_compute_tflops"], values["peak_bandwidth_gb_s"], key
        return None, None, device_name

    return None, None, "unavailable"


def classify_bottleneck(
    arithmetic_intensity: float,
    peak_compute_tflops: Optional[float],
    peak_bandwidth_gb_s: Optional[float],
) -> str:
    if peak_compute_tflops is not None and peak_bandwidth_gb_s is not None and peak_bandwidth_gb_s > 0:
        ridge_point = peak_compute_tflops * 1000.0 / peak_bandwidth_gb_s
        return "memory_bound" if arithmetic_intensity < ridge_point else "compute_bound"
    return "memory_bound" if arithmetic_intensity < 10 else "compute_bound"


def compare_outputs(
    actual: Any,
    expected: Any,
    atol: float,
    rtol: float,
) -> Tuple[bool, float, str]:
    actual_tensors = _flatten_tensors(actual)
    expected_tensors = _flatten_tensors(expected)
    if len(actual_tensors) != len(expected_tensors):
        return False, float("inf"), "output arity mismatch"

    max_abs_diff = 0.0
    for index, (actual_tensor, expected_tensor) in enumerate(zip(actual_tensors, expected_tensors)):
        if actual_tensor.shape != expected_tensor.shape:
            return False, float("inf"), f"tensor {index} shape mismatch"
        if actual_tensor.dtype != expected_tensor.dtype:
            expected_tensor = expected_tensor.to(actual_tensor.dtype)
        diff = float((actual_tensor - expected_tensor).abs().max().item())
        max_abs_diff = max(max_abs_diff, diff)
        if not torch.allclose(actual_tensor, expected_tensor, atol=atol, rtol=rtol):
            return False, max_abs_diff, f"tensor {index} exceeded tolerance"
    return True, max_abs_diff, "ok"


def run_correctness_pass(
    contract: KernelContract,
    reference_fn: Callable[..., Any],
    inputs: Dict[str, Any],
    atol: float,
    rtol: float,
) -> Tuple[bool, float, str]:
    expected = reference_fn(**_clone_tree(inputs))
    actual = contract.kernel_fn(**_clone_tree(inputs))
    return compare_outputs(actual, expected, atol=atol, rtol=rtol)


def run_correctness_checks(
    contract: KernelContract,
    config: BenchmarkConfig,
    device: str,
    quick: bool,
) -> CorrectnessSummary:
    reference_fn = REFERENCE_FUNCTIONS[contract.kernel_type]
    stages: List[StageResult] = []
    max_abs_diff = 0.0

    smoke_inputs = config.input_builder(contract, device, config.quick_shapes[0])
    passed, diff, detail = run_correctness_pass(contract, reference_fn, smoke_inputs, config.atol, config.rtol)
    max_abs_diff = max(max_abs_diff, diff)
    stages.append(StageResult("smoke", "PASS" if passed else "FAIL", detail, diff))
    if not passed:
        return CorrectnessSummary("FAIL", stages, max_abs_diff)

    shapes = config.quick_shapes if quick else config.full_shapes
    for shape in shapes:
        shape_inputs = config.input_builder(contract, device, shape)
        passed, diff, detail = run_correctness_pass(contract, reference_fn, shape_inputs, config.atol, config.rtol)
        max_abs_diff = max(max_abs_diff, diff)
        if not passed:
            stages.append(StageResult("shape_sweep", "FAIL", f"shape={shape} {detail}", diff))
            return CorrectnessSummary("FAIL", stages, max_abs_diff)
    stages.append(StageResult("shape_sweep", "PASS", f"tested={len(shapes)}", max_abs_diff))

    if quick:
        stages.append(StageResult("numerical_stability", "SKIP", "quick mode", 0.0))
        stages.append(StageResult("determinism", "SKIP", "quick mode", 0.0))
        stages.append(StageResult("edge_cases", "SKIP", "quick mode", 0.0))
        return CorrectnessSummary("PASS", stages, max_abs_diff)

    stability_inputs = config.input_builder(contract, device, config.timing_shape)
    for scale in (1e-3, 1.0, 1e3):
        scaled_inputs = _clone_tree(stability_inputs)
        for key, value in scaled_inputs.items():
            if isinstance(value, torch.Tensor) and torch.is_floating_point(value) and key != "weight":
                scaled_inputs[key] = value * scale
        passed, diff, detail = run_correctness_pass(contract, reference_fn, scaled_inputs, config.atol, config.rtol)
        max_abs_diff = max(max_abs_diff, diff)
        if not passed:
            stages.append(StageResult("numerical_stability", "FAIL", f"scale={scale} {detail}", diff))
            return CorrectnessSummary("FAIL", stages, max_abs_diff)
    stages.append(StageResult("numerical_stability", "PASS", "scales=1e-3,1,1e3", max_abs_diff))

    determinism_inputs = config.input_builder(contract, device, config.timing_shape)
    first = contract.kernel_fn(**_clone_tree(determinism_inputs))
    second = contract.kernel_fn(**_clone_tree(determinism_inputs))
    passed, diff, detail = compare_outputs(first, second, config.atol, config.rtol)
    max_abs_diff = max(max_abs_diff, diff)
    stages.append(StageResult("determinism", "PASS" if passed else "FAIL", detail, diff))
    if not passed:
        return CorrectnessSummary("FAIL", stages, max_abs_diff)

    for case_name, case_inputs in config.edge_case_builder(contract, device):
        passed, diff, detail = run_correctness_pass(contract, reference_fn, case_inputs, config.atol, config.rtol)
        max_abs_diff = max(max_abs_diff, diff)
        if not passed:
            stages.append(StageResult("edge_cases", "FAIL", f"{case_name} {detail}", diff))
            return CorrectnessSummary("FAIL", stages, max_abs_diff)
    stages.append(StageResult("edge_cases", "PASS", "all configured cases", max_abs_diff))
    return CorrectnessSummary("PASS", stages, max_abs_diff)


def benchmark_callable(fn: Callable[[], Any], device: str, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
        _sync_if_needed(device)

    timings = []
    for _ in range(iterations):
        start = perf_counter()
        fn()
        _sync_if_needed(device)
        timings.append(perf_counter() - start)
    return sum(timings) / len(timings)


def benchmark_kernel(
    contract: KernelContract,
    config: BenchmarkConfig,
    device: str,
    iterations: int,
    warmup: int,
) -> BenchmarkMetrics:
    inputs = config.input_builder(contract, device, config.timing_shape)
    reference_fn = REFERENCE_FUNCTIONS[contract.kernel_type]

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    kernel_seconds = benchmark_callable(lambda: contract.kernel_fn(**_clone_tree(inputs)), device, warmup, iterations)
    reference_seconds = benchmark_callable(lambda: reference_fn(**_clone_tree(inputs)), device, warmup, iterations)

    flops = call_metadata_fn(contract.get_flops, inputs)
    bytes_accessed = call_metadata_fn(contract.get_bytes, inputs)
    throughput_tflops = flops / kernel_seconds / 1e12
    bandwidth_gb_s = bytes_accessed / kernel_seconds / 1e9
    arithmetic_intensity = 0.0 if bytes_accessed == 0 else flops / bytes_accessed
    peak_compute_tflops, peak_bandwidth_gb_s, peak_source = infer_peak_targets(device)
    bottleneck = classify_bottleneck(arithmetic_intensity, peak_compute_tflops, peak_bandwidth_gb_s)
    pct_peak_compute = None
    pct_peak_bandwidth = None
    if peak_compute_tflops:
        pct_peak_compute = throughput_tflops / peak_compute_tflops * 100.0
    if peak_bandwidth_gb_s:
        pct_peak_bandwidth = bandwidth_gb_s / peak_bandwidth_gb_s * 100.0
    peak_vram_mb = 0.0
    if device.startswith("cuda") and torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return BenchmarkMetrics(
        time_ms=kernel_seconds * 1000.0,
        reference_time_ms=reference_seconds * 1000.0,
        throughput_tflops=throughput_tflops,
        bandwidth_gb_s=bandwidth_gb_s,
        pct_peak_compute=pct_peak_compute,
        pct_peak_bandwidth=pct_peak_bandwidth,
        bottleneck=bottleneck,
        speedup_vs_pytorch=reference_seconds / kernel_seconds if kernel_seconds > 0 else float("inf"),
        peak_vram_mb=peak_vram_mb,
        arithmetic_intensity=arithmetic_intensity,
        peak_source=peak_source,
    )


def summarize_correctness(summary: CorrectnessSummary) -> str:
    return ",".join(f"{stage.name}:{stage.status}" for stage in summary.stages)


def summarize_throughput(metrics: BenchmarkMetrics) -> str:
    parts = [
        f"throughput_tflops={metrics.throughput_tflops:.4f}",
        f"bandwidth_gb_s={metrics.bandwidth_gb_s:.4f}",
        f"pct_peak_compute={_format_optional(metrics.pct_peak_compute)}",
        f"pct_peak_bandwidth={_format_optional(metrics.pct_peak_bandwidth)}",
        f"bottleneck={metrics.bottleneck}",
        f"speedup_vs_pytorch={metrics.speedup_vs_pytorch:.4f}",
    ]
    return ";".join(parts)


def append_result_row(path: Path, row: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "\t".join(RESULT_COLUMNS)
    if not path.exists():
        path.write_text(f"{header}\n", encoding="utf-8")

    with path.open("a", encoding="utf-8") as handle:
        line = "\t".join(str(row.get(column, "")) for column in RESULT_COLUMNS)
        handle.write(f"{line}\n")


def append_memory_note(path: Path, title: str, lines: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    body = "\n".join(f"- {key}: {value}" for key, value in lines.items())
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n## {title} ({timestamp})\n\n{body}\n")


def default_experiment_id() -> str:
    return datetime.now(timezone.utc).strftime("exp-%Y%m%d-%H%M%S")


def load_active_context() -> Dict[str, object]:
    manifest_path = active_kernel_manifest_path(Path("workspace"))
    if not manifest_path.exists():
        return {}
    return read_json(manifest_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the active kernel.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--hypothesis", default="baseline")
    parser.add_argument("--kept", default="pending", choices=["pending", "yes", "no"])
    parser.add_argument("--results-path", default="results.tsv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    module = load_kernel_module("kernel.py")
    contract = validate_kernel_module(module)
    config = get_benchmark_config(contract.kernel_type)

    iterations = args.iterations if args.iterations is not None else (
        config.quick_iterations if args.quick else config.full_iterations
    )
    warmup = args.warmup if args.warmup is not None else config.default_warmup
    experiment_id = args.experiment_id or default_experiment_id()
    active_context = load_active_context()

    correctness = run_correctness_checks(contract, config, device, quick=args.quick)
    metrics = benchmark_kernel(contract, config, device, iterations=iterations, warmup=warmup)
    correctness_label = correctness.overall
    throughput_summary = summarize_throughput(metrics)

    append_result_row(
        Path(args.results_path),
        {
            "experiment_id": experiment_id,
            "kernel_type": contract.kernel_type,
            "target_name": active_context.get("name", ""),
            "hypothesis": args.hypothesis,
            "correctness": f"{correctness_label}|{summarize_correctness(correctness)}",
            "time_ms": f"{metrics.time_ms:.4f}",
            "throughput": throughput_summary,
            "peak_vram_mb": f"{metrics.peak_vram_mb:.2f}",
            "kept": args.kept,
            "speedup_vs_reference": f"{metrics.speedup_vs_pytorch:.4f}",
        },
    )

    memory_lines = {
        "kernel_type": contract.kernel_type,
        "experiment_id": experiment_id,
        "hypothesis": args.hypothesis,
        "correctness": correctness_label,
        "correctness_details": summarize_correctness(correctness),
        "time_ms": f"{metrics.time_ms:.4f}",
        "reference_time_ms": f"{metrics.reference_time_ms:.4f}",
        "throughput_tflops": f"{metrics.throughput_tflops:.4f}",
        "bandwidth_gb_s": f"{metrics.bandwidth_gb_s:.4f}",
        "pct_peak_compute": _format_optional(metrics.pct_peak_compute),
        "pct_peak_bandwidth": _format_optional(metrics.pct_peak_bandwidth),
        "bottleneck": metrics.bottleneck,
        "speedup_vs_pytorch": f"{metrics.speedup_vs_pytorch:.4f}",
        "peak_vram_mb": f"{metrics.peak_vram_mb:.2f}",
        "kept": args.kept,
        "mode": "quick" if args.quick else "full",
    }
    append_memory_note(Path("MEMORY.md"), f"Experiment {experiment_id}", memory_lines)
    append_memory_note(Path("memory") / f"{contract.kernel_type}.md", f"Experiment {experiment_id}", memory_lines)

    artifact_path = ""
    if args.kept == "yes" and correctness_label == "PASS" and active_context:
        updated_context = persist_optimized_kernel(
            workspace_root=Path("workspace"),
            kernel_file=Path("kernel.py"),
            active_manifest=active_context,
            speedup_vs_reference=metrics.speedup_vs_pytorch,
            experiment_id=experiment_id,
        )
        artifact_path = str(updated_context.get("optimized_kernel_path", ""))

    print(f"kernel={contract.kernel_type}")
    print(f"device={device}")
    print(f"experiment_id={experiment_id}")
    print(f"hypothesis={args.hypothesis}")
    print(f"correctness={correctness_label}")
    print(f"correctness_details={summarize_correctness(correctness)}")
    print(f"time_ms={metrics.time_ms:.4f}")
    print(f"reference_time_ms={metrics.reference_time_ms:.4f}")
    print(f"throughput_tflops={metrics.throughput_tflops:.4f}")
    print(f"bandwidth_gb_s={metrics.bandwidth_gb_s:.4f}")
    print(f"pct_peak_compute={_format_optional(metrics.pct_peak_compute)}")
    print(f"pct_peak_bandwidth={_format_optional(metrics.pct_peak_bandwidth)}")
    print(f"bottleneck={metrics.bottleneck}")
    print(f"speedup_vs_pytorch={metrics.speedup_vs_pytorch:.4f}")
    print(f"peak_vram_mb={metrics.peak_vram_mb:.2f}")
    print(f"peak_source={metrics.peak_source}")
    print(f"kept={args.kept}")
    if artifact_path:
        print(f"optimized_kernel_path={artifact_path}")
    return 0 if correctness_label == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
