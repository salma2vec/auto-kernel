import argparse
import json
import importlib.util
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Optional

from auto_kernel.model_adapters import parse_input_shape, trace_model_adapter
from prepare import collect_environment_report, resolve_device
from auto_kernel.workloads import load_workload_module, workload_entries, workload_name


def _load_stdlib_profile_module():
    stdlib_path = Path(sysconfig.get_path("stdlib")) / "profile.py"
    spec = importlib.util.spec_from_file_location("_stdlib_profile", stdlib_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load stdlib profile module from {stdlib_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_STDLIB_PROFILE = _load_stdlib_profile_module()
run = _STDLIB_PROFILE.run
runctx = _STDLIB_PROFILE.runctx
Profile = _STDLIB_PROFILE.Profile


@dataclass
class ProfileClassification:
    arithmetic_intensity: float
    time_ms: float
    throughput_tflops: float
    bandwidth_gb_s: float
    pct_peak_compute: Optional[float]
    pct_peak_bandwidth: Optional[float]
    bottleneck: str
    ridge_point: Optional[float]
    peak_source: str


def _format_optional(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def profile_kernel(contract, device: str, iterations: int, warmup: int) -> ProfileClassification:
    from bench import (
        _clone_tree,
        _sync_if_needed,
        call_metadata_fn,
        classify_bottleneck,
        get_benchmark_config,
        infer_peak_targets,
    )

    config = get_benchmark_config(contract.kernel_type)
    inputs = config.input_builder(contract, device, config.timing_shape)

    def run_once() -> None:
        contract.kernel_fn(**_clone_tree(inputs))
        _sync_if_needed(device)

    for _ in range(warmup):
        run_once()

    start = perf_counter()
    for _ in range(iterations):
        run_once()
    elapsed_seconds = (perf_counter() - start) / iterations

    flops = call_metadata_fn(contract.get_flops, inputs)
    bytes_accessed = call_metadata_fn(contract.get_bytes, inputs)
    arithmetic_intensity = 0.0 if bytes_accessed == 0 else flops / bytes_accessed
    throughput_tflops = flops / elapsed_seconds / 1e12
    bandwidth_gb_s = bytes_accessed / elapsed_seconds / 1e9
    peak_compute_tflops, peak_bandwidth_gb_s, peak_source = infer_peak_targets(device)
    pct_peak_compute = None
    pct_peak_bandwidth = None
    ridge_point = None
    if peak_compute_tflops:
        pct_peak_compute = throughput_tflops / peak_compute_tflops * 100.0
    if peak_bandwidth_gb_s:
        pct_peak_bandwidth = bandwidth_gb_s / peak_bandwidth_gb_s * 100.0
    if peak_compute_tflops and peak_bandwidth_gb_s:
        ridge_point = peak_compute_tflops * 1000.0 / peak_bandwidth_gb_s
    bottleneck = classify_bottleneck(arithmetic_intensity, peak_compute_tflops, peak_bandwidth_gb_s)
    return ProfileClassification(
        arithmetic_intensity=arithmetic_intensity,
        time_ms=elapsed_seconds * 1000.0,
        throughput_tflops=throughput_tflops,
        bandwidth_gb_s=bandwidth_gb_s,
        pct_peak_compute=pct_peak_compute,
        pct_peak_bandwidth=pct_peak_bandwidth,
        bottleneck=bottleneck,
        ridge_point=ridge_point,
        peak_source=peak_source,
    )


def profile_workload_entry(
    entry: dict,
    source_dir: Path,
    device: str,
    iterations: int,
    warmup: int,
) -> dict:
    if "baseline_time_ms" in entry:
        return {
            "name": entry["name"],
            "kernel_type": entry["kernel_type"],
            "kernel_path": str(entry["kernel_path"]),
            "baseline_time_ms": float(entry["baseline_time_ms"]),
            "input_kwargs": dict(entry.get("input_kwargs", {})),
        }

    kernel_path = Path(str(entry["kernel_path"]))
    if not kernel_path.is_absolute():
        kernel_path = (source_dir / kernel_path).resolve()

    from bench import _clone_tree, _sync_if_needed, load_kernel_module, validate_kernel_module

    module = load_kernel_module(str(kernel_path))
    contract = validate_kernel_module(module)

    input_kwargs = dict(entry.get("input_kwargs", {}))
    inputs = contract.get_inputs(device=device, **input_kwargs)

    def run_once() -> None:
        contract.kernel_fn(**_clone_tree(inputs))
        _sync_if_needed(device)

    for _ in range(warmup):
        run_once()

    start = perf_counter()
    for _ in range(iterations):
        run_once()
    elapsed_seconds = (perf_counter() - start) / iterations

    return {
        "name": entry["name"],
        "kernel_type": entry.get("kernel_type", contract.kernel_type),
        "kernel_path": str(kernel_path),
        "baseline_time_ms": elapsed_seconds * 1000.0,
        "input_kwargs": input_kwargs,
    }


def build_workload_manifest(
    workload_path: Path,
    device: str,
    iterations: int,
    warmup: int,
) -> dict:
    module = load_workload_module(str(workload_path))
    entries = workload_entries(module)
    workload_records = [
        profile_workload_entry(entry, workload_path.parent, device, iterations, warmup)
        for entry in entries
    ]
    ordered = sorted(workload_records, key=lambda record: record["baseline_time_ms"], reverse=True)
    total_time_ms = sum(record["baseline_time_ms"] for record in ordered)
    bottlenecks = []
    for index, record in enumerate(ordered, start=1):
        time_share = 0.0 if total_time_ms <= 0 else record["baseline_time_ms"] / total_time_ms
        bottlenecks.append(
            {
                "rank": index,
                "name": record["name"],
                "kernel_type": record["kernel_type"],
                "kernel_path": record["kernel_path"],
                "baseline_time_ms": record["baseline_time_ms"],
                "time_share": time_share,
                "input_kwargs": record.get("input_kwargs", {}),
            }
        )

    return {
        "workload_name": workload_name(module, workload_path),
        "device": device,
        "total_time_ms": total_time_ms,
        "bottlenecks": bottlenecks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile the active kernel.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--workload", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--class-name", default=None)
    parser.add_argument("--input-shape", default=None)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--manifest-out", default=None)
    parser.add_argument("--name", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    if args.model:
        model_manifest = trace_model_adapter(
            model_path=Path(args.model).resolve(),
            device=device,
            input_shape=parse_input_shape(args.input_shape),
            dtype=args.dtype,
            iterations=args.iterations,
            warmup=args.warmup,
            class_name=args.class_name,
        )
        print(f"workload_name={model_manifest['workload_name']}")
        print(f"device={device}")
        print("mode=model")
        print(f"total_time_ms={model_manifest['total_time_ms']:.4f}")
        print(f"bottleneck_count={len(model_manifest['bottlenecks'])}")
        if model_manifest["bottlenecks"]:
            print(f"top_kernel_type={model_manifest['bottlenecks'][0]['kernel_type']}")
            print(f"top_target_name={model_manifest['bottlenecks'][0]['name']}")
        if args.manifest_out:
            manifest_path = Path(args.manifest_out)
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(model_manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"manifest_path={manifest_path}")
        return 0

    if args.workload:
        workload_manifest = build_workload_manifest(
            workload_path=Path(args.workload).resolve(),
            device=device,
            iterations=args.iterations,
            warmup=args.warmup,
        )
        print(f"workload_name={workload_manifest['workload_name']}")
        print(f"device={device}")
        print(f"mode=workload")
        print(f"total_time_ms={workload_manifest['total_time_ms']:.4f}")
        print(f"bottleneck_count={len(workload_manifest['bottlenecks'])}")
        if workload_manifest["bottlenecks"]:
            print(f"top_kernel_type={workload_manifest['bottlenecks'][0]['kernel_type']}")
            print(f"top_target_name={workload_manifest['bottlenecks'][0]['name']}")
        if args.manifest_out:
            manifest_path = Path(args.manifest_out)
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(workload_manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"manifest_path={manifest_path}")
        return 0

    from bench import load_kernel_module, validate_kernel_module

    module = load_kernel_module("kernel.py")
    contract = validate_kernel_module(module)
    report = profile_kernel(contract, device=device, iterations=args.iterations, warmup=args.warmup)
    environment = collect_environment_report()
    mode = "ncu_assisted" if environment["ncu_available"] and device.startswith("cuda") else "analytical"
    print(f"kernel={contract.kernel_type}")
    print(f"device={device}")
    print(f"mode={mode}")
    print(f"arithmetic_intensity={report.arithmetic_intensity:.4f}")
    print(f"throughput_tflops={report.throughput_tflops:.4f}")
    print(f"bandwidth_gb_s={report.bandwidth_gb_s:.4f}")
    print(f"pct_peak_compute={_format_optional(report.pct_peak_compute)}")
    print(f"pct_peak_bandwidth={_format_optional(report.pct_peak_bandwidth)}")
    print(f"ridge_point={_format_optional(report.ridge_point)}")
    print(f"bottleneck={report.bottleneck}")
    print(f"peak_source={report.peak_source}")
    if args.manifest_out:
        manifest = {
            "workload_name": args.name or contract.kernel_type,
            "device": device,
            "total_time_ms": report.time_ms,
            "bottlenecks": [
                {
                    "rank": 1,
                    "name": args.name or contract.kernel_type,
                    "kernel_type": contract.kernel_type,
                    "baseline_time_ms": report.time_ms,
                    "time_share": 1.0,
                    "kernel_path": "kernel.py",
                }
            ],
        }
        manifest_path = Path(args.manifest_out)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"manifest_path={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
