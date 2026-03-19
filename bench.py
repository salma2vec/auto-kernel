import argparse
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from types import ModuleType
from typing import Any, Callable, Dict, Mapping

import torch

from reference import REFERENCE_FUNCTIONS


@dataclass
class KernelContract:
    module: ModuleType
    kernel_type: str
    kernel_fn: Callable[..., Any]
    get_inputs: Callable[..., Dict[str, Any]]
    get_flops: Callable[[Dict[str, Any]], int]
    get_bytes: Callable[[Dict[str, Any]], int]


@dataclass
class BenchmarkResult:
    kernel_type: str
    device: str
    correct: bool
    mean_seconds: float
    max_abs_diff: float


DEFAULT_COLUMNS = [
    "timestamp",
    "kernel_type",
    "device",
    "status",
    "correct",
    "mean_seconds",
    "notes",
]


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


def append_result_row(path: Path, row: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "\t".join(DEFAULT_COLUMNS)
    if not path.exists():
        path.write_text(f"{header}\n", encoding="utf-8")

    normalized = {column: row.get(column, "") for column in DEFAULT_COLUMNS}
    if not normalized["timestamp"]:
        from datetime import datetime, timezone

        normalized["timestamp"] = datetime.now(timezone.utc).isoformat()

    line = "\t".join(str(normalized[column]) for column in DEFAULT_COLUMNS)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{line}\n")


def append_memory_note(path: Path, title: str, lines: Mapping[str, object]) -> None:
    from datetime import datetime, timezone

    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    body = "\n".join(f"- {key}: {value}" for key, value in lines.items())
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n## {title} ({timestamp})\n\n{body}\n")


def benchmark_kernel(
    contract: KernelContract,
    device: str,
    iterations: int = 10,
    warmup: int = 3,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> BenchmarkResult:
    if contract.kernel_type not in REFERENCE_FUNCTIONS:
        raise KeyError(f"Missing reference function for kernel type: {contract.kernel_type}")

    inputs = contract.get_inputs(device=device)
    reference_fn = REFERENCE_FUNCTIONS[contract.kernel_type]
    expected = reference_fn(**inputs)
    actual = contract.kernel_fn(**inputs)

    correct = torch.allclose(actual, expected, atol=atol, rtol=rtol)
    max_abs_diff = float((actual - expected).abs().max().item())

    def run_once() -> None:
        contract.kernel_fn(**inputs)
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    for _ in range(warmup):
        run_once()

    timings = []
    for _ in range(iterations):
        start = perf_counter()
        run_once()
        timings.append(perf_counter() - start)

    return BenchmarkResult(
        kernel_type=contract.kernel_type,
        device=device,
        correct=correct,
        mean_seconds=sum(timings) / len(timings),
        max_abs_diff=max_abs_diff,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the active kernel.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    module = load_kernel_module("kernel.py")
    contract = validate_kernel_module(module)
    result = benchmark_kernel(
        contract,
        device=args.device,
        iterations=args.iterations,
        warmup=args.warmup,
    )

    status = "accepted" if result.correct else "rejected"
    append_result_row(
        Path("results.tsv"),
        {
            "kernel_type": result.kernel_type,
            "device": result.device,
            "status": status,
            "correct": result.correct,
            "mean_seconds": f"{result.mean_seconds:.8f}",
            "notes": f"max_abs_diff={result.max_abs_diff:.6e}",
        },
    )
    append_memory_note(
        Path("MEMORY.md"),
        "Benchmark",
        {
            "kernel_type": result.kernel_type,
            "device": result.device,
            "status": status,
            "mean_seconds": f"{result.mean_seconds:.8f}",
        },
    )
    append_memory_note(
        Path("memory") / f"{result.kernel_type}.md",
        "Benchmark",
        {
            "device": result.device,
            "status": status,
            "mean_seconds": f"{result.mean_seconds:.8f}",
        },
    )

    print(
        f"kernel={result.kernel_type} device={result.device} "
        f"correct={result.correct} mean_seconds={result.mean_seconds:.8f}"
    )
    return 0 if result.correct else 1


if __name__ == "__main__":
    raise SystemExit(main())
