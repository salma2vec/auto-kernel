import argparse
import importlib.util
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from types import ModuleType
from typing import Any, Callable, Dict

import torch

from prepare import collect_environment_report, resolve_device


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
class KernelContract:
    module: ModuleType
    kernel_type: str
    kernel_fn: Callable[..., Any]
    get_inputs: Callable[..., Dict[str, Any]]
    get_flops: Callable[[Dict[str, Any]], int]
    get_bytes: Callable[[Dict[str, Any]], int]


@dataclass
class ProfileClassification:
    arithmetic_intensity: float
    throughput_flops: float
    throughput_bytes: float
    bound: str


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


def classify_profile(flops: int, bytes_accessed: int, elapsed_seconds: float) -> ProfileClassification:
    arithmetic_intensity = 0.0 if bytes_accessed == 0 else flops / bytes_accessed
    throughput_flops = 0.0 if elapsed_seconds == 0 else flops / elapsed_seconds
    throughput_bytes = 0.0 if elapsed_seconds == 0 else bytes_accessed / elapsed_seconds
    bound = "memory-bound" if arithmetic_intensity < 10 else "compute-bound"
    return ProfileClassification(
        arithmetic_intensity=arithmetic_intensity,
        throughput_flops=throughput_flops,
        throughput_bytes=throughput_bytes,
        bound=bound,
    )


def profile_kernel(
    contract: KernelContract,
    device: str,
    iterations: int = 5,
    warmup: int = 2,
) -> ProfileClassification:
    inputs = contract.get_inputs(device=device)

    def run_once() -> None:
        contract.kernel_fn(**inputs)
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    for _ in range(warmup):
        run_once()

    start = perf_counter()
    for _ in range(iterations):
        run_once()
    elapsed = (perf_counter() - start) / iterations

    return classify_profile(
        flops=contract.get_flops(inputs),
        bytes_accessed=contract.get_bytes(inputs),
        elapsed_seconds=elapsed,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile the active kernel analytically.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    module = load_kernel_module("kernel.py")
    contract = validate_kernel_module(module)
    report = profile_kernel(contract, device=device, iterations=args.iterations, warmup=args.warmup)
    environment = collect_environment_report()
    mode = "ncu" if environment["ncu_available"] and device.startswith("cuda") else "analytical"
    print(
        " ".join(
            [
                f"kernel={contract.kernel_type}",
                f"device={device}",
                f"mode={mode}",
                f"arithmetic_intensity={report.arithmetic_intensity:.4f}",
                f"throughput_flops={report.throughput_flops:.4f}",
                f"throughput_bytes={report.throughput_bytes:.4f}",
                f"bound={report.bound}",
            ]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
