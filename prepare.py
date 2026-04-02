import importlib.util
import shutil
import sys
from typing import Optional


def _module_available(name: str) -> bool:
    if name in sys.modules:
        return True
    return importlib.util.find_spec(name) is not None


def collect_environment_report():
    torch_available = _module_available("torch")
    cuda_available = False
    if torch_available:
        import torch

        cuda_available = bool(torch.cuda.is_available())

    return {
        "python_version": sys.version.split()[0],
        "torch_available": torch_available,
        "cuda_available": cuda_available,
        "triton_available": _module_available("triton"),
        "ncu_available": shutil.which("ncu") is not None,
        "nvcc_available": shutil.which("nvcc") is not None,
    }


def resolve_device(requested_device: Optional[str] = None) -> str:
    normalized = (requested_device or "auto").strip().lower()
    if normalized == "auto":
        report = collect_environment_report()
        return "cuda" if report["cuda_available"] else "cpu"

    if normalized.startswith("cuda"):
        report = collect_environment_report()
        if not report["torch_available"]:
            raise RuntimeError("CUDA was requested, but PyTorch is not installed in this environment.")
        if not report["cuda_available"]:
            raise RuntimeError("CUDA was requested, but no CUDA-capable device is available.")

    return normalized


def main() -> int:
    report = collect_environment_report()
    for key, value in report.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
