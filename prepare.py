import importlib.util
import shutil
import sys


def _module_available(name: str) -> bool:
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


def main() -> int:
    report = collect_environment_report()
    for key, value in report.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
