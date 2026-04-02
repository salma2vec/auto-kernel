import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Dict, List


def load_workload_module(path: str) -> ModuleType:
    workload_path = Path(path).resolve()
    if not workload_path.exists():
        raise FileNotFoundError(f"Workload module not found: {workload_path}")

    spec = importlib.util.spec_from_file_location("active_workload", workload_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create module spec for {workload_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def workload_name(module: ModuleType, source_path: Path) -> str:
    return getattr(module, "WORKLOAD_NAME", source_path.stem)


def workload_entries(module: ModuleType) -> List[Dict[str, object]]:
    if hasattr(module, "get_workload"):
        entries = module.get_workload()
    elif hasattr(module, "WORKLOAD"):
        entries = module.WORKLOAD
    else:
        raise AttributeError("Workload module must export get_workload() or WORKLOAD")
    if not isinstance(entries, list):
        raise TypeError("Workload entries must be returned as a list")
    return entries
