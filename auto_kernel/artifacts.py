import re
import math
from pathlib import Path
from typing import Any, Dict, Mapping

from auto_kernel.workspace import active_kernel_manifest_path, ensure_workspace, write_bytes_atomic, write_json


def sanitize_target_name(name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return normalized or "kernel"


def optimized_kernel_path(workspace_root: Path, kernel_type: str, target_name: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_]+", kernel_type):
        raise ValueError("Invalid kernel_type")
    root = ensure_workspace(workspace_root)
    filename = f"{kernel_type}--{sanitize_target_name(target_name)}.py"
    return root / "optimized" / filename


def resolve_optimized_kernel(
    workspace_root: Path,
    kernel_type: str,
    target_name: str,
) -> Path:
    return optimized_kernel_path(workspace_root, kernel_type, target_name)


def persist_optimized_kernel(
    workspace_root: Path,
    kernel_file: Path,
    active_manifest: Mapping[str, Any],
    speedup_vs_reference: float,
    experiment_id: str,
) -> Dict[str, Any]:
    if not active_manifest.get("kernel_type") or not active_manifest.get("name"):
        return dict(active_manifest)

    if not math.isfinite(speedup_vs_reference) or speedup_vs_reference <= 0:
        raise ValueError("speedup must be finite and positive")
    if speedup_vs_reference < float(active_manifest.get("best_speedup_vs_reference", 0)):
        return dict(active_manifest)
    workspace_root = ensure_workspace(workspace_root)
    destination = optimized_kernel_path(
        workspace_root,
        str(active_manifest["kernel_type"]),
        str(active_manifest["name"]),
    )
    write_bytes_atomic(destination, kernel_file.read_bytes(), mode=kernel_file.stat().st_mode & 0o777)

    updated = dict(active_manifest)
    updated["optimized_kernel_path"] = str(destination)
    updated["best_speedup_vs_reference"] = float(speedup_vs_reference)
    updated["last_experiment_id"] = experiment_id
    write_json(active_kernel_manifest_path(workspace_root), updated)
    return updated
