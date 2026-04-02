import json
from pathlib import Path
from typing import Any, Dict


def ensure_workspace(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "optimized").mkdir(parents=True, exist_ok=True)
    return root


def profile_manifest_path(root: Path) -> Path:
    return root / "profile.json"


def active_kernel_manifest_path(root: Path) -> Path:
    return root / "active_kernel.json"


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
