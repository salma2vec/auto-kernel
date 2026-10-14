import json
import tempfile
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
    write_bytes_atomic(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def write_bytes_atomic(path: Path, data: bytes, mode: int = 0o600) -> None:
    """Replace a file without following an existing destination/temp symlink.

    The parent directory is trusted: this is not isolation from an attacker who
    can rename directories or modify the workspace concurrently.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".write-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
        temporary.chmod(mode)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
