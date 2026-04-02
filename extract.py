import argparse
import shutil
from pathlib import Path
from typing import Dict, List

from auto_kernel.workspace import active_kernel_manifest_path, ensure_workspace, read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select a profiled kernel and make it the active target.")
    parser.add_argument("--profile-path", default="workspace/profile.json")
    parser.add_argument("--workspace-root", default="workspace")
    parser.add_argument("--kernel-file", default="kernel.py")
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--kernel-type", default=None)
    return parser.parse_args()


def select_bottleneck(records: List[Dict[str, object]], rank: int, kernel_type: str) -> Dict[str, object]:
    if kernel_type:
        for record in records:
            if str(record.get("kernel_type")) == kernel_type:
                return record
        raise KeyError(f"Unable to find kernel type in profile manifest: {kernel_type}")

    ordered = sorted(records, key=lambda record: int(record.get("rank", 0) or 0))
    for record in ordered:
        if int(record.get("rank", 0) or 0) == rank:
            return record
    if rank <= len(ordered):
        return ordered[rank - 1]
    raise IndexError(f"Unable to find ranked bottleneck #{rank}")


def resolve_source_kernel(profile_path: Path, record: Dict[str, object]) -> Path:
    kernel_path = Path(str(record["kernel_path"]))
    if kernel_path.is_absolute():
        return kernel_path
    return (profile_path.parent / kernel_path).resolve()


def main() -> int:
    args = parse_args()
    profile_path = Path(args.profile_path).resolve()
    workspace_root = ensure_workspace(Path(args.workspace_root).resolve())
    profile = read_json(profile_path)
    record = select_bottleneck(profile.get("bottlenecks", []), args.rank, args.kernel_type)

    source_kernel = resolve_source_kernel(profile_path, record)
    destination_kernel = Path(args.kernel_file).resolve()
    shutil.copy2(source_kernel, destination_kernel)

    active_manifest = {
        "workload_name": profile.get("workload_name", "unknown"),
        "name": record.get("name", record.get("kernel_type")),
        "kernel_type": record["kernel_type"],
        "kernel_path": str(source_kernel),
        "rank": int(record.get("rank", 0) or 0),
        "time_share": float(record.get("time_share", 0.0) or 0.0),
    }
    write_json(active_kernel_manifest_path(workspace_root), active_manifest)

    print(f"selected_kernel_type={active_manifest['kernel_type']}")
    print(f"selected_target_name={active_manifest['name']}")
    print(f"workspace_root={workspace_root}")
    print(f"kernel_file={destination_kernel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
