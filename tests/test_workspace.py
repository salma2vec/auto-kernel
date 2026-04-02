import json
import shutil
import unittest
from pathlib import Path

from auto_kernel.workspace import (
    active_kernel_manifest_path,
    ensure_workspace,
    profile_manifest_path,
    read_json,
    write_json,
)


class WorkspaceTests(unittest.TestCase):
    def _scratch_dir(self, name: str) -> Path:
        base = Path(__file__).resolve().parent.parent / ".tmp" / "test_workspaces"
        base.mkdir(parents=True, exist_ok=True)
        root = base / name
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda root=root: shutil.rmtree(root, ignore_errors=True))
        return root

    def test_ensure_workspace_creates_expected_layout(self) -> None:
        root = self._scratch_dir("layout") / "workspace"
        resolved = ensure_workspace(root)

        self.assertEqual(resolved, root)
        self.assertTrue(root.exists())
        self.assertTrue((root / "optimized").exists())
        self.assertEqual(profile_manifest_path(root), root / "profile.json")
        self.assertEqual(active_kernel_manifest_path(root), root / "active_kernel.json")

    def test_write_and_read_json_round_trip(self) -> None:
        path = self._scratch_dir("json") / "workspace" / "profile.json"
        payload = {
            "workload_name": "toy",
            "bottlenecks": [
                {"rank": 1, "kernel_type": "rms_norm", "time_share": 0.7},
            ],
        }

        write_json(path, payload)

        self.assertEqual(read_json(path), payload)
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), payload)


if __name__ == "__main__":
    unittest.main()
