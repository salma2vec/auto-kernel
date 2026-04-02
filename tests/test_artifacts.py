import shutil
import unittest
from pathlib import Path

from auto_kernel.artifacts import optimized_kernel_path, persist_optimized_kernel
from auto_kernel.workspace import read_json


class ArtifactTests(unittest.TestCase):
    def _scratch_dir(self, name: str) -> Path:
        root = Path(__file__).resolve().parent.parent / ".tmp_test_workspaces" / name
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        return root

    def test_optimized_kernel_path_is_stable_and_named_by_target(self) -> None:
        workspace = self._scratch_dir("artifact-paths") / "workspace"
        path = optimized_kernel_path(workspace, "rms_norm", "encoder.block0.rms_norm")

        self.assertEqual(path, workspace / "optimized" / "rms_norm--encoder_block0_rms_norm.py")

    def test_persist_optimized_kernel_copies_kernel_and_updates_manifest(self) -> None:
        root = self._scratch_dir("artifact-persist")
        workspace = root / "workspace"
        kernel_file = root / "kernel.py"
        kernel_file.write_text("KERNEL_TYPE = 'rms_norm'\n", encoding="utf-8")
        manifest_path = workspace / "active_kernel.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            (
                "{\n"
                '  "workload_name": "toy",\n'
                '  "name": "encoder.block0.rms_norm",\n'
                '  "kernel_type": "rms_norm"\n'
                "}\n"
            ),
            encoding="utf-8",
        )

        updated = persist_optimized_kernel(
            workspace_root=workspace,
            kernel_file=kernel_file,
            active_manifest=read_json(manifest_path),
            speedup_vs_reference=1.75,
            experiment_id="exp-123",
        )

        optimized_path = workspace / "optimized" / "rms_norm--encoder_block0_rms_norm.py"
        self.assertTrue(optimized_path.exists())
        self.assertEqual(optimized_path.read_text(encoding="utf-8"), kernel_file.read_text(encoding="utf-8"))
        self.assertEqual(updated["optimized_kernel_path"], str(optimized_path))
        self.assertEqual(updated["best_speedup_vs_reference"], 1.75)
        self.assertEqual(updated["last_experiment_id"], "exp-123")
        self.assertEqual(read_json(manifest_path)["optimized_kernel_path"], str(optimized_path))


if __name__ == "__main__":
    unittest.main()
