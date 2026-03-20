import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _copy_workflow_fixture(destination: Path) -> None:
    files_to_copy = [
        "prepare.py",
        "bench.py",
        "profile.py",
        "kernel.py",
        "reference.py",
        "pyproject.toml",
    ]
    for relative_path in files_to_copy:
        shutil.copy2(REPO_ROOT / relative_path, destination / relative_path)

    shutil.copytree(REPO_ROOT / "kernels", destination / "kernels")


class EndToEndWorkflowTests(unittest.TestCase):
    def test_prepare_bench_and_profile_run_successfully(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            _copy_workflow_fixture(workspace)

            prepare = subprocess.run(
                [sys.executable, "prepare.py"],
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(prepare.returncode, 0, prepare.stdout + prepare.stderr)
            self.assertIn("torch_available=", prepare.stdout)

            bench = subprocess.run(
                [sys.executable, "bench.py", "--iterations", "1", "--warmup", "0"],
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(bench.returncode, 0, bench.stdout + bench.stderr)
            self.assertIn("kernel=rms_norm", bench.stdout)
            self.assertIn("correct=True", bench.stdout)

            profile = subprocess.run(
                [sys.executable, "profile.py", "--iterations", "1", "--warmup", "0"],
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(profile.returncode, 0, profile.stdout + profile.stderr)
            self.assertIn("kernel=rms_norm", profile.stdout)
            self.assertIn("mode=", profile.stdout)


if __name__ == "__main__":
    unittest.main()
