import shutil
import subprocess
import sys
import tempfile
import unittest
import os
import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


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
    shutil.copytree(
        REPO_ROOT / "auto_kernel",
        destination / "auto_kernel",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )


class EndToEndWorkflowTests(unittest.TestCase):
    @unittest.skipUnless(TORCH_AVAILABLE, "torch is required for end-to-end workflow tests")
    def test_prepare_bench_and_profile_run_successfully(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            _copy_workflow_fixture(workspace)
            tmp_path = workspace / ".tmp"
            pycache_path = workspace / ".pycache"
            tmp_path.mkdir(parents=True, exist_ok=True)
            pycache_path.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env["TMPDIR"] = str(tmp_path)
            env["PYTHONPYCACHEPREFIX"] = str(pycache_path)
            env["PYTHONDONTWRITEBYTECODE"] = "1"

            prepare = subprocess.run(
                [sys.executable, "prepare.py"],
                cwd=workspace,
                env=env,
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
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(bench.returncode, 0, bench.stdout + bench.stderr)
            self.assertIn("kernel=rms_norm", bench.stdout)
            self.assertIn("correctness=PASS", bench.stdout)

            profile = subprocess.run(
                [sys.executable, "profile.py", "--iterations", "1", "--warmup", "0"],
                cwd=workspace,
                env=env,
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
