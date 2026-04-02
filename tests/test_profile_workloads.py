import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_profile(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    tmp_path = cwd / ".tmp"
    pycache_path = cwd / ".pycache"
    tmp_path.mkdir(parents=True, exist_ok=True)
    pycache_path.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["TMPDIR"] = str(tmp_path)
    env["PYTHONPYCACHEPREFIX"] = str(pycache_path)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(cwd) if not existing_pythonpath else f"{cwd}{os.pathsep}{existing_pythonpath}"
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "profile.py"), *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


class WorkloadProfileTests(unittest.TestCase):
    def _scratch_dir(self, name: str) -> Path:
        base = REPO_ROOT / ".tmp" / "test_workspaces"
        base.mkdir(parents=True, exist_ok=True)
        root = base / name
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda root=root: shutil.rmtree(root, ignore_errors=True))
        return root

    def test_profile_workload_writes_ranked_manifest(self) -> None:
        workspace = self._scratch_dir("profile-workload")
        (workspace / "torch.py").write_text(
            "\n".join(
                [
                    "class Tensor:",
                    "    pass",
                    "",
                    "class _Cuda:",
                    "    @staticmethod",
                    "    def is_available():",
                    "        return False",
                    "",
                    "cuda = _Cuda()",
                    "Tensor = Tensor",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (workspace / "toy_workload.py").write_text(
            "\n".join(
                [
                    'WORKLOAD_NAME = "toy-workload"',
                    "",
                    "def get_workload():",
                    "    return [",
                    "        {",
                    '            "name": "encoder.rms_norm",',
                    '            "kernel_type": "rms_norm",',
                    '            "kernel_path": "kernels/rms_norm.py",',
                    '            "baseline_time_ms": 7.0,',
                    "        },",
                    "        {",
                    '            "name": "decoder.softmax",',
                    '            "kernel_type": "softmax",',
                    '            "kernel_path": "kernels/rms_norm.py",',
                    '            "baseline_time_ms": 3.0,',
                    "        },",
                    "    ]",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        shutil.copytree(
            REPO_ROOT / "kernels",
            workspace / "kernels",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        manifest_path = workspace / "workspace" / "profile.json"

        result = _run_profile(
            workspace,
            "--workload",
            "toy_workload.py",
            "--manifest-out",
            str(manifest_path),
        )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("workload_name=toy-workload", result.stdout)
        self.assertIn("top_kernel_type=rms_norm", result.stdout)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["workload_name"], "toy-workload")
        self.assertEqual(manifest["bottlenecks"][0]["name"], "encoder.rms_norm")
        self.assertEqual(manifest["bottlenecks"][0]["rank"], 1)
        self.assertAlmostEqual(manifest["bottlenecks"][0]["time_share"], 0.7, places=6)


if __name__ == "__main__":
    unittest.main()
