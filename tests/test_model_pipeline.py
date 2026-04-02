import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_script(script_name: str, cwd: Path, *args: str) -> subprocess.CompletedProcess:
    tmp_path = REPO_ROOT / ".tmp"
    pycache_path = REPO_ROOT / ".pycache_local"
    tmp_path.mkdir(parents=True, exist_ok=True)
    pycache_path.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["TMPDIR"] = str(tmp_path)
    env["PYTHONPYCACHEPREFIX"] = str(pycache_path)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(cwd) if not existing_pythonpath else f"{cwd}{os.pathsep}{existing_pythonpath}"
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / script_name), *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


class ModelPipelineTests(unittest.TestCase):
    def _scratch_dir(self, name: str) -> Path:
        base = REPO_ROOT / ".tmp" / "test_workspaces"
        base.mkdir(parents=True, exist_ok=True)
        root = base / name
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda root=root: shutil.rmtree(root, ignore_errors=True))
        return root

    def test_profile_model_adapter_writes_manifest(self) -> None:
        workspace = self._scratch_dir("profile-model-adapter")
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
        (workspace / "toy_model_adapter.py").write_text(
            "\n".join(
                [
                    'WORKLOAD_NAME = "toy-model-adapter"',
                    "",
                    "def trace_model(device='cpu', input_shape=None, dtype='float32', iterations=1, warmup=0, class_name=None):",
                    "    return {",
                    '        "workload_name": "toy-model-adapter",',
                    '        "device": device,',
                    '        "bottlenecks": [',
                    "            {",
                    '                "name": "encoder.rms_norm",',
                    '                "kernel_type": "rms_norm",',
                    '                "kernel_path": "kernels/rms_norm.py",',
                    '                "baseline_time_ms": 8.0,',
                    "            },",
                    "            {",
                    '                "name": "decoder.rms_norm",',
                    '                "kernel_type": "rms_norm",',
                    '                "kernel_path": "kernels/rms_norm.py",',
                    '                "baseline_time_ms": 2.0,',
                    "            },",
                    "        ],",
                    "    }",
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

        result = _run_script(
            "profile.py",
            workspace,
            "--model",
            str((workspace / "toy_model_adapter.py").resolve()),
            "--manifest-out",
            str(manifest_path),
        )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("mode=model", result.stdout)
        self.assertIn("workload_name=toy-model-adapter", result.stdout)
        self.assertIn("top_kernel_type=rms_norm", result.stdout)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["workload_name"], "toy-model-adapter")
        self.assertEqual(manifest["bottlenecks"][0]["name"], "encoder.rms_norm")
        self.assertAlmostEqual(manifest["bottlenecks"][0]["time_share"], 0.8, places=6)

    def test_verify_model_adapter_reports_measured_reintegration(self) -> None:
        workspace = self._scratch_dir("verify-model-adapter")
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
        (workspace / "toy_model_adapter.py").write_text(
            "\n".join(
                [
                    'WORKLOAD_NAME = "toy-model-adapter"',
                    "",
                    "def trace_model(device='cpu', input_shape=None, dtype='float32', iterations=1, warmup=0, class_name=None):",
                    "    return {",
                    '        "workload_name": "toy-model-adapter",',
                    '        "device": device,',
                    '        "bottlenecks": [',
                    "            {",
                    '                "name": "encoder.rms_norm",',
                    '                "kernel_type": "rms_norm",',
                    '                "kernel_path": "kernels/rms_norm.py",',
                    '                "baseline_time_ms": 8.0,',
                    "            },",
                    "        ],",
                    "    }",
                    "",
                    "def verify_model(optimized_kernels, profile, device='cpu', iterations=1, class_name=None):",
                    "    optimized = 'encoder.rms_norm' in optimized_kernels",
                    "    return {",
                    '        "correct": True,',
                    '        "max_abs_diff": 0.0,',
                    '        "baseline_total_ms": 8.0,',
                    '        "optimized_total_ms": 4.0 if optimized else 8.0,',
                    "    }",
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
        profile_path = workspace / "profile.json"
        profile_path.write_text(
            json.dumps(
                {
                    "workload_name": "toy-model-adapter",
                    "bottlenecks": [
                        {
                            "rank": 1,
                            "name": "encoder.rms_norm",
                            "kernel_type": "rms_norm",
                            "time_share": 1.0,
                            "kernel_path": "kernels/rms_norm.py",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        optimized_dir = workspace / "workspace" / "optimized"
        optimized_dir.mkdir(parents=True, exist_ok=True)
        (optimized_dir / "rms_norm--encoder_rms_norm.py").write_text(
            "KERNEL_TYPE = 'rms_norm'\n",
            encoding="utf-8",
        )

        result = _run_script(
            "verify.py",
            workspace,
            "--profile-path",
            str(profile_path.resolve()),
            "--model",
            str((workspace / "toy_model_adapter.py").resolve()),
            "--workspace-root",
            str((workspace / "workspace").resolve()),
        )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("mode=model_verify", result.stdout)
        self.assertIn("model_correctness=PASS", result.stdout)
        self.assertIn("model_baseline_total_ms=8.0000", result.stdout)
        self.assertIn("model_optimized_total_ms=4.0000", result.stdout)
        self.assertIn("model_end_to_end_speedup=2.0000", result.stdout)


if __name__ == "__main__":
    unittest.main()
