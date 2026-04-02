import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_script(script_name: str, cwd: Path, *args: str) -> subprocess.CompletedProcess:
    tmp_path = cwd / ".tmp"
    pycache_path = cwd / ".pycache"
    tmp_path.mkdir(parents=True, exist_ok=True)
    pycache_path.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["TMPDIR"] = str(tmp_path)
    env["PYTHONPYCACHEPREFIX"] = str(pycache_path)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / script_name), *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


class PipelineScriptTests(unittest.TestCase):
    def _scratch_dir(self, name: str) -> Path:
        root = REPO_ROOT / ".tmp_test_workspaces" / name
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        return root

    def test_extract_writes_active_manifest_and_copies_kernel(self) -> None:
        workspace = self._scratch_dir("extract-script")
        shutil.copytree(REPO_ROOT / "kernels", workspace / "kernels")
        (workspace / "kernel.py").write_text("# placeholder\n", encoding="utf-8")
        profile_path = workspace / "profile.json"
        profile_path.write_text(
            json.dumps(
                {
                    "workload_name": "toy",
                    "bottlenecks": [
                        {
                            "rank": 1,
                            "name": "block0.rms_norm",
                            "kernel_type": "rms_norm",
                            "time_share": 0.7,
                            "kernel_path": "kernels/rms_norm.py",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        result = _run_script(
            "extract.py",
            workspace,
            "--profile-path",
            "profile.json",
            "--workspace-root",
            "workspace",
            "--kernel-file",
            "kernel.py",
        )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("selected_kernel_type=rms_norm", result.stdout)
        active_manifest = json.loads((workspace / "workspace" / "active_kernel.json").read_text(encoding="utf-8"))
        self.assertEqual(active_manifest["kernel_type"], "rms_norm")
        self.assertIn('KERNEL_TYPE = "rms_norm"', (workspace / "kernel.py").read_text(encoding="utf-8"))

    def test_orchestrate_and_verify_report_projected_speedup(self) -> None:
        workspace = self._scratch_dir("orchestrate-script")
        profile_path = workspace / "profile.json"
        profile_path.write_text(
            json.dumps(
                {
                    "workload_name": "toy",
                    "bottlenecks": [
                        {"rank": 1, "name": "block0.rms_norm", "kernel_type": "rms_norm", "time_share": 0.6},
                        {"rank": 2, "name": "attn.softmax", "kernel_type": "softmax", "time_share": 0.2},
                    ],
                }
            ),
            encoding="utf-8",
        )
        results_path = workspace / "results.tsv"
        results_path.write_text(
            "\t".join(
                [
                    "experiment_id",
                    "kernel_type",
                    "target_name",
                    "hypothesis",
                    "correctness",
                    "time_ms",
                    "throughput",
                    "peak_vram_mb",
                    "kept",
                    "speedup_vs_reference",
                ]
            )
            + "\n"
            + "\t".join(
                [
                    "exp-1",
                    "rms_norm",
                    "block0.rms_norm",
                    "tile tweak",
                    "PASS|smoke:PASS",
                    "1.0",
                    "throughput_tflops=0.1",
                    "0.0",
                    "yes",
                    "1.5",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        orchestrate = _run_script(
            "orchestrate.py",
            workspace,
            "--profile-path",
            "profile.json",
            "--results-path",
            "results.tsv",
        )
        self.assertEqual(orchestrate.returncode, 0, orchestrate.stdout + orchestrate.stderr)
        self.assertIn("recommended_kernel_type=softmax", orchestrate.stdout)

        verify = _run_script(
            "verify.py",
            workspace,
            "--profile-path",
            "profile.json",
            "--results-path",
            "results.tsv",
        )
        self.assertEqual(verify.returncode, 0, verify.stdout + verify.stderr)
        self.assertIn("projected_end_to_end_speedup=1.2500", verify.stdout)


if __name__ == "__main__":
    unittest.main()
