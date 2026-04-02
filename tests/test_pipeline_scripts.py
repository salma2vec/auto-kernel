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
        base = REPO_ROOT / ".tmp" / "test_workspaces"
        base.mkdir(parents=True, exist_ok=True)
        root = base / name
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda root=root: shutil.rmtree(root, ignore_errors=True))
        return root

    def test_extract_writes_active_manifest_and_copies_kernel(self) -> None:
        workspace = self._scratch_dir("extract-script")
        shutil.copytree(
            REPO_ROOT / "kernels",
            workspace / "kernels",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
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
        self.assertEqual(active_manifest["source_kind"], "baseline")

    def test_extract_prefers_saved_optimized_kernel_when_requested(self) -> None:
        workspace = self._scratch_dir("extract-optimized-script")
        shutil.copytree(
            REPO_ROOT / "kernels",
            workspace / "kernels",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        optimized_dir = workspace / "workspace" / "optimized"
        optimized_dir.mkdir(parents=True, exist_ok=True)
        optimized_kernel = optimized_dir / "rms_norm--encoder_block0_rms_norm.py"
        optimized_kernel.write_text("# optimized variant\nKERNEL_TYPE = 'rms_norm'\n", encoding="utf-8")
        (workspace / "kernel.py").write_text("# placeholder\n", encoding="utf-8")
        profile_path = workspace / "profile.json"
        profile_path.write_text(
            json.dumps(
                {
                    "workload_name": "toy",
                    "bottlenecks": [
                        {
                            "rank": 1,
                            "name": "encoder.block0.rms_norm",
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
            "--prefer-optimized",
        )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("source_kind=optimized", result.stdout)
        self.assertIn("# optimized variant", (workspace / "kernel.py").read_text(encoding="utf-8"))

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

    def test_verify_measured_mode_benchmarks_baseline_and_optimized_variants(self) -> None:
        workspace = self._scratch_dir("verify-measured-script")
        (workspace / "toy_kernel.py").write_text(
            "\n".join(
                [
                    'KERNEL_TYPE = "toy"',
                    "",
                    "def kernel_fn(value, inner_loops=5000):",
                    "    total = 0",
                    "    for _ in range(inner_loops):",
                    "        total += value",
                    "    return total",
                    "",
                    "def get_inputs(device=None, value=1, inner_loops=5000):",
                    '    return {"value": value, "inner_loops": inner_loops}',
                    "",
                    "def get_flops(inputs):",
                    '    return inputs["inner_loops"]',
                    "",
                    "def get_bytes(inputs):",
                    "    return 1",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        optimized_dir = workspace / "workspace" / "optimized"
        optimized_dir.mkdir(parents=True, exist_ok=True)
        (optimized_dir / "toy--toy_entry.py").write_text(
            "\n".join(
                [
                    'KERNEL_TYPE = "toy"',
                    "",
                    "def kernel_fn(value, inner_loops=1000):",
                    "    total = 0",
                    "    for _ in range(inner_loops):",
                    "        total += value",
                    "    return total",
                    "",
                    "def get_inputs(device=None, value=1, inner_loops=1000):",
                    '    return {"value": value, "inner_loops": inner_loops}',
                    "",
                    "def get_flops(inputs):",
                    '    return inputs["inner_loops"]',
                    "",
                    "def get_bytes(inputs):",
                    "    return 1",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        profile_path = workspace / "profile.json"
        profile_path.write_text(
            json.dumps(
                {
                    "workload_name": "toy",
                    "bottlenecks": [
                        {
                            "rank": 1,
                            "name": "toy.entry",
                            "kernel_type": "toy",
                            "time_share": 1.0,
                            "kernel_path": str((workspace / "toy_kernel.py").resolve()),
                            "input_kwargs": {"value": 2, "inner_loops": 4000},
                        }
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
                    "exp-2",
                    "toy",
                    "toy.entry",
                    "reduce loop count",
                    "PASS|smoke:PASS",
                    "1.0",
                    "throughput_tflops=0.1",
                    "0.0",
                    "yes",
                    "2.0",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        verify = _run_script(
            "verify.py",
            workspace,
            "--profile-path",
            "profile.json",
            "--results-path",
            "results.tsv",
            "--workspace-root",
            "workspace",
            "--measure-workload",
            "--iterations",
            "3",
        )

        self.assertEqual(verify.returncode, 0, verify.stdout + verify.stderr)
        self.assertIn("measured_baseline_total_ms=", verify.stdout)
        self.assertIn("measured_optimized_total_ms=", verify.stdout)
        self.assertIn("measured_end_to_end_speedup=", verify.stdout)


if __name__ == "__main__":
    unittest.main()
