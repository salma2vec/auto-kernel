import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from auto_kernel.model_adapters import _compare_values


class VerificationIntegrityTests(unittest.TestCase):
    def test_tensor_comparison_rejects_broadcasting_dtype_and_nonfinite_values(self):
        for actual, expected in [
            (torch.ones(1), torch.ones(2, 3)),
            (torch.ones(2).half(), torch.ones(2)),
            (torch.tensor([math.inf]), torch.tensor([math.inf])),
            (torch.tensor([math.nan]), torch.tensor([math.nan])),
        ]:
            with self.subTest(actual=actual, expected=expected):
                self.assertFalse(_compare_values(actual, expected, 1e-5, 1e-4)[0])
        self.assertEqual(_compare_values(torch.empty(0), torch.empty(0), 1e-5, 1e-4), (True, 0.0))
        self.assertTrue(_compare_values(torch.ones(2), torch.ones(2), 1e-5, 1e-4)[0])

    def test_cli_fails_closed_on_failed_or_malformed_model_verification(self):
        repo = Path(__file__).resolve().parent.parent
        cases = [
            {"correct": False, "max_abs_diff": 9.0},
            {"correct": "false", "max_abs_diff": 0.0},
            {"correct": True, "max_abs_diff": float("nan")},
            {"correct": True, "max_abs_diff": 0.0, "optimized_total_ms": 0.0},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "profile.json").write_text(json.dumps({"bottlenecks": []}))
            for case in cases:
                result = {"baseline_total_ms": 2.0, "optimized_total_ms": 1.0, **case}
                # JSON parses nonfinite values without needing executable fixture text.
                (root / "model.py").write_text(
                    "import json\ndef verify_model(**kwargs):\n    return json.loads(" + repr(json.dumps(result)) + ")\n"
                )
                completed = subprocess.run(
                    [sys.executable, str(repo / "verify.py"), "--profile-path", str(root / "profile.json"),
                     "--results-path", str(root / "missing.tsv"), "--workspace-root", str(root / "workspace"),
                     "--model", str(root / "model.py")], capture_output=True, text=True, timeout=30,
                )
                with self.subTest(case=case):
                    self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                    self.assertIn("model_correctness=FAIL", completed.stdout)
