import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from experiments import accelerator_validation as validation


class AcceleratorValidationTests(unittest.TestCase):
    def test_cpu_parity_failure_is_recorded_without_running_optimizer(self):
        contract = SimpleNamespace(kernel_fn=lambda: torch.zeros(2))
        config = SimpleNamespace(input_builder=lambda *args: {}, timing_shape=(), atol=1e-5, rtol=1e-4)
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(validation, "REFERENCE_FUNCTIONS", {"rms_norm": lambda: torch.ones(2)}), \
             patch.object(validation, "get_benchmark_config", return_value=config), \
             patch.object(validation, "validate_kernel_module", return_value=contract), \
             patch.object(validation, "optimize") as optimizer:
            result = validation.run(Path(tmp) / "run", "cpu", seeds=(42,))
            optimizer.assert_not_called()
            self.assertFalse(result["passed"])
            self.assertTrue(result["complete"])
            self.assertEqual(result["runs"][0]["status"], "failed")
            saved = json.loads((Path(tmp) / "run" / "summary.json").read_text())
            self.assertFalse(saved["runs"][0]["cpu_reference"]["passed"])

    def test_failed_suite_has_unsuccessful_exit_status(self):
        with patch.object(validation, "run", return_value={"passed": False}):
            self.assertEqual(validation.main(["--output", "unused", "--device", "cpu"]), 1)
