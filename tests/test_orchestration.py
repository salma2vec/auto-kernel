import unittest

from auto_kernel.orchestration import (
    aggregate_speedup_by_kernel_type,
    amdahl_speedup,
    choose_next_kernel,
    project_end_to_end_speedup,
)


class OrchestrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = {
            "workload_name": "toy",
            "bottlenecks": [
                {"name": "block0.rms_norm", "kernel_type": "rms_norm", "time_share": 0.5},
                {"name": "block1.rms_norm", "kernel_type": "rms_norm", "time_share": 0.2},
                {"name": "attn.softmax", "kernel_type": "softmax", "time_share": 0.2},
            ],
        }

    def test_aggregate_speedup_groups_by_kernel_type(self) -> None:
        grouped = aggregate_speedup_by_kernel_type(self.profile)

        self.assertEqual(grouped["rms_norm"]["time_share"], 0.7)
        self.assertEqual(grouped["softmax"]["time_share"], 0.2)

    def test_amdahl_speedup_matches_expected_formula(self) -> None:
        self.assertAlmostEqual(amdahl_speedup(0.7, 2.0), 1.5384615384, places=6)

    def test_project_end_to_end_speedup_uses_all_optimized_kernels(self) -> None:
        grouped = aggregate_speedup_by_kernel_type(self.profile)
        projected = project_end_to_end_speedup(grouped, {"rms_norm": 2.0, "softmax": 1.5})

        self.assertAlmostEqual(projected, 1.7142857142, places=6)

    def test_choose_next_kernel_prefers_highest_remaining_share(self) -> None:
        grouped = aggregate_speedup_by_kernel_type(self.profile)
        choice = choose_next_kernel(grouped, {})

        self.assertEqual(choice["kernel_type"], "rms_norm")
        self.assertAlmostEqual(choice["time_share"], 0.7, places=6)

    def test_choose_next_kernel_skips_optimized_kernels(self) -> None:
        grouped = aggregate_speedup_by_kernel_type(self.profile)
        choice = choose_next_kernel(grouped, {"rms_norm": 1.8})

        self.assertEqual(choice["kernel_type"], "softmax")


if __name__ == "__main__":
    unittest.main()
