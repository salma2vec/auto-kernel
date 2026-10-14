import copy
import unittest

import torch

from experiments.workload_validation import (
    PROTOCOL,
    acceptance_gate,
    check_correctness,
    choose_from_discovery,
    make_input,
    measure_case,
    replace_norms,
    summarize_run,
)
from models.rms_norm_model import build_model


def cases(original=3.0, multiply=2.5, native=1.0, shapes=None):
    return [
        {"shape": shape, "blocks": [
            {"milliseconds": {"original": original, "multiply_square": multiply,
                              "native_rms": native}}
            for _ in range(5)
        ]}
        for shape in (shapes or PROTOCOL["confirmation_shapes"])
    ]


class WorkloadValidationTests(unittest.TestCase):
    def test_native_selection_is_not_added_search_value(self):
        discovery = cases(shapes=PROTOCOL["discovery_shapes"])
        selected, decisions = choose_from_discovery(discovery)
        self.assertEqual(selected, "native_rms")
        self.assertTrue(decisions[-1]["accepted"])
        summaries = [summarize_run(seed, selected, cases(), [{"accepted": False}] * 6)
                     for seed in PROTOCOL["seeds"]]
        result = acceptance_gate(summaries)
        self.assertTrue(result["engineering_gate"])
        self.assertFalse(result["added_search_value_gate"])
        self.assertEqual(result["search_value_passing_runs"], 0)

    def test_confirmation_does_not_reselect_a_better_native(self):
        selected, _ = choose_from_discovery(cases(multiply=1.0, native=2.0))
        self.assertEqual(selected, "multiply_square")
        # A reversed ranking on new inputs is a failure to transfer, not an
        # opportunity to select again and score the new best method.
        result = summarize_run(142, selected, cases(multiply=4.0, native=1.0),
                               [{"accepted": False}] * 6)
        self.assertEqual(result["selected"], "multiply_square")
        self.assertEqual(result["geomean_speedup_vs_original"], 0.75)

    def test_same_code_false_promotion_blocks_both_gates(self):
        summaries = [summarize_run(seed, "multiply_square", cases(multiply=0.5),
                                   [{"accepted": False}] * 6)
                     for seed in PROTOCOL["seeds"]]
        summaries[0]["false_promotions"] = 1
        result = acceptance_gate(summaries)
        self.assertFalse(result["engineering_gate"])
        self.assertFalse(result["added_search_value_gate"])

    def test_native_case_regression_blocks_added_value_only(self):
        measured = cases(original=4.0, multiply=0.5, native=1.0)
        for block in measured[1]["blocks"]:
            block["milliseconds"]["multiply_square"] = 1.2
        summaries = [summarize_run(seed, "multiply_square", measured,
                                   [{"accepted": False}] * 6)
                     for seed in PROTOCOL["seeds"]]
        result = acceptance_gate(summaries)
        self.assertTrue(result["engineering_gate"])
        self.assertEqual(result["search_value_passing_runs"], 5)
        self.assertFalse(result["no_large_case_regression_vs_native"])
        self.assertFalse(result["added_search_value_gate"])

    def test_missing_runs_cases_and_controls_are_rejected(self):
        summaries = [summarize_run(seed, "native_rms", cases(), [{"accepted": False}] * 6)
                     for seed in PROTOCOL["seeds"]]
        with self.assertRaises(ValueError):
            acceptance_gate(summaries[:-1])
        incomplete = copy.deepcopy(summaries)
        incomplete[0]["confirmation_cases"].pop()
        with self.assertRaises(ValueError):
            acceptance_gate(incomplete)
        incomplete = copy.deepcopy(summaries)
        incomplete[0]["control_count"] = 5
        with self.assertRaises(ValueError):
            acceptance_gate(incomplete)

    def test_nonfinite_confirmation_cannot_pass(self):
        summaries = [summarize_run(seed, "native_rms", cases(), [{"accepted": False}] * 6)
                     for seed in PROTOCOL["seeds"]]
        summaries[0]["geomean_speedup_vs_original"] = float("nan")
        with self.assertRaises(ValueError):
            acceptance_gate(summaries)
        summaries[0]["geomean_speedup_vs_original"] = 1.0
        summaries[0]["confirmation_cases"][0]["speedup_vs_always_native"] = float("inf")
        with self.assertRaises(ValueError):
            acceptance_gate(summaries)

    def test_strided_input_has_matching_values_and_nonunit_last_stride(self):
        value = make_input((1, 2, 512), 142, "cpu", layout="strided")
        self.assertEqual(value.shape, (1, 2, 512))
        self.assertEqual(value.stride()[-1], 2)
        self.assertFalse(value.is_contiguous())
        torch.testing.assert_close(value, make_input((1, 2, 512), 142, "cpu", layout="strided"))

    def test_complete_model_native_correctness_and_cpu_timing(self):
        torch.set_num_threads(1)
        original = build_model().eval()
        native = replace_norms(original, lambda x, weight, eps:
                               torch.nn.functional.rms_norm(x, (x.shape[-1],), weight, eps))
        models = {"original": original, "native_rms": native}
        evidence = check_correctness(models, (1, 2, 512), 142, "cpu")
        self.assertEqual(len(evidence), 16)
        self.assertTrue(all(row["passed"] for row in evidence))
        measured = measure_case(models, (1, 2, 512), 142, "cpu", repeats=3,
                                iterations=2, warmup=1)
        self.assertEqual(len(measured["blocks"]), 3)
        self.assertTrue(all(set(row["order"]) == set(models) for row in measured["blocks"]))

    def test_mutating_or_wrong_candidate_is_rejected(self):
        original = build_model().eval()

        def corrupt(x, weight, eps):
            x.add_(1)
            return x

        candidate = replace_norms(original, corrupt)
        with self.assertRaisesRegex(ValueError, "correctness failed"):
            check_correctness({"corrupt": candidate}, (1, 2, 512), 142, "cpu")


if __name__ == "__main__":
    unittest.main()
