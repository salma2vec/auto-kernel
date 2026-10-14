import copy
import math
import unittest

from experiments.search_calibration import (
    PROTOCOL,
    acceptance_gate,
    check_control_correctness,
    make_inputs,
    measure_implementation,
    native_rms,
)


def measurement(value):
    return {
        "candidate_ms": [value] * PROTOCOL["repeats"],
        "reference_ms": [2 * value] * PROTOCOL["repeats"],
        "candidate_median_ms": value,
        "reference_median_ms": 2 * value,
        "speedup": 2.0,
    }


def complete_trials():
    trials = []
    for kind in ("null", "positive"):
        seeds = PROTOCOL[f"{kind}_seeds"]
        for index, seed in enumerate(seeds):
            trials.append({
                "kind": kind,
                "seed": seed,
                "shape": PROTOCOL["shapes"][index % len(PROTOCOL["shapes"])],
                "accepted": kind == "positive",
                "candidate": measurement(1.0),
                "incumbent": measurement(2.0 if kind == "positive" else 1.0),
            })
    return trials


class SearchCalibrationTests(unittest.TestCase):
    def test_complete_controls_pass_frozen_gate(self):
        result = acceptance_gate(complete_trials())
        self.assertTrue(result["passed"])
        self.assertEqual(result["false_promotions"], 0)
        self.assertEqual(result["positive_detections"], 20)

    def test_false_promotion_and_sensitivity_fail_separately(self):
        trials = complete_trials()
        trials[0]["accepted"] = True
        result = acceptance_gate(trials)
        self.assertFalse(result["no_false_promotions_gate"])
        self.assertFalse(result["passed"])

        trials = complete_trials()
        for row in trials:
            if row["kind"] == "positive" and row["shape"] == PROTOCOL["shapes"][0]:
                row["accepted"] = False
        result = acceptance_gate(trials)
        self.assertTrue(result["no_false_promotions_gate"])
        self.assertFalse(result["positive_sensitivity_gate"])
        self.assertFalse(result["passed"])

    def test_incomplete_duplicate_and_nonfinite_trials_are_rejected(self):
        trials = complete_trials()
        with self.assertRaisesRegex(ValueError, "every fixed trial"):
            acceptance_gate(trials[:-1])
        duplicate = copy.deepcopy(trials)
        duplicate[-1]["seed"] = duplicate[-2]["seed"]
        with self.assertRaisesRegex(ValueError, "every fixed trial"):
            acceptance_gate(duplicate)
        invalid = copy.deepcopy(trials)
        invalid[0]["candidate"]["candidate_ms"][0] = math.nan
        with self.assertRaisesRegex(ValueError, "finite positive"):
            acceptance_gate(invalid)

    def test_slowdown_control_is_exact_and_does_not_mutate(self):
        evidence = check_control_correctness((1, 2, 512), 2000, "cpu")
        self.assertTrue(evidence["exact_match"])
        self.assertTrue(evidence["input_unchanged"])

    def test_cpu_measurement_retains_orders_and_finite_samples(self):
        inputs, weight = make_inputs((1, 2, 32), 3000, "cpu")
        measured = measure_implementation(native_rms, native_rms, inputs, weight, "cpu", 3000)
        self.assertEqual(len(measured["orders"]), PROTOCOL["repeats"])
        self.assertTrue(all(set(order) == {"candidate_ms", "reference_ms"}
                            for order in measured["orders"]))
        values = measured["candidate_ms"] + measured["reference_ms"]
        self.assertTrue(all(math.isfinite(value) and value > 0 for value in values))


if __name__ == "__main__":
    unittest.main()
