import copy
import math
import unittest

from experiments.search_calibration import make_inputs, native_rms
from experiments.search_calibration_paired import PROTOCOL, acceptance_gate, measure_paired
from optimize import exact_paired_sign_pvalue


class PairedSearchCalibrationTests(unittest.TestCase):
    def test_measurement_retains_complete_interleaved_pairs(self):
        inputs, weight = make_inputs((1, 2, 32), 3000, "cpu")
        incumbent, candidate = measure_paired(
            native_rms, native_rms, inputs, weight, "cpu", 3000
        )
        self.assertEqual(len(candidate["orders"]), PROTOCOL["repeats"])
        self.assertTrue(all(set(order) == {"candidate", "incumbent", "reference"}
                            for order in candidate["orders"]))
        self.assertEqual(len(candidate["paired_speedups"]), PROTOCOL["repeats"])
        self.assertTrue(all(math.isfinite(value) and value > 0
                            for value in candidate["paired_speedups"]))
        self.assertEqual(candidate["paired_sign_pvalue"],
                         exact_paired_sign_pvalue(candidate["paired_speedups"]))
        self.assertEqual(candidate["reference_ms"], incumbent["reference_ms"])

    def test_gate_recomputes_complete_v2_evidence(self):
        result = acceptance_gate(complete_trials())
        self.assertTrue(result["passed"])
        self.assertEqual(result["false_promotions"], 0)
        self.assertEqual(result["positive_detections"], 20)

    def test_gate_rejects_tampered_decision_pair_pvalue_and_order(self):
        mutations = []
        trials = complete_trials()
        trials[0]["accepted"] = True
        mutations.append((trials, "decision"))
        trials = complete_trials()
        trials[-1]["candidate"]["paired_speedups"][0] = 99
        mutations.append((trials, "paired evidence"))
        trials = complete_trials()
        trials[-1]["candidate"]["paired_sign_pvalue"] = 0.9
        mutations.append((trials, "paired evidence"))
        trials = complete_trials()
        trials[-1]["candidate"]["orders"][0] = ["candidate", "candidate", "reference"]
        mutations.append((trials, "orders"))
        for tampered, message in mutations:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                acceptance_gate(tampered)

    def test_gate_rejects_missing_or_nonfinite_raw_trials(self):
        with self.assertRaisesRegex(ValueError, "every fixed v2 trial"):
            acceptance_gate(complete_trials()[:-1])
        trials = complete_trials()
        trials[0]["candidate"]["candidate_ms"][0] = math.nan
        with self.assertRaisesRegex(ValueError, "raw timings"):
            acceptance_gate(trials)
        trials = complete_trials()
        trials[0]["candidate"]["candidate_median_ms"] = True
        with self.assertRaisesRegex(ValueError, "summaries"):
            acceptance_gate(trials)


def complete_trials():
    trials = []
    orders = [["candidate", "incumbent", "reference"] for _ in range(PROTOCOL["repeats"])]
    for kind in ("null", "positive"):
        seeds = PROTOCOL[f"{kind}_seeds"]
        for index, seed in enumerate(seeds):
            candidate_values = [1.0] * PROTOCOL["repeats"]
            incumbent_values = [1.0 if kind == "null" else 2.0] * PROTOCOL["repeats"]
            reference_values = [3.0] * PROTOCOL["repeats"]
            paired = [incumbent / candidate for incumbent, candidate
                      in zip(incumbent_values, candidate_values)]
            candidate = {
                "candidate_ms": candidate_values,
                "reference_ms": reference_values,
                "candidate_median_ms": 1.0,
                "reference_median_ms": 3.0,
                "speedup": 3.0,
                "orders": copy.deepcopy(orders),
                "paired_speedups": paired,
                "paired_median_speedup": 1.0 if kind == "null" else 2.0,
                "paired_sign_pvalue": exact_paired_sign_pvalue(paired),
            }
            incumbent_median = 1.0 if kind == "null" else 2.0
            incumbent = {
                "candidate_ms": incumbent_values,
                "reference_ms": reference_values,
                "candidate_median_ms": incumbent_median,
                "reference_median_ms": 3.0,
                "speedup": 3.0 / incumbent_median,
                "orders": copy.deepcopy(orders),
            }
            trials.append({
                "kind": kind,
                "seed": seed,
                "shape": PROTOCOL["shapes"][index % len(PROTOCOL["shapes"])],
                "candidate": candidate,
                "incumbent": incumbent,
                "accepted": kind == "positive",
            })
    return trials


if __name__ == "__main__":
    unittest.main()
