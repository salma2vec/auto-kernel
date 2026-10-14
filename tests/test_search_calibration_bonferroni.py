import copy
import unittest

from experiments.search_calibration_bonferroni import PROTOCOL
from experiments.search_calibration_paired import acceptance_gate
from optimize import balanced_orders, exact_paired_sign_pvalue, select_candidate


def complete_trials():
    trials = []
    for kind in ("null", "positive"):
        seeds = PROTOCOL[f"{kind}_seeds"]
        for index, seed in enumerate(seeds):
            candidate_values = [1.0] * PROTOCOL["repeats"]
            incumbent_values = [1.0 if kind == "null" else 2.0] * PROTOCOL["repeats"]
            reference_values = [3.0] * PROTOCOL["repeats"]
            paired = [incumbent / candidate for incumbent, candidate
                      in zip(incumbent_values, candidate_values)]
            orders = balanced_orders(seed, PROTOCOL["repeats"])
            candidate = {
                "candidate_ms": candidate_values, "reference_ms": reference_values,
                "candidate_median_ms": 1.0, "reference_median_ms": 3.0,
                "speedup": 3.0, "orders": copy.deepcopy(orders),
                "paired_speedups": paired,
                "paired_median_speedup": 1.0 if kind == "null" else 2.0,
                "paired_sign_pvalue": exact_paired_sign_pvalue(paired),
            }
            incumbent_median = 1.0 if kind == "null" else 2.0
            incumbent = {
                "candidate_ms": incumbent_values, "reference_ms": reference_values,
                "candidate_median_ms": incumbent_median, "reference_median_ms": 3.0,
                "speedup": 3.0 / incumbent_median, "orders": copy.deepcopy(orders),
            }
            trials.append({
                "kind": kind, "seed": seed,
                "shape": PROTOCOL["shapes"][index % len(PROTOCOL["shapes"])],
                "candidate": candidate, "incumbent": incumbent,
                "accepted": kind == "positive",
            })
    return trials


class FamilywiseSearchCalibrationTests(unittest.TestCase):
    def test_eleven_of_eleven_passes_but_ten_does_not(self):
        incumbent = {"candidate_ms": [2.0] * 11, "candidate_median_ms": 2.0}
        candidate = {"candidate_ms": [1.0] * 11, "candidate_median_ms": 1.0,
                     "paired_speedups": [2.0] * 11}
        self.assertEqual(exact_paired_sign_pvalue(candidate["paired_speedups"]), 1 / 2048)
        self.assertTrue(select_candidate(
            incumbent, candidate, PROTOCOL["minimum_paired_speedup"],
            PROTOCOL["maximum_one_sided_sign_pvalue"],
        ))
        candidate["paired_speedups"][-1] = 0.9
        self.assertEqual(exact_paired_sign_pvalue(candidate["paired_speedups"]), 12 / 2048)
        self.assertFalse(select_candidate(
            incumbent, candidate, PROTOCOL["minimum_paired_speedup"],
            PROTOCOL["maximum_one_sided_sign_pvalue"],
        ))

    def test_balanced_order_uses_each_position_three_or_four_times(self):
        orders = balanced_orders(5000, PROTOCOL["repeats"])
        self.assertEqual(len(orders), 11)
        for name in ("candidate", "incumbent", "reference"):
            counts = [sum(order[position] == name for order in orders) for position in range(3)]
            self.assertEqual(sorted(counts), [3, 4, 4])

    def test_complete_fresh_schedule_passes_and_tampering_fails(self):
        result = acceptance_gate(complete_trials(), PROTOCOL)
        self.assertTrue(result["passed"])
        trials = complete_trials()
        trials[0]["candidate"]["orders"][0], trials[0]["candidate"]["orders"][1] = (
            trials[0]["candidate"]["orders"][1], trials[0]["candidate"]["orders"][0]
        )
        trials[0]["incumbent"]["orders"] = copy.deepcopy(trials[0]["candidate"]["orders"])
        with self.assertRaisesRegex(ValueError, "balanced schedule"):
            acceptance_gate(trials, PROTOCOL)
        trials = complete_trials()
        trials[-1]["accepted"] = False
        with self.assertRaisesRegex(ValueError, "decision"):
            acceptance_gate(trials, PROTOCOL)

    def test_protocol_family_and_threshold_must_agree(self):
        protocol = {**PROTOCOL, "null_decision_family_size": 99}
        with self.assertRaisesRegex(ValueError, "family size"):
            acceptance_gate(complete_trials(), protocol)


if __name__ == "__main__":
    unittest.main()
