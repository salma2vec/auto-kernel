import math
import unittest

from experiments.search_calibration import make_inputs, native_rms
from experiments.search_calibration_paired import PROTOCOL, measure_paired
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


if __name__ == "__main__":
    unittest.main()
