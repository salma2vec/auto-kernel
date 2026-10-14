import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import optimize
from auto_kernel.artifacts import optimized_kernel_path
from auto_kernel.results import best_speedups_by_kernel_type
from bench import append_result_row


class SearchTests(unittest.TestCase):
    def test_rejected_and_failed_trials_preserve_incumbent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            active = root / 'kernel.py'
            source = 'from kernels.rms_norm import *\n'
            active.write_text(source)
            result = {'candidate_ms': [2, 2, 2], 'candidate_median_ms': 2, 'speedup': 1}
            rejected = {**result, 'paired_speedups': [1, 1, 1, 1, 1]}
            with (patch.object(optimize, 'evaluate', return_value=result),
                  patch.object(optimize, 'evaluate_pair',
                               side_effect=[(result, rejected), RuntimeError('bad')])):
                report = optimize.optimize(active, root / 'run')
            self.assertFalse(report['changed'])
            self.assertEqual(active.read_text(), source)
            self.assertEqual(len(report['trials']), 2)
            self.assertIn('RuntimeError', report['trials'][1]['error'])
            self.assertTrue((root / 'run/report.json').exists())

    def test_material_improvement_is_promoted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            active = root / 'kernel.py'
            active.write_text('from kernels.rms_norm import *\n')
            slow = {'candidate_ms': [2, 2.1, 2.2], 'candidate_median_ms': 2.1, 'speedup': 1.0}
            fast = {'candidate_ms': [1, 1.1, 1.2], 'candidate_median_ms': 1.1,
                    'speedup': 2.0, 'paired_speedups': [2, 2, 2, 2, 2]}
            rejected = {**slow, 'paired_speedups': [0.5, 0.5, 0.5, 0.5, 0.5]}
            with (patch.object(optimize, 'evaluate', return_value=slow),
                  patch.object(optimize, 'evaluate_pair',
                               side_effect=[(slow, fast), (fast, rejected)])):
                report = optimize.optimize(active, root / 'run')
            self.assertTrue(report['changed'])
            self.assertEqual(active.read_bytes(), (root / 'run/selected.py').read_bytes())
            self.assertTrue(report['trials'][0]['accepted'])

    def test_path_traversal_kernel_type_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                optimized_kernel_path(Path(tmp), '../outside', 'name')

    def test_promotion_does_not_follow_predictable_temp_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            active = root / 'kernel.py'
            active.write_text('from kernels.rms_norm import *\n')
            unrelated = root / 'unrelated.txt'
            unrelated.write_text('preserve me')
            active.with_suffix('.candidate.tmp').symlink_to(unrelated)
            slow = {'candidate_ms': [2, 2.1, 2.2], 'candidate_median_ms': 2.1, 'speedup': 1.0}
            fast = {'candidate_ms': [1, 1.1, 1.2], 'candidate_median_ms': 1.1,
                    'speedup': 2.0, 'paired_speedups': [2, 2, 2, 2, 2]}
            rejected = {**slow, 'paired_speedups': [0.5, 0.5, 0.5, 0.5, 0.5]}
            with (patch.object(optimize, 'evaluate', return_value=slow),
                  patch.object(optimize, 'evaluate_pair',
                               side_effect=[(slow, fast), (fast, rejected)])):
                report = optimize.optimize(active, root / 'run')
            self.assertTrue(report['changed'])
            self.assertEqual(unrelated.read_text(), 'preserve me')
            self.assertFalse(active.is_symlink())
            self.assertEqual(active.read_bytes(), (root / 'run/selected.py').read_bytes())

    def test_paired_gate_requires_effect_and_all_five_consistent_pairs(self):
        incumbent = {'candidate_ms': [2] * 5, 'candidate_median_ms': 2, 'speedup': 1}
        candidate = {'candidate_ms': [1] * 5, 'candidate_median_ms': 1, 'speedup': 2,
                     'paired_speedups': [1.2] * 5}
        self.assertTrue(optimize.select_candidate(incumbent, candidate, 1.05))
        candidate['paired_speedups'][-1] = 0.99
        self.assertFalse(optimize.select_candidate(incumbent, candidate, 1.05))
        candidate['paired_speedups'] = [1.01] * 5
        self.assertFalse(optimize.select_candidate(incumbent, candidate, 1.05))

    def test_exact_sign_pvalue_handles_ties_and_direction(self):
        self.assertEqual(optimize.exact_paired_sign_pvalue([1.2] * 5), 1 / 32)
        self.assertEqual(optimize.exact_paired_sign_pvalue([0.8] * 5), 1.0)
        self.assertEqual(optimize.exact_paired_sign_pvalue([1.0] * 5), 1.0)

    def test_pending_and_nonfinite_results_not_projected(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / 'results.tsv'
            for kept, speed in [('pending', 9), ('yes', float('inf')), ('yes', 2)]:
                append_result_row(results, {'kernel_type': 'rms_norm', 'correctness': 'PASS', 'kept': kept,
                                           'speedup_vs_reference': speed, 'hypothesis': 'tabs\tand\nnewlines'})
            self.assertEqual(best_speedups_by_kernel_type(results), {'rms_norm': 2})
