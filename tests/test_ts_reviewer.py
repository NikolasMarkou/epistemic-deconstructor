#!/usr/bin/env python3
"""Tests for src/scripts/ts_reviewer.py"""

import math
import os
import sys
import unittest

# Allow importing from src/scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'scripts'))

from ts_reviewer import (
    TimeSeriesReviewer,
    ReviewReport,
    Verdict,
    Severity,
    Finding,
    PhaseResult,
    _to_floats,
    _mean,
    _std,
    _diff,
    _r_squared,
    _mae,
    _rmse,
    walk_forward_split,
    quick_review,
)


class TestToFloats(unittest.TestCase):

    def test_basic_conversion(self):
        result = _to_floats([1, 2.5, 3])
        self.assertEqual(result, [1.0, 2.5, 3.0])

    def test_filters_none(self):
        result = _to_floats([1, None, 3])
        self.assertEqual(result, [1.0, 3.0])

    def test_filters_non_numeric(self):
        result = _to_floats([1, "abc", 3, "def"])
        self.assertEqual(result, [1.0, 3.0])

    def test_filters_inf_nan(self):
        result = _to_floats([1, float('inf'), float('-inf'), float('nan'), 3])
        self.assertEqual(result, [1.0, 3.0])

    def test_empty_input(self):
        self.assertEqual(_to_floats([]), [])

    def test_string_numbers(self):
        result = _to_floats(["1.5", "2.0", "bad"])
        self.assertEqual(result, [1.5, 2.0])


class TestConstructor(unittest.TestCase):

    def test_basic_construction(self):
        r = TimeSeriesReviewer([1, 2, 3], name="test")
        self.assertEqual(r.name, "test")
        self.assertEqual(len(r._clean), 3)

    def test_filters_none_and_non_float(self):
        r = TimeSeriesReviewer([1, None, "abc", 3])
        self.assertEqual(len(r._clean), 2)
        self.assertEqual(r._clean, [1.0, 3.0])

    def test_frequency_setting(self):
        r = TimeSeriesReviewer([1, 2, 3], frequency=12)
        self.assertEqual(r.frequency, 12)

    def test_empty_input(self):
        r = TimeSeriesReviewer([])
        self.assertEqual(len(r._clean), 0)


class TestPhaseCoherence(unittest.TestCase):

    def test_reject_short_series(self):
        """Less than 10 valid observations -> REJECT."""
        r = TimeSeriesReviewer([1, 2, 3, 4, 5])
        ph = r.phase_coherence()
        self.assertEqual(ph.verdict, Verdict.REJECT)

    def test_warn_marginal_series(self):
        """10-29 observations -> WARN (marginal)."""
        data = list(range(20))
        r = TimeSeriesReviewer(data)
        ph = r.phase_coherence()
        min_len_finding = [f for f in ph.findings if f.check == "min_length"][0]
        self.assertEqual(min_len_finding.verdict, Verdict.WARN)

    def test_pass_normal_series(self):
        """30+ observations -> PASS."""
        data = list(range(50))
        r = TimeSeriesReviewer(data)
        ph = r.phase_coherence()
        min_len_finding = [f for f in ph.findings if f.check == "min_length"][0]
        self.assertEqual(min_len_finding.verdict, Verdict.PASS)

    def test_reject_constant_series(self):
        """All same value -> REJECT."""
        data = [42.0] * 50
        r = TimeSeriesReviewer(data)
        ph = r.phase_coherence()
        constant_finding = [f for f in ph.findings if f.check == "constant_series"]
        self.assertTrue(len(constant_finding) > 0)
        self.assertEqual(constant_finding[0].verdict, Verdict.REJECT)


class TestPhaseDataQuality(unittest.TestCase):

    def test_clean_pass(self):
        """Clean data should pass missing values check."""
        data = [float(i) for i in range(50)]
        r = TimeSeriesReviewer(data)
        ph = r.phase_data_quality()
        missing_finding = [f for f in ph.findings if f.check == "missing_values"][0]
        self.assertEqual(missing_finding.verdict, Verdict.PASS)

    def test_high_missing(self):
        """Over 20% missing -> FAIL."""
        data = [1.0] * 30 + [None] * 20
        r = TimeSeriesReviewer(data)
        ph = r.phase_data_quality()
        missing_finding = [f for f in ph.findings if f.check == "missing_values"][0]
        self.assertIn(missing_finding.verdict, (Verdict.FAIL, Verdict.WARN))


class TestPhaseStationarity(unittest.TestCase):

    def test_skip_short(self):
        """Less than 20 observations -> SKIP."""
        r = TimeSeriesReviewer(list(range(15)))
        ph = r.phase_stationarity()
        self.assertEqual(ph.verdict, Verdict.SKIP)

    def test_runs_on_sufficient_data(self):
        """20+ observations should produce findings (not all SKIP)."""
        import random
        random.seed(42)
        data = [random.gauss(0, 1) for _ in range(100)]
        r = TimeSeriesReviewer(data)
        ph = r.phase_stationarity()
        non_skip = [f for f in ph.findings if f.verdict != Verdict.SKIP]
        self.assertTrue(len(non_skip) > 0)


class TestPhaseBaselineBenchmarks(unittest.TestCase):

    def test_skip_short(self):
        """Less than 10 observations -> SKIP."""
        r = TimeSeriesReviewer([1, 2, 3])
        ph = r.phase_baseline_benchmarks()
        self.assertEqual(ph.verdict, Verdict.SKIP)

    def test_computes_on_sufficient_data(self):
        """100+ values produce baseline findings."""
        import random
        random.seed(42)
        data = [100 + random.gauss(0, 5) for _ in range(150)]
        r = TimeSeriesReviewer(data)
        ph = r.phase_baseline_benchmarks()
        baseline_finding = [f for f in ph.findings if f.check == "baseline_errors"]
        self.assertTrue(len(baseline_finding) > 0)
        self.assertEqual(baseline_finding[0].verdict, Verdict.PASS)


class TestFullReview(unittest.TestCase):

    def test_returns_report(self):
        """full_review returns a ReviewReport."""
        import random
        random.seed(42)
        data = [100 + random.gauss(0, 3) for _ in range(200)]
        r = TimeSeriesReviewer(data, name="test_series")
        report = r.full_review()
        self.assertIsInstance(report, ReviewReport)
        self.assertEqual(report.series_name, "test_series")
        self.assertTrue(len(report.phases) > 0)

    def test_short_circuit_on_reject(self):
        """full_review short-circuits after phase 1 REJECT."""
        r = TimeSeriesReviewer([1, 2, 3])
        report = r.full_review()
        self.assertEqual(report.overall_verdict, Verdict.REJECT)
        # Only phase 1 should be present
        self.assertEqual(len(report.phases), 1)


class TestReviewReportToDict(unittest.TestCase):

    def test_structure(self):
        """to_dict returns expected top-level keys."""
        import random
        random.seed(42)
        data = [random.gauss(50, 10) for _ in range(100)]
        r = TimeSeriesReviewer(data, name="dict_test")
        report = r.full_review()
        d = report.to_dict()
        self.assertIn("series_name", d)
        self.assertIn("n_observations", d)
        self.assertIn("overall_verdict", d)
        self.assertIn("phases", d)
        self.assertIn("red_flag_count", d)
        self.assertIn("warning_count", d)
        self.assertIsInstance(d["phases"], list)


class TestHelperFunctions(unittest.TestCase):

    def test_mean(self):
        self.assertAlmostEqual(_mean([1, 2, 3]), 2.0)

    def test_mean_empty(self):
        self.assertAlmostEqual(_mean([]), 0.0)

    def test_std(self):
        result = _std([1, 2, 3, 4, 5])
        self.assertGreater(result, 0)

    def test_std_single(self):
        self.assertAlmostEqual(_std([5]), 0.0)

    def test_diff(self):
        self.assertEqual(_diff([1, 3, 6, 10]), [2, 3, 4])

    def test_r_squared_perfect(self):
        actual = [1, 2, 3, 4, 5]
        predicted = [1, 2, 3, 4, 5]
        self.assertAlmostEqual(_r_squared(actual, predicted), 1.0)

    def test_mae(self):
        actual = [1, 2, 3]
        predicted = [1.5, 2.5, 3.5]
        self.assertAlmostEqual(_mae(actual, predicted), 0.5)

    def test_rmse(self):
        actual = [1, 2, 3]
        predicted = [1, 2, 3]
        self.assertAlmostEqual(_rmse(actual, predicted), 0.0)


class TestWalkForwardSplit(unittest.TestCase):

    def test_basic_split(self):
        data = list(range(100))
        splits = walk_forward_split(data, n_folds=3, min_train=30, horizon=10)
        self.assertGreater(len(splits), 0)
        for train, test in splits:
            self.assertGreater(len(train), 0)
            self.assertGreater(len(test), 0)

    def test_short_data_fallback(self):
        """Data shorter than min_train + horizon returns single split."""
        data = list(range(31))
        splits = walk_forward_split(data, n_folds=5, min_train=30, horizon=5)
        self.assertEqual(len(splits), 1)


class TestQuickReview(unittest.TestCase):

    def test_returns_report(self):
        import random
        random.seed(42)
        data = [random.gauss(0, 1) for _ in range(100)]
        report = quick_review(data, name="quick_test")
        self.assertIsInstance(report, ReviewReport)
        self.assertEqual(report.series_name, "quick_test")


if __name__ == '__main__':
    unittest.main()
