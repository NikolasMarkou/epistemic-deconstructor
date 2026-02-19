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
    _rmsse,
    _wape,
    _me_bias,
    _pinball_loss,
    _fva,
    _permutation_entropy,
    walk_forward_split,
    quick_review,
    conformal_intervals,
    cqr_intervals,
    compare_models,
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


class TestPermutationEntropy(unittest.TestCase):

    def test_deterministic_series(self):
        """Monotonically increasing series should have PE near 0."""
        data = list(range(100))
        pe = _permutation_entropy(data, order=3, delay=1)
        self.assertLess(pe, 0.05)

    def test_random_series(self):
        """Random series should have PE near 1."""
        import random
        random.seed(123)
        data = [random.gauss(0, 1) for _ in range(1000)]
        pe = _permutation_entropy(data, order=3, delay=1)
        self.assertGreater(pe, 0.9)

    def test_short_series_nan(self):
        """Too-short series should return nan."""
        pe = _permutation_entropy([1.0, 2.0], order=3, delay=1)
        self.assertTrue(math.isnan(pe))

    def test_order_4(self):
        """Higher order still works on deterministic data."""
        data = list(range(200))
        pe = _permutation_entropy(data, order=4, delay=1)
        self.assertLess(pe, 0.05)

    def test_normalized_range(self):
        """PE should be in [0, 1]."""
        import random
        random.seed(42)
        data = [random.gauss(0, 1) for _ in range(200)]
        pe = _permutation_entropy(data, order=3, delay=1)
        self.assertGreaterEqual(pe, 0.0)
        self.assertLessEqual(pe, 1.0)


class TestNewMetrics(unittest.TestCase):

    def test_rmsse_perfect(self):
        actual = [10.0, 20.0, 30.0]
        predicted = [10.0, 20.0, 30.0]
        train = [5.0, 10.0, 15.0, 20.0, 25.0]
        self.assertAlmostEqual(_rmsse(actual, predicted, train, 1), 0.0)

    def test_rmsse_nonzero(self):
        actual = [10.0, 20.0, 30.0]
        predicted = [12.0, 22.0, 28.0]
        train = [5.0, 10.0, 15.0, 20.0, 25.0]
        result = _rmsse(actual, predicted, train, 1)
        self.assertGreater(result, 0)
        self.assertTrue(math.isfinite(result))

    def test_rmsse_nan_short_train(self):
        result = _rmsse([1.0], [1.0], [1.0], 1)
        self.assertTrue(math.isnan(result))

    def test_wape_basic(self):
        actual = [100.0, 200.0, 300.0]
        predicted = [110.0, 190.0, 310.0]
        result = _wape(actual, predicted)
        # total error = 10+10+10 = 30, total actual = 600, WAPE = 5%
        self.assertAlmostEqual(result, 5.0)

    def test_wape_zero_actual(self):
        result = _wape([0.0, 0.0], [1.0, 1.0])
        self.assertTrue(math.isnan(result))

    def test_me_bias_positive(self):
        """Over-forecasting should give positive ME."""
        actual = [10.0, 20.0, 30.0]
        predicted = [15.0, 25.0, 35.0]
        self.assertAlmostEqual(_me_bias(actual, predicted), 5.0)

    def test_me_bias_negative(self):
        """Under-forecasting should give negative ME."""
        actual = [10.0, 20.0, 30.0]
        predicted = [8.0, 18.0, 28.0]
        self.assertAlmostEqual(_me_bias(actual, predicted), -2.0)

    def test_me_bias_unbiased(self):
        actual = [10.0, 20.0]
        predicted = [10.0, 20.0]
        self.assertAlmostEqual(_me_bias(actual, predicted), 0.0)

    def test_pinball_loss_median(self):
        """At tau=0.5, pinball loss = 0.5 * MAE."""
        actual = [10.0, 20.0, 30.0]
        predicted = [12.0, 18.0, 30.0]
        result = _pinball_loss(actual, predicted, 0.5)
        expected = 0.5 * (2 + 2 + 0) / 3
        self.assertAlmostEqual(result, expected)

    def test_pinball_loss_high_quantile(self):
        """Under-predictions penalized more at tau=0.9."""
        actual = [20.0]
        predicted = [10.0]  # under-prediction by 10
        result = _pinball_loss(actual, predicted, 0.9)
        # residual = 20 - 10 = 10 >= 0, so loss = 0.9 * 10 = 9
        self.assertAlmostEqual(result, 9.0)

    def test_fva_positive(self):
        result = _fva(8.0, 10.0)
        self.assertAlmostEqual(result, 20.0)

    def test_fva_negative(self):
        result = _fva(12.0, 10.0)
        self.assertAlmostEqual(result, -20.0)

    def test_fva_zero_naive(self):
        result = _fva(5.0, 0.0)
        self.assertTrue(math.isnan(result))


class TestCQRIntervals(unittest.TestCase):

    def test_basic_cqr(self):
        """CQR should produce intervals with positive width."""
        # Calibration where some actuals fall outside quantile bounds
        cal_actual = [10.0, 20.0, 30.0, 40.0, 55.0]
        cal_lower = [8.0, 18.0, 28.0, 38.0, 48.0]
        cal_upper = [12.0, 22.0, 32.0, 42.0, 52.0]
        test_lower = [58.0, 68.0]
        test_upper = [62.0, 72.0]

        intervals = cqr_intervals(cal_actual, cal_lower, cal_upper,
                                   test_lower, test_upper, coverage=0.95)
        self.assertEqual(len(intervals), 2)
        for lo, hi in intervals:
            self.assertLess(lo, hi)

    def test_cqr_widens_intervals(self):
        """CQR correction should widen (or keep same) the original intervals."""
        cal_actual = [10.0, 20.0, 15.0, 25.0, 30.0]
        cal_lower = [12.0, 18.0, 17.0, 23.0, 28.0]  # some miss low
        cal_upper = [14.0, 22.0, 19.0, 27.0, 32.0]

        test_lower = [50.0]
        test_upper = [55.0]

        intervals = cqr_intervals(cal_actual, cal_lower, cal_upper,
                                   test_lower, test_upper, coverage=0.95)
        lo, hi = intervals[0]
        # CQR should widen: lo <= test_lower, hi >= test_upper
        self.assertLessEqual(lo, 50.0)
        self.assertGreaterEqual(hi, 55.0)

    def test_cqr_empty_calibration(self):
        """Empty calibration returns unadjusted intervals."""
        intervals = cqr_intervals([], [], [], [10.0], [20.0])
        self.assertEqual(len(intervals), 1)
        self.assertAlmostEqual(intervals[0][0], 10.0)
        self.assertAlmostEqual(intervals[0][1], 20.0)


class TestPhase4PermutationEntropy(unittest.TestCase):

    def test_phase4_includes_pe(self):
        """Phase 4 should produce a permutation_entropy finding on sufficient data."""
        import random
        random.seed(42)
        data = [100 + 10 * math.sin(2 * math.pi * i / 50) + random.gauss(0, 2)
                for i in range(300)]
        r = TimeSeriesReviewer(data, name="pe_test")
        ph = r.phase_forecastability()
        pe_findings = [f for f in ph.findings if f.check == "permutation_entropy"]
        self.assertEqual(len(pe_findings), 1)
        self.assertIn(pe_findings[0].verdict, (Verdict.PASS, Verdict.WARN))

    def test_phase4_skip_pe_short(self):
        """Phase 4 with <10 obs should skip entirely."""
        r = TimeSeriesReviewer(list(range(5)))
        ph = r.phase_forecastability()
        pe_findings = [f for f in ph.findings if f.check == "permutation_entropy"]
        self.assertEqual(len(pe_findings), 0)


class TestPhase6FVA(unittest.TestCase):

    def test_phase6_includes_fva(self):
        """Phase 6 should include FVA when predictions are supplied."""
        import random
        random.seed(42)
        data = [100 + random.gauss(0, 5) for _ in range(150)]
        actuals = data[-20:]
        # Good predictions (close to actual)
        predictions = [a + random.gauss(0, 1) for a in actuals]
        r = TimeSeriesReviewer(data, name="fva_test")
        ph = r.phase_baseline_benchmarks(predictions=predictions, actuals=actuals)
        fva_findings = [f for f in ph.findings if f.check == "fva"]
        self.assertEqual(len(fva_findings), 1)

    def test_phase6_no_fva_without_predictions(self):
        """Phase 6 without predictions should not have FVA."""
        import random
        random.seed(42)
        data = [100 + random.gauss(0, 5) for _ in range(150)]
        r = TimeSeriesReviewer(data, name="fva_test")
        ph = r.phase_baseline_benchmarks()
        fva_findings = [f for f in ph.findings if f.check == "fva"]
        self.assertEqual(len(fva_findings), 0)

    def test_phase6_bad_model_fva_fails(self):
        """Phase 6 with bad model should produce FVA < 0 (FAIL)."""
        import random
        random.seed(42)
        data = [100 + random.gauss(0, 2) for _ in range(150)]
        actuals = data[-20:]
        # Bad predictions — far off
        predictions = [a + 50 for a in actuals]
        r = TimeSeriesReviewer(data, name="fva_bad")
        ph = r.phase_baseline_benchmarks(predictions=predictions, actuals=actuals)
        fva_findings = [f for f in ph.findings if f.check == "fva"]
        self.assertEqual(len(fva_findings), 1)
        self.assertEqual(fva_findings[0].verdict, Verdict.FAIL)


class TestCompareModelsExtended(unittest.TestCase):

    def test_includes_new_metrics(self):
        """compare_models should include wape and me_bias."""
        actuals = [10.0, 20.0, 30.0, 40.0, 50.0]
        models = {
            "model_a": [12.0, 22.0, 28.0, 42.0, 48.0],
        }
        results = compare_models(actuals, models)
        self.assertIn("model_a", results)
        self.assertIn("wape", results["model_a"])
        self.assertIn("me_bias", results["model_a"])


if __name__ == '__main__':
    unittest.main()
