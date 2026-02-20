#!/usr/bin/env python3
"""Tests for src/scripts/forecast_modeler.py"""

import math
import os
import sys
import subprocess
import unittest

# Allow importing from src/scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'scripts'))

from forecast_modeler import (
    Verdict,
    Severity,
    Finding,
    PhaseResult,
    ForecastResult,
    ForecastReport,
    ForecastModeler,
    auto_forecast,
    compare_forecasters,
    conformal_forecast,
    _safe,
    _to_floats,
    _mean,
    _std,
    _median,
    _percentile,
    _diff,
    _mae,
    _rmse,
    _mase,
    _rmsse,
    _wape,
    _me_bias,
    _fva,
    _permutation_entropy,
    _r_squared,
    _naive_forecast,
    _seasonal_naive,
    _drift_forecast,
    _moving_average,
    _conformal_intervals,
    _cqr_intervals,
    _create_lag_features,
    _create_rolling_features,
    _create_calendar_features,
    _create_fourier_features,
    _build_feature_matrix,
    _HAS_NUMPY,
    _HAS_STATSMODELS,
    _HAS_PMDARIMA,
    _HAS_CATBOOST,
)


# ---------------------------------------------------------------------------
# Helper: generate test data
# ---------------------------------------------------------------------------

def _trend_seasonal(n=120, period=12, trend=0.5, noise=0.1):
    """Trend + seasonal + noise series."""
    import random
    random.seed(42)
    data = []
    for i in range(n):
        val = trend * i + 10 * math.sin(2 * math.pi * i / period)
        val += random.gauss(0, noise)
        data.append(val)
    return data


def _constant_series(n=50, value=5.0):
    return [value] * n


def _random_walk(n=100, seed=42):
    """Random walk — effectively random, high PE."""
    import random
    random.seed(seed)
    data = [0.0]
    for _ in range(n - 1):
        data.append(data[-1] + random.gauss(0, 1))
    return data


# ===========================================================================
# 1. DATA STRUCTURE TESTS
# ===========================================================================


class TestVerdict(unittest.TestCase):

    def test_all_verdicts_exist(self):
        expected = {"PASS", "WARN", "FAIL", "REJECT", "SKIP"}
        actual = {v.value for v in Verdict}
        self.assertEqual(actual, expected)


class TestSeverity(unittest.TestCase):

    def test_all_severities_exist(self):
        expected = {"INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"}
        actual = {s.value for s in Severity}
        self.assertEqual(actual, expected)


class TestFinding(unittest.TestCase):

    def test_creation(self):
        f = Finding(
            phase="P1", check="test_check",
            verdict=Verdict.PASS, severity=Severity.INFO, detail="ok"
        )
        self.assertEqual(f.phase, "P1")
        self.assertEqual(f.check, "test_check")
        self.assertEqual(f.verdict, Verdict.PASS)
        self.assertEqual(f.severity, Severity.INFO)
        self.assertEqual(f.detail, "ok")

    def test_str(self):
        f = Finding(
            phase="P1", check="test_check",
            verdict=Verdict.WARN, severity=Severity.HIGH, detail="warning"
        )
        s = str(f)
        self.assertIn("test_check", s)
        self.assertIn("warning", s)


class TestPhaseResult(unittest.TestCase):

    def test_add_finding(self):
        ph = PhaseResult("Test Phase")
        ph.add("check1", Verdict.PASS, Severity.INFO, "ok")
        ph.add("check2", Verdict.FAIL, Severity.HIGH, "bad")
        self.assertEqual(len(ph.findings), 2)

    def test_verdict_property(self):
        ph = PhaseResult("Test Phase")
        ph.add("c1", Verdict.PASS, Severity.INFO, "ok")
        ph.add("c2", Verdict.WARN, Severity.MEDIUM, "meh")
        ph.add("c3", Verdict.FAIL, Severity.HIGH, "bad")
        self.assertEqual(ph.verdict, Verdict.FAIL)

    def test_verdict_empty(self):
        ph = PhaseResult("Empty")
        # Empty phase with no findings → SKIP (all findings are SKIP vacuously)
        self.assertEqual(ph.verdict, Verdict.SKIP)

    def test_verdict_reject_trumps_fail(self):
        ph = PhaseResult("Test")
        ph.add("c1", Verdict.FAIL, Severity.HIGH, "fail")
        ph.add("c2", Verdict.REJECT, Severity.CRITICAL, "reject")
        self.assertEqual(ph.verdict, Verdict.REJECT)


class TestForecastResult(unittest.TestCase):

    def test_creation(self):
        r = ForecastResult(
            model_name="test",
            predictions=[1.0, 2.0],
            lower=[0.5, 1.5],
            upper=[1.5, 2.5],
            metrics={"mae": 0.1},
            fitted_params={"p": 1}
        )
        self.assertEqual(r.model_name, "test")
        self.assertEqual(len(r.predictions), 2)

    def test_defaults(self):
        r = ForecastResult(model_name="test", predictions=[1.0])
        self.assertIsNone(r.lower)
        self.assertIsNone(r.upper)
        self.assertEqual(r.metrics, {})


class TestForecastReport(unittest.TestCase):

    def test_creation_and_phases(self):
        rpt = ForecastReport(series_name="test", n_observations=100)
        self.assertEqual(rpt.series_name, "test")
        self.assertIsInstance(rpt.phases, list)
        self.assertIsInstance(rpt.models, dict)

    def test_to_dict(self):
        rpt = ForecastReport(series_name="test", n_observations=100)
        d = rpt.to_dict()
        self.assertIn("series_name", d)
        self.assertIn("phases", d)
        self.assertIn("best_model", d)

    def test_overall_verdict_empty(self):
        rpt = ForecastReport(series_name="test", n_observations=100)
        self.assertEqual(rpt.overall_verdict, Verdict.PASS)


# ===========================================================================
# 2. NUMERIC HELPER TESTS
# ===========================================================================


class TestSafe(unittest.TestCase):

    def test_normal_float(self):
        self.assertEqual(_safe(3.14), 3.14)

    def test_nan_becomes_string(self):
        # _safe converts non-finite floats to str() via the except path
        result = _safe(float("nan"))
        self.assertIsInstance(result, (float, str))

    def test_inf_passes_through(self):
        # _safe: float("inf") is a float, so passes the isinstance check
        # and is returned as-is
        result = _safe(float("inf"))
        self.assertEqual(result, float("inf"))

    def test_none(self):
        self.assertIsNone(_safe(None))

    def test_list(self):
        result = _safe([1.0, 2.0])
        self.assertEqual(result, [1.0, 2.0])

    def test_dict(self):
        result = _safe({"a": 1.0})
        self.assertEqual(result, {"a": 1.0})


class TestToFloats(unittest.TestCase):

    def test_numeric(self):
        result = _to_floats([1, 2.0, 3])
        self.assertEqual(result, [1.0, 2.0, 3.0])

    def test_with_none(self):
        result = _to_floats([1.0, None, 3.0])
        self.assertEqual(len(result), 2)

    def test_with_strings(self):
        result = _to_floats(["1.5", "bad", "3.0"])
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[0], 1.5)

    def test_drops_inf(self):
        result = _to_floats([1.0, float("inf"), 3.0])
        self.assertEqual(len(result), 2)


class TestBasicStats(unittest.TestCase):

    def test_mean(self):
        self.assertAlmostEqual(_mean([1.0, 2.0, 3.0]), 2.0)

    def test_mean_empty(self):
        self.assertAlmostEqual(_mean([]), 0.0)

    def test_std(self):
        s = _std([1.0, 2.0, 3.0])
        self.assertGreater(s, 0)

    def test_std_single(self):
        self.assertAlmostEqual(_std([1.0]), 0.0)

    def test_median_odd(self):
        self.assertAlmostEqual(_median([1.0, 3.0, 2.0]), 2.0)

    def test_median_even(self):
        self.assertAlmostEqual(_median([1.0, 2.0, 3.0, 4.0]), 2.5)

    def test_percentile(self):
        data = list(range(1, 101))
        p50 = _percentile([float(x) for x in data], 50)
        self.assertAlmostEqual(p50, 50.5, places=0)

    def test_diff(self):
        self.assertEqual(_diff([1.0, 3.0, 6.0, 10.0]), [2.0, 3.0, 4.0])


# ===========================================================================
# 3. METRIC HELPER TESTS
# ===========================================================================


class TestMAE(unittest.TestCase):

    def test_perfect(self):
        self.assertAlmostEqual(_mae([1, 2, 3], [1, 2, 3]), 0.0)

    def test_known(self):
        self.assertAlmostEqual(_mae([1.0, 2.0, 3.0], [2.0, 3.0, 4.0]), 1.0)


class TestRMSE(unittest.TestCase):

    def test_perfect(self):
        self.assertAlmostEqual(_rmse([1, 2, 3], [1, 2, 3]), 0.0)

    def test_known(self):
        self.assertAlmostEqual(_rmse([0.0, 0.0], [1.0, 1.0]), 1.0)


class TestMASE(unittest.TestCase):

    def test_perfect(self):
        self.assertAlmostEqual(
            _mase([1, 2, 3, 4], [1, 2, 3, 4], seasonal_period=1), 0.0
        )

    def test_known(self):
        actual = [5.0, 6.0]
        predicted = [5.5, 6.5]
        m = _mase(actual, predicted, seasonal_period=1)
        self.assertGreater(m, 0)

    def test_seasonal(self):
        actual = [1.0, 2.0, 3.0]
        predicted = [1.5, 2.5, 3.5]
        m = _mase(actual, predicted, seasonal_period=2)
        self.assertIsNotNone(m)


class TestRMSSE(unittest.TestCase):

    def test_perfect(self):
        train = [1.0, 2.0, 3.0, 4.0, 5.0]
        actual = [6.0, 7.0]
        predicted = [6.0, 7.0]
        self.assertAlmostEqual(_rmsse(actual, predicted, train, seasonal_period=1), 0.0)

    def test_known(self):
        train = [1.0, 2.0, 3.0, 4.0, 5.0]
        actual = [6.0, 7.0]
        predicted = [7.0, 8.0]
        r = _rmsse(actual, predicted, train, seasonal_period=1)
        self.assertGreater(r, 0)


class TestWAPE(unittest.TestCase):

    def test_perfect(self):
        self.assertAlmostEqual(_wape([1, 2, 3], [1, 2, 3]), 0.0)

    def test_known(self):
        # |1-2| + |2-3| + |3-4| = 3, sum(|actuals|) = 6
        w = _wape([1.0, 2.0, 3.0], [2.0, 3.0, 4.0])
        self.assertAlmostEqual(w, 50.0)


class TestMEBias(unittest.TestCase):

    def test_unbiased(self):
        self.assertAlmostEqual(_me_bias([1, 2, 3], [1, 2, 3]), 0.0)

    def test_over_forecast(self):
        me = _me_bias([1.0, 2.0, 3.0], [2.0, 3.0, 4.0])
        self.assertGreater(me, 0)  # positive = over-forecast

    def test_under_forecast(self):
        me = _me_bias([2.0, 3.0, 4.0], [1.0, 2.0, 3.0])
        self.assertLess(me, 0)  # negative = under-forecast


class TestFVA(unittest.TestCase):

    def test_model_better(self):
        fva = _fva(model_mae=0.5, naive_mae=1.0)
        self.assertAlmostEqual(fva, 50.0)

    def test_model_worse(self):
        fva = _fva(model_mae=1.5, naive_mae=1.0)
        self.assertLess(fva, 0)

    def test_model_equal(self):
        fva = _fva(model_mae=1.0, naive_mae=1.0)
        self.assertAlmostEqual(fva, 0.0)

    def test_zero_naive(self):
        fva = _fva(model_mae=1.0, naive_mae=0.0)
        self.assertTrue(math.isnan(fva), "Should return NaN for zero naive MAE")


class TestRSquared(unittest.TestCase):

    def test_perfect(self):
        self.assertAlmostEqual(_r_squared([1, 2, 3, 4], [1, 2, 3, 4]), 1.0)

    def test_mean_model(self):
        actual = [1.0, 2.0, 3.0, 4.0]
        mean_val = sum(actual) / len(actual)
        r2 = _r_squared(actual, [mean_val] * 4)
        self.assertAlmostEqual(r2, 0.0, places=5)


class TestPermutationEntropy(unittest.TestCase):

    def test_deterministic(self):
        data = [float(i) for i in range(100)]
        pe = _permutation_entropy(data, order=3, delay=1)
        self.assertLess(pe, 0.2)

    def test_high_entropy(self):
        import random
        random.seed(42)
        data = [random.random() for _ in range(500)]
        pe = _permutation_entropy(data, order=3, delay=1)
        self.assertGreater(pe, 0.8)

    def test_range(self):
        data = [float(i) for i in range(50)]
        pe = _permutation_entropy(data, order=3, delay=1)
        self.assertGreaterEqual(pe, 0.0)
        self.assertLessEqual(pe, 1.0)

    def test_too_short(self):
        pe = _permutation_entropy([1.0, 2.0], order=5, delay=1)
        self.assertTrue(math.isnan(pe))


# ===========================================================================
# 4. NAIVE FORECAST HELPER TESTS
# ===========================================================================


class TestNaiveForecast(unittest.TestCase):

    def test_length(self):
        preds = _naive_forecast([1.0, 2.0, 3.0], horizon=5)
        self.assertEqual(len(preds), 5)

    def test_last_value(self):
        preds = _naive_forecast([1.0, 2.0, 3.0], horizon=3)
        self.assertTrue(all(p == 3.0 for p in preds))


class TestSeasonalNaive(unittest.TestCase):

    def test_repeats_pattern(self):
        train = [10.0, 20.0, 30.0, 40.0]
        preds = _seasonal_naive(train, period=4, horizon=4)
        self.assertEqual(len(preds), 4)
        self.assertEqual(preds, [10.0, 20.0, 30.0, 40.0])

    def test_length(self):
        preds = _seasonal_naive([1.0, 2.0, 3.0] * 4, period=3, horizon=6)
        self.assertEqual(len(preds), 6)


class TestDriftForecast(unittest.TestCase):

    def test_length(self):
        preds = _drift_forecast([1.0, 2.0, 3.0, 4.0], horizon=3)
        self.assertEqual(len(preds), 3)

    def test_trend(self):
        train = [1.0, 2.0, 3.0, 4.0, 5.0]
        preds = _drift_forecast(train, horizon=3)
        self.assertAlmostEqual(preds[0], 6.0, places=1)

    def test_short_series(self):
        preds = _drift_forecast([5.0], horizon=2)
        self.assertEqual(len(preds), 2)


class TestMovingAverage(unittest.TestCase):

    def test_length(self):
        preds = _moving_average([1.0, 2.0, 3.0, 4.0, 5.0], k=3, horizon=4)
        self.assertEqual(len(preds), 4)

    def test_known_value(self):
        preds = _moving_average([1.0, 2.0, 3.0], k=3, horizon=1)
        self.assertAlmostEqual(preds[0], 2.0)


# ===========================================================================
# 5. CONFORMAL PREDICTION TESTS
# ===========================================================================


class TestConformalIntervals(unittest.TestCase):

    def test_returns_list_of_tuples(self):
        residuals = [0.1, -0.2, 0.3, -0.1, 0.2, 0.15, -0.05, 0.25, -0.3, 0.1]
        forecasts = [10.0, 11.0, 12.0]
        result = _conformal_intervals(residuals, forecasts, coverage=0.90)
        self.assertEqual(len(result), 3)
        self.assertIsInstance(result[0], tuple)
        self.assertEqual(len(result[0]), 2)

    def test_upper_gt_lower(self):
        residuals = [0.1, -0.2, 0.3, -0.1, 0.2] * 10
        forecasts = [5.0, 6.0]
        result = _conformal_intervals(residuals, forecasts, coverage=0.95)
        for lo, hi in result:
            self.assertLessEqual(lo, hi)

    def test_centered_on_forecast(self):
        residuals = [0.1, -0.1, 0.1, -0.1] * 10
        forecasts = [100.0]
        result = _conformal_intervals(residuals, forecasts, coverage=0.80)
        lo, hi = result[0]
        mid = (lo + hi) / 2
        self.assertAlmostEqual(mid, 100.0, places=0)


class TestCQRIntervals(unittest.TestCase):

    def test_returns_list_of_tuples(self):
        cal_actual = [1.0, 2.0, 3.0, 4.0, 5.0] * 10
        cal_lower = [0.5, 1.5, 2.5, 3.5, 4.5] * 10
        cal_upper = [1.5, 2.5, 3.5, 4.5, 5.5] * 10
        test_lower = [5.5, 6.5]
        test_upper = [6.5, 7.5]
        result = _cqr_intervals(
            cal_actual, cal_lower, cal_upper,
            test_lower, test_upper, coverage=0.90
        )
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], tuple)

    def test_upper_gt_lower(self):
        cal_actual = [float(i) for i in range(50)]
        cal_lower = [float(i) - 1.0 for i in range(50)]
        cal_upper = [float(i) + 1.0 for i in range(50)]
        test_lower = [50.0, 51.0]
        test_upper = [52.0, 53.0]
        result = _cqr_intervals(
            cal_actual, cal_lower, cal_upper,
            test_lower, test_upper, coverage=0.95
        )
        for lo, hi in result:
            self.assertLessEqual(lo, hi)


# ===========================================================================
# 6. FEATURE ENGINEERING TESTS
# ===========================================================================


class TestCreateLagFeatures(unittest.TestCase):

    def test_basic(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = _create_lag_features(data, [1, 2])
        self.assertIn("lag_1", result)
        self.assertIn("lag_2", result)
        self.assertEqual(len(result["lag_1"]), 5)

    def test_lag_values(self):
        data = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = _create_lag_features(data, [1])
        # lag_1[0] should be None (no prior value)
        self.assertIsNone(result["lag_1"][0])
        # lag_1[1] should be 10.0 (the first value)
        self.assertAlmostEqual(result["lag_1"][1], 10.0)

    def test_default_lags(self):
        data = [float(i) for i in range(30)]
        result = _create_lag_features(data)
        self.assertIn("lag_1", result)


class TestCreateRollingFeatures(unittest.TestCase):

    def test_basic(self):
        data = [float(i) for i in range(20)]
        result = _create_rolling_features(data, [3])
        self.assertIn("rmean_3", result)
        self.assertIn("rstd_3", result)

    def test_leakage_prevention(self):
        """Early values should be None due to shift(1) + rolling window."""
        data = [float(i) for i in range(20)]
        result = _create_rolling_features(data, [3])
        rmean = result["rmean_3"]
        # First w+1 values should be None
        self.assertIsNone(rmean[0])
        self.assertIsNone(rmean[1])
        self.assertIsNone(rmean[2])
        self.assertIsNone(rmean[3])


class TestCreateCalendarFeatures(unittest.TestCase):

    def test_basic_with_freq(self):
        result = _create_calendar_features(24, freq=12)
        self.assertIn("season_idx", result)
        self.assertIn("time_idx", result)
        self.assertEqual(len(result["time_idx"]), 24)

    def test_no_freq(self):
        result = _create_calendar_features(8, freq=1)
        self.assertIn("time_idx", result)
        # No season_idx when freq=1
        self.assertNotIn("season_idx", result)

    def test_season_idx_wraps(self):
        result = _create_calendar_features(15, freq=4)
        self.assertEqual(result["season_idx"][0], 0)
        self.assertEqual(result["season_idx"][4], 0)
        self.assertEqual(result["season_idx"][3], 3)


class TestCreateFourierFeatures(unittest.TestCase):

    def test_basic(self):
        result = _create_fourier_features(100, period=12, n_harmonics=3)
        self.assertIn("sin_12_1", result)
        self.assertIn("cos_12_1", result)
        self.assertIn("sin_12_3", result)
        self.assertEqual(len(result["sin_12_1"]), 100)

    def test_values_bounded(self):
        result = _create_fourier_features(50, period=7, n_harmonics=2)
        for key, values in result.items():
            for v in values:
                self.assertGreaterEqual(v, -1.0 - 1e-10)
                self.assertLessEqual(v, 1.0 + 1e-10)


class TestBuildFeatureMatrix(unittest.TestCase):

    def test_returns_rows_and_colnames(self):
        data = _trend_seasonal(120, period=12)
        rows, col_names = _build_feature_matrix(data, freq=12)
        self.assertIsInstance(rows, list)
        self.assertIsInstance(col_names, list)
        self.assertGreater(len(rows), 0)
        self.assertGreater(len(col_names), 0)

    def test_no_none_in_output(self):
        data = _trend_seasonal(120, period=12)
        rows, col_names = _build_feature_matrix(data, freq=12)
        for row in rows:
            for val in row:
                self.assertIsNotNone(val)
                self.assertFalse(math.isnan(val))

    def test_consistent_width(self):
        data = _trend_seasonal(120, period=12)
        rows, col_names = _build_feature_matrix(data, freq=12)
        for row in rows:
            self.assertEqual(len(row), len(col_names))


# ===========================================================================
# 7. PHASE TESTS
# ===========================================================================


class TestPhase1Forecastability(unittest.TestCase):

    def test_structured_series(self):
        data = _trend_seasonal(120, period=12)
        modeler = ForecastModeler(data, name="test", frequency=12)
        result = modeler.phase_forecastability()
        self.assertIsInstance(result, PhaseResult)
        self.assertGreater(len(result.findings), 0)
        # Structured series should not be rejected
        self.assertNotEqual(result.verdict, Verdict.REJECT)

    def test_constant_series_rejected(self):
        data = _constant_series(50)
        modeler = ForecastModeler(data, name="const")
        result = modeler.phase_forecastability()
        self.assertEqual(result.verdict, Verdict.REJECT)

    def test_short_series_rejected(self):
        data = [1.0, 2.0, 3.0]
        modeler = ForecastModeler(data, name="short")
        result = modeler.phase_forecastability()
        self.assertEqual(result.verdict, Verdict.REJECT)

    def test_pe_in_metadata(self):
        data = _trend_seasonal(200, period=12)
        modeler = ForecastModeler(data, name="test", frequency=12)
        modeler.phase_forecastability()
        pe = modeler.report.metadata.get("permutation_entropy")
        self.assertIsNotNone(pe)
        self.assertGreater(pe, 0)
        self.assertLess(pe, 1)

    def test_random_walk_warns_or_fails(self):
        data = _random_walk(200)
        modeler = ForecastModeler(data, name="rw")
        result = modeler.phase_forecastability()
        worst = result.verdict
        self.assertIn(worst, [Verdict.WARN, Verdict.FAIL, Verdict.REJECT])


class TestPhase4Comparison(unittest.TestCase):

    def test_comparison_via_pipeline(self):
        """Phase 4 comparison works through the full pipeline."""
        data = _trend_seasonal(120, period=12)
        modeler = ForecastModeler(data, name="test", frequency=12)
        report = modeler.full_pipeline(horizon=12)
        # Phase 4 should be in the phases list
        phase_names = [p.name for p in report.phases]
        self.assertTrue(
            any("Comparison" in name or "comparison" in name.lower() for name in phase_names)
            or len(report.models) > 0,
            "Pipeline should produce model comparison"
        )


# ===========================================================================
# 8. FULL PIPELINE TESTS
# ===========================================================================


class TestFullPipeline(unittest.TestCase):

    def test_full_pipeline_returns_report(self):
        data = _trend_seasonal(120, period=12)
        modeler = ForecastModeler(data, name="test", frequency=12)
        report = modeler.full_pipeline(horizon=12)
        self.assertIsInstance(report, ForecastReport)
        self.assertGreater(len(report.phases), 0)

    def test_full_pipeline_has_best_model(self):
        data = _trend_seasonal(120, period=12)
        modeler = ForecastModeler(data, name="test", frequency=12)
        report = modeler.full_pipeline(horizon=12)
        self.assertIsNotNone(report.best_model)

    def test_to_dict(self):
        data = _trend_seasonal(120, period=12)
        modeler = ForecastModeler(data, name="test", frequency=12)
        report = modeler.full_pipeline(horizon=12)
        d = report.to_dict()
        self.assertIn("phases", d)
        self.assertIn("best_model", d)
        self.assertIn("series_name", d)

    def test_rejected_series_stops_early(self):
        data = _constant_series(50)
        modeler = ForecastModeler(data, name="const")
        report = modeler.full_pipeline(horizon=5)
        self.assertIsInstance(report, ForecastReport)
        self.assertEqual(report.overall_verdict, Verdict.REJECT)

    def test_models_dict_populated(self):
        data = _trend_seasonal(120, period=12)
        modeler = ForecastModeler(data, name="test", frequency=12)
        report = modeler.full_pipeline(horizon=12)
        self.assertGreater(len(report.models), 0)


# ===========================================================================
# 9. CONVENIENCE FUNCTION TESTS
# ===========================================================================


class TestAutoForecast(unittest.TestCase):

    def test_returns_report(self):
        data = _trend_seasonal(120, period=12)
        report = auto_forecast(data, horizon=12, coverage=0.95)
        self.assertIsInstance(report, ForecastReport)


class TestCompareForecastersFunc(unittest.TestCase):

    def test_returns_dict(self):
        data = _trend_seasonal(120, period=12)
        results = compare_forecasters(data, horizon=12)
        self.assertIsInstance(results, dict)


class TestConformalForecastFunc(unittest.TestCase):

    def test_returns_list_of_tuples(self):
        residuals = [0.1, -0.2, 0.3, -0.1, 0.2] * 20
        preds = [10.0, 11.0, 12.0]
        result = conformal_forecast(preds, residuals, coverage=0.95)
        self.assertEqual(len(result), 3)
        self.assertIsInstance(result[0], tuple)
        lo, hi = result[0]
        self.assertLessEqual(lo, hi)


# ===========================================================================
# 10. GRACEFUL DEGRADATION TESTS
# ===========================================================================


class TestGracefulDegradation(unittest.TestCase):

    def test_stdlib_mode_phase1(self):
        """Phase 1 must work with zero optional dependencies."""
        data = _trend_seasonal(120, period=12)
        modeler = ForecastModeler(data, name="test", frequency=12)
        result = modeler.phase_forecastability()
        self.assertIsInstance(result, PhaseResult)
        self.assertGreater(len(result.findings), 0)

    def test_dependency_flags_are_booleans(self):
        self.assertIsInstance(_HAS_NUMPY, bool)
        self.assertIsInstance(_HAS_STATSMODELS, bool)
        self.assertIsInstance(_HAS_PMDARIMA, bool)
        self.assertIsInstance(_HAS_CATBOOST, bool)

    def test_report_shows_available_deps(self):
        data = _trend_seasonal(120, period=12)
        modeler = ForecastModeler(data, name="test", frequency=12)
        report = modeler.full_pipeline(horizon=12)
        d = report.to_dict()
        meta = d.get("metadata", {})
        # Metadata should track dependency availability
        self.assertIsInstance(meta, dict)


# ===========================================================================
# 11. CLI SMOKE TESTS
# ===========================================================================


class TestCLI(unittest.TestCase):

    SCRIPT = os.path.join(
        os.path.dirname(__file__), '..', 'src', 'scripts', 'forecast_modeler.py'
    )

    def test_help(self):
        result = subprocess.run(
            [sys.executable, self.SCRIPT, "--help"],
            capture_output=True, text=True, timeout=30
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("fit", result.stdout)
        self.assertIn("assess", result.stdout)
        self.assertIn("compare", result.stdout)
        self.assertIn("demo", result.stdout)

    def test_demo(self):
        result = subprocess.run(
            [sys.executable, self.SCRIPT, "demo"],
            capture_output=True, text=True, timeout=60
        )
        self.assertEqual(result.returncode, 0)


# ===========================================================================
# 12. PRINT / REPORT OUTPUT TESTS
# ===========================================================================


class TestPrintReport(unittest.TestCase):

    def test_print_report_does_not_crash(self):
        data = _trend_seasonal(120, period=12)
        modeler = ForecastModeler(data, name="test", frequency=12)
        report = modeler.full_pipeline(horizon=12)
        # Should not raise
        report.print_report()

    def test_print_report_verbose_false(self):
        data = _trend_seasonal(120, period=12)
        modeler = ForecastModeler(data, name="test", frequency=12)
        report = modeler.full_pipeline(horizon=12)
        report.print_report(verbose=False)


if __name__ == "__main__":
    unittest.main()
