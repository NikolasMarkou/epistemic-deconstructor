#!/usr/bin/env python3
"""
ts_reviewer.py — Time Series Review & Evaluation Framework
============================================================
Systematic validation framework for unknown time series produced by systems.

Synthesizes:
  - Forecast-checker skill: phased evaluation, red flags, overfitting/leakage
    detection, baseline requirements, domain calibration, uncertainty checks
  - Visual forecasting guide: decomposition, stationarity, forecastability
    spectrum, residual diagnostics, walk-forward validation, regime analysis

Usage:
    from ts_reviewer import TimeSeriesReviewer

    reviewer = TimeSeriesReviewer(data, name="sensor_output")
    report = reviewer.full_review()
    report.print_report()
    d = report.to_dict()

    # Or run individual phases:
    reviewer.phase_coherence()
    reviewer.phase_data_quality()
    reviewer.phase_stationarity()
    reviewer.phase_forecastability()
    reviewer.phase_decomposition()
    reviewer.phase_baseline_benchmarks()
    reviewer.phase_overfitting_screen(predictions, actuals, train_preds, train_actuals)
    reviewer.phase_residual_diagnostics(residuals)
    reviewer.phase_uncertainty(pred_intervals_lower, pred_intervals_upper, actuals)
    reviewer.phase_regime_analysis()

Dependencies (graceful degradation):
    Required : (none — pure stdlib fallback works)
    Optional : numpy, scipy, statsmodels  (enables full statistical tests)
"""

from __future__ import annotations

import math
import statistics
import warnings
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Optional imports — framework degrades gracefully
# ---------------------------------------------------------------------------
try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    _HAS_NUMPY = False

try:
    from scipy import stats as sp_stats
    from scipy.signal import periodogram as _periodogram

    _HAS_SCIPY = True
except ImportError:
    sp_stats = None  # type: ignore
    _HAS_SCIPY = False

try:
    from statsmodels.tsa.stattools import adfuller, kpss, acf as sm_acf
    from statsmodels.tsa.seasonal import STL
    from statsmodels.stats.diagnostic import acorr_ljungbox

    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False


# ===========================================================================
# 1. ENUMS & DATA STRUCTURES
# ===========================================================================


class Verdict(Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    REJECT = "REJECT"
    SKIP = "SKIP"


class Severity(Enum):
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class Finding:
    """Single evaluation finding."""

    phase: str
    check: str
    verdict: Verdict
    severity: Severity
    detail: str
    value: Any = None

    def __str__(self) -> str:
        icons = {
            "PASS": "[OK]",
            "WARN": "[!!]",
            "FAIL": "[XX]",
            "REJECT": "[RJ]",
            "SKIP": "[--]",
        }
        return f"  {icons[self.verdict.value]} [{self.severity.value:8s}] {self.check}: {self.detail}"


@dataclass
class PhaseResult:
    """Aggregated result for one evaluation phase."""

    name: str
    findings: List[Finding] = field(default_factory=list)

    @property
    def verdict(self) -> Verdict:
        for v in (Verdict.REJECT, Verdict.FAIL, Verdict.WARN):
            if any(f.verdict == v for f in self.findings):
                return v
        if all(f.verdict == Verdict.SKIP for f in self.findings):
            return Verdict.SKIP
        return Verdict.PASS

    def add(
        self,
        check: str,
        verdict: Verdict,
        severity: Severity,
        detail: str,
        value: Any = None,
    ) -> Finding:
        f = Finding(self.name, check, verdict, severity, detail, value)
        self.findings.append(f)
        return f


@dataclass
class ReviewReport:
    """Complete review output across all phases."""

    series_name: str
    n_observations: int
    phases: List[PhaseResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- verdicts ----------------------------------------------------------

    @property
    def overall_verdict(self) -> Verdict:
        for v in (Verdict.REJECT, Verdict.FAIL, Verdict.WARN):
            if any(p.verdict == v for p in self.phases):
                return v
        return Verdict.PASS

    @property
    def all_findings(self) -> List[Finding]:
        return [f for p in self.phases for f in p.findings]

    @property
    def red_flags(self) -> List[Finding]:
        return [f for f in self.all_findings if f.verdict in (Verdict.FAIL, Verdict.REJECT)]

    @property
    def warnings(self) -> List[Finding]:
        return [f for f in self.all_findings if f.verdict == Verdict.WARN]

    # -- output ------------------------------------------------------------

    def print_report(self, *, verbose: bool = True) -> None:
        w = 72
        print(f"\n{'=' * w}")
        print(f"  TIME SERIES REVIEW: {self.series_name}")
        print(f"  Observations : {self.n_observations}")
        print(f"  Overall      : {self.overall_verdict.value}")
        print(f"  Red flags    : {len(self.red_flags)}")
        print(f"  Warnings     : {len(self.warnings)}")
        print(f"{'=' * w}")

        for phase in self.phases:
            print(f"\n--- {phase.name} [{phase.verdict.value}] ---")
            items = phase.findings if verbose else [
                f for f in phase.findings if f.verdict != Verdict.PASS
            ]
            for f in items:
                print(str(f))

        flags = self.red_flags
        if flags:
            print(f"\n{'!' * w}")
            print(f"  RED FLAGS ({len(flags)}):")
            for f in flags:
                print(f"    [{f.phase}] {f.check}: {f.detail}")
            print(f"{'!' * w}")
        print()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "series_name": self.series_name,
            "n_observations": self.n_observations,
            "overall_verdict": self.overall_verdict.value,
            "red_flag_count": len(self.red_flags),
            "warning_count": len(self.warnings),
            "metadata": _safe(self.metadata),
            "phases": [
                {
                    "name": p.name,
                    "verdict": p.verdict.value,
                    "findings": [
                        {
                            "check": f.check,
                            "verdict": f.verdict.value,
                            "severity": f.severity.value,
                            "detail": f.detail,
                            "value": _safe(f.value),
                        }
                        for f in p.findings
                    ],
                }
                for p in self.phases
            ],
        }


def _safe(v: Any) -> Any:
    if v is None or isinstance(v, (int, float, str, bool)):
        return v
    if isinstance(v, (list, tuple)):
        return [_safe(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _safe(val) for k, val in v.items()}
    try:
        f = float(v)
        return f if math.isfinite(f) else str(v)
    except Exception:
        return str(v)


# ===========================================================================
# 2. NUMERIC HELPERS (stdlib fallbacks when numpy/scipy absent)
# ===========================================================================


def _to_floats(data: Sequence) -> List[float]:
    """Coerce to list[float], dropping None / NaN / Inf."""
    out: List[float] = []
    for x in data:
        if x is None:
            continue
        try:
            val = float(x)
        except (TypeError, ValueError):
            continue
        if math.isfinite(val):
            out.append(val)
    return out


def _arr(data: List[float]):
    return np.asarray(data, dtype=np.float64) if _HAS_NUMPY else data


def _mean(d: List[float]) -> float:
    if not d:
        return 0.0
    return float(np.mean(d)) if _HAS_NUMPY else statistics.mean(d)


def _std(d: List[float], *, ddof: int = 1) -> float:
    if len(d) < 2:
        return 0.0
    if _HAS_NUMPY:
        return float(np.std(d, ddof=ddof))
    return statistics.stdev(d) if ddof == 1 else statistics.pstdev(d)


def _median(d: List[float]) -> float:
    if not d:
        return 0.0
    return float(np.median(d)) if _HAS_NUMPY else statistics.median(d)


def _percentile(d: List[float], q: float) -> float:
    if not d:
        return 0.0
    if _HAS_NUMPY:
        return float(np.percentile(d, q))
    sd = sorted(d)
    k = (len(sd) - 1) * q / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sd[int(k)]
    return sd[f] * (c - k) + sd[c] * (k - f)


def _diff(d: List[float]) -> List[float]:
    return [d[i] - d[i - 1] for i in range(1, len(d))]


def _log_returns(d: List[float]) -> List[float]:
    out: List[float] = []
    for i in range(1, len(d)):
        if d[i - 1] > 0 and d[i] > 0:
            out.append(math.log(d[i] / d[i - 1]))
    return out


def _autocorr(d: List[float], max_lag: int = 40) -> List[float]:
    """Sample autocorrelation via statsmodels or manual."""
    if _HAS_STATSMODELS:
        n_lags = min(max_lag, len(d) // 2 - 1)
        if n_lags < 1:
            return []
        return list(sm_acf(d, nlags=n_lags, fft=len(d) > 500))
    # manual fallback
    n = len(d)
    m = _mean(d)
    var = sum((x - m) ** 2 for x in d)
    if var == 0:
        return [1.0] + [0.0] * min(max_lag, n // 2)
    acfs = [1.0]
    for lag in range(1, min(max_lag, n // 2) + 1):
        c = sum((d[i] - m) * (d[i - lag] - m) for i in range(lag, n))
        acfs.append(c / var)
    return acfs


def _r_squared(actual: List[float], predicted: List[float]) -> float:
    n = min(len(actual), len(predicted))
    if n < 2:
        return float("nan")
    a, p = actual[:n], predicted[:n]
    m = _mean(a)
    ss_res = sum((a[i] - p[i]) ** 2 for i in range(n))
    ss_tot = sum((a[i] - m) ** 2 for i in range(n))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _mae(actual: List[float], predicted: List[float]) -> float:
    n = min(len(actual), len(predicted))
    return sum(abs(actual[i] - predicted[i]) for i in range(n)) / n


def _rmse(actual: List[float], predicted: List[float]) -> float:
    n = min(len(actual), len(predicted))
    return math.sqrt(sum((actual[i] - predicted[i]) ** 2 for i in range(n)) / n)


def _mase(actual: List[float], predicted: List[float], seasonal_period: int = 1) -> float:
    """MASE: MAE of model / MAE of naive (seasonal) baseline."""
    n = min(len(actual), len(predicted))
    if n < seasonal_period + 1:
        return float("nan")
    model_mae = sum(abs(actual[i] - predicted[i]) for i in range(n)) / n
    naive_errors = [
        abs(actual[i] - actual[i - seasonal_period])
        for i in range(seasonal_period, len(actual))
    ]
    if not naive_errors:
        return float("nan")
    naive_mae = sum(naive_errors) / len(naive_errors)
    if naive_mae == 0:
        return float("nan")
    return model_mae / naive_mae


def _iqr_bounds(d: List[float], k: float = 1.5) -> Tuple[float, float]:
    q1, q3 = _percentile(d, 25), _percentile(d, 75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def _rmsse(
    actual: List[float],
    predicted: List[float],
    train_data: List[float],
    seasonal_period: int = 1,
) -> float:
    """Root Mean Squared Scaled Error (M5 competition metric)."""
    n = min(len(actual), len(predicted))
    if n < 1 or len(train_data) <= seasonal_period:
        return float("nan")
    mse_model = sum((actual[i] - predicted[i]) ** 2 for i in range(n)) / n
    naive_sq = [
        (train_data[i] - train_data[i - seasonal_period]) ** 2
        for i in range(seasonal_period, len(train_data))
    ]
    if not naive_sq:
        return float("nan")
    scale = sum(naive_sq) / len(naive_sq)
    if scale == 0:
        return float("nan")
    return math.sqrt(mse_model / scale)


def _wape(actual: List[float], predicted: List[float]) -> float:
    """Weighted Absolute Percentage Error."""
    n = min(len(actual), len(predicted))
    if n < 1:
        return float("nan")
    total_abs_actual = sum(abs(actual[i]) for i in range(n))
    if total_abs_actual == 0:
        return float("nan")
    total_abs_error = sum(abs(actual[i] - predicted[i]) for i in range(n))
    return total_abs_error / total_abs_actual * 100


def _me_bias(actual: List[float], predicted: List[float]) -> float:
    """Mean Error / forecast bias. Positive = over-forecast."""
    n = min(len(actual), len(predicted))
    if n < 1:
        return float("nan")
    return sum(predicted[i] - actual[i] for i in range(n)) / n


def _pinball_loss(actual: List[float], quantile_pred: List[float], tau: float) -> float:
    """Pinball (quantile) loss for a single quantile level tau."""
    n = min(len(actual), len(quantile_pred))
    if n < 1:
        return float("nan")
    total = 0.0
    for i in range(n):
        residual = actual[i] - quantile_pred[i]
        if residual >= 0:
            total += tau * residual
        else:
            total += (tau - 1) * residual
    return total / n


def _fva(model_mae: float, naive_mae: float) -> float:
    """Forecast Value Added (%). Positive = model beats naive."""
    if naive_mae == 0:
        return float("nan")
    return (naive_mae - model_mae) / naive_mae * 100


def _permutation_entropy(d: List[float], order: int = 3, delay: int = 1) -> float:
    """
    Permutation entropy, normalized to [0, 1].

    Measures ordinal-pattern complexity. Lower = more predictable.
    Pure-stdlib implementation (no numpy/scipy required).

    Parameters
    ----------
    d : list of float
        Time series values.
    order : int
        Embedding dimension (number of consecutive values per pattern).
    delay : int
        Embedding time delay.

    Returns
    -------
    float
        Normalized permutation entropy in [0, 1], or nan if insufficient data.
    """
    n = len(d)
    # Need at least (order - 1) * delay + 1 values to form one pattern
    if n < (order - 1) * delay + 1:
        return float("nan")

    # Count ordinal patterns
    pattern_counts: Dict[Tuple[int, ...], int] = {}
    total = 0
    for i in range(n - (order - 1) * delay):
        # Extract subsequence with given delay
        subseq = [d[i + j * delay] for j in range(order)]
        # Compute rank pattern (argsort)
        indexed = sorted(range(order), key=lambda k: subseq[k])
        pattern = tuple(indexed)
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        total += 1

    if total == 0:
        return float("nan")

    # Shannon entropy over pattern frequencies
    entropy = 0.0
    for count in pattern_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log(p)

    # Normalize by max possible entropy = log(order!)
    max_entropy = math.log(math.factorial(order))
    if max_entropy == 0:
        return float("nan")
    return entropy / max_entropy


# ===========================================================================
# 3. THE REVIEWER
# ===========================================================================


class TimeSeriesReviewer:
    """
    Systematic evaluator for unknown time series.

    Phases (mirrors forecast-checker + visual-guide workflow):
        1. Coherence         — basic sanity, length, type, constant checks
        2. Data Quality      — missing %, outliers, gaps, duplicates
        3. Stationarity      — ADF / KPSS / visual heuristics
        4. Forecastability    — autocorrelation strength, entropy proxy,
                               signal-to-noise, forecastability spectrum
        5. Decomposition     — trend strength, seasonality strength,
                               residual fraction, structural breaks
        6. Baseline Benchmarks — naive, seasonal naive, drift; MASE
        7. Overfitting Screen — R-squared, train-vs-test gap, memorization,
                               leakage indicators  (needs predictions)
        8. Residual Diagnostics — zero mean, no autocorrelation,
                               homoscedasticity, normality
        9. Uncertainty        — coverage calibration, interval width,
                               sharpness
       10. Regime Analysis    — volatility clustering, structural breaks,
                               distribution shifts across windows

    Parameters
    ----------
    data : sequence of float-like
        The raw time series values in temporal order.
    name : str
        Human-readable label for reports.
    frequency : int | None
        Known seasonal period (e.g. 12 for monthly, 7 for daily-weekly).
        If None, the framework tries to infer it.
    timestamps : sequence | None
        Optional aligned timestamps (for gap detection).
    """

    def __init__(
        self,
        data: Sequence,
        *,
        name: str = "unknown_series",
        frequency: Optional[int] = None,
        timestamps: Optional[Sequence] = None,
    ):
        self._raw = list(data)
        self._clean: List[float] = _to_floats(data)
        self.name = name
        self.frequency = frequency
        self.timestamps = list(timestamps) if timestamps else None
        self._report = ReviewReport(series_name=name, n_observations=len(self._clean))
        self._diffs: Optional[List[float]] = None
        self._acf: Optional[List[float]] = None

    # -- cached helpers ----------------------------------------------------

    @property
    def diffs(self) -> List[float]:
        if self._diffs is None:
            self._diffs = _diff(self._clean)
        return self._diffs

    @property
    def acf_values(self) -> List[float]:
        if self._acf is None:
            self._acf = _autocorr(self._clean)
        return self._acf

    @property
    def report(self) -> ReviewReport:
        return self._report

    # ======================================================================
    # PHASE 1: COHERENCE
    # ======================================================================

    def phase_coherence(self) -> PhaseResult:
        """Basic sanity: length, type, constant, range plausibility."""
        ph = PhaseResult("Phase 1: Coherence")

        raw_len = len(self._raw)
        clean_len = len(self._clean)
        dropped = raw_len - clean_len

        # minimum length
        if clean_len < 10:
            ph.add("min_length", Verdict.REJECT, Severity.CRITICAL,
                    f"Only {clean_len} valid observations — cannot evaluate", clean_len)
        elif clean_len < 30:
            ph.add("min_length", Verdict.WARN, Severity.MEDIUM,
                    f"{clean_len} observations is marginal; most methods need >=50", clean_len)
        else:
            ph.add("min_length", Verdict.PASS, Severity.INFO,
                    f"{clean_len} observations", clean_len)

        # non-numeric drop rate
        drop_pct = dropped / raw_len * 100 if raw_len > 0 else 0
        if drop_pct > 20:
            ph.add("parse_failures", Verdict.FAIL, Severity.HIGH,
                    f"{drop_pct:.1f}% of values unparseable — data pipeline issue", drop_pct)
        elif drop_pct > 5:
            ph.add("parse_failures", Verdict.WARN, Severity.MEDIUM,
                    f"{drop_pct:.1f}% of values unparseable", drop_pct)
        else:
            ph.add("parse_failures", Verdict.PASS, Severity.INFO,
                    f"{drop_pct:.1f}% unparseable", drop_pct)

        # constant series
        if clean_len > 1:
            uniq = len(set(self._clean))
            if uniq == 1:
                ph.add("constant_series", Verdict.REJECT, Severity.CRITICAL,
                        "Series is constant — zero information content")
            elif uniq <= 3:
                ph.add("low_cardinality", Verdict.WARN, Severity.MEDIUM,
                        f"Only {uniq} unique values — may be categorical, not continuous", uniq)
            else:
                ph.add("value_diversity", Verdict.PASS, Severity.INFO,
                        f"{uniq} unique values", uniq)

        # range plausibility (detect clipped / stuck sensors)
        if clean_len > 10:
            mn, mx = min(self._clean), max(self._clean)
            rng = mx - mn
            if rng == 0:
                pass  # handled above
            else:
                top_pct = sum(1 for v in self._clean if v == mx) / clean_len
                bot_pct = sum(1 for v in self._clean if v == mn) / clean_len
                if top_pct > 0.10 or bot_pct > 0.10:
                    ph.add("value_clipping", Verdict.WARN, Severity.MEDIUM,
                            f"Min or max repeated >10% of the time — possible sensor clipping",
                            {"min_pct": round(bot_pct * 100, 1), "max_pct": round(top_pct * 100, 1)})
                else:
                    ph.add("value_clipping", Verdict.PASS, Severity.INFO, "No clipping detected")

        self._report.phases.append(ph)
        return ph

    # ======================================================================
    # PHASE 2: DATA QUALITY
    # ======================================================================

    def phase_data_quality(self) -> PhaseResult:
        """Missing values, outliers, duplicate timestamps, gap analysis."""
        ph = PhaseResult("Phase 2: Data Quality")
        d = self._clean
        n = len(d)

        # --- missing / None / NaN in raw --------------------------------
        n_missing = len(self._raw) - n
        miss_pct = n_missing / len(self._raw) * 100 if self._raw else 0
        if miss_pct > 20:
            ph.add("missing_values", Verdict.FAIL, Severity.HIGH,
                    f"{miss_pct:.1f}% missing — reconsider data source", miss_pct)
        elif miss_pct > 5:
            ph.add("missing_values", Verdict.WARN, Severity.MEDIUM,
                    f"{miss_pct:.1f}% missing — model-based imputation recommended", miss_pct)
        else:
            ph.add("missing_values", Verdict.PASS, Severity.INFO,
                    f"{miss_pct:.1f}% missing", miss_pct)

        # --- outliers (IQR) ---------------------------------------------
        if n >= 20:
            lo, hi = _iqr_bounds(d)
            outliers = [v for v in d if v < lo or v > hi]
            out_pct = len(outliers) / n * 100
            if out_pct > 10:
                ph.add("outliers_iqr", Verdict.WARN, Severity.MEDIUM,
                        f"{out_pct:.1f}% outliers (IQR method) — heavy tails or data issues",
                        {"pct": round(out_pct, 2), "bounds": (round(lo, 4), round(hi, 4))})
            else:
                ph.add("outliers_iqr", Verdict.PASS, Severity.INFO,
                        f"{out_pct:.1f}% outliers (IQR)", round(out_pct, 2))

            # extreme outliers (3x IQR)
            lo3, hi3 = _iqr_bounds(d, k=3.0)
            extremes = [v for v in d if v < lo3 or v > hi3]
            if extremes:
                ph.add("extreme_outliers", Verdict.WARN, Severity.HIGH,
                        f"{len(extremes)} extreme outliers (3xIQR) — investigate individually",
                        len(extremes))
            else:
                ph.add("extreme_outliers", Verdict.PASS, Severity.INFO,
                        "No extreme outliers (3xIQR)")
        else:
            ph.add("outliers_iqr", Verdict.SKIP, Severity.INFO,
                    "Too few observations for outlier analysis")

        # --- duplicate timestamps ----------------------------------------
        if self.timestamps and len(self.timestamps) == len(self._raw):
            dupes = len(self.timestamps) - len(set(self.timestamps))
            if dupes > 0:
                ph.add("duplicate_timestamps", Verdict.WARN, Severity.MEDIUM,
                        f"{dupes} duplicate timestamps detected", dupes)
            else:
                ph.add("duplicate_timestamps", Verdict.PASS, Severity.INFO,
                        "No duplicate timestamps")

        # --- consecutive identical values (stuck sensor) -----------------
        if n > 10:
            max_run = 1
            cur_run = 1
            for i in range(1, n):
                if d[i] == d[i - 1]:
                    cur_run += 1
                    max_run = max(max_run, cur_run)
                else:
                    cur_run = 1
            stuck_pct = max_run / n * 100
            if stuck_pct > 10:
                ph.add("stuck_values", Verdict.WARN, Severity.HIGH,
                        f"Longest run of identical values: {max_run} ({stuck_pct:.1f}%) — possible stuck sensor",
                        max_run)
            elif max_run > max(5, n * 0.02):
                ph.add("stuck_values", Verdict.WARN, Severity.LOW,
                        f"Longest identical-value run: {max_run}", max_run)
            else:
                ph.add("stuck_values", Verdict.PASS, Severity.INFO,
                        f"Max identical-value run: {max_run}", max_run)

        self._report.phases.append(ph)
        return ph

    # ======================================================================
    # PHASE 3: STATIONARITY
    # ======================================================================

    def phase_stationarity(self) -> PhaseResult:
        """ADF / KPSS tests, or heuristic fallback."""
        ph = PhaseResult("Phase 3: Stationarity")
        d = self._clean

        if len(d) < 20:
            ph.add("stationarity", Verdict.SKIP, Severity.INFO,
                    "Insufficient data for stationarity testing")
            self._report.phases.append(ph)
            return ph

        # --- statsmodels path (full tests) --------------------------------
        if _HAS_STATSMODELS:
            # ADF (H0: unit root exists = non-stationary)
            try:
                adf_stat, adf_p, *_ = adfuller(d, autolag="AIC")
                if adf_p < 0.05:
                    ph.add("adf_test", Verdict.PASS, Severity.INFO,
                            f"ADF p={adf_p:.4f} — stationary at 5%",
                            {"statistic": round(adf_stat, 4), "p": round(adf_p, 4)})
                else:
                    ph.add("adf_test", Verdict.WARN, Severity.MEDIUM,
                            f"ADF p={adf_p:.4f} — cannot reject unit root (non-stationary). "
                            "Differencing or transformation needed.",
                            {"statistic": round(adf_stat, 4), "p": round(adf_p, 4)})
            except Exception as e:
                ph.add("adf_test", Verdict.SKIP, Severity.INFO, f"ADF failed: {e}")

            # KPSS (H0: series is stationary)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    kpss_stat, kpss_p, *_ = kpss(d, regression="c", nlags="auto")
                if kpss_p > 0.05:
                    ph.add("kpss_test", Verdict.PASS, Severity.INFO,
                            f"KPSS p={kpss_p:.4f} — stationary at 5%",
                            {"statistic": round(kpss_stat, 4), "p": round(kpss_p, 4)})
                else:
                    ph.add("kpss_test", Verdict.WARN, Severity.MEDIUM,
                            f"KPSS p={kpss_p:.4f} — rejects stationarity. "
                            "Differencing or detrending recommended.",
                            {"statistic": round(kpss_stat, 4), "p": round(kpss_p, 4)})
            except Exception as e:
                ph.add("kpss_test", Verdict.SKIP, Severity.INFO, f"KPSS failed: {e}")

            # differenced series
            dd = _diff(d)
            if len(dd) >= 20:
                try:
                    adf2_stat, adf2_p, *_ = adfuller(dd, autolag="AIC")
                    if adf2_p < 0.05:
                        ph.add("diff_stationarity", Verdict.PASS, Severity.INFO,
                                f"First difference is stationary (ADF p={adf2_p:.4f})",
                                round(adf2_p, 4))
                    else:
                        ph.add("diff_stationarity", Verdict.WARN, Severity.MEDIUM,
                                f"Even first difference is non-stationary (p={adf2_p:.4f}) — "
                                "may need d=2 or log transform",
                                round(adf2_p, 4))
                except Exception:
                    pass
        else:
            # --- heuristic fallback (no statsmodels) ----------------------
            half = len(d) // 2
            m1, m2 = _mean(d[:half]), _mean(d[half:])
            s1, s2 = _std(d[:half]), _std(d[half:])
            mean_shift = abs(m2 - m1) / max(_std(d), 1e-12)
            var_ratio = max(s1, 1e-12) / max(s2, 1e-12)

            if mean_shift > 1.5:
                ph.add("mean_shift_heuristic", Verdict.WARN, Severity.MEDIUM,
                        f"Large mean shift between halves ({mean_shift:.2f}s) — likely non-stationary",
                        round(mean_shift, 3))
            else:
                ph.add("mean_shift_heuristic", Verdict.PASS, Severity.INFO,
                        f"Mean shift between halves: {mean_shift:.2f}s", round(mean_shift, 3))

            if var_ratio > 2.0 or var_ratio < 0.5:
                ph.add("variance_ratio_heuristic", Verdict.WARN, Severity.MEDIUM,
                        f"Variance ratio between halves: {var_ratio:.2f} — heteroscedastic",
                        round(var_ratio, 3))
            else:
                ph.add("variance_ratio_heuristic", Verdict.PASS, Severity.INFO,
                        f"Variance ratio between halves: {var_ratio:.2f}", round(var_ratio, 3))

        self._report.phases.append(ph)
        return ph

    # ======================================================================
    # PHASE 4: FORECASTABILITY ASSESSMENT
    # ======================================================================

    def phase_forecastability(self) -> PhaseResult:
        """
        Autocorrelation strength, approximate entropy, signal-to-noise,
        forecastability spectrum placement.
        From forecast-checker: 'If not assessed, author doesn't understand fundamentals.'
        """
        ph = PhaseResult("Phase 4: Forecastability")
        d = self._clean

        if len(d) < 20:
            ph.add("forecastability", Verdict.SKIP, Severity.INFO, "Insufficient data")
            self._report.phases.append(ph)
            return ph

        # --- autocorrelation strength ------------------------------------
        acf_vals = self.acf_values
        if len(acf_vals) > 1:
            # significant lags (|acf| > 2/sqrt(n))
            threshold = 2.0 / math.sqrt(len(d))
            sig_lags = [i for i, v in enumerate(acf_vals[1:], 1) if abs(v) > threshold]
            acf_energy = sum(v ** 2 for v in acf_vals[1:min(len(acf_vals), 21)])

            if not sig_lags:
                ph.add("acf_structure", Verdict.WARN, Severity.HIGH,
                        "No significant autocorrelation detected — series may be white noise. "
                        "Forecastability is LOW.",
                        {"significant_lags": [], "acf_energy": round(acf_energy, 4)})
            else:
                desc = "HIGH" if acf_energy > 2.0 else "MODERATE" if acf_energy > 0.5 else "LOW"
                ph.add("acf_structure", Verdict.PASS, Severity.INFO,
                        f"Forecastability signal: {desc} (ACF energy={acf_energy:.3f}, "
                        f"{len(sig_lags)} significant lags up to lag {max(sig_lags)})",
                        {"significant_lags": sig_lags[:10], "acf_energy": round(acf_energy, 4)})

            # lag-1 autocorrelation (persistence)
            lag1 = acf_vals[1] if len(acf_vals) > 1 else 0
            if abs(lag1) < threshold:
                ph.add("lag1_persistence", Verdict.WARN, Severity.MEDIUM,
                        f"Lag-1 autocorrelation {lag1:.4f} is not significant — "
                        "no short-term persistence", round(lag1, 4))
            else:
                ph.add("lag1_persistence", Verdict.PASS, Severity.INFO,
                        f"Lag-1 autocorrelation: {lag1:.4f}", round(lag1, 4))

        # --- approximate entropy proxy (crude) ---------------------------
        if len(d) >= 50:
            ups = sum(1 for i in range(1, len(d)) if d[i] > d[i - 1])
            downs = sum(1 for i in range(1, len(d)) if d[i] < d[i - 1])
            flats = len(d) - 1 - ups - downs
            total = ups + downs + flats
            probs = [c / total for c in (ups, downs, flats) if c > 0]
            direction_entropy = -sum(p * math.log2(p) for p in probs)
            max_entropy = math.log2(3)
            norm_entropy = direction_entropy / max_entropy

            if norm_entropy > 0.95:
                ph.add("direction_entropy", Verdict.WARN, Severity.MEDIUM,
                        f"Direction entropy {norm_entropy:.3f} near max — nearly random walk",
                        round(norm_entropy, 4))
            else:
                ph.add("direction_entropy", Verdict.PASS, Severity.INFO,
                        f"Direction entropy: {norm_entropy:.3f} (1.0 = pure random)",
                        round(norm_entropy, 4))

        # --- permutation entropy (proper forecastability measure) --------
        if len(d) >= 10:
            # Adaptive order: 3 for short, 4 for medium, 5 for long
            pe_order = 3 if len(d) < 200 else (4 if len(d) < 1000 else 5)
            pe = _permutation_entropy(d, order=pe_order, delay=1)
            if math.isfinite(pe):
                if pe > 0.95:
                    ph.add("permutation_entropy", Verdict.WARN, Severity.HIGH,
                            f"Permutation entropy {pe:.3f} (order={pe_order}) — "
                            "effectively random; forecastability is near-zero",
                            {"pe": round(pe, 4), "order": pe_order})
                elif pe > 0.85:
                    ph.add("permutation_entropy", Verdict.WARN, Severity.MEDIUM,
                            f"Permutation entropy {pe:.3f} (order={pe_order}) — "
                            "high complexity; forecastability is low",
                            {"pe": round(pe, 4), "order": pe_order})
                elif pe < 0.5:
                    ph.add("permutation_entropy", Verdict.PASS, Severity.INFO,
                            f"Permutation entropy {pe:.3f} (order={pe_order}) — "
                            "strong structure; forecastability is high",
                            {"pe": round(pe, 4), "order": pe_order})
                else:
                    ph.add("permutation_entropy", Verdict.PASS, Severity.INFO,
                            f"Permutation entropy {pe:.3f} (order={pe_order}) — "
                            "moderate complexity",
                            {"pe": round(pe, 4), "order": pe_order})

        # --- signal-to-noise ratio on differences -----------------------
        dd = self.diffs
        if len(dd) >= 10:
            snr_mean = abs(_mean(dd))
            snr_std = _std(dd)
            snr = snr_mean / snr_std if snr_std > 0 else float("inf")
            if snr < 0.05:
                ph.add("snr_diffs", Verdict.WARN, Severity.MEDIUM,
                        f"SNR of first differences: {snr:.4f} — dominated by noise", round(snr, 5))
            else:
                ph.add("snr_diffs", Verdict.PASS, Severity.INFO,
                        f"SNR of first differences: {snr:.4f}", round(snr, 5))

        # --- Ljung-Box test for white noise ------------------------------
        if _HAS_STATSMODELS and len(d) >= 30:
            try:
                lb_result = acorr_ljungbox(d, lags=[min(10, len(d) // 5)], return_df=True)
                lb_p = float(lb_result["lb_pvalue"].iloc[0])
                if lb_p > 0.05:
                    ph.add("ljung_box", Verdict.WARN, Severity.HIGH,
                            f"Ljung-Box p={lb_p:.4f} — cannot reject white noise. "
                            "Series may be unforecastable.", round(lb_p, 4))
                else:
                    ph.add("ljung_box", Verdict.PASS, Severity.INFO,
                            f"Ljung-Box p={lb_p:.4f} — significant autocorrelation exists",
                            round(lb_p, 4))
            except Exception:
                pass

        # --- infer seasonality if not provided ---------------------------
        if self.frequency is None and len(d) >= 50:
            self.frequency = self._infer_seasonality(d, acf_vals)
            if self.frequency:
                ph.add("inferred_seasonality", Verdict.PASS, Severity.INFO,
                        f"Inferred seasonal period: {self.frequency}", self.frequency)

        self._report.phases.append(ph)
        return ph

    @staticmethod
    def _infer_seasonality(d: List[float], acf_vals: List[float]) -> Optional[int]:
        """Detect dominant seasonal period from ACF peaks."""
        if len(acf_vals) < 5:
            return None
        threshold = 2.0 / math.sqrt(len(d))
        peaks = []
        for i in range(2, len(acf_vals) - 1):
            if (acf_vals[i] > acf_vals[i - 1]
                    and acf_vals[i] > acf_vals[i + 1]
                    and acf_vals[i] > threshold):
                peaks.append((i, acf_vals[i]))
        if not peaks:
            return None
        best = max(peaks, key=lambda x: x[1])
        return best[0] if best[1] > 0.1 else None

    # ======================================================================
    # PHASE 5: DECOMPOSITION
    # ======================================================================

    def phase_decomposition(self) -> PhaseResult:
        """Trend / seasonality strength, residual ratio, structural breaks."""
        ph = PhaseResult("Phase 5: Decomposition")
        d = self._clean
        n = len(d)

        if n < 30:
            ph.add("decomposition", Verdict.SKIP, Severity.INFO, "Too short to decompose")
            self._report.phases.append(ph)
            return ph

        period = self.frequency or 2

        if _HAS_STATSMODELS and _HAS_NUMPY and n >= 2 * period + 1 and period >= 2:
            try:
                stl = STL(np.array(d), period=period, robust=True)
                result = stl.fit()
                trend_comp = list(result.trend)
                season_comp = list(result.seasonal)
                resid_comp = list(result.resid)

                var_resid = _std(resid_comp, ddof=0) ** 2
                var_trend_resid = _std([trend_comp[i] + resid_comp[i] for i in range(n)], ddof=0) ** 2
                trend_strength = max(0, 1 - var_resid / var_trend_resid) if var_trend_resid > 0 else 0

                var_season_resid = _std([season_comp[i] + resid_comp[i] for i in range(n)], ddof=0) ** 2
                season_strength = max(0, 1 - var_resid / var_season_resid) if var_season_resid > 0 else 0

                resid_frac = _std(resid_comp, ddof=0) / _std(d, ddof=0) if _std(d, ddof=0) > 0 else 1.0

                ph.add("trend_strength", Verdict.PASS, Severity.INFO,
                        f"Trend strength: {trend_strength:.3f} "
                        f"({'strong' if trend_strength > 0.6 else 'moderate' if trend_strength > 0.3 else 'weak'})",
                        round(trend_strength, 4))
                ph.add("seasonality_strength", Verdict.PASS, Severity.INFO,
                        f"Seasonality strength: {season_strength:.3f} (period={period}) "
                        f"({'strong' if season_strength > 0.6 else 'moderate' if season_strength > 0.3 else 'weak'})",
                        round(season_strength, 4))
                ph.add("residual_fraction", Verdict.PASS if resid_frac < 0.8 else Verdict.WARN,
                        Severity.MEDIUM if resid_frac >= 0.8 else Severity.INFO,
                        f"Residual fraction of total variance: {resid_frac:.3f}",
                        round(resid_frac, 4))

                self._report.metadata["trend_strength"] = round(trend_strength, 4)
                self._report.metadata["season_strength"] = round(season_strength, 4)
            except Exception as e:
                ph.add("stl_decomposition", Verdict.SKIP, Severity.INFO, f"STL failed: {e}")
        else:
            # heuristic: moving-average trend extraction
            w = min(period if period >= 3 else 5, n // 3)
            if w >= 3:
                trend_approx = []
                for i in range(n):
                    lo_idx = max(0, i - w // 2)
                    hi_idx = min(n, i + w // 2 + 1)
                    trend_approx.append(_mean(d[lo_idx:hi_idx]))
                detrended = [d[i] - trend_approx[i] for i in range(n)]
                var_d = _std(d, ddof=0) ** 2
                var_dt = _std(detrended, ddof=0) ** 2
                trend_strength = max(0, 1 - var_dt / var_d) if var_d > 0 else 0

                ph.add("trend_strength_heuristic", Verdict.PASS, Severity.INFO,
                        f"Approximate trend strength: {trend_strength:.3f}", round(trend_strength, 4))

        # --- structural break detection (variance-based) -----------------
        if n >= 60:
            window = max(20, n // 5)
            variances = []
            for start in range(0, n - window + 1, window // 2):
                seg = d[start:start + window]
                variances.append(_std(seg, ddof=0) ** 2)
            if len(variances) >= 3:
                max_var = max(variances)
                min_var = min(variances)
                ratio = max_var / min_var if min_var > 0 else float("inf")
                if ratio > 4.0:
                    ph.add("structural_break", Verdict.WARN, Severity.HIGH,
                            f"Variance ratio across windows: {ratio:.2f} — "
                            "possible structural break or regime change", round(ratio, 2))
                elif ratio > 2.0:
                    ph.add("structural_break", Verdict.WARN, Severity.LOW,
                            f"Variance ratio across windows: {ratio:.2f} — moderate heteroscedasticity",
                            round(ratio, 2))
                else:
                    ph.add("structural_break", Verdict.PASS, Severity.INFO,
                            f"Variance ratio across windows: {ratio:.2f} — stable", round(ratio, 2))

        self._report.phases.append(ph)
        return ph

    # ======================================================================
    # PHASE 6: BASELINE BENCHMARKS
    # ======================================================================

    def phase_baseline_benchmarks(
        self,
        predictions: Optional[List[float]] = None,
        actuals: Optional[List[float]] = None,
    ) -> PhaseResult:
        """
        Compute naive / seasonal-naive / drift baselines.
        If predictions+actuals provided, compare model against baselines.
        From forecast-checker: 'If you can't beat naive, you can't forecast.'
        """
        ph = PhaseResult("Phase 6: Baseline Benchmarks")
        d = self._clean
        n = len(d)

        if n < 10:
            ph.add("baselines", Verdict.SKIP, Severity.INFO, "Insufficient data")
            self._report.phases.append(ph)
            return ph

        # --- naive baseline (Y[t+1] = Y[t]) -----------------------------
        naive_errors = [abs(d[i] - d[i - 1]) for i in range(1, n)]
        naive_mae = _mean(naive_errors)

        # --- seasonal naive (Y[t+s] = Y[t]) ----------------------------
        sp = self.frequency or 1
        snaive_mae = None
        if sp > 1 and n > 2 * sp:
            snaive_errors = [abs(d[i] - d[i - sp]) for i in range(sp, n)]
            snaive_mae = _mean(snaive_errors)

        # --- drift baseline (linear extrapolation) ----------------------
        drift_errors = []
        for i in range(2, n):
            pred = d[i - 1] + (d[i - 1] - d[0]) / (i - 1)
            drift_errors.append(abs(d[i] - pred))
        drift_mae = _mean(drift_errors) if drift_errors else None

        # --- mean baseline (Y[t+1] = mean(Y[:t])) ----------------------
        mean_errors = []
        for i in range(max(10, n // 5), n):
            pred = _mean(d[:i])
            mean_errors.append(abs(d[i] - pred))
        mean_mae = _mean(mean_errors) if mean_errors else None

        baselines = {"naive_mae": round(naive_mae, 6)}
        best_baseline = ("naive", naive_mae)
        if snaive_mae is not None:
            baselines["seasonal_naive_mae"] = round(snaive_mae, 6)
            if snaive_mae < best_baseline[1]:
                best_baseline = ("seasonal_naive", snaive_mae)
        if drift_mae is not None:
            baselines["drift_mae"] = round(drift_mae, 6)
            if drift_mae < best_baseline[1]:
                best_baseline = ("drift", drift_mae)
        if mean_mae is not None:
            baselines["mean_mae"] = round(mean_mae, 6)
            if mean_mae < best_baseline[1]:
                best_baseline = ("mean", mean_mae)

        ph.add("baseline_errors", Verdict.PASS, Severity.INFO,
                f"Best baseline: {best_baseline[0]} (MAE={best_baseline[1]:.6f})", baselines)

        # --- compare model if provided ----------------------------------
        if predictions is not None and actuals is not None:
            pred = _to_floats(predictions)
            act = _to_floats(actuals)
            n_cmp = min(len(pred), len(act))
            if n_cmp >= 5:
                model_mae = _mae(act, pred)
                model_rmse = _rmse(act, pred)
                model_mase = _mase(act, pred, seasonal_period=sp)

                ph.add("model_mae", Verdict.PASS, Severity.INFO,
                        f"Model MAE: {model_mae:.6f}", round(model_mae, 6))
                ph.add("model_rmse", Verdict.PASS, Severity.INFO,
                        f"Model RMSE: {model_rmse:.6f}", round(model_rmse, 6))

                if math.isfinite(model_mase):
                    if model_mase >= 1.5:
                        ph.add("model_mase", Verdict.FAIL, Severity.HIGH,
                                f"MASE={model_mase:.3f} — much worse than naive. "
                                "Model adds negative value.", round(model_mase, 4))
                    elif model_mase >= 1.0:
                        ph.add("model_mase", Verdict.WARN, Severity.MEDIUM,
                                f"MASE={model_mase:.3f} — does not beat naive baseline. "
                                "Complexity not justified.", round(model_mase, 4))
                    elif model_mase >= 0.8:
                        ph.add("model_mase", Verdict.PASS, Severity.INFO,
                                f"MASE={model_mase:.3f} — acceptable improvement over naive",
                                round(model_mase, 4))
                    else:
                        ph.add("model_mase", Verdict.PASS, Severity.INFO,
                                f"MASE={model_mase:.3f} — good improvement over naive",
                                round(model_mase, 4))

                # --- Forecast Value Added (FVA) --------------------------
                fva_val = _fva(model_mae, best_baseline[1])
                if math.isfinite(fva_val):
                    if fva_val < 0:
                        ph.add("fva", Verdict.FAIL, Severity.HIGH,
                                f"FVA={fva_val:.1f}% — model destroys value vs "
                                f"best baseline ({best_baseline[0]}). Use naive.",
                                round(fva_val, 2))
                    elif fva_val < 10:
                        ph.add("fva", Verdict.WARN, Severity.MEDIUM,
                                f"FVA={fva_val:.1f}% — marginal improvement over "
                                f"best baseline ({best_baseline[0]}). "
                                "Complexity may not be justified.",
                                round(fva_val, 2))
                    else:
                        ph.add("fva", Verdict.PASS, Severity.INFO,
                                f"FVA={fva_val:.1f}% — model adds substantial value "
                                f"over best baseline ({best_baseline[0]})",
                                round(fva_val, 2))

        self._report.phases.append(ph)
        return ph

    # ======================================================================
    # PHASE 7: OVERFITTING SCREEN
    # ======================================================================

    def phase_overfitting_screen(
        self,
        test_predictions: List[float],
        test_actuals: List[float],
        train_predictions: Optional[List[float]] = None,
        train_actuals: Optional[List[float]] = None,
    ) -> PhaseResult:
        """
        R-squared checks, train-vs-test gap, memorization, leakage indicators.
        From forecast-checker: 'R2=1.0 train = memorization, R2>0.95 test = leakage.'
        """
        ph = PhaseResult("Phase 7: Overfitting Screen")

        tp = _to_floats(test_predictions)
        ta = _to_floats(test_actuals)
        n_test = min(len(tp), len(ta))

        if n_test < 5:
            ph.add("overfitting", Verdict.SKIP, Severity.INFO, "Too few test samples")
            self._report.phases.append(ph)
            return ph

        test_r2 = _r_squared(ta, tp)
        test_mae = _mae(ta, tp)

        # --- test R-squared plausibility ---------------------------------
        if math.isfinite(test_r2):
            if test_r2 > 0.99:
                ph.add("test_r2", Verdict.REJECT, Severity.CRITICAL,
                        f"Test R2={test_r2:.6f} — near-certain data leakage or fabrication",
                        round(test_r2, 6))
            elif test_r2 > 0.95:
                ph.add("test_r2", Verdict.FAIL, Severity.HIGH,
                        f"Test R2={test_r2:.6f} — suspiciously high. "
                        "Verify no future leakage, target leakage, or train-test contamination.",
                        round(test_r2, 6))
            elif test_r2 < 0:
                ph.add("test_r2", Verdict.FAIL, Severity.HIGH,
                        f"Test R2={test_r2:.4f} — model is worse than predicting the mean",
                        round(test_r2, 6))
            else:
                ph.add("test_r2", Verdict.PASS, Severity.INFO,
                        f"Test R2={test_r2:.4f}", round(test_r2, 6))

        # --- train-vs-test gap -------------------------------------------
        if train_predictions is not None and train_actuals is not None:
            trp = _to_floats(train_predictions)
            tra = _to_floats(train_actuals)
            n_train = min(len(trp), len(tra))
            if n_train >= 5:
                train_r2 = _r_squared(tra, trp)
                train_mae = _mae(tra, trp)

                if math.isfinite(train_r2):
                    if train_r2 >= 0.9999:
                        ph.add("train_r2_memorization", Verdict.REJECT, Severity.CRITICAL,
                                f"Train R2={train_r2:.6f} — model has memorized training data",
                                round(train_r2, 6))
                    elif train_r2 > 0.99:
                        ph.add("train_r2", Verdict.WARN, Severity.HIGH,
                                f"Train R2={train_r2:.6f} — very tight fit, check regularization",
                                round(train_r2, 6))

                    # impossible: test > train
                    if math.isfinite(test_r2) and test_r2 > train_r2 + 0.01:
                        ph.add("test_better_than_train", Verdict.REJECT, Severity.CRITICAL,
                                f"Test R2({test_r2:.4f}) > Train R2({train_r2:.4f}) — "
                                "statistically impossible without data leakage",
                                {"train_r2": round(train_r2, 4), "test_r2": round(test_r2, 4)})

                    # overfitting gap
                    if math.isfinite(test_r2):
                        gap = train_r2 - test_r2
                        if gap > 0.3:
                            ph.add("overfit_gap", Verdict.FAIL, Severity.HIGH,
                                    f"Train-test R2 gap: {gap:.4f} — severe overfitting",
                                    round(gap, 4))
                        elif gap > 0.15:
                            ph.add("overfit_gap", Verdict.WARN, Severity.MEDIUM,
                                    f"Train-test R2 gap: {gap:.4f} — moderate overfitting",
                                    round(gap, 4))
                        else:
                            ph.add("overfit_gap", Verdict.PASS, Severity.INFO,
                                    f"Train-test R2 gap: {gap:.4f}", round(gap, 4))

                # MAE gap
                if test_mae > 0 and train_mae > 0:
                    mae_ratio = test_mae / train_mae
                    if mae_ratio > 3.0:
                        ph.add("mae_gap", Verdict.FAIL, Severity.HIGH,
                                f"Test/Train MAE ratio: {mae_ratio:.2f} — heavy overfitting",
                                round(mae_ratio, 3))
                    elif mae_ratio > 1.5:
                        ph.add("mae_gap", Verdict.WARN, Severity.MEDIUM,
                                f"Test/Train MAE ratio: {mae_ratio:.2f}", round(mae_ratio, 3))
                    else:
                        ph.add("mae_gap", Verdict.PASS, Severity.INFO,
                                f"Test/Train MAE ratio: {mae_ratio:.2f}", round(mae_ratio, 3))

        self._report.phases.append(ph)
        return ph

    # ======================================================================
    # PHASE 8: RESIDUAL DIAGNOSTICS
    # ======================================================================

    def phase_residual_diagnostics(self, residuals: Sequence[float]) -> PhaseResult:
        """
        Zero mean, no autocorrelation, homoscedasticity, normality.
        From visual guide: 'Good model residuals should be white noise.'
        """
        ph = PhaseResult("Phase 8: Residual Diagnostics")
        r = _to_floats(residuals)
        n = len(r)

        if n < 10:
            ph.add("residuals", Verdict.SKIP, Severity.INFO, "Too few residuals")
            self._report.phases.append(ph)
            return ph

        # --- zero mean (bias) -------------------------------------------
        r_mean = _mean(r)
        r_std = _std(r)
        bias_t = abs(r_mean) / (r_std / math.sqrt(n)) if r_std > 0 else float("inf")

        if bias_t > 2.576:
            ph.add("residual_bias", Verdict.FAIL, Severity.HIGH,
                    f"Mean residual={r_mean:.6f} (t={bias_t:.2f}) — significant systematic bias",
                    {"mean": round(r_mean, 6), "t_stat": round(bias_t, 3)})
        elif bias_t > 1.96:
            ph.add("residual_bias", Verdict.WARN, Severity.MEDIUM,
                    f"Mean residual={r_mean:.6f} (t={bias_t:.2f}) — marginal bias",
                    {"mean": round(r_mean, 6), "t_stat": round(bias_t, 3)})
        else:
            ph.add("residual_bias", Verdict.PASS, Severity.INFO,
                    f"Mean residual={r_mean:.6f} — no significant bias", round(r_mean, 6))

        # --- autocorrelation in residuals --------------------------------
        r_acf = _autocorr(r, max_lag=min(20, n // 3))
        if len(r_acf) > 1:
            threshold = 2.0 / math.sqrt(n)
            sig_lags = [i for i, v in enumerate(r_acf[1:], 1) if abs(v) > threshold]
            if sig_lags:
                ph.add("residual_acf", Verdict.FAIL, Severity.HIGH,
                        f"Residuals have significant autocorrelation at lags {sig_lags[:5]} — "
                        "model is missing temporal structure", sig_lags[:10])
            else:
                ph.add("residual_acf", Verdict.PASS, Severity.INFO,
                        "No significant residual autocorrelation — residuals approximate white noise")

        # --- Ljung-Box on residuals -------------------------------------
        if _HAS_STATSMODELS and n >= 20:
            try:
                lb = acorr_ljungbox(r, lags=[min(10, n // 5)], return_df=True)
                lb_p = float(lb["lb_pvalue"].iloc[0])
                if lb_p < 0.05:
                    ph.add("residual_ljungbox", Verdict.FAIL, Severity.HIGH,
                            f"Ljung-Box p={lb_p:.4f} — residuals are not white noise",
                            round(lb_p, 4))
                else:
                    ph.add("residual_ljungbox", Verdict.PASS, Severity.INFO,
                            f"Ljung-Box p={lb_p:.4f} — residuals consistent with white noise",
                            round(lb_p, 4))
            except Exception:
                pass

        # --- homoscedasticity (variance stability) -----------------------
        half = n // 2
        std1 = _std(r[:half])
        std2 = _std(r[half:])
        var_ratio = max(std1, 1e-15) / max(std2, 1e-15)
        if var_ratio > 2.0 or var_ratio < 0.5:
            ph.add("residual_heteroscedasticity", Verdict.WARN, Severity.MEDIUM,
                    f"Residual variance ratio (1st half / 2nd half): {var_ratio:.2f} — "
                    "heteroscedastic; consider GARCH or variance-stabilizing transform",
                    round(var_ratio, 3))
        else:
            ph.add("residual_heteroscedasticity", Verdict.PASS, Severity.INFO,
                    f"Residual variance ratio: {var_ratio:.2f} — approximately homoscedastic",
                    round(var_ratio, 3))

        # --- normality --------------------------------------------------
        if _HAS_SCIPY and n >= 20:
            try:
                _, shap_p = sp_stats.shapiro(r[:min(n, 5000)])
                if shap_p < 0.01:
                    ph.add("residual_normality", Verdict.WARN, Severity.LOW,
                            f"Shapiro-Wilk p={shap_p:.4f} — residuals non-normal "
                            "(prediction intervals may be miscalibrated)", round(shap_p, 4))
                else:
                    ph.add("residual_normality", Verdict.PASS, Severity.INFO,
                            f"Shapiro-Wilk p={shap_p:.4f} — residuals approximately normal",
                            round(shap_p, 4))
            except Exception:
                pass

        self._report.phases.append(ph)
        return ph

    # ======================================================================
    # PHASE 9: UNCERTAINTY / INTERVAL CALIBRATION
    # ======================================================================

    def phase_uncertainty(
        self,
        lower: Sequence[float],
        upper: Sequence[float],
        actuals: Sequence[float],
        nominal_coverage: float = 0.95,
    ) -> PhaseResult:
        """
        Coverage calibration, interval width, sharpness evaluation.
        From visual guide: 'Conformal prediction is the ONLY method with
        finite-sample validity guarantees.'
        """
        ph = PhaseResult("Phase 9: Uncertainty Quantification")
        lo = _to_floats(lower)
        hi = _to_floats(upper)
        act = _to_floats(actuals)
        n = min(len(lo), len(hi), len(act))

        if n < 10:
            ph.add("uncertainty", Verdict.SKIP, Severity.INFO, "Too few samples")
            self._report.phases.append(ph)
            return ph

        # --- coverage ---------------------------------------------------
        covered = sum(1 for i in range(n) if lo[i] <= act[i] <= hi[i])
        actual_coverage = covered / n

        cov_gap = actual_coverage - nominal_coverage
        if cov_gap < -0.10:
            ph.add("coverage", Verdict.FAIL, Severity.HIGH,
                    f"Actual coverage {actual_coverage:.1%} vs nominal {nominal_coverage:.0%} — "
                    "severely under-covered. Intervals are dangerously narrow.",
                    {"actual": round(actual_coverage, 4), "nominal": nominal_coverage})
        elif cov_gap < -0.03:
            ph.add("coverage", Verdict.WARN, Severity.MEDIUM,
                    f"Actual coverage {actual_coverage:.1%} vs nominal {nominal_coverage:.0%} — "
                    "under-covered; intervals need widening.",
                    {"actual": round(actual_coverage, 4), "nominal": nominal_coverage})
        elif cov_gap > 0.10:
            ph.add("coverage", Verdict.WARN, Severity.LOW,
                    f"Actual coverage {actual_coverage:.1%} vs nominal {nominal_coverage:.0%} — "
                    "over-covered; intervals are unnecessarily wide (poor sharpness).",
                    {"actual": round(actual_coverage, 4), "nominal": nominal_coverage})
        else:
            ph.add("coverage", Verdict.PASS, Severity.INFO,
                    f"Actual coverage {actual_coverage:.1%} vs nominal {nominal_coverage:.0%} — well-calibrated",
                    round(actual_coverage, 4))

        # --- interval width (sharpness) ---------------------------------
        widths = [hi[i] - lo[i] for i in range(n)]
        mean_width = _mean(widths)
        act_range = max(act[:n]) - min(act[:n])
        relative_width = mean_width / act_range if act_range > 0 else float("inf")

        if relative_width > 1.0:
            ph.add("interval_width", Verdict.WARN, Severity.MEDIUM,
                    f"Mean interval width ({mean_width:.4f}) exceeds data range — "
                    "intervals are uninformative",
                    {"mean_width": round(mean_width, 4), "relative": round(relative_width, 3)})
        else:
            ph.add("interval_width", Verdict.PASS, Severity.INFO,
                    f"Mean interval width: {mean_width:.4f} ({relative_width:.1%} of data range)",
                    {"mean_width": round(mean_width, 4), "relative": round(relative_width, 3)})

        # --- interval width consistency ----------------------------------
        if len(widths) >= 20:
            w_std = _std(widths)
            cv = w_std / mean_width if mean_width > 0 else 0
            if cv > 0.5:
                ph.add("interval_consistency", Verdict.WARN, Severity.LOW,
                        f"Interval width CV={cv:.3f} — highly variable. "
                        "Consider adaptive conformal prediction.", round(cv, 3))
            else:
                ph.add("interval_consistency", Verdict.PASS, Severity.INFO,
                        f"Interval width CV={cv:.3f}", round(cv, 3))

        # --- Winkler score (combined calibration + sharpness) -----------
        alpha = 1.0 - nominal_coverage
        winkler_scores = []
        for i in range(n):
            w = hi[i] - lo[i]
            if act[i] < lo[i]:
                w += (2.0 / alpha) * (lo[i] - act[i])
            elif act[i] > hi[i]:
                w += (2.0 / alpha) * (act[i] - hi[i])
            winkler_scores.append(w)
        mean_winkler = _mean(winkler_scores)
        ph.add("winkler_score", Verdict.PASS, Severity.INFO,
                f"Mean Winkler score: {mean_winkler:.4f} (lower = better calibration + sharpness)",
                round(mean_winkler, 4))

        self._report.phases.append(ph)
        return ph

    # ======================================================================
    # PHASE 10: REGIME ANALYSIS
    # ======================================================================

    def phase_regime_analysis(self, n_windows: int = 5) -> PhaseResult:
        """
        Volatility clustering, distribution shifts, regime detection.
        From forecast-checker: 'If model only tested in one regime,
        results don't generalize.'
        """
        ph = PhaseResult("Phase 10: Regime Analysis")
        d = self._clean
        n = len(d)

        if n < 50:
            ph.add("regimes", Verdict.SKIP, Severity.INFO, "Too short for regime analysis")
            self._report.phases.append(ph)
            return ph

        # --- split into windows -----------------------------------------
        wsize = n // n_windows
        windows = [d[i * wsize:(i + 1) * wsize] for i in range(n_windows)]
        rem = d[n_windows * wsize:]
        if rem:
            windows[-1].extend(rem)

        means = [_mean(w) for w in windows]
        stds = [_std(w) for w in windows]

        # --- mean stability across regimes --------------------------------
        overall_std = _std(d)
        mean_range = max(means) - min(means)
        mean_drift = mean_range / overall_std if overall_std > 0 else 0

        if mean_drift > 2.0:
            ph.add("regime_mean_drift", Verdict.WARN, Severity.HIGH,
                    f"Mean drifts {mean_drift:.2f}s across windows — "
                    "strong non-stationarity / regime changes",
                    {"means": [round(m, 4) for m in means], "drift_sigma": round(mean_drift, 3)})
        elif mean_drift > 1.0:
            ph.add("regime_mean_drift", Verdict.WARN, Severity.MEDIUM,
                    f"Mean drifts {mean_drift:.2f}s across windows",
                    {"means": [round(m, 4) for m in means], "drift_sigma": round(mean_drift, 3)})
        else:
            ph.add("regime_mean_drift", Verdict.PASS, Severity.INFO,
                    f"Mean drift: {mean_drift:.2f}s — stable across windows",
                    round(mean_drift, 3))

        # --- volatility clustering ----------------------------------------
        vol_range = max(stds) / min(stds) if min(stds) > 0 else float("inf")
        if vol_range > 3.0:
            ph.add("volatility_clustering", Verdict.WARN, Severity.HIGH,
                    f"Volatility ratio across windows: {vol_range:.2f} — "
                    "strong volatility clustering; GARCH or regime-switching models recommended",
                    {"stds": [round(s, 4) for s in stds], "ratio": round(vol_range, 2)})
        elif vol_range > 1.5:
            ph.add("volatility_clustering", Verdict.WARN, Severity.LOW,
                    f"Volatility ratio: {vol_range:.2f}", round(vol_range, 2))
        else:
            ph.add("volatility_clustering", Verdict.PASS, Severity.INFO,
                    f"Volatility ratio: {vol_range:.2f} — stable", round(vol_range, 2))

        # --- KS test between consecutive windows -------------------------
        if _HAS_SCIPY and all(len(w) >= 10 for w in windows):
            min_ks_p = 1.0
            worst_pair = (0, 1)
            for i in range(len(windows) - 1):
                _, ks_p = sp_stats.ks_2samp(windows[i], windows[i + 1])
                if ks_p < min_ks_p:
                    min_ks_p = ks_p
                    worst_pair = (i, i + 1)

            if min_ks_p < 0.01:
                ph.add("regime_ks_test", Verdict.WARN, Severity.HIGH,
                        f"KS test detects significant distribution shift between "
                        f"windows {worst_pair[0]}->{worst_pair[1]} (p={min_ks_p:.4f}). "
                        "Walk-forward validation across regimes is essential.",
                        {"worst_p": round(min_ks_p, 4), "pair": worst_pair})
            elif min_ks_p < 0.05:
                ph.add("regime_ks_test", Verdict.WARN, Severity.MEDIUM,
                        f"Marginal distribution shift detected (p={min_ks_p:.4f})",
                        round(min_ks_p, 4))
            else:
                ph.add("regime_ks_test", Verdict.PASS, Severity.INFO,
                        f"No significant distribution shift between consecutive windows "
                        f"(min p={min_ks_p:.4f})", round(min_ks_p, 4))

        # --- recommendation based on regimes ----------------------------
        if mean_drift > 1.5 or vol_range > 2.5:
            ph.add("regime_recommendation", Verdict.WARN, Severity.MEDIUM,
                    "Multiple regimes detected. Recommendations: "
                    "(1) Use rolling/expanding window validation, not fixed split. "
                    "(2) Report performance per-regime, not just aggregate. "
                    "(3) Consider regime-switching models or adaptive methods.",
                    {"mean_drift": round(mean_drift, 3), "vol_range": round(vol_range, 2)})

        self._report.phases.append(ph)
        return ph

    # ======================================================================
    # FULL REVIEW (orchestrator)
    # ======================================================================

    def full_review(
        self,
        *,
        predictions: Optional[List[float]] = None,
        actuals: Optional[List[float]] = None,
        train_predictions: Optional[List[float]] = None,
        train_actuals: Optional[List[float]] = None,
        residuals: Optional[List[float]] = None,
        pred_lower: Optional[List[float]] = None,
        pred_upper: Optional[List[float]] = None,
        pred_actuals: Optional[List[float]] = None,
    ) -> ReviewReport:
        """
        Run all applicable phases. Supply optional args to enable
        model-evaluation phases (7-9).
        """
        self._report = ReviewReport(
            series_name=self.name,
            n_observations=len(self._clean),
        )

        # Always-run phases (only need the raw series)
        self.phase_coherence()
        if self._report.phases[-1].verdict == Verdict.REJECT:
            return self._report

        self.phase_data_quality()
        self.phase_stationarity()
        self.phase_forecastability()
        self.phase_decomposition()
        self.phase_baseline_benchmarks(predictions=predictions, actuals=actuals)

        # Model-evaluation phases (optional)
        if predictions is not None and actuals is not None:
            self.phase_overfitting_screen(
                test_predictions=predictions,
                test_actuals=actuals,
                train_predictions=train_predictions,
                train_actuals=train_actuals,
            )

        if residuals is not None:
            self.phase_residual_diagnostics(residuals)

        if pred_lower is not None and pred_upper is not None and pred_actuals is not None:
            self.phase_uncertainty(pred_lower, pred_upper, pred_actuals)

        self.phase_regime_analysis()

        # metadata summary
        self._report.metadata.update({
            "n_raw": len(self._raw),
            "n_clean": len(self._clean),
            "mean": round(_mean(self._clean), 6) if self._clean else None,
            "std": round(_std(self._clean), 6) if len(self._clean) > 1 else None,
            "min": round(min(self._clean), 6) if self._clean else None,
            "max": round(max(self._clean), 6) if self._clean else None,
            "frequency": self.frequency,
            "libs": {
                "numpy": _HAS_NUMPY,
                "scipy": _HAS_SCIPY,
                "statsmodels": _HAS_STATSMODELS,
            },
        })

        return self._report


# ===========================================================================
# 4. CONVENIENCE FUNCTIONS
# ===========================================================================


def quick_review(data: Sequence, *, name: str = "series", **kwargs) -> ReviewReport:
    """One-liner: full review with default settings."""
    return TimeSeriesReviewer(data, name=name, **kwargs).full_review()


def compare_models(
    actuals: List[float],
    models: Dict[str, List[float]],
    seasonal_period: int = 1,
) -> Dict[str, Dict[str, float]]:
    """Compare multiple model predictions against actuals. Returns metrics dict ranked by MASE."""
    act = _to_floats(actuals)
    results = {}
    for name, preds in models.items():
        p = _to_floats(preds)
        n = min(len(act), len(p))
        if n < 2:
            continue
        results[name] = {
            "mae": round(_mae(act[:n], p[:n]), 6),
            "rmse": round(_rmse(act[:n], p[:n]), 6),
            "r2": round(_r_squared(act[:n], p[:n]), 6),
            "mase": round(_mase(act[:n], p[:n], seasonal_period), 6),
            "wape": round(_wape(act[:n], p[:n]), 6),
            "me_bias": round(_me_bias(act[:n], p[:n]), 6),
        }
    ranked = sorted(results.items(), key=lambda x: x[1].get("mase", float("inf")))
    return {k: v for k, v in ranked}


def walk_forward_split(
    data: List[float],
    *,
    n_folds: int = 5,
    min_train: int = 30,
    horizon: int = 1,
    expanding: bool = True,
) -> List[Tuple[List[float], List[float]]]:
    """
    Generate walk-forward (expanding or rolling) train/test splits.
    Returns list of (train, test) tuples.
    """
    n = len(data)
    if n < min_train + horizon:
        return [(data[:-horizon], data[-horizon:])]

    test_total = n - min_train
    step = max(1, test_total // n_folds)
    splits = []

    for fold in range(n_folds):
        test_start = min_train + fold * step
        test_end = min(test_start + horizon, n)
        if test_start >= n:
            break

        if expanding:
            train = data[:test_start]
        else:
            train_start = max(0, test_start - min_train)
            train = data[train_start:test_start]

        test = data[test_start:test_end]
        if train and test:
            splits.append((train, test))

    return splits


def conformal_intervals(
    calibration_residuals: List[float],
    point_forecasts: List[float],
    coverage: float = 0.95,
) -> List[Tuple[float, float]]:
    """
    Split conformal prediction intervals.
    Uses calibration residuals to size intervals with guaranteed coverage.
    """
    abs_residuals = sorted(abs(r) for r in calibration_residuals)
    n = len(abs_residuals)
    q_idx = min(int(math.ceil((n + 1) * coverage)) - 1, n - 1)
    q_idx = max(0, q_idx)
    threshold = abs_residuals[q_idx]
    return [(f - threshold, f + threshold) for f in point_forecasts]


def cqr_intervals(
    calibration_actuals: List[float],
    calibration_lower: List[float],
    calibration_upper: List[float],
    test_lower: List[float],
    test_upper: List[float],
    coverage: float = 0.95,
) -> List[Tuple[float, float]]:
    """
    Conformalized Quantile Regression (CQR) intervals.

    Takes pre-computed quantile predictions from any quantile regression model
    and adjusts them using conformal calibration to guarantee coverage.

    Parameters
    ----------
    calibration_actuals : list of float
        True values for the calibration set.
    calibration_lower : list of float
        Lower quantile predictions on calibration set.
    calibration_upper : list of float
        Upper quantile predictions on calibration set.
    test_lower : list of float
        Lower quantile predictions on test set.
    test_upper : list of float
        Upper quantile predictions on test set.
    coverage : float
        Desired coverage level (e.g. 0.95).

    Returns
    -------
    list of (float, float)
        Adjusted (lower, upper) intervals for the test set.
    """
    n_cal = min(len(calibration_actuals), len(calibration_lower), len(calibration_upper))
    if n_cal < 1:
        return [(lo, hi) for lo, hi in zip(test_lower, test_upper)]

    # Compute conformity scores: max(lower - actual, actual - upper)
    scores = []
    for i in range(n_cal):
        score = max(
            calibration_lower[i] - calibration_actuals[i],
            calibration_actuals[i] - calibration_upper[i],
        )
        scores.append(score)

    # Compute correction quantile
    scores_sorted = sorted(scores)
    q_idx = min(int(math.ceil((n_cal + 1) * coverage)) - 1, n_cal - 1)
    q_idx = max(0, q_idx)
    q_correction = scores_sorted[q_idx]

    # Adjust test intervals
    n_test = min(len(test_lower), len(test_upper))
    return [
        (test_lower[i] - q_correction, test_upper[i] + q_correction)
        for i in range(n_test)
    ]


# ===========================================================================
# 5. CLI DEMO
# ===========================================================================


def _demo():
    """Run a self-contained demo with synthetic data."""
    import random

    random.seed(42)
    n = 500

    # synthetic: trend + seasonality + noise + regime shift at t=350
    data = []
    for i in range(n):
        trend = 0.02 * i
        season = 10 * math.sin(2 * math.pi * i / 52)
        noise = random.gauss(0, 3 if i < 350 else 8)
        data.append(100 + trend + season + noise)

    print("=" * 72)
    print("  ts_reviewer.py — DEMO")
    print("=" * 72)

    reviewer = TimeSeriesReviewer(data, name="synthetic_demo", frequency=52)
    report = reviewer.full_review()
    report.print_report()

    # walk-forward splits
    splits = walk_forward_split(data, n_folds=5, min_train=200, horizon=26)
    print(f"Walk-forward splits: {len(splits)} folds")
    for i, (train, test) in enumerate(splits):
        print(f"  Fold {i + 1}: train={len(train)}, test={len(test)}")

    # conformal intervals demo
    cal_resids = [random.gauss(0, 4) for _ in range(100)]
    forecasts = [data[-1] + 0.02 * h for h in range(1, 11)]
    intervals = conformal_intervals(cal_resids, forecasts, coverage=0.95)
    print(f"\nConformal 95% intervals for 10-step-ahead forecast:")
    for h, (lo, hi) in enumerate(intervals, 1):
        print(f"  h={h}: [{lo:.2f}, {hi:.2f}]")

    print("\n--- JSON summary (truncated) ---")
    import json
    print(json.dumps(report.to_dict(), indent=2, default=str)[:2000], "...\n")


# ===========================================================================
# 6. CLI INTERFACE
# ===========================================================================


def main():
    import argparse
    import csv
    import sys

    parser = argparse.ArgumentParser(
        description="Time Series Reviewer — systematic signal diagnostics"
    )
    subparsers = parser.add_subparsers(dest="cmd", help="Commands")

    # review command — full 10-phase review from CSV
    review_p = subparsers.add_parser("review", help="Full 10-phase review from CSV")
    review_p.add_argument("file", help="CSV file path")
    review_p.add_argument("--column", required=True, help="Column name to analyze")
    review_p.add_argument("--freq", type=int, default=None,
                          help="Known seasonal period (e.g. 12 for monthly)")

    # quick command — phases 1-6 only (no model needed)
    quick_p = subparsers.add_parser("quick", help="Quick review (phases 1-6 only)")
    quick_p.add_argument("file", help="CSV file path")
    quick_p.add_argument("--column", required=True, help="Column name to analyze")
    quick_p.add_argument("--freq", type=int, default=None,
                         help="Known seasonal period")

    # demo command
    subparsers.add_parser("demo", help="Run built-in synthetic data demo")

    args = parser.parse_args()

    if args.cmd == "demo":
        _demo()

    elif args.cmd in ("review", "quick"):
        # Read CSV and extract column
        try:
            with open(args.file, newline="") as f:
                reader = csv.DictReader(f)
                if args.column not in reader.fieldnames:
                    print(f"Error: column '{args.column}' not found. "
                          f"Available: {', '.join(reader.fieldnames)}")
                    sys.exit(1)
                raw_values = [row[args.column] for row in reader]
        except FileNotFoundError:
            print(f"Error: file not found: {args.file}")
            sys.exit(1)
        except Exception as e:
            print(f"Error reading CSV: {e}")
            sys.exit(1)

        # Convert to floats (ts_reviewer handles None/NaN gracefully)
        data = []
        for v in raw_values:
            try:
                data.append(float(v))
            except (TypeError, ValueError):
                data.append(None)

        reviewer = TimeSeriesReviewer(data, name=args.column, frequency=args.freq)

        if args.cmd == "quick":
            # Run phases 1-6 only
            reviewer.phase_coherence()
            if reviewer.report.phases[-1].verdict != Verdict.REJECT:
                reviewer.phase_data_quality()
                reviewer.phase_stationarity()
                reviewer.phase_forecastability()
                reviewer.phase_decomposition()
                reviewer.phase_baseline_benchmarks()
            reviewer.report.print_report()
        else:
            # Full review (phases 1-6 + regime analysis)
            report = reviewer.full_review()
            report.print_report()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
