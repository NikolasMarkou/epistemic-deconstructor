#!/usr/bin/env python3
"""
forecast_modeler.py — Forecasting Model Fitting & Selection Framework
=====================================================================
Fits forecasting models to time-series data and produces predictions with
calibrated uncertainty via conformal prediction.

Complements ts_reviewer.py (diagnoses signals) and fourier_analyst.py
(analyzes spectra) by providing the actual model-fitting step.

Usage:
    from forecast_modeler import ForecastModeler

    modeler = ForecastModeler(data, name="sales", frequency=12)
    report = modeler.full_pipeline(horizon=12)
    report.print_report()

    # Or run individual phases:
    modeler.phase_forecastability()
    modeler.phase_classical_fitting(horizon=12)
    modeler.phase_ml_fitting(horizon=12)
    modeler.phase_model_comparison()
    modeler.phase_conformal_prediction(coverage=0.95)
    modeler.phase_forecast_generation(horizon=12)
    modeler.phase_report()

Dependencies (graceful degradation):
    Required : (none — pure stdlib works for Phase 1 + naive + conformal)
    Optional : numpy        (better numerics, feature engineering)
               statsmodels  (ARIMA/ETS fitting)
               pmdarima     (auto-ARIMA parameter search)
               catboost     (gradient boosting with quantile regression)
               scipy        (advanced statistics)
"""

from __future__ import annotations

import math
import statistics
import warnings
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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

    _HAS_SCIPY = True
except ImportError:
    sp_stats = None  # type: ignore
    _HAS_SCIPY = False

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    from statsmodels.tsa.stattools import adfuller

    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False

try:
    from pmdarima import auto_arima as _auto_arima

    _HAS_PMDARIMA = True
except ImportError:
    _HAS_PMDARIMA = False

try:
    from catboost import CatBoostRegressor, Pool

    _HAS_CATBOOST = True
except ImportError:
    _HAS_CATBOOST = False


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
class ForecastResult:
    """Result from a single forecasting model."""

    model_name: str
    predictions: List[float]
    lower: Optional[List[float]] = None
    upper: Optional[List[float]] = None
    train_predictions: Optional[List[float]] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    fitted_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ForecastReport:
    """Complete forecasting pipeline output."""

    series_name: str
    n_observations: int
    horizon: int = 0
    phases: List[PhaseResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    models: Dict[str, ForecastResult] = field(default_factory=dict)
    best_model: Optional[str] = None

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

    def print_report(self, *, verbose: bool = True) -> None:
        w = 72
        print(f"\n{'=' * w}")
        print(f"  FORECAST MODELER: {self.series_name}")
        print(f"  Observations : {self.n_observations}")
        print(f"  Horizon      : {self.horizon}")
        print(f"  Best Model   : {self.best_model or 'N/A'}")
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

        if self.models:
            print(f"\n--- Model Summary ---")
            for name, res in self.models.items():
                marker = " << BEST" if name == self.best_model else ""
                metrics_str = ", ".join(
                    f"{k}={v:.4f}" for k, v in res.metrics.items()
                    if isinstance(v, (int, float)) and math.isfinite(v)
                )
                print(f"  {name}: {metrics_str}{marker}")

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
            "horizon": self.horizon,
            "best_model": self.best_model,
            "overall_verdict": self.overall_verdict.value,
            "red_flag_count": len(self.red_flags),
            "warning_count": len(self.warnings),
            "metadata": _safe(self.metadata),
            "models": {
                name: {
                    "model_name": r.model_name,
                    "predictions": _safe(r.predictions[:20]),
                    "metrics": _safe(r.metrics),
                    "fitted_params": _safe(r.fitted_params),
                }
                for name, r in self.models.items()
            },
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


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _mae(actual: List[float], predicted: List[float]) -> float:
    n = min(len(actual), len(predicted))
    if n < 1:
        return float("nan")
    return sum(abs(actual[i] - predicted[i]) for i in range(n)) / n


def _rmse(actual: List[float], predicted: List[float]) -> float:
    n = min(len(actual), len(predicted))
    if n < 1:
        return float("nan")
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
    total_abs = sum(abs(actual[i]) for i in range(n))
    if total_abs == 0:
        return float("nan")
    total_err = sum(abs(actual[i] - predicted[i]) for i in range(n))
    return total_err / total_abs * 100


def _me_bias(actual: List[float], predicted: List[float]) -> float:
    """Mean Error / forecast bias. Positive = over-forecast."""
    n = min(len(actual), len(predicted))
    if n < 1:
        return float("nan")
    return sum(predicted[i] - actual[i] for i in range(n)) / n


def _fva(model_mae: float, naive_mae: float) -> float:
    """Forecast Value Added (%). Positive = model beats naive."""
    if naive_mae == 0:
        return float("nan")
    return (naive_mae - model_mae) / naive_mae * 100


def _permutation_entropy(d: List[float], order: int = 3, delay: int = 1) -> float:
    """Permutation entropy, normalized to [0, 1]. Lower = more predictable."""
    n = len(d)
    if n < (order - 1) * delay + 1:
        return float("nan")
    pattern_counts: Dict[Tuple[int, ...], int] = {}
    total = 0
    for i in range(n - (order - 1) * delay):
        subseq = [d[i + j * delay] for j in range(order)]
        indexed = sorted(range(order), key=lambda k: subseq[k])
        pattern = tuple(indexed)
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        total += 1
    if total == 0:
        return float("nan")
    entropy = 0.0
    for count in pattern_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log(p)
    max_entropy = math.log(math.factorial(order))
    if max_entropy == 0:
        return float("nan")
    return entropy / max_entropy


def _r_squared(actual: List[float], predicted: List[float]) -> float:
    n = min(len(actual), len(predicted))
    if n < 2:
        return float("nan")
    m = _mean(actual[:n])
    ss_res = sum((actual[i] - predicted[i]) ** 2 for i in range(n))
    ss_tot = sum((actual[i] - m) ** 2 for i in range(n))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# Naive forecast helpers
# ---------------------------------------------------------------------------


def _naive_forecast(train: List[float], horizon: int) -> List[float]:
    """Naive: repeat last value."""
    return [train[-1]] * horizon


def _seasonal_naive(train: List[float], period: int, horizon: int) -> List[float]:
    """Seasonal naive: repeat last season."""
    season = train[-period:]
    reps = horizon // period + 1
    return (season * reps)[:horizon]


def _drift_forecast(train: List[float], horizon: int) -> List[float]:
    """Drift: naive with linear trend."""
    n = len(train)
    if n < 2:
        return [train[-1]] * horizon
    slope = (train[-1] - train[0]) / (n - 1)
    return [train[-1] + slope * h for h in range(1, horizon + 1)]


def _moving_average(train: List[float], k: int, horizon: int) -> List[float]:
    """Moving average: mean of last k values."""
    ma = _mean(train[-k:])
    return [ma] * horizon


# ---------------------------------------------------------------------------
# Conformal prediction helpers
# ---------------------------------------------------------------------------


def _conformal_intervals(
    calibration_residuals: List[float],
    point_forecasts: List[float],
    coverage: float = 0.95,
) -> List[Tuple[float, float]]:
    """Split conformal prediction intervals."""
    abs_residuals = sorted(abs(r) for r in calibration_residuals)
    n = len(abs_residuals)
    if n < 1:
        return [(f, f) for f in point_forecasts]
    q_idx = min(int(math.ceil((n + 1) * coverage)) - 1, n - 1)
    q_idx = max(0, q_idx)
    threshold = abs_residuals[q_idx]
    return [(f - threshold, f + threshold) for f in point_forecasts]


def _cqr_intervals(
    cal_actuals: List[float],
    cal_lower: List[float],
    cal_upper: List[float],
    test_lower: List[float],
    test_upper: List[float],
    coverage: float = 0.95,
) -> List[Tuple[float, float]]:
    """Conformalized Quantile Regression intervals."""
    n_cal = min(len(cal_actuals), len(cal_lower), len(cal_upper))
    if n_cal < 1:
        return [(lo, hi) for lo, hi in zip(test_lower, test_upper)]
    scores = []
    for i in range(n_cal):
        score = max(cal_lower[i] - cal_actuals[i], cal_actuals[i] - cal_upper[i])
        scores.append(score)
    scores_sorted = sorted(scores)
    q_idx = min(int(math.ceil((n_cal + 1) * coverage)) - 1, n_cal - 1)
    q_idx = max(0, q_idx)
    q_correction = scores_sorted[q_idx]
    n_test = min(len(test_lower), len(test_upper))
    return [
        (test_lower[i] - q_correction, test_upper[i] + q_correction)
        for i in range(n_test)
    ]


# ---------------------------------------------------------------------------
# Feature engineering helpers (for CatBoost)
# ---------------------------------------------------------------------------


def _create_lag_features(
    series: List[float], lags: Optional[List[int]] = None
) -> Dict[str, List[Optional[float]]]:
    """Create lagged values of the series."""
    if lags is None:
        lags = [1, 2, 3, 7, 14]
    n = len(series)
    features: Dict[str, List[Optional[float]]] = {}
    for lag in lags:
        col = f"lag_{lag}"
        features[col] = [None] * lag + [series[i] for i in range(n - lag)]
    return features


def _create_rolling_features(
    series: List[float], windows: Optional[List[int]] = None
) -> Dict[str, List[Optional[float]]]:
    """Create rolling mean/std with shift(1) to prevent leakage."""
    if windows is None:
        windows = [7, 14, 28]
    n = len(series)
    features: Dict[str, List[Optional[float]]] = {}
    for w in windows:
        rmean_col = f"rmean_{w}"
        rstd_col = f"rstd_{w}"
        rmean: List[Optional[float]] = []
        rstd: List[Optional[float]] = []
        for i in range(n):
            # shift(1): use series[max(0,i-w):i] (excludes current point)
            if i < w + 1:
                rmean.append(None)
                rstd.append(None)
            else:
                window_data = series[i - w:i]
                rmean.append(_mean(window_data))
                rstd.append(_std(window_data) if len(window_data) > 1 else 0.0)
        features[rmean_col] = rmean
        features[rstd_col] = rstd
    return features


def _create_calendar_features(n: int, freq: int = 1) -> Dict[str, List[int]]:
    """Create simple periodic index features."""
    features: Dict[str, List[int]] = {}
    if freq > 1:
        features["season_idx"] = [i % freq for i in range(n)]
        features["season_half"] = [0 if (i % freq) < freq // 2 else 1 for i in range(n)]
    features["time_idx"] = list(range(n))
    return features


def _create_fourier_features(
    n: int, period: int, n_harmonics: int = 3
) -> Dict[str, List[float]]:
    """Fourier sin/cos terms for seasonal patterns."""
    features: Dict[str, List[float]] = {}
    for k in range(1, n_harmonics + 1):
        features[f"sin_{period}_{k}"] = [
            math.sin(2 * math.pi * k * t / period) for t in range(n)
        ]
        features[f"cos_{period}_{k}"] = [
            math.cos(2 * math.pi * k * t / period) for t in range(n)
        ]
    return features


def _build_feature_matrix(
    series: List[float],
    freq: int = 1,
    lags: Optional[List[int]] = None,
    windows: Optional[List[int]] = None,
    n_harmonics: int = 3,
) -> Tuple[List[List[float]], List[str]]:
    """
    Build tabular feature matrix from time series for ML models.
    Returns (rows, column_names). Rows with None values are dropped.
    """
    n = len(series)
    all_features: Dict[str, List] = {}

    # Lag features
    lag_feats = _create_lag_features(series, lags)
    all_features.update(lag_feats)

    # Rolling features
    roll_feats = _create_rolling_features(series, windows)
    all_features.update(roll_feats)

    # Calendar features
    cal_feats = _create_calendar_features(n, freq)
    all_features.update(cal_feats)

    # Fourier features
    if freq > 1:
        fourier_feats = _create_fourier_features(n, freq, n_harmonics)
        all_features.update(fourier_feats)

    col_names = list(all_features.keys())

    # Build rows, drop any with None
    valid_rows: List[Tuple[int, List[float]]] = []
    for i in range(n):
        row = []
        has_none = False
        for col in col_names:
            val = all_features[col][i]
            if val is None:
                has_none = True
                break
            row.append(float(val))
        if not has_none:
            valid_rows.append((i, row))

    return [r for _, r in valid_rows], col_names


# ===========================================================================
# 3. THE FORECAST MODELER
# ===========================================================================


class ForecastModeler:
    """
    Forecasting model fitting and selection framework.

    Phases:
        1. Forecastability Gate  — PE, naive baselines, FVA go/no-go
        2. Classical Fitting     — Auto-ARIMA, Auto-ETS (statsmodels/pmdarima)
        3. ML Fitting            — CatBoost with feature engineering
        4. Model Comparison      — MASE, RMSSE, WAPE, ME bias, FVA, decision tree
        5. Conformal Prediction  — Split conformal (ICP), CQR
        6. Forecast Generation   — Point forecasts + intervals
        7. Report                — Metrics suite, model recommendation

    Parameters
    ----------
    data : sequence of float-like
        Time series values in temporal order.
    name : str
        Human-readable label.
    frequency : int | None
        Seasonal period (e.g. 12 for monthly, 7 for daily-weekly).
    """

    def __init__(
        self,
        data: Sequence,
        *,
        name: str = "series",
        frequency: Optional[int] = None,
    ):
        self._raw = list(data)
        self._clean: List[float] = _to_floats(data)
        self.name = name
        self.frequency = frequency or 1
        self._report = ForecastReport(
            series_name=name,
            n_observations=len(self._clean),
        )
        # Internal state
        self._train: Optional[List[float]] = None
        self._test: Optional[List[float]] = None
        self._calib: Optional[List[float]] = None
        self._naive_baselines: Dict[str, ForecastResult] = {}
        self._fitted_models: Dict[str, ForecastResult] = {}
        self._best_model_name: Optional[str] = None

    @property
    def report(self) -> ForecastReport:
        return self._report

    def _split_data(self, horizon: int) -> None:
        """Split data into train / calibration / test."""
        d = self._clean
        n = len(d)
        # Reserve horizon for test, 20% of remaining for calibration
        test_size = min(horizon, max(1, n // 5))
        remaining = n - test_size
        calib_size = max(1, remaining // 5)
        train_size = remaining - calib_size

        if train_size < 10:
            # Not enough data — use all for train, no split
            self._train = d
            self._calib = d[-max(10, n // 4):]
            self._test = d[-test_size:]
        else:
            self._train = d[:train_size]
            self._calib = d[train_size:train_size + calib_size]
            self._test = d[train_size + calib_size:]

    # ======================================================================
    # PHASE 1: FORECASTABILITY GATE
    # ======================================================================

    def phase_forecastability(self) -> PhaseResult:
        """
        Assess forecastability: PE, naive baselines, FVA gate.
        Always runs (stdlib only). Determines go/no-go.
        """
        ph = PhaseResult("Phase 1: Forecastability Gate")
        d = self._clean
        n = len(d)

        if n < 10:
            ph.add("data_length", Verdict.REJECT, Severity.CRITICAL,
                    f"Only {n} observations — cannot forecast", n)
            self._report.phases.append(ph)
            return ph

        if n < 30:
            ph.add("data_length", Verdict.WARN, Severity.MEDIUM,
                    f"{n} observations is marginal for model fitting (need >=50 ideally)", n)
        else:
            ph.add("data_length", Verdict.PASS, Severity.INFO,
                    f"{n} observations available", n)

        # --- Permutation Entropy ---
        pe_order = 3 if n < 200 else (4 if n < 1000 else 5)
        pe = _permutation_entropy(d, order=pe_order, delay=1)
        if math.isfinite(pe):
            if pe > 0.95:
                ph.add("permutation_entropy", Verdict.FAIL, Severity.HIGH,
                        f"PE={pe:.3f} (order={pe_order}) — effectively random. "
                        "Series likely unforecastable by any method.",
                        {"pe": round(pe, 4), "order": pe_order})
            elif pe > 0.85:
                ph.add("permutation_entropy", Verdict.WARN, Severity.MEDIUM,
                        f"PE={pe:.3f} (order={pe_order}) — high complexity. "
                        "Expect limited forecastability.",
                        {"pe": round(pe, 4), "order": pe_order})
            elif pe < 0.5:
                ph.add("permutation_entropy", Verdict.PASS, Severity.INFO,
                        f"PE={pe:.3f} (order={pe_order}) — strong structure. "
                        "Good forecastability expected.",
                        {"pe": round(pe, 4), "order": pe_order})
            else:
                ph.add("permutation_entropy", Verdict.PASS, Severity.INFO,
                        f"PE={pe:.3f} (order={pe_order}) — moderate complexity.",
                        {"pe": round(pe, 4), "order": pe_order})
            self._report.metadata["permutation_entropy"] = round(pe, 4)

        # --- Naive baselines (in-sample) ---
        sp = self.frequency
        eval_start = max(sp + 1, n // 4)
        if n > eval_start + 5:
            eval_actual = d[eval_start:]

            # Naive (random walk)
            naive_pred = [d[i - 1] for i in range(eval_start, n)]
            naive_mae = _mae(eval_actual, naive_pred)

            # Seasonal naive
            snaive_mae = float("nan")
            if sp > 1 and n > 2 * sp:
                snaive_pred = [d[i - sp] for i in range(eval_start, n)]
                snaive_mae = _mae(eval_actual, snaive_pred)

            # Drift
            drift_pred = []
            for i in range(eval_start, n):
                slope = (d[i - 1] - d[0]) / max(1, i - 1)
                drift_pred.append(d[i - 1] + slope)
            drift_mae = _mae(eval_actual, drift_pred)

            baselines = {"naive": naive_mae, "drift": drift_mae}
            best_name, best_mae = "naive", naive_mae
            if math.isfinite(snaive_mae):
                baselines["seasonal_naive"] = snaive_mae
                if snaive_mae < best_mae:
                    best_name, best_mae = "seasonal_naive", snaive_mae
            if drift_mae < best_mae:
                best_name, best_mae = "drift", drift_mae

            ph.add("naive_baselines", Verdict.PASS, Severity.INFO,
                    f"Best naive baseline: {best_name} (MAE={best_mae:.6f})",
                    {k: round(v, 6) for k, v in baselines.items()})

            self._report.metadata["best_naive"] = best_name
            self._report.metadata["best_naive_mae"] = round(best_mae, 6)
            self._report.metadata["naive_baselines"] = {
                k: round(v, 6) for k, v in baselines.items()
            }

        # --- Constant series check ---
        if len(set(d)) <= 1:
            ph.add("constant_series", Verdict.REJECT, Severity.CRITICAL,
                    "Series is constant — zero information content")
        elif len(set(d)) <= 3:
            ph.add("low_cardinality", Verdict.WARN, Severity.MEDIUM,
                    f"Only {len(set(d))} unique values")

        # --- Stationarity hint (heuristic, no deps needed) ---
        if n >= 40:
            half = n // 2
            m1, m2 = _mean(d[:half]), _mean(d[half:])
            s1, s2 = _std(d[:half]), _std(d[half:])
            overall_std = _std(d)
            mean_shift = abs(m2 - m1) / max(overall_std, 1e-12)
            if mean_shift > 2.0:
                ph.add("stationarity_hint", Verdict.WARN, Severity.LOW,
                        f"Large mean shift ({mean_shift:.2f}s) — differencing may help",
                        round(mean_shift, 3))
            else:
                ph.add("stationarity_hint", Verdict.PASS, Severity.INFO,
                        f"Mean shift: {mean_shift:.2f}s", round(mean_shift, 3))

        self._report.phases.append(ph)
        return ph

    # ======================================================================
    # PHASE 2: CLASSICAL FITTING (ARIMA / ETS)
    # ======================================================================

    def phase_classical_fitting(self, horizon: int = 1) -> PhaseResult:
        """
        Fit ARIMA and ETS models. Requires statsmodels/pmdarima.
        """
        ph = PhaseResult("Phase 2: Classical Fitting")

        if not self._train:
            self._split_data(horizon)

        train = self._train
        test = self._test
        n_train = len(train)

        if n_train < 20:
            ph.add("classical_skip", Verdict.SKIP, Severity.INFO,
                    "Insufficient training data for classical models")
            self._report.phases.append(ph)
            return ph

        sp = self.frequency
        h = len(test) if test else horizon

        # --- Auto-ARIMA ---
        if _HAS_PMDARIMA:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    arima_model = _auto_arima(
                        train,
                        seasonal=sp > 1,
                        m=sp if sp > 1 else 1,
                        d=None, D=None,
                        start_p=0, max_p=3,
                        start_q=0, max_q=3,
                        start_P=0, max_P=2,
                        start_Q=0, max_Q=2,
                        information_criterion="aicc",
                        stepwise=True,
                        suppress_warnings=True,
                        error_action="ignore",
                    )
                arima_preds = list(arima_model.predict(n_periods=h))
                arima_order = arima_model.order
                arima_seasonal = getattr(arima_model, "seasonal_order", None)
                arima_aic = float(arima_model.aicc()) if hasattr(arima_model, "aicc") else float("nan")

                # In-sample predictions for calibration
                arima_insample = list(arima_model.predict_in_sample())

                metrics = {}
                if test and len(test) >= 2:
                    t = test[:len(arima_preds)]
                    p = arima_preds[:len(t)]
                    metrics = {
                        "mae": round(_mae(t, p), 6),
                        "rmse": round(_rmse(t, p), 6),
                        "mase": round(_mase(t, p, sp), 6),
                    }

                result = ForecastResult(
                    model_name="arima",
                    predictions=arima_preds,
                    train_predictions=arima_insample,
                    metrics=metrics,
                    fitted_params={
                        "order": list(arima_order),
                        "seasonal_order": list(arima_seasonal) if arima_seasonal else None,
                        "aicc": round(arima_aic, 2) if math.isfinite(arima_aic) else None,
                    },
                )
                self._fitted_models["arima"] = result

                ph.add("arima_fit", Verdict.PASS, Severity.INFO,
                        f"Auto-ARIMA{arima_order} fitted"
                        + (f" seasonal{arima_seasonal}" if arima_seasonal and arima_seasonal[3] > 1 else ""),
                        result.fitted_params)
            except Exception as e:
                ph.add("arima_fit", Verdict.WARN, Severity.MEDIUM,
                        f"Auto-ARIMA failed: {e}")

        elif _HAS_STATSMODELS:
            # Fallback: fit a simple ARIMA(1,1,1)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    order = (1, 1, 1)
                    s_order = (1, 1, 1, sp) if sp > 1 and n_train > 2 * sp else (0, 0, 0, 0)
                    model = SARIMAX(train, order=order, seasonal_order=s_order,
                                    enforce_stationarity=False, enforce_invertibility=False)
                    fitted = model.fit(disp=False, maxiter=200)
                    arima_preds = list(fitted.forecast(steps=h))
                    arima_insample = list(fitted.fittedvalues)

                    metrics = {}
                    if test and len(test) >= 2:
                        t = test[:len(arima_preds)]
                        p = arima_preds[:len(t)]
                        metrics = {
                            "mae": round(_mae(t, p), 6),
                            "rmse": round(_rmse(t, p), 6),
                            "mase": round(_mase(t, p, sp), 6),
                        }

                    result = ForecastResult(
                        model_name="arima",
                        predictions=arima_preds,
                        train_predictions=arima_insample,
                        metrics=metrics,
                        fitted_params={"order": list(order), "seasonal_order": list(s_order)},
                    )
                    self._fitted_models["arima"] = result
                    ph.add("arima_fit", Verdict.PASS, Severity.INFO,
                            f"SARIMAX{order} fitted (no auto-ARIMA; install pmdarima for auto search)",
                            result.fitted_params)
            except Exception as e:
                ph.add("arima_fit", Verdict.WARN, Severity.MEDIUM,
                        f"ARIMA fitting failed: {e}")
        else:
            ph.add("arima_skip", Verdict.SKIP, Severity.INFO,
                    "statsmodels/pmdarima not available — ARIMA fitting skipped")

        # --- Auto-ETS ---
        if _HAS_STATSMODELS:
            best_ets = None
            best_ets_aic = float("inf")

            # Try common ETS configurations
            ets_configs = [
                {"error": "add", "trend": "add", "seasonal": None, "damped_trend": False},
                {"error": "add", "trend": "add", "seasonal": None, "damped_trend": True},
            ]
            if sp > 1 and n_train >= 2 * sp:
                ets_configs.extend([
                    {"error": "add", "trend": "add", "seasonal": "add", "damped_trend": True,
                     "seasonal_periods": sp},
                    {"error": "add", "trend": "add", "seasonal": "mul", "damped_trend": True,
                     "seasonal_periods": sp},
                ])
                # Only try multiplicative error if all values positive
                if all(v > 0 for v in train):
                    ets_configs.extend([
                        {"error": "mul", "trend": "add", "seasonal": "mul", "damped_trend": True,
                         "seasonal_periods": sp},
                    ])

            for cfg in ets_configs:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        ets_model = ETSModel(train, **cfg)
                        ets_fitted = ets_model.fit(disp=False, maxiter=200)
                        aic = ets_fitted.aic
                        if math.isfinite(aic) and aic < best_ets_aic:
                            best_ets_aic = aic
                            best_ets = (ets_fitted, cfg)
                except Exception:
                    continue

            if best_ets is not None:
                ets_fitted, ets_cfg = best_ets
                try:
                    ets_forecast = ets_fitted.forecast(steps=h)
                    ets_preds = list(ets_forecast)
                    ets_insample = list(ets_fitted.fittedvalues)

                    metrics = {}
                    if test and len(test) >= 2:
                        t = test[:len(ets_preds)]
                        p = ets_preds[:len(t)]
                        metrics = {
                            "mae": round(_mae(t, p), 6),
                            "rmse": round(_rmse(t, p), 6),
                            "mase": round(_mase(t, p, sp), 6),
                        }

                    cfg_str = f"E={ets_cfg['error']},T={ets_cfg['trend']}" \
                              f"{'d' if ets_cfg.get('damped_trend') else ''}" \
                              f",S={ets_cfg.get('seasonal', 'N')}"
                    result = ForecastResult(
                        model_name="ets",
                        predictions=ets_preds,
                        train_predictions=ets_insample,
                        metrics=metrics,
                        fitted_params={
                            "config": cfg_str,
                            "aic": round(best_ets_aic, 2),
                        },
                    )
                    self._fitted_models["ets"] = result
                    ph.add("ets_fit", Verdict.PASS, Severity.INFO,
                            f"ETS({cfg_str}) fitted (AIC={best_ets_aic:.1f})",
                            result.fitted_params)
                except Exception as e:
                    ph.add("ets_fit", Verdict.WARN, Severity.MEDIUM,
                            f"ETS forecast generation failed: {e}")
            else:
                ph.add("ets_fit", Verdict.WARN, Severity.LOW,
                        "No ETS configuration converged")
        else:
            ph.add("ets_skip", Verdict.SKIP, Severity.INFO,
                    "statsmodels not available — ETS fitting skipped")

        self._report.phases.append(ph)
        return ph

    # ======================================================================
    # PHASE 3: ML FITTING (CatBoost)
    # ======================================================================

    def phase_ml_fitting(self, horizon: int = 1) -> PhaseResult:
        """
        Fit CatBoost with feature engineering. Requires catboost + numpy.
        """
        ph = PhaseResult("Phase 3: ML Fitting")

        if not _HAS_CATBOOST or not _HAS_NUMPY:
            missing = []
            if not _HAS_CATBOOST:
                missing.append("catboost")
            if not _HAS_NUMPY:
                missing.append("numpy")
            ph.add("ml_skip", Verdict.SKIP, Severity.INFO,
                    f"ML fitting skipped — missing: {', '.join(missing)}")
            self._report.phases.append(ph)
            return ph

        if not self._train:
            self._split_data(horizon)

        train = self._train
        test = self._test
        calib = self._calib
        sp = self.frequency
        h = len(test) if test else horizon

        if len(train) < 50:
            ph.add("ml_skip", Verdict.SKIP, Severity.INFO,
                    "Insufficient training data for CatBoost (<50 observations)")
            self._report.phases.append(ph)
            return ph

        # --- Feature Engineering ---
        # Determine lags based on frequency
        if sp > 1:
            lags = sorted(set([1, 2, 3] + [sp, 2 * sp]))
        else:
            lags = [1, 2, 3, 7, 14]
        lags = [l for l in lags if l < len(train) // 3]

        windows = [7, 14]
        if sp > 1:
            windows = sorted(set([sp, 2 * sp]))
        windows = [w for w in windows if w < len(train) // 3]

        try:
            # Build features for full series (train + calib + test)
            full_series = train + (calib or []) + (test or [])
            rows, col_names = _build_feature_matrix(
                full_series, freq=sp, lags=lags, windows=windows,
                n_harmonics=min(3, max(1, sp // 4)) if sp > 1 else 1,
            )

            if len(rows) < 30:
                ph.add("ml_features", Verdict.SKIP, Severity.INFO,
                        "Too few valid feature rows after engineering")
                self._report.phases.append(ph)
                return ph

            # Map rows back to original indices
            n_full = len(full_series)
            n_train = len(train)
            n_calib = len(calib) if calib else 0

            # Determine valid row start index (rows with None were dropped)
            warmup = max(lags) + max(windows) + 1 if windows else max(lags) + 1

            # Build target (y = value at time t)
            target = full_series[warmup:]
            X_data = rows[:len(target)]

            if len(X_data) != len(target):
                # Align
                min_len = min(len(X_data), len(target))
                X_data = X_data[:min_len]
                target = target[:min_len]

            X_arr = np.array(X_data, dtype=np.float64)
            y_arr = np.array(target, dtype=np.float64)

            # Split into train/calib/test portions
            train_end = max(0, n_train - warmup)
            calib_end = train_end + n_calib

            if train_end < 20:
                ph.add("ml_data", Verdict.SKIP, Severity.INFO,
                        "Insufficient valid training rows for CatBoost")
                self._report.phases.append(ph)
                return ph

            X_train_ml = X_arr[:train_end]
            y_train_ml = y_arr[:train_end]
            X_calib_ml = X_arr[train_end:calib_end] if calib_end > train_end else X_arr[train_end:]
            y_calib_ml = y_arr[train_end:calib_end] if calib_end > train_end else y_arr[train_end:]
            X_test_ml = X_arr[calib_end:] if calib_end < len(X_arr) else None
            y_test_ml = y_arr[calib_end:] if calib_end < len(y_arr) else None

            # --- Point forecast model ---
            model_point = CatBoostRegressor(
                iterations=500,
                learning_rate=0.05,
                depth=6,
                l2_leaf_reg=3.0,
                loss_function="RMSE",
                random_seed=42,
                verbose=0,
            )
            model_point.fit(X_train_ml, y_train_ml)

            # --- Quantile models (for CQR) ---
            alpha = 0.10  # For 90% intervals
            model_lower = CatBoostRegressor(
                iterations=500,
                learning_rate=0.05,
                depth=6,
                loss_function=f"Quantile:alpha={alpha / 2}",
                random_seed=42,
                verbose=0,
            )
            model_upper = CatBoostRegressor(
                iterations=500,
                learning_rate=0.05,
                depth=6,
                loss_function=f"Quantile:alpha={1 - alpha / 2}",
                random_seed=42,
                verbose=0,
            )
            model_lower.fit(X_train_ml, y_train_ml)
            model_upper.fit(X_train_ml, y_train_ml)

            # Generate predictions
            cb_preds = []
            cb_lower_preds = []
            cb_upper_preds = []
            cb_train_preds = list(model_point.predict(X_train_ml))

            if X_test_ml is not None and len(X_test_ml) > 0:
                cb_preds = list(model_point.predict(X_test_ml))
                cb_lower_preds = list(model_lower.predict(X_test_ml))
                cb_upper_preds = list(model_upper.predict(X_test_ml))
            elif X_calib_ml is not None and len(X_calib_ml) > 0:
                cb_preds = list(model_point.predict(X_calib_ml))
                cb_lower_preds = list(model_lower.predict(X_calib_ml))
                cb_upper_preds = list(model_upper.predict(X_calib_ml))

            # Calibration residuals for CQR
            if len(X_calib_ml) > 0:
                calib_point = list(model_point.predict(X_calib_ml))
                calib_lower_q = list(model_lower.predict(X_calib_ml))
                calib_upper_q = list(model_upper.predict(X_calib_ml))
                calib_actual = list(y_calib_ml)
            else:
                calib_point = cb_train_preds[-20:]
                calib_actual = list(y_train_ml[-20:])
                calib_lower_q = calib_upper_q = None

            metrics = {}
            test_actual = list(y_test_ml) if y_test_ml is not None and len(y_test_ml) > 0 else None
            if test_actual and cb_preds:
                t = test_actual[:len(cb_preds)]
                p = cb_preds[:len(t)]
                metrics = {
                    "mae": round(_mae(t, p), 6),
                    "rmse": round(_rmse(t, p), 6),
                    "mase": round(_mase(t, p, sp), 6),
                }

            result = ForecastResult(
                model_name="catboost",
                predictions=cb_preds,
                lower=cb_lower_preds if cb_lower_preds else None,
                upper=cb_upper_preds if cb_upper_preds else None,
                train_predictions=cb_train_preds,
                metrics=metrics,
                fitted_params={
                    "n_features": len(col_names),
                    "feature_names": col_names[:10],
                    "train_rows": len(X_train_ml),
                    "calib_rows": len(X_calib_ml),
                },
            )
            self._fitted_models["catboost"] = result

            # Store calibration data for CQR
            if calib_lower_q and calib_upper_q:
                self._report.metadata["catboost_cqr_calib"] = {
                    "actuals": [round(v, 6) for v in calib_actual[:50]],
                    "lower": [round(v, 6) for v in calib_lower_q[:50]],
                    "upper": [round(v, 6) for v in calib_upper_q[:50]],
                }

            n_feats = len(col_names)
            ph.add("catboost_fit", Verdict.PASS, Severity.INFO,
                    f"CatBoost fitted ({n_feats} features, "
                    f"{len(X_train_ml)} train rows)",
                    result.fitted_params)

        except Exception as e:
            ph.add("catboost_fit", Verdict.WARN, Severity.MEDIUM,
                    f"CatBoost fitting failed: {e}")

        self._report.phases.append(ph)
        return ph

    # ======================================================================
    # PHASE 4: MODEL COMPARISON
    # ======================================================================

    def phase_model_comparison(self) -> PhaseResult:
        """
        Compare all fitted models against naive baselines.
        Uses MASE, RMSSE, WAPE, ME bias, FVA.
        """
        ph = PhaseResult("Phase 4: Model Comparison")

        if not self._fitted_models:
            # Only naive baselines available
            ph.add("no_models", Verdict.WARN, Severity.MEDIUM,
                    "No models fitted — only naive baselines available. "
                    "Install statsmodels, pmdarima, or catboost for model fitting.")
            self._report.phases.append(ph)
            return ph

        test = self._test
        train = self._train
        sp = self.frequency
        naive_mae = self._report.metadata.get("best_naive_mae", float("nan"))

        # Compute additional metrics for each model
        for name, result in self._fitted_models.items():
            if not result.predictions or not test:
                continue
            n_cmp = min(len(result.predictions), len(test))
            if n_cmp < 2:
                continue

            t = test[:n_cmp]
            p = result.predictions[:n_cmp]

            result.metrics["mae"] = round(_mae(t, p), 6)
            result.metrics["rmse"] = round(_rmse(t, p), 6)
            result.metrics["mase"] = round(_mase(t, p, sp), 6)
            result.metrics["wape"] = round(_wape(t, p), 6)
            result.metrics["me_bias"] = round(_me_bias(t, p), 6)
            result.metrics["r2"] = round(_r_squared(t, p), 6)

            if train:
                result.metrics["rmsse"] = round(_rmsse(t, p, train, sp), 6)

            # FVA against best naive
            if math.isfinite(naive_mae) and naive_mae > 0:
                fva_val = _fva(result.metrics["mae"], naive_mae)
                result.metrics["fva"] = round(fva_val, 2)

        # --- Rank by MASE ---
        ranked = sorted(
            self._fitted_models.items(),
            key=lambda x: x[1].metrics.get("mase", float("inf")),
        )

        if ranked:
            best_name, best_result = ranked[0]
            best_mase = best_result.metrics.get("mase", float("nan"))
            best_fva = best_result.metrics.get("fva", float("nan"))

            # Decision tree evaluation
            if math.isfinite(best_mase):
                if best_mase >= 1.5:
                    ph.add("best_model_quality", Verdict.FAIL, Severity.HIGH,
                            f"Best model ({best_name}) MASE={best_mase:.3f} — "
                            "much worse than naive. Use naive baseline instead.",
                            {"model": best_name, "mase": best_mase})
                elif best_mase >= 1.0:
                    ph.add("best_model_quality", Verdict.WARN, Severity.MEDIUM,
                            f"Best model ({best_name}) MASE={best_mase:.3f} — "
                            "does not beat naive. Complexity not justified.",
                            {"model": best_name, "mase": best_mase})
                elif best_mase >= 0.9:
                    ph.add("best_model_quality", Verdict.PASS, Severity.INFO,
                            f"Best model ({best_name}) MASE={best_mase:.3f} — "
                            "marginal improvement over naive.",
                            {"model": best_name, "mase": best_mase})
                else:
                    ph.add("best_model_quality", Verdict.PASS, Severity.INFO,
                            f"Best model ({best_name}) MASE={best_mase:.3f} — "
                            "good improvement over naive.",
                            {"model": best_name, "mase": best_mase})

            # FVA check
            if math.isfinite(best_fva):
                if best_fva < 0:
                    ph.add("fva_check", Verdict.FAIL, Severity.HIGH,
                            f"FVA={best_fva:.1f}% — model destroys value vs naive",
                            round(best_fva, 2))
                elif best_fva < 10:
                    ph.add("fva_check", Verdict.WARN, Severity.MEDIUM,
                            f"FVA={best_fva:.1f}% — marginal improvement",
                            round(best_fva, 2))
                else:
                    ph.add("fva_check", Verdict.PASS, Severity.INFO,
                            f"FVA={best_fva:.1f}% — model adds substantial value",
                            round(best_fva, 2))

            # Simplicity check: if simple model within 5% of complex
            if len(ranked) >= 2:
                simple_models = ["arima", "ets"]
                complex_models = ["catboost"]
                simple_best = None
                complex_best = None
                for n, r in ranked:
                    if n in simple_models and simple_best is None:
                        simple_best = (n, r)
                    if n in complex_models and complex_best is None:
                        complex_best = (n, r)
                if simple_best and complex_best:
                    s_mase = simple_best[1].metrics.get("mase", float("inf"))
                    c_mase = complex_best[1].metrics.get("mase", float("inf"))
                    if math.isfinite(s_mase) and math.isfinite(c_mase) and c_mase > 0:
                        improvement = (s_mase - c_mase) / c_mase * 100
                        if improvement < 5:
                            ph.add("simplicity_check", Verdict.PASS, Severity.INFO,
                                    f"Simple model ({simple_best[0]}) within 5% of "
                                    f"complex ({complex_best[0]}) — prefer simple (Occam's razor).",
                                    {"improvement_pct": round(improvement, 2)})

            self._best_model_name = best_name
            self._report.best_model = best_name

            # Summary table
            comparison = {}
            for n, r in ranked:
                comparison[n] = {k: v for k, v in r.metrics.items() if isinstance(v, (int, float))}
            ph.add("model_ranking", Verdict.PASS, Severity.INFO,
                    f"Model ranking (by MASE): {' > '.join(n for n, _ in ranked)}",
                    comparison)

        self._report.phases.append(ph)
        return ph

    # ======================================================================
    # PHASE 5: CONFORMAL PREDICTION
    # ======================================================================

    def phase_conformal_prediction(self, coverage: float = 0.95) -> PhaseResult:
        """
        Produce conformal prediction intervals for the best model.
        Uses split conformal (ICP) + CQR if quantile predictions available.
        """
        ph = PhaseResult("Phase 5: Conformal Prediction")

        best_name = self._best_model_name
        if not best_name or best_name not in self._fitted_models:
            # Fall back to naive
            if self._train and self._test:
                naive_preds = _naive_forecast(self._train, len(self._test))
                calib_data = self._calib or self._train[-50:]
                calib_preds = [self._train[max(0, len(self._train) - len(calib_data) + i - 1)]
                               for i in range(len(calib_data))]
                residuals = [calib_data[i] - calib_preds[i] for i in range(len(calib_data))]
                intervals = _conformal_intervals(residuals, naive_preds, coverage)

                self._fitted_models["naive_conformal"] = ForecastResult(
                    model_name="naive_conformal",
                    predictions=naive_preds,
                    lower=[lo for lo, _ in intervals],
                    upper=[hi for _, hi in intervals],
                )
                ph.add("conformal_naive", Verdict.PASS, Severity.INFO,
                        f"Split conformal intervals on naive forecast (coverage={coverage:.0%})",
                        {"method": "ICP", "coverage_target": coverage})
            else:
                ph.add("conformal_skip", Verdict.SKIP, Severity.INFO,
                        "No model or data available for conformal prediction")
            self._report.phases.append(ph)
            return ph

        best_result = self._fitted_models[best_name]

        # --- Split Conformal (ICP) ---
        if best_result.train_predictions and self._calib:
            # Compute calibration residuals
            # Use calibration set: actual - predicted
            train = self._train
            calib = self._calib

            # For classical models, get predictions on calibration period
            if best_name in ("arima", "ets") and best_result.train_predictions:
                # In-sample predictions cover the training period
                # We need residuals on the calibration set
                # Approximate with last-train residuals
                insample = best_result.train_predictions
                n_use = min(len(insample), len(train))
                cal_residuals = [train[i] - insample[i] for i in range(n_use)][-len(calib):]
                if len(cal_residuals) < 10:
                    cal_residuals = [train[i] - insample[i] for i in range(n_use)]
            else:
                # For CatBoost or others with explicit calibration
                cal_residuals = []
                if hasattr(best_result, "train_predictions") and best_result.train_predictions:
                    insample = best_result.train_predictions
                    n_use = min(len(insample), len(train))
                    cal_residuals = [train[i] - insample[i] for i in range(n_use)]

            if len(cal_residuals) >= 5:
                intervals = _conformal_intervals(
                    cal_residuals, best_result.predictions, coverage
                )
                best_result.lower = [lo for lo, _ in intervals]
                best_result.upper = [hi for _, hi in intervals]

                # Evaluate coverage on test
                if self._test:
                    test = self._test
                    n_eval = min(len(test), len(best_result.lower))
                    covered = sum(
                        1 for i in range(n_eval)
                        if best_result.lower[i] <= test[i] <= best_result.upper[i]
                    )
                    actual_cov = covered / n_eval if n_eval > 0 else 0
                    widths = [best_result.upper[i] - best_result.lower[i] for i in range(n_eval)]
                    avg_width = _mean(widths) if widths else 0

                    cov_gap = actual_cov - coverage
                    if cov_gap < -0.10:
                        ph.add("conformal_coverage", Verdict.FAIL, Severity.HIGH,
                                f"Coverage {actual_cov:.1%} vs target {coverage:.0%} — "
                                "severely under-covered",
                                {"actual": round(actual_cov, 4), "target": coverage})
                    elif cov_gap < -0.05:
                        ph.add("conformal_coverage", Verdict.WARN, Severity.MEDIUM,
                                f"Coverage {actual_cov:.1%} vs target {coverage:.0%}",
                                {"actual": round(actual_cov, 4), "target": coverage})
                    else:
                        ph.add("conformal_coverage", Verdict.PASS, Severity.INFO,
                                f"Coverage {actual_cov:.1%} vs target {coverage:.0%}",
                                {"actual": round(actual_cov, 4), "target": coverage})

                    ph.add("interval_width", Verdict.PASS, Severity.INFO,
                            f"Mean interval width: {avg_width:.4f}",
                            round(avg_width, 4))

                    self._report.metadata["conformal_coverage"] = round(actual_cov, 4)
                    self._report.metadata["conformal_width"] = round(avg_width, 4)
                else:
                    ph.add("conformal_icp", Verdict.PASS, Severity.INFO,
                            f"Split conformal intervals computed (coverage={coverage:.0%})",
                            {"method": "ICP", "n_calibration": len(cal_residuals)})
            else:
                ph.add("conformal_skip", Verdict.WARN, Severity.LOW,
                        "Insufficient calibration residuals for conformal prediction")

        # --- CQR (if CatBoost quantile predictions available) ---
        cqr_calib = self._report.metadata.get("catboost_cqr_calib")
        if cqr_calib and best_name == "catboost" and best_result.lower and best_result.upper:
            try:
                cqr_intervals_result = _cqr_intervals(
                    cqr_calib["actuals"],
                    cqr_calib["lower"],
                    cqr_calib["upper"],
                    best_result.lower,
                    best_result.upper,
                    coverage,
                )
                # Use CQR intervals (they're better calibrated)
                best_result.lower = [lo for lo, _ in cqr_intervals_result]
                best_result.upper = [hi for _, hi in cqr_intervals_result]

                ph.add("cqr_applied", Verdict.PASS, Severity.INFO,
                        "CQR correction applied to CatBoost quantile intervals",
                        {"method": "CQR"})
            except Exception as e:
                ph.add("cqr_failed", Verdict.WARN, Severity.LOW,
                        f"CQR correction failed: {e}")

        self._report.phases.append(ph)
        return ph

    # ======================================================================
    # PHASE 6: FORECAST GENERATION
    # ======================================================================

    def phase_forecast_generation(self, horizon: int = 1) -> PhaseResult:
        """
        Generate final forecasts. Store in report.models.
        Also generate naive baseline forecasts for comparison.
        """
        ph = PhaseResult("Phase 6: Forecast Generation")

        train = self._train or self._clean
        sp = self.frequency

        # --- Naive baselines ---
        naive_preds = _naive_forecast(train, horizon)
        self._report.models["naive"] = ForecastResult(
            model_name="naive", predictions=naive_preds)

        if sp > 1 and len(train) >= sp:
            sn_preds = _seasonal_naive(train, sp, horizon)
            self._report.models["seasonal_naive"] = ForecastResult(
                model_name="seasonal_naive", predictions=sn_preds)

        drift_preds = _drift_forecast(train, horizon)
        self._report.models["drift"] = ForecastResult(
            model_name="drift", predictions=drift_preds)

        # --- Fitted models ---
        for name, result in self._fitted_models.items():
            self._report.models[name] = result

        n_models = len(self._report.models)
        best = self._best_model_name or "naive"
        ph.add("forecasts_generated", Verdict.PASS, Severity.INFO,
                f"{n_models} forecasts generated (best: {best}, horizon: {horizon})",
                {"n_models": n_models, "best": best, "horizon": horizon})

        self._report.horizon = horizon
        self._report.phases.append(ph)
        return ph

    # ======================================================================
    # PHASE 7: REPORT
    # ======================================================================

    def phase_report(self) -> PhaseResult:
        """
        Final summary: metrics suite, recommendations, warnings.
        """
        ph = PhaseResult("Phase 7: Report Summary")

        best = self._best_model_name
        pe = self._report.metadata.get("permutation_entropy")
        naive_mae = self._report.metadata.get("best_naive_mae")

        # --- Recommendation ---
        if best and best in self._fitted_models:
            result = self._fitted_models[best]
            mase = result.metrics.get("mase", float("nan"))
            fva = result.metrics.get("fva", float("nan"))

            rec_parts = [f"Recommended model: {best}"]
            if math.isfinite(mase):
                rec_parts.append(f"MASE={mase:.3f}")
            if math.isfinite(fva):
                rec_parts.append(f"FVA={fva:.1f}%")
            ph.add("recommendation", Verdict.PASS, Severity.INFO,
                    ". ".join(rec_parts))

            # Uncertainty status
            if result.lower and result.upper:
                cov = self._report.metadata.get("conformal_coverage")
                if cov:
                    ph.add("uncertainty_status", Verdict.PASS, Severity.INFO,
                            f"Conformal intervals available (coverage={cov:.1%})")
                else:
                    ph.add("uncertainty_status", Verdict.PASS, Severity.INFO,
                            "Conformal intervals available")
            else:
                ph.add("uncertainty_status", Verdict.WARN, Severity.MEDIUM,
                        "No prediction intervals produced — add conformal prediction layer")
        else:
            ph.add("recommendation", Verdict.WARN, Severity.MEDIUM,
                    "No model outperforms naive baselines. "
                    "Use naive/seasonal-naive forecast. "
                    "Consider: (1) more data, (2) domain features, "
                    "(3) different model class, (4) accept low forecastability.")

        # --- Forecastability summary ---
        if pe is not None:
            if pe > 0.85:
                ph.add("forecastability_summary", Verdict.WARN, Severity.MEDIUM,
                        f"Low forecastability (PE={pe:.3f}). "
                        "Invest in data collection/feature engineering before model tuning.")
            else:
                ph.add("forecastability_summary", Verdict.PASS, Severity.INFO,
                        f"Forecastability OK (PE={pe:.3f})")

        # --- Available libraries ---
        libs = {
            "numpy": _HAS_NUMPY,
            "scipy": _HAS_SCIPY,
            "statsmodels": _HAS_STATSMODELS,
            "pmdarima": _HAS_PMDARIMA,
            "catboost": _HAS_CATBOOST,
        }
        available = [k for k, v in libs.items() if v]
        missing = [k for k, v in libs.items() if not v]
        if missing:
            ph.add("library_status", Verdict.WARN, Severity.LOW,
                    f"Missing optional libraries: {', '.join(missing)}. "
                    f"Available: {', '.join(available) if available else 'none'}.",
                    libs)
        else:
            ph.add("library_status", Verdict.PASS, Severity.INFO,
                    "All optional libraries available", libs)

        self._report.metadata.update({
            "libs": libs,
            "n_models_fitted": len(self._fitted_models),
            "best_model": best,
        })

        self._report.phases.append(ph)
        return ph

    # ======================================================================
    # FULL PIPELINE (orchestrator)
    # ======================================================================

    def full_pipeline(
        self,
        *,
        horizon: int = 1,
        coverage: float = 0.95,
    ) -> ForecastReport:
        """
        Run all phases: assess → fit → compare → conformal → forecast → report.
        """
        self._report = ForecastReport(
            series_name=self.name,
            n_observations=len(self._clean),
            horizon=horizon,
        )
        self._split_data(horizon)

        # Phase 1: Forecastability gate
        self.phase_forecastability()
        if self._report.phases[-1].verdict == Verdict.REJECT:
            return self._report

        # Phase 2: Classical fitting
        self.phase_classical_fitting(horizon=horizon)

        # Phase 3: ML fitting
        self.phase_ml_fitting(horizon=horizon)

        # Phase 4: Model comparison
        self.phase_model_comparison()

        # Phase 5: Conformal prediction
        self.phase_conformal_prediction(coverage=coverage)

        # Phase 6: Forecast generation
        self.phase_forecast_generation(horizon=horizon)

        # Phase 7: Report
        self.phase_report()

        return self._report


# ===========================================================================
# 4. CONVENIENCE FUNCTIONS
# ===========================================================================


def auto_forecast(
    data: Sequence,
    horizon: int = 1,
    *,
    name: str = "series",
    frequency: Optional[int] = None,
    coverage: float = 0.95,
) -> ForecastReport:
    """One-liner: full forecasting pipeline."""
    modeler = ForecastModeler(data, name=name, frequency=frequency)
    return modeler.full_pipeline(horizon=horizon, coverage=coverage)


def compare_forecasters(
    data: Sequence,
    horizon: int = 1,
    *,
    frequency: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """Run full pipeline and return model comparison dict ranked by MASE."""
    report = auto_forecast(data, horizon, frequency=frequency)
    results = {}
    for name, res in report.models.items():
        if res.metrics:
            results[name] = {k: v for k, v in res.metrics.items()
                            if isinstance(v, (int, float))}
    return dict(sorted(results.items(), key=lambda x: x[1].get("mase", float("inf"))))


def conformal_forecast(
    point_forecasts: List[float],
    calibration_residuals: List[float],
    coverage: float = 0.95,
) -> List[Tuple[float, float]]:
    """Convenience wrapper for split conformal intervals."""
    return _conformal_intervals(calibration_residuals, point_forecasts, coverage)


# ===========================================================================
# 5. CLI DEMO
# ===========================================================================


def _demo():
    """Run a self-contained demo with synthetic data."""
    import random

    random.seed(42)
    n = 300

    # Synthetic: trend + seasonality + noise
    data = []
    for i in range(n):
        trend = 0.05 * i
        season = 15 * math.sin(2 * math.pi * i / 12)
        noise = random.gauss(0, 5)
        data.append(100 + trend + season + noise)

    print("=" * 72)
    print("  forecast_modeler.py — DEMO")
    print("=" * 72)
    print(f"  Synthetic data: {n} observations, trend + 12-period seasonality + noise")
    print(f"  Available libraries: numpy={_HAS_NUMPY}, statsmodels={_HAS_STATSMODELS}, "
          f"pmdarima={_HAS_PMDARIMA}, catboost={_HAS_CATBOOST}")
    print()

    modeler = ForecastModeler(data, name="synthetic_demo", frequency=12)
    report = modeler.full_pipeline(horizon=12, coverage=0.95)
    report.print_report()

    # Show forecasts
    if report.best_model and report.best_model in report.models:
        best = report.models[report.best_model]
        print(f"\n--- Forecast (best model: {report.best_model}) ---")
        for h, pred in enumerate(best.predictions[:12], 1):
            lo = best.lower[h - 1] if best.lower and h - 1 < len(best.lower) else "?"
            hi = best.upper[h - 1] if best.upper and h - 1 < len(best.upper) else "?"
            lo_str = f"{lo:.2f}" if isinstance(lo, float) else lo
            hi_str = f"{hi:.2f}" if isinstance(hi, float) else hi
            print(f"  h={h:2d}: {pred:.2f}  [{lo_str}, {hi_str}]")

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
        description="Forecast Modeler — fit forecasting models with conformal intervals"
    )
    subparsers = parser.add_subparsers(dest="cmd", help="Commands")

    # fit command — full pipeline
    fit_p = subparsers.add_parser("fit", help="Full forecasting pipeline from CSV")
    fit_p.add_argument("file", help="CSV file path")
    fit_p.add_argument("--column", required=True, help="Column name to forecast")
    fit_p.add_argument("--horizon", type=int, default=12, help="Forecast horizon (default: 12)")
    fit_p.add_argument("--freq", type=int, default=None, help="Seasonal period (e.g. 12 for monthly)")
    fit_p.add_argument("--coverage", type=float, default=0.95, help="Conformal coverage (default: 0.95)")
    fit_p.add_argument("--output", type=str, default=None, help="Output JSON file path")

    # assess command — forecastability only
    assess_p = subparsers.add_parser("assess", help="Forecastability assessment only (Phase 1)")
    assess_p.add_argument("file", help="CSV file path")
    assess_p.add_argument("--column", required=True, help="Column name")
    assess_p.add_argument("--freq", type=int, default=None, help="Seasonal period")

    # compare command — multi-model comparison
    compare_p = subparsers.add_parser("compare", help="Multi-model comparison from CSV")
    compare_p.add_argument("file", help="CSV file path")
    compare_p.add_argument("--column", required=True, help="Column name")
    compare_p.add_argument("--horizon", type=int, default=12, help="Forecast horizon")
    compare_p.add_argument("--freq", type=int, default=None, help="Seasonal period")

    # demo command
    subparsers.add_parser("demo", help="Run built-in synthetic data demo")

    args = parser.parse_args()

    if args.cmd == "demo":
        _demo()

    elif args.cmd in ("fit", "assess", "compare"):
        # Read CSV
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

        data = []
        for v in raw_values:
            try:
                data.append(float(v))
            except (TypeError, ValueError):
                data.append(None)

        freq = getattr(args, "freq", None)
        modeler = ForecastModeler(data, name=args.column, frequency=freq)

        if args.cmd == "assess":
            modeler.phase_forecastability()
            modeler.report.print_report()

        elif args.cmd == "compare":
            horizon = getattr(args, "horizon", 12)
            report = modeler.full_pipeline(horizon=horizon)
            # Print comparison table
            print(f"\n{'Model':<20} {'MAE':>10} {'RMSE':>10} {'MASE':>10} {'FVA%':>10}")
            print("-" * 62)
            for name, res in sorted(
                report.models.items(),
                key=lambda x: x[1].metrics.get("mase", float("inf")),
            ):
                mae = res.metrics.get("mae", float("nan"))
                rmse = res.metrics.get("rmse", float("nan"))
                mase = res.metrics.get("mase", float("nan"))
                fva = res.metrics.get("fva", float("nan"))
                marker = " *" if name == report.best_model else ""
                print(f"  {name:<18} {mae:>10.4f} {rmse:>10.4f} "
                      f"{mase:>10.4f} {fva:>10.1f}{marker}")
            print(f"\n  * = best model")

        elif args.cmd == "fit":
            horizon = args.horizon
            coverage = args.coverage
            report = modeler.full_pipeline(horizon=horizon, coverage=coverage)
            report.print_report()

            if args.output:
                try:
                    # Try common.py save_json for file locking
                    try:
                        from common import save_json
                        save_json(args.output, report.to_dict())
                    except ImportError:
                        import json
                        with open(args.output, "w") as f:
                            json.dump(report.to_dict(), f, indent=2, default=str)
                    print(f"Results saved to {args.output}")
                except Exception as e:
                    print(f"Error saving output: {e}", file=sys.stderr)
                    sys.exit(1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
