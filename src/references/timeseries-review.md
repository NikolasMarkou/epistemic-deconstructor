# Time-Series Signal Review Guide

Reference for using `ts_reviewer.py` during Epistemic Deconstruction analyses.

## Table of Contents

- [When to Use](#when-to-use)
- [Phase Mapping](#phase-mapping)
- [Interpreting Verdicts](#interpreting-verdicts)
- [Quick Reference](#quick-reference)
- [Relationship to Financial Validation](#relationship-to-financial-validation)
- [New Capabilities](#new-capabilities)
- [Validating Simulation Output](#validating-simulation-output)
- [Utility Functions for Protocol Phases](#utility-functions-for-protocol-phases)
- [Cross-References](#cross-references)

---

## When to Use

Use `ts_reviewer.py` whenever a system under investigation produces **time-ordered numeric output** — sensor readings, log metrics, price series, telemetry, periodic measurements, or any sequential signal.

| Protocol Phase | When ts_reviewer Helps |
|---|---|
| **Phase 1: Boundary Mapping** | Characterize I/O signals: stationarity, data quality, forecastability |
| **Phase 3: Parametric ID** | Validate model residuals, detect overfitting, compute baselines |
| **Phase 5: Validation** | Residual diagnostics, uncertainty calibration, regime analysis |
| **RAPID Tier** | Quick coherence check on claimed time-series results |

**Applicable tiers**: STANDARD, COMPREHENSIVE, RAPID (quick mode), LITE (phases 1-6).

---

## Phase Mapping

| ts_reviewer Phase | Protocol Phase | Purpose |
|---|---|---|
| 1. Coherence | Phase 1 | Signal length, type, constant/clipping checks |
| 2. Data Quality | Phase 1 | Missing values, outliers, stuck sensors |
| 3. Stationarity | Phase 1, 3 | ADF/KPSS tests — determines if differencing needed |
| 4. Forecastability | Phase 1 | ACF structure, entropy, permutation entropy (PE), signal-to-noise |
| 5. Decomposition | Phase 1, 3 | Trend/seasonality strength, structural breaks |
| 6. Baselines | Phase 3, 5 | Naive/seasonal/drift MAE, Forecast Value Added (FVA) — model must beat these |
| 7. Overfitting | Phase 5 | Train-vs-test gap, R2 plausibility, leakage detection |
| 8. Residuals | Phase 5 | Zero mean, no autocorrelation, homoscedasticity |
| 9. Uncertainty | Phase 5 | Coverage calibration, interval sharpness, Winkler score |
| 10. Regimes | Phase 5 | Volatility clustering, distribution shifts across windows |

**Quick mode** (`quick` subcommand) runs phases 1-6 only — sufficient for signal characterization without a model.

---

## Interpreting Verdicts

Each check produces a verdict with severity:

| Verdict | Meaning | Action |
|---|---|---|
| **PASS** | Check OK | Continue |
| **WARN** | Potential issue | Investigate, document in hypothesis tracker |
| **FAIL** | Significant problem | Address before trusting model results |
| **REJECT** | Fatal flaw | Data or model is fundamentally unsuitable |
| **SKIP** | Cannot evaluate | Insufficient data or missing dependencies |

**Escalation rule**: If the overall verdict is FAIL or REJECT, do not proceed to model synthesis (Phase 4) without addressing the flagged issues. Document any overrides in the hypothesis tracker with justification.

**Severity levels**: INFO < LOW < MEDIUM < HIGH < CRITICAL. Focus on HIGH and CRITICAL findings first.

---

## Quick Reference

```bash
# Full review from CSV file
python scripts/ts_reviewer.py review data.csv --column value

# Quick review (phases 1-6 only, no model needed)
python scripts/ts_reviewer.py quick data.csv --column temperature --freq 12

# Built-in demo with synthetic data
python scripts/ts_reviewer.py demo
```

**Programmatic use** (within Python):

```python
from ts_reviewer import TimeSeriesReviewer, quick_review

# One-liner
report = quick_review(data, name="sensor_A", frequency=24)
report.print_report()

# With model evaluation
reviewer = TimeSeriesReviewer(data, name="sensor_A")
report = reviewer.full_review(
    predictions=test_preds, actuals=test_actual,
    residuals=model_residuals
)
```

---

## Relationship to Financial Validation

`ts_reviewer.py` is a **general-purpose** signal diagnostics tool. For financial time series specifically, also consult `financial-validation.md` which adds:

- Financial disqualifiers (martingale baseline, price-level R2 trap)
- Economic significance tests (transaction costs, capacity)
- Financial-specific domain calibration bounds
- Regime decomposition into bull/bear/high-vol/low-vol

**Workflow**: Run `ts_reviewer.py` first for signal-level diagnostics, then apply `financial-validation.md` checks for domain-specific validation.

---

## New Capabilities

### Permutation Entropy (Phase 4)

Phase 4 now computes **Permutation Entropy (PE)** alongside the existing direction entropy. PE is a proper forecastability measure that captures ordinal-pattern complexity in the time series. Unlike direction entropy (which only counts up/down/flat proportions), PE considers the full ordering structure.

- Adaptive order: D=3 for <200 obs, D=4 for 200-999, D=5 for ≥1000
- Normalized to [0, 1]: PE > 0.95 warns "effectively random", PE < 0.5 indicates strong structure
- Pure stdlib implementation — no numpy/scipy required

### Forecast Value Added (Phase 6)

Phase 6 now reports **FVA** when model predictions are supplied. FVA measures the percentage improvement of the model over the best baseline:

```
FVA = (Naive_MAE - Model_MAE) / Naive_MAE × 100%
```

- FVA < 0%: FAIL — model destroys value
- FVA 0-10%: WARN — marginal improvement, complexity may not be justified
- FVA > 10%: PASS — model adds substantial value

### Additional Metrics

New helper functions available for programmatic use:

| Function | Purpose |
|---|---|
| `_rmsse(actual, predicted, train_data, sp)` | Root Mean Squared Scaled Error (M5 metric) |
| `_wape(actual, predicted)` | Weighted Absolute Percentage Error |
| `_me_bias(actual, predicted)` | Mean Error / forecast bias |
| `_pinball_loss(actual, quantile_pred, tau)` | Quantile loss |
| `_fva(model_mae, naive_mae)` | Forecast Value Added (%) |
| `_permutation_entropy(d, order, delay)` | Permutation entropy [0, 1] |

`compare_models()` now includes `wape` and `me_bias` in its output alongside existing metrics.

### Conformalized Quantile Regression (CQR)

New `cqr_intervals()` function complements the existing `conformal_intervals()`:

```python
from ts_reviewer import cqr_intervals

# Adjust quantile predictions with conformal calibration
intervals = cqr_intervals(
    calibration_actuals, calibration_lower, calibration_upper,
    test_lower, test_upper, coverage=0.95
)
```

CQR produces adaptive-width intervals (wider where model is uncertain) unlike standard split conformal which gives uniform-width intervals.

---

## Validating Simulation Output

When `simulator.py` generates time-series output (SD trajectories, MC fan charts, ABM macro traces), route through ts_reviewer for signal quality validation:

1. **Quick mode** (`ts_reviewer.py quick`): Run phases 1-6 on simulated output. Check stationarity, forecastability (PE), and baselines against the observed data from Phase 1.
2. **Residual validation** (phases 7-10): Compute residuals as `simulated - observed` and run `ts_reviewer.py review` on the residual series. Zero mean, no autocorrelation, and homoscedasticity indicate good model fit.
3. **Comparison**: Use `compare_models(observed, {"simulation": simulated, "naive": naive_predictions})` to verify simulation beats naive baseline (FVA > 0%).

**Red flags**: If simulated PE differs from observed PE by more than 0.2, the simulation may not reproduce the system's temporal structure. If FVA < 0%, the simulation is worse than naive — return to Phase 3/4 and iterate.

---

## Utility Functions for Protocol Phases

ts_reviewer.py provides standalone functions usable outside the 10-phase review:

| Function | Protocol Phase | Usage |
|---|---|---|
| `compare_models(actuals, models_dict, seasonal_period)` | Phase 3, 5 | Rank candidate models by MASE; select simplest within 5% of best |
| `walk_forward_split(data, n_folds, min_train, horizon, expanding)` | Phase 3, 5 | Generate temporal CV splits — use instead of random K-fold for time series |
| `conformal_intervals(calibration_residuals, point_forecasts, coverage)` | Phase 5 | Distribution-free prediction intervals with finite-sample coverage guarantee |
| `cqr_intervals(cal_actuals, cal_lower, cal_upper, test_lower, test_upper, coverage)` | Phase 5 | Adaptive-width conformal intervals — wider where model is uncertain |
| `quick_review(data, name, **kwargs)` | Phase 1, 4 | One-liner full review; use on raw signals (Phase 1) or simulation output (Phase 4) |

**Model selection workflow** (Phase 3):
```python
from ts_reviewer import compare_models, walk_forward_split

splits = walk_forward_split(data, n_folds=5, min_train=100)
# ... fit each candidate model on each fold ...
results = compare_models(test_actuals, {"arx": arx_preds, "armax": armax_preds})
# Pick simplest model within 5% of best MASE; see forecasting-science.md hierarchy
```

**Uncertainty quantification** (Phase 5):
```python
from ts_reviewer import conformal_intervals, cqr_intervals

# Simple: symmetric intervals from calibration residuals
intervals = conformal_intervals(cal_residuals, point_forecasts, coverage=0.95)

# Adaptive: use with quantile regression model output
intervals = cqr_intervals(cal_actual, cal_lo, cal_hi, test_lo, test_hi, coverage=0.95)
```

---

## Cross-References

- `system-identification.md` — Model structure selection (ARX, ARMAX, state-space) that ts_reviewer validates
- `validation-checklist.md` — Consolidates all validation requirements including residual checks
- `domain-calibration.md` — Plausibility bounds referenced by ts_reviewer's baseline comparisons
- `financial-validation.md` — Finance-specific extensions to general time-series validation
- `coherence-checks.md` — RAPID tier coherence checks that ts_reviewer automates for time-series data
- `forecasting-science.md` — Forecasting science reference (PE, FVA, metrics framework, conformal prediction, model selection)
- `forecasting-tools.md` — Forecast model fitting guide; after ts_reviewer diagnoses a forecastable series, use `forecast_modeler.py fit` for ARIMA/ETS/CatBoost with conformal intervals
- `simulation-guide.md` — Simulation for model validation (forward projection from identified models)
