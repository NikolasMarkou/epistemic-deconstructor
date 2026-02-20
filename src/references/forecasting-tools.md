# Forecasting Tools Reference

Usage guide for `forecast_modeler.py`. Complements `ts_reviewer.py` (signal diagnostics) and `fourier_analyst.py` (spectral analysis) by providing the actual model-fitting step.

## Table of Contents

- [When to Use](#when-to-use)
- [Phase Mapping](#phase-mapping)
- [CLI Quick Reference](#cli-quick-reference)
- [Phases Overview](#phases-overview)
- [Verdict Interpretation](#verdict-interpretation)
- [Graceful Degradation](#graceful-degradation)
- [Utility Functions](#utility-functions)
- [Workflow Examples](#workflow-examples)
- [Cross-References](#cross-references)

---

## When to Use

| Protocol Phase | When | What forecast_modeler Does |
|---|---|---|
| Phase 3: Parametric ID | Need to fit forecasting models to time-series data | ARIMA, ETS, CatBoost fitting + model comparison |
| Phase 5: Validation | Need calibrated prediction intervals | Conformal prediction (ICP, CQR), coverage verification |
| Phase 1: Boundary Mapping | Quick forecastability check before committing to modeling | PE, naive baselines, go/no-go gate |
| RAPID tier | Quick signal forecastability assessment | `assess` mode (Phase 1 only) |

**Use forecast_modeler when:**
- System produces time-series output that needs predictive modeling
- Model selection is needed (ARIMA vs ETS vs CatBoost vs naive)
- Calibrated prediction intervals are required (conformal prediction)
- Forecast Value Added (FVA) must be verified against naive baselines

**Use ts_reviewer instead when:**
- Signal diagnostics only (stationarity, data quality, residuals)
- No model fitting needed — just characterizing the signal
- Validating residuals of an already-fitted model

**Use both when:**
- Phase 3: ts_reviewer first (signal quality check) → forecast_modeler (model fitting)
- Phase 5: forecast_modeler (generate predictions) → ts_reviewer (residual validation)

---

## Phase Mapping

| forecast_modeler Phase | Protocol Phase | Purpose |
|---|---|---|
| 1. Forecastability Gate | Phase 1, 3 | PE, naive baselines (RW, seasonal, drift), FVA threshold, go/no-go |
| 2. Classical Fitting | Phase 3 | Auto-ARIMA, Auto-ETS, best by AICc |
| 3. ML Fitting | Phase 3 | CatBoost + feature engineering (lag, rolling, calendar, Fourier) |
| 4. Model Comparison | Phase 3 | MASE, RMSSE, WAPE, ME bias, FVA, decision tree |
| 5. Conformal Prediction | Phase 5 | Split conformal (ICP), CQR if quantile models available |
| 6. Forecast Generation | Phase 5 | Point forecasts + intervals for specified horizon |
| 7. Report Summary | Phase 5 | Metrics suite, model recommendation, warnings |

**assess mode** (Phase 1 only): Use for quick forecastability screening without fitting models.

---

## CLI Quick Reference

```bash
# Full pipeline from CSV
python3 forecast_modeler.py fit data.csv --column value --horizon 12 --freq 12 --coverage 0.95

# Forecastability assessment only (Phase 1)
python3 forecast_modeler.py assess data.csv --column value --freq 12

# Multi-model comparison table
python3 forecast_modeler.py compare data.csv --column value --horizon 12 --freq 12

# Built-in demo with synthetic data
python3 forecast_modeler.py demo
```

### CLI Flags

| Flag | Command | Description |
|---|---|---|
| `--column` | fit, assess, compare | Column name in CSV to forecast (required) |
| `--horizon` | fit, compare | Forecast horizon in steps (default: 12) |
| `--freq` | fit, assess, compare | Seasonal period (e.g. 12 for monthly, 52 for weekly) |
| `--coverage` | fit | Conformal prediction coverage (default: 0.95) |
| `--output` | fit | Save JSON report to file |

---

## Phases Overview

### Phase 1: Forecastability Gate
- **Permutation Entropy (PE)**: Ordinal-pattern complexity; PE > 0.95 = effectively random (REJECT), PE < 0.5 = strong structure
- **Naive baselines**: Random walk, seasonal naive, drift, moving average — computed as FVA reference
- **Series length check**: < 30 observations = REJECT (insufficient for model fitting)
- **Constant series check**: Zero variance = REJECT
- **Go/no-go decision**: REJECT if PE > 0.95 or series too short; WARN if PE > 0.85

### Phase 2: Classical Fitting
- **Auto-ARIMA**: Automatic (p,d,q)(P,D,Q,m) selection via pmdarima; falls back to manual grid if pmdarima unavailable
- **ETS**: Automatic error/trend/seasonal type selection
- **AICc ranking**: Best classical model selected by information criterion
- **Skip conditions**: No statsmodels installed, or Phase 1 returned REJECT

### Phase 3: ML Fitting
- **CatBoost**: Gradient boosting with quantile regression for native prediction intervals
- **Feature engineering**: Lag features, rolling statistics (with shift(1) leakage prevention), calendar features, Fourier harmonics
- **Walk-forward validation**: Temporal train/test split — no data leakage
- **Skip conditions**: No catboost/numpy installed, or Phase 1 returned REJECT

### Phase 4: Model Comparison
- **Metrics**: MAE, RMSE, MASE, RMSSE, WAPE, ME bias, FVA over best naive
- **Selection rule**: Lowest MASE; ties broken by simplicity (naive < drift < ARIMA < ETS < CatBoost)
- **FVA gate**: Model must beat naive (FVA > 0%) to be recommended; FVA < 0% = FAIL
- **Bias check**: |ME bias| > 10% of mean = WARN (systematic over/under-prediction)

### Phase 5: Conformal Prediction
- **Split conformal (ICP)**: Distribution-free intervals from calibration residuals; guaranteed finite-sample coverage
- **CQR**: Conformalized Quantile Regression — adaptive-width intervals (wider where model is uncertain); requires quantile model output
- **Coverage verification**: Empirical coverage on held-out calibration set; flags if > 5% deviation from target

### Phase 6: Forecast Generation
- **Point forecasts**: From best model, for specified horizon
- **Prediction intervals**: From Phase 5 conformal procedure
- **Ensemble option**: Average of top models if within 5% MASE of best

### Phase 7: Report Summary
- **Model recommendation**: Best model with justification
- **Metrics summary**: Full metrics table for all fitted models
- **Warnings**: Aggregated findings from all phases
- **Dependency report**: Which optional libraries were available

---

## Verdict Interpretation

| Verdict | Meaning | Action |
|---|---|---|
| **PASS** | Check OK | Continue |
| **WARN** | Potential issue | Investigate, document in hypothesis tracker |
| **FAIL** | Significant problem | Address before trusting forecasts |
| **REJECT** | Fatal flaw | Data unsuitable for modeling (e.g. random, too short) |
| **SKIP** | Cannot evaluate | Missing dependencies or insufficient data |

**Escalation rule**: If Phase 1 returns REJECT, the pipeline stops — no model fitting is attempted. FAIL findings in Phase 4 (FVA < 0%) mean the model adds no value over naive.

---

## Graceful Degradation

| Dependencies Available | Phases Available | Capability |
|---|---|---|
| None (pure stdlib) | 1, 4, 5, 6, 7 | Forecastability assessment + naive baselines + conformal intervals |
| + numpy | 1, 3*, 4, 5, 6, 7 | Better numerics, feature engineering (*CatBoost still needs catboost) |
| + statsmodels | 1, 2, 4, 5, 6, 7 | ARIMA + ETS fitting |
| + pmdarima | 1, 2, 4, 5, 6, 7 | Auto-ARIMA parameter search (enhances Phase 2) |
| + catboost + numpy | All | Full pipeline including ML fitting |

The tool always works — stdlib-only mode provides Phase 1 forecastability assessment plus naive baselines with conformal intervals. This is often sufficient for RAPID tier analyses.

---

## Utility Functions

For programmatic use (Python API, not CLI):

| Function | Protocol Phase | Usage |
|---|---|---|
| `auto_forecast(data, horizon, coverage)` | Phase 3, 5 | One-liner full pipeline — returns ForecastReport |
| `compare_forecasters(data, horizon)` | Phase 3 | Multi-model comparison dict with metrics |
| `conformal_forecast(predictions, calibration_residuals, coverage)` | Phase 5 | Conformal interval wrapper for any model's output |

**One-liner workflow** (Phase 3):
```python
from forecast_modeler import auto_forecast

report = auto_forecast(data, horizon=12, coverage=0.95)
print(f"Best model: {report.best_model}")
print(f"FVA: {report.models[report.best_model].metrics.get('fva', 'N/A')}%")
```

**Model comparison** (Phase 3):
```python
from forecast_modeler import compare_forecasters

results = compare_forecasters(data, horizon=12)
for name, result in results.items():
    print(f"{name}: MASE={result.metrics.get('mase', 'N/A'):.3f}")
```

**Conformal intervals** (Phase 5):
```python
from forecast_modeler import conformal_forecast

# Get calibrated intervals for any point forecasts
lower, upper = conformal_forecast(
    predictions=[100, 105, 110],
    calibration_residuals=residuals_from_validation,
    coverage=0.95
)
```

---

## Workflow Examples

### Standard Tier — Phase 3 Model Fitting

```
1. ts_reviewer.py quick data.csv --column value --freq 12
   → Signal diagnostics: stationarity, data quality, forecastability (PE)
   → If PE > 0.95: STOP — series is effectively random

2. forecast_modeler.py fit data.csv --column value --horizon 12 --freq 12
   → Fits ARIMA, ETS, CatBoost; selects best by MASE
   → Produces point forecasts + conformal intervals

3. ts_reviewer.py review residuals.csv --column residual
   → Validate residuals: zero mean, no autocorrelation, homoscedasticity
```

### RAPID Tier — Quick Forecastability Check

```
1. forecast_modeler.py assess data.csv --column value --freq 12
   → Phase 1 only: PE, naive baselines, go/no-go
   → Result feeds into rapid_checker coherence checks
```

### Phase 5 — Validation Pipeline

```
1. forecast_modeler.py fit data.csv --column value --horizon 12 --coverage 0.95 --output forecast.json
2. Verify: FVA > 0%, conformal coverage within 5% of target
3. Compare against domain calibration bounds (domain-calibration.md)
4. Document in validation.md
```

---

## Cross-References

- `timeseries-review.md` — Signal diagnostics (ts_reviewer.py) — use before and after forecast_modeler
- `spectral-analysis.md` — Frequency-domain analysis (fourier_analyst.py) — complements time-domain forecasting
- `forecasting-science.md` — Methodological foundation: PE, FVA, metrics framework, conformal prediction theory
- `system-identification.md` — Model structure selection (ARX, ARMAX, state-space) — parametric models that forecast_modeler validates
- `financial-validation.md` — Finance-specific forecasting validation (FVA thresholds, martingale baseline)
- `validation-checklist.md` — Consolidated validation requirements including forecast checks
- `simulation-guide.md` — Forward simulation from identified models — simulation output can be compared to forecasts
- `tool-catalog.md` — Tool recommendations by phase/domain
