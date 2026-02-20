# Forecasting Science Reference

Distilled framework for forecastability assessment, model selection, error metrics, and conformal prediction. Synthesized from Manokhin's research and practical forecasting science.

## Table of Contents

- [Forecastability Assessment](#forecastability-assessment)
- [Model Selection Hierarchy](#model-selection-hierarchy)
- [ARIMA Identification Guide](#arima-identification-guide)
- [ETS Model Selection](#ets-model-selection)
- [CatBoost Feature Engineering](#catboost-feature-engineering)
- [Forecast Error Metrics Framework](#forecast-error-metrics-framework)
- [Conformal Prediction Overview](#conformal-prediction-overview)
- [Common Pitfalls](#common-pitfalls)
- [Protocol Phase Integration](#protocol-phase-integration)
- [Cross-References](#cross-references)

---

## Forecastability Assessment

**Core principle**: Assess forecastability *before* model selection. If not assessed, the analyst doesn't understand fundamentals.

### Coefficient of Variation (CoV) — Why to Avoid

| Failure | Problem |
|---|---|
| Assumes normality | Real data exhibit skewness, multimodality, heavy tails |
| Ignores temporal structure | Treats time series as unordered sets — a rising trend reads as "volatile" |
| Scale sensitivity | Misleading near zero; meaningless for intermittent demand |
| Stability ≠ Predictability | Gauges *stability*, not *forecastability* — operationally distinct concepts |

**Operational definition**: Forecastability = the range of forecast errors achievable in the long run, not just historical stability. It is method-dependent, requires actual forecasting experiments, and connects to whether any method significantly beats naive benchmarks.

### Permutation Entropy (PE)

Quantifies predictability by measuring ordinal-pattern complexity in time series.

**Advantages over CoV**: Non-parametric, robust to noise, invariant under nonlinear monotonic transforms, captures temporal ordering.

**Parameters**:
- **Embedding dimension (D)**: Consecutive values grouped into vectors. Adaptive selection: D=3 for <200 obs, D=4 for 200-999, D=5 for ≥1000.
- **Embedding delay (τ)**: Step size for phase-space vectors. Default τ=1 for most applications.

**Interpretation (normalized PE ∈ [0, 1])**:

| PE Range | Interpretation | Forecastability |
|---|---|---|
| PE = 0 | Completely deterministic | Maximum |
| PE < 0.5 | Strong regular structure | High |
| 0.5 ≤ PE < 0.85 | Moderate complexity | Moderate |
| 0.85 ≤ PE < 0.95 | High complexity | Low |
| PE ≥ 0.95 | Effectively random | Near-zero |

**Algorithm sketch** (stdlib, no dependencies):
1. Extract all overlapping subsequences of length D with delay τ
2. For each subsequence, compute the ordinal pattern (rank permutation)
3. Count frequency of each permutation
4. Compute Shannon entropy over permutation frequencies
5. Normalize by log(D!) to get PE ∈ [0, 1]

### Forecast Value Added (FVA)

**Formula**:
```
FVA = (Naive_Error - Model_Error) / Naive_Error × 100%
```

| FVA Range | Interpretation |
|---|---|
| FVA > 10% | Model adds substantial value |
| 0% ≤ FVA ≤ 10% | Marginal improvement — complexity may not be justified |
| FVA < 0% | Model destroys value — use naive forecast instead |

**Principle**: If even the best algorithms barely beat naive, the series has genuinely low forecastability. If a simple model equals a complex model, the complex model is overkill.

### Naive Benchmark Principle

Every forecasting model must be benchmarked against naive methods before deployment. Standard baselines:

| Baseline | Method | Use Case |
|---|---|---|
| Naive | Y[t+1] = Y[t] | Non-seasonal data |
| Seasonal naive | Y[t+1] = Y[t-m] | Seasonal data |
| Drift | Y[t+1] = Y[t] + slope | Trending data |
| Moving average | Y[t+1] = mean(Y[t-k:t]) | Noisy, no trend/season |

---

## Model Selection Hierarchy

### Decision Tree

```
1. Does model beat naive by >10% (FVA > 10%)?
   NO  → Series has low forecastability OR model is misspecified
   YES → Continue

2. Does model beat seasonal naive (if seasonal data)?
   NO  → Model fails to capture seasonality
   YES → Continue

3. Does complexity justify improvement?
   (Simple model within 5% of complex model → use simple)
   YES → Use complex model
   NO  → Use simple model (Occam's razor)

4. Does model produce valid prediction intervals?
   NO  → Add conformal prediction layer
   YES → Deploy
```

### Ranked Model Hierarchy

Evaluate in this order — stop when additional complexity doesn't justify improvement:

| Rank | Model | Strengths | When to Prefer |
|---|---|---|---|
| 1 | Naive baselines | Zero complexity, hard-to-beat benchmark | Always compute first |
| 2 | ETS (esp. M,Ad,M) | Damped trend, wins competitions, fast | Short series, multiplicative seasonality |
| 3 | ARIMA/SARIMA | Well-understood, handles exogenous vars (SARIMAX) | Strong autocorrelation, covariates available |
| 4 | CatBoost | Nonlinear, native categoricals, ordered boosting | Rich features, nonlinear dynamics, >200 obs |
| 5 | Specialized | N-BEATS, N-HiTS, TSMixer | Very large datasets, complex patterns |
| 6 | Ensembles | Combine top 2-3 models | When diversity improves robustness |

### ARIMA vs ETS Quick Comparison

| Scenario | ARIMA | ETS |
|---|---|---|
| Exogenous variables | Yes (SARIMAX) | No |
| Multiplicative seasonality | Log-transform | Native |
| Interpretable components | No | Yes (level, trend, season) |
| Very short series (<30) | Poor | Better (fewer params) |
| Best single-model default | — | ETS(M,Ad,M) |

**Best practice**: Fit both, select by cross-validated forecast accuracy, or combine. `forecast_modeler.py` Phase 2 fits both and Phase 4 selects.

### Anti-Patterns

| Anti-Pattern | Problem |
|---|---|
| Prophet as default | Designed for business metrics with holidays, not general time series; poor on series without clear trend+seasonality; no valid uncertainty without conformal wrapping |
| Transformer models for small data | Require massive datasets (>10k obs); overfit catastrophically on typical business series; inference cost unjustified for most applications |
| Skipping naive baseline | Cannot assess whether model adds value; complexity without justification |
| MAPE-driven model selection | Systematically biases toward under-forecasting (see Metrics section) |

---

## ARIMA Identification Guide

### ACF/PACF Signature Table

Use on the **stationary** (differenced) series to identify candidate (p, q) orders:

| Model | ACF Pattern | PACF Pattern |
|---|---|---|
| AR(p) | Exponential/oscillatory decay | Sharp cutoff after lag p |
| MA(q) | Sharp cutoff after lag q | Exponential/oscillatory decay |
| ARMA(p,q) | Decays after lag q | Decays after lag p |

### Box-Jenkins Quick Reference

1. **Stationarity test**: ADF (H₀: non-stationary) + KPSS (H₀: stationary) as cross-check. Non-stationary → increment d.
2. **ACF/PACF analysis**: Identify (p, q) from signature table above.
3. **Estimation**: MLE or CSS. Compare candidates by AICc (preferred for small samples) or BIC (parsimony).
4. **Diagnostics**: Residuals must pass — zero mean, constant variance, no autocorrelation (Ljung-Box test).
5. **Forecasting**: Point forecasts + Gaussian intervals (limited — wrap with conformal prediction for validity).

### Seasonal ARIMA: SARIMA(p,d,q)(P,D,Q)[m]

| Data Pattern | Starting Model |
|---|---|
| Stationary, no seasonality | ARIMA(1,0,1) or ARIMA(2,0,2) |
| Trending, no seasonality | ARIMA(1,1,1) |
| Seasonal (monthly) | SARIMA(1,1,1)(1,1,1)[12] |
| Seasonal (quarterly) | SARIMA(1,1,1)(1,1,1)[4] |
| Weekly with daily data | SARIMA(1,1,1)(1,1,1)[7] |

**Auto-ARIMA** (`pmdarima.auto_arima`): Automates stepwise search over (p,d,q)(P,D,Q)[m]. Used by `forecast_modeler.py` Phase 2 when pmdarima is available.

### SARIMAX — Exogenous Variables

ARIMA + external regressors. **Caution**: exogenous variables must be available at forecast time. If you must forecast the covariates too, error compounds.

---

## ETS Model Selection

### The ETS(E, T, S) Framework

| Component | Options | Code |
|---|---|---|
| **Error** | Additive, Multiplicative | A, M |
| **Trend** | None, Additive, Additive Damped, Multiplicative, Multiplicative Damped | N, A, Ad, M, Md |
| **Seasonality** | None, Additive, Multiplicative | N, A, M |

30 possible configurations. Key models:

| Model | ETS Code | Use Case |
|---|---|---|
| ETS(A,N,N) | SES | No trend/seasonality, level only |
| ETS(A,Ad,N) | Damped Holt's | Trending, no seasonality |
| ETS(A,A,A) | Holt-Winters Additive | Constant seasonal amplitude |
| ETS(M,Ad,M) | Damped Multiplicative | **Best single-model default** — wins competitions |

### Additive vs Multiplicative Selection

| Criterion | Additive | Multiplicative |
|---|---|---|
| Seasonal amplitude | Constant regardless of level | Proportional to level |
| Data values | Can include zeros/negatives | Strictly positive only |
| Visual diagnostic | Seasonal swings same height | Seasonal swings grow with level |
| Typical domains | Temperature, some financial | Sales, demand, production |

**Decision rule**: If seasonal fluctuations scale with the series level, use multiplicative. If constant, use additive. When uncertain, let AICc decide — `forecast_modeler.py` Phase 2 searches all valid configurations.

### Damped Trend

Multiplies trend by damping factor φ ∈ (0.8, 1.0) at each step, preventing unrealistic long-horizon extrapolation. As h→∞, cumulative trend approaches φ/(1-φ)×b_t. Damped models consistently outperform non-damped in M3/M4/M5 competitions.

### ARIMA vs ETS: When to Use Which

| Scenario | Prefer ARIMA | Prefer ETS |
|---|---|---|
| Exogenous variables available | Yes (SARIMAX) | No |
| Multiplicative seasonality | Possible (log transform) | Native |
| Strong autocorrelation in residuals | Yes | No |
| Very short series (< 30 obs) | No | Yes (fewer params) |
| Best single-model default | — | ETS(M,Ad,M) |
| Ensemble/combination | Include both | Include both |

---

## CatBoost Feature Engineering

### Why CatBoost for Time Series

1. **Ordered boosting**: Computes gradients using only preceding examples — structurally similar to temporal ordering, prevents target leakage
2. **Native categoricals**: Handles day_of_week, month, etc. without one-hot encoding
3. **Oblivious trees**: Implicit regularization, reduces overfitting on noisy series
4. **Built-in quantile regression**: Native prediction intervals for conformal prediction integration

### Feature Engineering Checklist

| Feature Type | Method | Leakage Prevention |
|---|---|---|
| **Lag features** | shift(lag) for lags 1..p + seasonal lags (m, 2m) | Inherently safe (past values) |
| **Rolling statistics** | mean/std/min/max over window w | **shift(1) before rolling** — critical |
| **Calendar features** | day_of_week, month, quarter, is_weekend | Mark as categorical in CatBoost |
| **Fourier harmonics** | sin/cos(2πk·t/period) for k=1..K | Deterministic — no leakage |

### Lag Selection Heuristics

- Include lags 1 through p (dominant PACF cutoff)
- Include seasonal lags: m, 2m (e.g., 7, 14 for daily with weekly pattern)
- For long-range dependencies: m×k for k = 1, 2, 3

### Rolling Statistics — Leakage Prevention

**Critical**: Always `shift(1)` before computing rolling statistics. The rolling window must only use information available at forecast time.

```
WRONG: df['y'].rolling(7).mean()        ← includes current value
RIGHT: df['y'].shift(1).rolling(7).mean() ← uses only past values
```

`forecast_modeler.py` Phase 3 implements all feature types with leakage prevention built in.

### When to Prefer CatBoost

- Rich feature space available (calendar, external regressors, lag structure)
- Nonlinear dynamics suspected
- Series length > 200 observations (smaller series → prefer ETS/ARIMA)
- Quantile regression needed for CQR conformal intervals

---

## Forecast Error Metrics Framework

### Metric Categories

| Category | Metrics | Use Case |
|---|---|---|
| Scale-dependent | MAE, RMSE | Single-series evaluation |
| Percentage-based | WAPE | Multi-series volume-weighted comparison |
| Scaled | MASE, RMSSE | Cross-series comparison vs naive |
| Bias | ME (Mean Error) | Systematic over/under detection |
| Probabilistic | Coverage, CRPS, Pinball Loss, Winkler | Interval/distribution evaluation |

### MAPE and sMAPE — Avoid

**MAPE** has five critical flaws:
1. Undefined at zero, explosive near zero
2. Asymmetric — systematically rewards under-forecasting
3. Disproportionate penalty for small-volume series
4. No consistent cross-series interpretation
5. Distorted decades of forecasting competition results

**sMAPE** reverses the asymmetry (favors over-forecasting) and has a confusing [0, 200%] range. Neither metric solves the fundamental problems.

**If stakeholders demand a percentage metric, use WAPE.**

### Key Metric Formulas

**Scale-dependent**:
```
MAE  = (1/n) Σ |yᵢ - ŷᵢ|
RMSE = √[(1/n) Σ (yᵢ - ŷᵢ)²]
```

**Percentage** (volume-weighted):
```
WAPE = Σ |yᵢ - ŷᵢ| / Σ |yᵢ| × 100
```

**Scaled** (vs naive benchmark):
```
MASE  = MAE_model / MAE_naive_in_sample
RMSSE = √[MSE_model / MSE_naive_in_sample]
```

**Bias**:
```
ME = (1/n) Σ (ŷᵢ - yᵢ)      positive = over-forecast
```

**Probabilistic**:
```
Pinball(τ) = τ(y - q̂)   if y ≥ q̂,   (τ-1)(y - q̂)   if y < q̂
FVA = (Naive_MAE - Model_MAE) / Naive_MAE × 100%
```

### Metric Selection Framework

**Step 1 — Scope**:

| Context | Recommended |
|---|---|
| Single series | MAE, RMSE, ME |
| Multi-series (equal weight) | MASE, RMSSE |
| Multi-series (volume-weighted) | WAPE |

**Step 2 — Forecast type**:

| Type | Recommended |
|---|---|
| Point forecast | MAE/RMSE + MASE/RMSSE + ME |
| Quantile forecasts | Pinball Loss |
| Prediction intervals | Coverage + Winkler Score |
| Full distributional | CRPS |

**Step 3 — Business incentives**:

| Priority | Use | Avoid |
|---|---|---|
| Minimize total error | MAE, WAPE | MAPE |
| Penalize large misses | RMSE, RMSSE | MAE |
| Detect bias | ME + MAE | MAPE (introduces bias) |
| Compare to naive | MASE, FVA | Raw MAE alone |

**Minimum recommended set**: MAE or RMSE + MASE or RMSSE + ME + Coverage/CRPS (if probabilistic).

---

## Conformal Prediction Overview

### Validity-First Principle

**Validity (coverage) must precede efficiency (sharpness).** If a model claims 95% intervals, they must cover ~95% of actuals. Only conformal prediction provides finite-sample validity guarantees — Bayesian, bootstrap, and Monte Carlo methods lack this property.

### Three Methods for Time Series

| Method | Key Property | When to Use |
|---|---|---|
| **ICP** (Inductive Conformal) | Simple split, fast, general | Large datasets (>1000 obs), production systems |
| **CQR** (Conformalized Quantile Regression) | Adaptive interval width, better conditional coverage | Heteroscedastic data, quantile models available |
| **EnbPI** (Ensemble Batch PI) | No exchangeability required, custom-built for time series | Time series with distribution shift, ensemble models |

### CQR Algorithm

Conformalized Quantile Regression combines quantile regression with conformal correction:

1. **Train quantile models** on proper training set: predict lower quantile q̂_{α/2}(x) and upper quantile q̂_{1-α/2}(x)
2. **Compute conformity scores** on calibration set: αᵢ = max(q̂_lower(xᵢ) - yᵢ, yᵢ - q̂_upper(xᵢ))
3. **Compute correction** Q = quantile at level ⌈(n+1)(1-α)⌉/n of the conformity scores
4. **Adjust test intervals**: [q̂_lower(x) - Q, q̂_upper(x) + Q]

**Key advantage**: Intervals adapt width to input features (wider where model is uncertain, narrower where confident), unlike standard split conformal which produces uniform-width intervals.

### Exchangeability Challenge

Time series violate the exchangeability assumption required by standard conformal prediction. Solutions:

- **EnbPI**: Uses bootstrap ensemble + out-of-sample residuals; requires only "strongly mixing stochastic errors"
- **Adaptive Conformal Inference (ACI)**: Dynamically adjusts α based on recent coverage: α_{t+1} = αₜ + γ(ε - errₜ). Adapts to distribution shift with low computational overhead.
- **Sliding window calibration**: Use only recent N calibration scores (discards stale data)

### Minimum Calibration Set Size

At least 500 calibration observations recommended for stable conformal intervals. Smaller sets produce valid but highly variable interval widths.

---

## Common Pitfalls

| Pitfall | Description | Remedy |
|---|---|---|
| CoV for forecastability | CoV ignores temporal structure; stable ≠ predictable | Use Permutation Entropy + FVA |
| Skipping naive baseline | No context for whether model adds value | Always compute naive/seasonal naive MAE first |
| MAPE-driven selection | Under-forecasting bias, undefined at zero | Use MASE/RMSSE + ME |
| Single metric evaluation | Model rankings change across metrics | Report MAE + MASE + ME minimum |
| Gaussian interval assumption | ARIMA/ETS intervals assume normality | Wrap with conformal prediction |
| Random K-fold for time series | Violates temporal ordering | Use walk-forward or expanding window CV |
| Prophet as universal default | Poor on general time series without holidays/trend | Use ETS or ARIMA first; benchmark against naive |
| Ignoring forecast bias | Bias compounds over time across products | Always report ME alongside accuracy metrics |
| Overfitting CatBoost on short series | Gradient boosting overfits when <200 observations | Prefer ETS/ARIMA; increase regularization |
| Raw quantile intervals | Quantile regression lacks coverage guarantees | Conformalize with CQR |

---

## Protocol Phase Integration

This reference supports specific Epistemic Deconstruction phases:

### Phase 3: Parametric Identification
- **Forecastability gate**: Run `forecast_modeler.py assess` or Phase 1 — PE > 0.95 means effectively random, complex models will overfit.
- **Model fitting**: Run `forecast_modeler.py fit` — fits ARIMA, ETS, CatBoost in sequence, selects best by MASE.
- **Model selection**: Use the [Ranked Model Hierarchy](#ranked-model-hierarchy). Evaluate: Naive → ETS → ARIMA → CatBoost. Stop when FVA improvement < 5%.
- **Comparison**: `forecast_modeler.py compare` or `compare_models()` from ts_reviewer.py to rank candidates. Select simplest within 5% of best.
- **Temporal CV**: Use `walk_forward_split()` — never random K-fold on time series.

### Phase 5: Validation & Adversarial
- **Hard gate**: FVA > 0% required. If model doesn't beat naive baseline, do not accept.
- **Metrics**: Report MAE + MASE + ME minimum. Use WAPE for multi-series. Avoid MAPE/sMAPE (see [Common Pitfalls](#common-pitfalls)).
- **Uncertainty**: `forecast_modeler.py` Phase 5 computes conformal intervals (ICP + CQR). Also available via `conformal_intervals()` / `cqr_intervals()` from ts_reviewer.py.
- **Coverage check**: ts_reviewer Phase 9 validates interval calibration. 95% intervals must cover ~95% of actuals.
- **Residual validation**: After fitting with forecast_modeler, run ts_reviewer on residuals (phases 7-10) — zero mean, no autocorrelation, homoscedasticity.

### RAPID Tier
- **Quick coherence**: Run `forecast_modeler.py assess` — PE + naive baselines in one command. PE > 0.95 + FVA < 0% = strong evidence the claim is unreliable.

---

## Cross-References

- `forecasting-tools.md` — forecast_modeler.py usage guide (model fitting, conformal prediction, CLI reference)
- `timeseries-review.md` — ts_reviewer.py usage guide (signal diagnostics, PE, FVA, baseline benchmarks, coverage checks)
- `financial-validation.md` — Finance-specific forecasting validation (martingale baseline, economic significance)
- `system-identification.md` — Parametric model estimation (ARX, ARMAX, state-space)
- `validation-checklist.md` — Consolidated validation requirements including residual checks
- `domain-calibration.md` — Plausibility bounds by domain for metric thresholds
- `simulation-guide.md` — Simulation paradigms (MC for uncertainty propagation, sensitivity analysis)
