# Financial Forecasting Validation

Specialized validation framework for financial prediction claims. Supplements generic validation infrastructure with finance-specific disqualifiers, stationarity requirements, forecastability assessment, and economic significance testing.

## Table of Contents

- [Financial Disqualifiers (Tier 0)](#financial-disqualifiers-tier-0)
- [Stationarity Requirements](#stationarity-requirements)
- [Forecastability Assessment](#forecastability-assessment)
- [Class Imbalance in Financial Context](#class-imbalance-in-financial-context)
- [Regime Testing](#regime-testing)
- [Economic Significance](#economic-significance)
- [Financial Statistical Tests](#financial-statistical-tests)
- [Financial Calibration Bounds](#financial-calibration-bounds)
- [The 5% Problem: Component Weights](#the-5-problem-component-weights)
- [Financial Validation Checklist](#financial-validation-checklist)
- [Cross-References](#cross-references)

---

## Financial Disqualifiers (Tier 0)

Instant-reject conditions specific to financial forecasting. If any of these are present, stop analysis immediately.

| Disqualifier | What It Means | Test |
|-------------|---------------|------|
| No Martingale baseline | Fails to compare against random walk / buy-and-hold | Check if naive baselines are reported |
| Price-level metrics only | Reports R² on prices instead of returns | Check if metrics use returns (Δp or log-returns) |
| Reconstructed R² trick | Reports R² on cumulated predictions vs. cumulated actuals | Verify R² is on step-ahead returns, not reconstructed levels |
| No out-of-sample test | All results are in-sample or use random train/test split | Verify temporal (walk-forward) split is used |
| Future information leakage | Features include future data (e.g., same-day close for open prediction) | Audit feature timestamps vs. prediction target timestamp |
| Survivorship bias | Dataset excludes delisted/bankrupt securities | Check if universe includes dead stocks |

### The Martingale Baseline Rule

Financial markets are approximately efficient. Any forecasting claim must demonstrate improvement over:

1. **Random walk**: Tomorrow's price = today's price (for prices)
2. **Historical mean**: Tomorrow's return = long-run average return (for returns)
3. **Buy-and-hold**: Simple market exposure with no trading

If a paper doesn't compare against these, the claimed performance is uninterpretable.

### The Price-Level Trap

R² on price levels is **always** high because prices are non-stationary (trending). Two independent random walks will show high R² on levels. This is the single most common trick in financial ML papers.

**Test**: If R² > 0.5 and the target is price levels → instant reject. Legitimate financial forecasting reports R² on **returns** (typically 0.01–0.15 for good models).

---

## Stationarity Requirements

Financial time series are non-stationary. Models trained on non-stationary data produce spurious results.

### Required Tests

| Test | Purpose | Passing Criterion |
|------|---------|-------------------|
| ADF (Augmented Dickey-Fuller) | Test for unit root | p < 0.05 (reject unit root) |
| KPSS | Test for stationarity | p > 0.05 (fail to reject stationarity) |
| Both ADF + KPSS | Confirm result | ADF rejects unit root AND KPSS fails to reject stationarity |

### Differencing Rules

| Data Type | Required Transform | Metric Allowed On |
|-----------|-------------------|-------------------|
| Price levels | First-difference (returns) or log-returns | Returns only |
| Volume | Log-transform or percentage change | Transformed series |
| Volatility | Often stationary; test first | Raw if stationary |
| Spreads | Often stationary; test first | Raw if stationary |

### Level-Based Metrics Prohibition

The following metrics are **prohibited** on non-stationary financial series:

- R² on price levels
- RMSE/MAE on price levels (without normalization)
- Correlation on price levels

Acceptable alternatives:
- R² on returns or log-returns
- RMSE/MAE on returns (or normalized by price)
- Directional accuracy on returns

---

## Forecastability Assessment

Before building a model, assess whether the target series contains exploitable structure.

### Autocorrelation Analysis

| Measure | Interpretation |
|---------|---------------|
| ACF/PACF at lag 1–5 | Significant autocorrelation = potential predictability |
| Ljung-Box test | p < 0.05 = series has autocorrelation structure |
| ACF of squared returns | Detects volatility clustering (ARCH effects) |

If daily returns show no significant autocorrelation at any lag, the series is approximately a random walk and **directional prediction is not feasible**.

### Entropy and Information Content

| Measure | What It Tests |
|---------|--------------|
| Approximate entropy (ApEn) | Regularity/predictability of series |
| Sample entropy (SampEn) | Improved version of ApEn |
| Permutation entropy | Ordinal pattern complexity |

Higher entropy = less predictable. If entropy is near theoretical maximum, the series is effectively random.

### Noise Floor Estimation

The noise floor is the irreducible prediction error. Estimate it via:

1. **Persistence model**: Predict tomorrow = today (for returns: predict 0)
2. **Expanding mean**: Predict expanding historical average
3. **GARCH(1,1)**: Capture volatility dynamics only

Any model must beat the noise floor significantly (not just marginally).

### Predictability Decay

Financial predictability decays rapidly with horizon. Expected pattern:

| Horizon | Typical R² (returns) | Notes |
|---------|---------------------|-------|
| 1 minute | 0.001–0.01 | HFT territory, requires tick data |
| 1 day | 0.001–0.05 | Most academic studies |
| 1 week | 0.005–0.08 | Momentum effects |
| 1 month | 0.01–0.10 | Macro factors matter more |
| 1 quarter | 0.02–0.15 | Value/macro signals |

Claims of R² > 0.15 at any horizon require extraordinary evidence.

---

## Class Imbalance in Financial Context

Financial direction prediction suffers from subtle class imbalance issues.

### The Resampling Delusion

Standard resampling techniques (SMOTE, oversampling, undersampling) are **inappropriate** for financial time series because:

1. **Temporal structure**: Resampling destroys time-ordering
2. **Synthetic samples**: SMOTE creates unrealistic return distributions
3. **Base rate manipulation**: Changing class frequencies distorts directional accuracy

### The "Always Predict Up" Test

Stock markets have an upward bias (~53-55% up days historically for major indices). Any directional model must be tested against:

- **Always predict up**: Baseline accuracy = historical up-day frequency
- **Predict with market drift**: Baseline = predict direction of long-run mean return

If a model achieves 56% accuracy but "always up" gives 54%, the model adds only 2% marginal value — possibly within noise.

### Recommended Metrics for Imbalanced Financial Prediction

| Metric | Why It's Better | Threshold for Significance |
|--------|----------------|---------------------------|
| MCC (Matthews Correlation Coefficient) | Balanced for all confusion matrix quadrants | > 0.05 (financial context) |
| Informedness (Youden's J) | TPR + TNR - 1; chance = 0 | > 0.05 |
| Balanced accuracy | Average of per-class accuracy | > historical up-day % + 2% |
| Profit factor | Gross profit / gross loss | > 1.0 after costs |

### Threshold Calibration vs. Resampling

Instead of resampling the data, calibrate the **decision threshold**:

1. Train model on original (imbalanced) data
2. Generate predicted probabilities on validation set
3. Sweep threshold from 0 to 1
4. Select threshold that maximizes risk-adjusted metric (e.g., Sharpe on predicted signals)
5. Evaluate on out-of-sample test set with fixed threshold

This preserves temporal structure and base rates while optimizing for economic utility.

---

## Regime Testing

Financial markets exhibit regime changes. A model validated in one regime may fail in another.

### Required Regime Decomposition

| Regime | Definition | Test Requirement |
|--------|-----------|------------------|
| Bull market | Sustained uptrend (e.g., 12-month return > 10%) | Report performance separately |
| Bear market | Sustained downtrend (e.g., 12-month return < -10%) | Report performance separately |
| High volatility | VIX > 25 or realized vol > 1.5× long-run average | Report performance separately |
| Low volatility | VIX < 15 or realized vol < 0.7× long-run average | Report performance separately |
| Structural break | Regime shift detected by CUSUM/Bai-Perron test | Test if model degrades post-break |

### Regime-Conditional Evaluation Protocol

1. **Identify regimes** in the test period using the criteria above
2. **Compute metrics separately** for each regime
3. **Report worst-regime performance** alongside aggregate
4. **Flag regime concentration**: If test period is >70% one regime, results are not generalizable

### Structural Break Detection

Methods for detecting structural breaks in financial series:

| Method | Use Case |
|--------|----------|
| CUSUM test | Detect shifts in mean |
| Bai-Perron test | Multiple structural break detection |
| Markov switching model | Probabilistic regime identification |
| Rolling window statistics | Visual regime identification |

A model that has not been tested across structural breaks has **unknown generalization properties**.

---

## Economic Significance

Statistical significance ≠ economic significance. A model may be statistically significant but unprofitable after real-world frictions.

### Transaction Cost Framework

| Cost Component | Typical Range | Must Be Included |
|----------------|---------------|------------------|
| Commission/fees | 0–10 bps per trade | Yes |
| Bid-ask spread | 1–50 bps (varies by asset) | Yes |
| Market impact | 5–100 bps (varies by size/liquidity) | Yes, for >$1M strategies |
| Slippage | 2–20 bps | Yes |
| Borrowing costs (short) | 0.5–10% annualized | Yes, if shorting |
| Funding costs | Risk-free rate | Yes, for leveraged strategies |

### Net Profitability Test

A strategy is economically significant only if:

```
Net Sharpe = (Gross Return - Total Costs) / Volatility > 0.5
```

Where Total Costs include all components above, applied per-trade and annualized.

### Capacity Constraints

| Strategy Type | Typical Capacity | Implication |
|---------------|-----------------|-------------|
| HFT/microstructure | $1M–$50M | Results don't scale |
| Statistical arbitrage | $50M–$500M | Market impact grows with size |
| Momentum/trend | $500M–$5B | Moderate scalability |
| Value/fundamental | $1B–$50B | Scales well |

Claims of high Sharpe (>2) without capacity analysis are suspect — high Sharpe strategies typically have low capacity.

### The Backtesting Illusion

Even with proper methodology, backtested results overstate live performance due to:

1. **Selection bias**: You test many ideas, report the best
2. **Parameter optimization**: Tuned to historical data
3. **Regime fit**: Backtest period may not represent future
4. **Execution assumptions**: Backtests assume perfect fills

**Haircut rule**: Apply 50% haircut to backtested Sharpe ratios when estimating live performance.

---

## Financial Statistical Tests

Specialized statistical tests for financial model comparison.

### Diebold-Mariano Test

Tests whether two forecasts have equal predictive accuracy.

| Parameter | Guidance |
|-----------|----------|
| Loss function | Use MSE for returns, or economic loss (profit/loss) |
| Null hypothesis | Equal predictive accuracy |
| Rejection | p < 0.05 means one forecast is significantly better |
| Variant | Use Harvey-Leybourne-Newbold correction for small samples |

### Reality Check and SPA Test

Controls for data snooping when comparing multiple strategies/models.

| Test | What It Does |
|------|-------------|
| White's Reality Check | Tests if best model beats benchmark, accounting for multiple comparisons |
| Hansen's SPA (Superior Predictive Ability) | Improved version; less conservative than Reality Check |
| StepM | Identifies which specific models beat benchmark |

### Multiple Testing Correction

When testing N strategies/models:

| Method | Formula | When to Use |
|--------|---------|-------------|
| Bonferroni | α/N | Conservative; few comparisons |
| Holm-Bonferroni | Step-down procedure | Moderate N |
| Benjamini-Hochberg (FDR) | Controls false discovery rate | Many comparisons (>20) |
| Bootstrap Reality Check | Simulation-based | Any N; accounts for correlation |

**Rule of thumb**: If you tested N models and report the best, multiply p-values by N (Bonferroni) or use bootstrap methods.

---

## Financial Calibration Bounds

Extended calibration bounds for financial prediction metrics. See `domain-calibration.md` for base table.

| Metric | Suspicious | Plausible | Excellent | Notes |
|--------|------------|-----------|-----------|-------|
| Directional accuracy (daily) | >65% | 52-58% | 58-62% | Base table |
| Sharpe ratio (after costs) | >3.0 | 0.5-1.5 | 1.5-2.5 | Base table |
| R² on returns (test) | >0.30 | 0.01-0.05 | 0.05-0.15 | Base table |
| R² on price levels (test) | >0.95 | N/A | N/A | Always suspicious |
| Annual alpha | >30% | 2-8% | 8-15% | After risk adjustment |
| MCC (direction prediction) | >0.60 | 0.02-0.10 | 0.10-0.25 | Chance = 0 |
| Max drawdown | <5% | 20-50% | 10-20% | Lower is better |
| R² on returns (train) | <5% | 50-80% | 30-50% | Lower is better; high = overfit |

### Interpretation Notes

- **Annual alpha >30%**: Top hedge funds sustain 10-15%. Claims above 30% without verifiable track record = reject.
- **MCC >0.60**: In financial direction prediction, MCC > 0.25 is exceptional. MCC > 0.60 implies near-perfect directional prediction.
- **Max drawdown <5%**: Virtually impossible for equity strategies over multi-year periods. Even Medallion has drawdowns.
- **R² on returns (train) <5%**: If training R² is very low but test R² is high, the model likely found a spurious pattern. If training R² is very high (>50%), the model is likely overfitting.

---

## The 5% Problem: Component Weights

The "5% problem" refers to the observation that most claimed improvements in financial prediction are within the noise range (~5% of baseline performance). Weight each validation component when assessing financial claims.

| Component | Weight | What It Catches |
|-----------|--------|----------------|
| Stationarity compliance | 15% | Price-level metrics, non-stationary features |
| Baseline comparison | 20% | No random walk baseline, no buy-and-hold |
| Walk-forward validation | 15% | Random splits, look-ahead bias |
| Regime robustness | 15% | Bull-only testing, single-regime validation |
| Economic significance | 15% | Pre-cost performance, no capacity analysis |
| Statistical rigor | 10% | No multiple testing correction, no DM test |
| Class imbalance handling | 10% | Resampling on time series, no MCC/informedness |

### Scoring Protocol

1. Assess each component: PASS (full weight), PARTIAL (half weight), FAIL (zero)
2. Sum weighted scores
3. Interpret:
   - **>80%**: Credible — proceed to deeper analysis
   - **60-80%**: Skeptical — significant gaps, request clarification
   - **40-60%**: Doubtful — multiple failures, likely unreliable
   - **<40%**: Reject — fundamental methodological failures

---

## Financial Validation Checklist

Consolidated checklist for financial forecasting claims. Check each item before accepting results.

### Tier 0: Instant Disqualifiers
- [ ] Compares against Martingale / random walk / buy-and-hold baseline
- [ ] Uses returns (not price levels) for evaluation metrics
- [ ] Uses temporal (walk-forward) train/test split
- [ ] No future information in features
- [ ] Addresses survivorship bias (if applicable)

### Tier 1: Stationarity and Forecastability
- [ ] Reports ADF/KPSS test results on target series
- [ ] Uses differenced or stationary series for modeling
- [ ] Assesses autocorrelation structure before modeling
- [ ] Acknowledges noise floor and predictability limits

### Tier 2: Methodology
- [ ] Walk-forward or expanding window validation
- [ ] No resampling (SMOTE/over/under) on time series
- [ ] Reports MCC or informedness alongside accuracy
- [ ] Threshold calibrated on validation set, not resampled
- [ ] Reports performance across bull/bear/high-vol/low-vol regimes

### Tier 3: Statistical Rigor
- [ ] Diebold-Mariano test vs. baseline
- [ ] Multiple testing correction if >1 model compared
- [ ] Confidence intervals on key metrics
- [ ] Reports worst-regime performance

### Tier 4: Economic Significance
- [ ] Includes transaction costs (commissions, spread, impact, slippage)
- [ ] Reports net Sharpe ratio (after all costs)
- [ ] Addresses capacity constraints
- [ ] Applies backtesting haircut or discusses overfitting risk

---

## Cross-References

| Topic | Reference File |
|-------|---------------|
| Generic coherence checks | `coherence-checks.md` |
| Generic red flags | `red-flags.md` |
| Generic validation (walk-forward, leakage, uncertainty) | `validation-checklist.md` |
| Base calibration bounds | `domain-calibration.md` |
| Cognitive traps (tool worship, confirmation bias) | `cognitive-traps.md` |
| Domain calibration data (JSON) | `../config/domains.json` |
