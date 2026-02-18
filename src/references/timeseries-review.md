# Time-Series Signal Review Guide

Reference for using `ts_reviewer.py` during Epistemic Deconstruction analyses.

## Table of Contents

- [When to Use](#when-to-use)
- [Phase Mapping](#phase-mapping)
- [Interpreting Verdicts](#interpreting-verdicts)
- [Quick Reference](#quick-reference)
- [Relationship to Financial Validation](#relationship-to-financial-validation)
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
| 4. Forecastability | Phase 1 | ACF structure, entropy, signal-to-noise |
| 5. Decomposition | Phase 1, 3 | Trend/seasonality strength, structural breaks |
| 6. Baselines | Phase 3, 5 | Naive/seasonal/drift MAE — model must beat these |
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

## Cross-References

- `system-identification.md` — Model structure selection (ARX, ARMAX, state-space) that ts_reviewer validates
- `validation-checklist.md` — Consolidates all validation requirements including residual checks
- `domain-calibration.md` — Plausibility bounds referenced by ts_reviewer's baseline comparisons
- `financial-validation.md` — Finance-specific extensions to general time-series validation
- `coherence-checks.md` — RAPID tier coherence checks that ts_reviewer automates for time-series data
