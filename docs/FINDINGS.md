# Findings

## v6.5.0 — Financial Forecasting Validation Framework (2026-02-18)

### Problem

Financial forecasting validation requires domain-specific checks that go beyond generic ML validation. Common failure modes in financial ML papers — price-level R², missing Martingale baselines, ignored transaction costs, single-regime testing — are not caught by general-purpose red flags or coherence checks.

### Analysis

Reviewed financial ML validation requirements against existing infrastructure:

**Already covered generically** (no duplication needed):
- Coherence checks, AI-slop detection, publication patterns (`coherence-checks.md`)
- Red flags for memorization, contamination, tool worship (`red-flags.md`)
- Overfitting basics, data leakage, walk-forward validation (`validation-checklist.md`)
- Base financial calibration bounds (`domain-calibration.md` + `domains.json`)

**Gaps identified** (9 categories of unique financial content):
1. Financial disqualifiers — instant-reject conditions specific to finance
2. Stationarity requirements — ADF/KPSS, differencing rules
3. Forecastability assessment — autocorrelation, entropy, noise floor
4. Class imbalance handling — resampling delusion, threshold calibration
5. Regime testing — bull/bear/vol/structural breaks
6. Economic significance — transaction costs, capacity, net profitability
7. Financial statistical tests — Diebold-Mariano, SPA, multiple testing
8. Additional calibration bounds — annual alpha, MCC, max drawdown, training R²
9. The 5% problem component weights — financial-specific scoring

### Changes Made

1. Created `src/references/financial-validation.md` (~370 lines) with all 9 categories
2. Added 4 new metrics to `domains.json` and `rapid_checker.py` fallback defaults
3. Extended lower-is-better logic in `rapid_checker.py` for `max_drawdown` and `r2_returns_train`
4. Added 4 new rows to `domain-calibration.md` Financial Prediction table
5. Updated repository structure in `CLAUDE.md` and `README.md`

### Verification

- `make validate` passes
- All 4 new metrics appear in `rapid_checker.py domains` output
- Calibration tests pass:
  - `annual_alpha 0.10` → PLAUSIBLE (within 0.02–0.15 range)
  - `annual_alpha 0.20` → EXCELLENT (>= 0.15 threshold)
  - `max_drawdown 0.03` → SUSPICIOUS (lower-is-better, <= 0.05)
  - `max_drawdown 0.08` → EXCELLENT (lower-is-better, <= 0.10)
  - `r2_returns_train 0.02` → SUSPICIOUS (lower-is-better, <= 0.05)
  - `mcc 0.70` → SUSPICIOUS (>= 0.60 threshold)

### Design Decisions

- **Single file, not scattered**: All financial content in one reference file rather than appending to multiple existing files. Prevents duplication and keeps the cross-reference chain clean: `SKILL.md` → `domain-calibration.md` → `financial-validation.md`.
- **No SKILL.md changes**: At 487/500 lines, no headroom. Financial content is discoverable via cross-reference chain.
- **Lower-is-better for max_drawdown and r2_returns_train**: Both metrics are "better when lower" — low drawdown is good, low training R² avoids overfitting. Added to the existing `mape` lower-is-better branch.
