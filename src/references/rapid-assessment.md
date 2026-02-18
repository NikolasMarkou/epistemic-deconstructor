# RAPID Assessment Reference

Consolidated guide for the RAPID tier: a 10-minute structured claim validation workflow.

## Table of Contents

- [Overview](#overview)
- [5-Step Workflow](#5-step-workflow)
  - [Step 1: Coherence Check (2 min)](#step-1-coherence-check-2-min)
  - [Step 2: Verifiability Check (2 min)](#step-2-verifiability-check-2-min)
  - [Step 3: Red Flag Scan (3 min)](#step-3-red-flag-scan-3-min)
  - [Step 4: Domain Calibration (3 min)](#step-4-domain-calibration-3-min)
  - [Step 5: Verdict](#step-5-verdict)
- [Verdict Logic](#verdict-logic)
- [RAPID to Next Tier Decision Tree](#rapid-to-next-tier-decision-tree)
- [CLI Quick Reference](#cli-quick-reference)
- [Cross-References](#cross-references)

---

## Overview

RAPID is the fastest tier in the Epistemic Deconstruction Protocol. Use it when:

- You need a quick credibility assessment of a claim, paper, or result
- Time budget is under 30 minutes
- The goal is a go/no-go decision, not deep understanding

RAPID uses Phase 0.5 (Coherence Screening) followed by a direct jump to verdict.
It does **not** perform boundary mapping, causal analysis, or parametric identification.

---

## 5-Step Workflow

### Step 1: Coherence Check (2 min)

Verify basic alignment between claims, data, and methodology.

| Check | What to Verify |
|-------|----------------|
| **Data-task match** | Input data appropriate for claimed output |
| **Metric-task match** | Metrics suitable for this type of problem |
| **Internal consistency** | Numbers, tables, and text don't contradict each other |
| **No AI-slop** | Not obviously AI-generated filler without substance |

**Quick test**: Can you trace raw data -> model input -> prediction target -> claimed task?

**Instant reject conditions** (any one is sufficient):

| Condition | Interpretation |
|-----------|----------------|
| Impossibility | Results violate known domain limits |
| Memorization | Perfect fit on training/calibration data |
| Contamination | Test > training performance |
| Incoherence | Internal contradictions in methodology or results |

### Step 2: Verifiability Check (2 min)

| Check | What to Verify |
|-------|----------------|
| **Data source specified** | Can you find/access the data? |
| **Method documented** | Enough detail to reproduce? |
| **Code/config available** | Can you run it? |

### Step 3: Red Flag Scan (3 min)

Scan across six categories. Each flag found increases scrutiny.

| Category | Example Flags |
|----------|---------------|
| **Methodology** | No baseline comparison, no cross-validation, train/test leak |
| **Documentation** | Missing method details, no error bars, vague data description |
| **Results** | Test > train performance, suspiciously round numbers, cherry-picked metrics |
| **Claims** | Overclaiming ("solves X"), no limitation discussion, extrapolation beyond data |
| **Tool worship** | "Used deep learning so results are valid", fancy tool justifies weak method |
| **Statistical** | p-hacking indicators, multiple comparisons without correction, tiny sample size |

**Meta-Rule**: If 3+ categories have flags, treat the entire work as suspect.

**Severity levels**:
- **Critical**: Instant reject (data contamination, impossibility)
- **Major**: Serious concern, needs explanation (no baseline, no error bars)
- **Minor**: Worth noting but not disqualifying (formatting issues, minor omissions)

### Step 4: Domain Calibration (3 min)

Compare claimed results to domain-specific plausibility bounds.

Use `scripts/rapid_checker.py calibrate` or consult `references/domain-calibration.md`.

| Assessment | Meaning |
|------------|---------|
| **Suspicious** | Beyond what state-of-art typically achieves |
| **Excellent** | Strong but plausible results |
| **Plausible** | Within expected range |
| **Unknown** | No calibration data for this domain/metric |

### Step 5: Verdict

Combine coherence, flags, and calibration into a single verdict.

---

## Verdict Logic

| Verdict | Criteria |
|---------|----------|
| **CREDIBLE** | 0 rejects, 0-1 flags, within calibration bounds |
| **SKEPTICAL** | 0 rejects, 2-3 flags, near bounds |
| **DOUBTFUL** | 4+ flags OR at calibration bounds OR Meta-Rule triggered (3+ categories) |
| **REJECT** | Any instant reject condition OR critical flag OR >5 flags OR beyond bounds |

**Priority rules** (evaluated in order):
1. Any critical flag or coherence failure -> REJECT
2. Any suspicious calibration -> REJECT
3. Total flags > 5 or 4+ categories with flags -> REJECT
4. Total flags >= 4 or 3+ categories (Meta-Rule) -> DOUBTFUL
5. Total flags >= 2 -> SKEPTICAL
6. Otherwise -> CREDIBLE

---

## RAPID to Next Tier Decision Tree

After completing a RAPID assessment, decide what comes next:

```
After RAPID verdict:
+-- CREDIBLE + no follow-up needed? --> DONE
+-- SKEPTICAL/DOUBTFUL + need more? --> STANDARD tier
+-- REJECT + must investigate?      --> STANDARD tier (find root cause)
+-- Complex system revealed?        --> COMPREHENSIVE tier
```

**Key considerations**:
- RAPID gives a credibility score, not an explanation
- If you need to understand *why* something fails, escalate to STANDARD
- If the system has >15 components or adversarial indicators, go COMPREHENSIVE

---

## CLI Quick Reference

```bash
# Start a new assessment session
python scripts/rapid_checker.py start "Paper: XYZ Claims"

# Record coherence checks (Step 1)
python scripts/rapid_checker.py coherence data-task-match --pass
python scripts/rapid_checker.py coherence metric-task-match --fail --notes "Wrong metrics"

# Add red flags (Step 3)
python scripts/rapid_checker.py flag methodology "No baseline comparison"
python scripts/rapid_checker.py flag results "Test > Train" --severity critical

# Check domain calibration (Step 4)
python scripts/rapid_checker.py calibrate accuracy 0.99 --domain ml_classification

# Get verdict (Step 5)
python scripts/rapid_checker.py verdict

# Full report
python scripts/rapid_checker.py report

# List available calibration domains
python scripts/rapid_checker.py domains
```

The `bayesian_tracker.py` also supports RAPID-style tracking:

```bash
# Red flags via bayesian_tracker
python scripts/bayesian_tracker.py flag add methodology "No baseline"
python scripts/bayesian_tracker.py flag report

# Coherence via bayesian_tracker
python scripts/bayesian_tracker.py coherence data-task-match --pass

# Verdict via bayesian_tracker
python scripts/bayesian_tracker.py verdict
```

---

## Cross-References

| Topic | Reference |
|-------|-----------|
| Coherence check details | `references/coherence-checks.md` |
| Full red flag catalog | `references/red-flags.md` |
| Domain calibration bounds | `references/domain-calibration.md` |
| Validation checklist | `references/validation-checklist.md` |
| Financial-specific validation | `references/financial-validation.md` |
| Time-series review | `references/timeseries-review.md` |
| Calibration config | `config/domains.json` |
