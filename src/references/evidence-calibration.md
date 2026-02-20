# Evidence Calibration Rules

Detailed guidance for likelihood ratio assignment, anti-bundling enforcement, and prior discipline. The compact rules in SKILL.md are the enforcement layer; this file provides rationale and edge-case guidance.

## Table of Contents

- [Bayesian Update Rule](#bayesian-update-rule)
- [Likelihood Ratio Scale](#likelihood-ratio-scale)
- [LR Cap Rules](#lr-cap-rules)
- [Anti-Bundling Rule](#anti-bundling-rule)
- [Prior Discipline](#prior-discipline)
- [Adversarial Hypothesis Requirement](#adversarial-hypothesis-requirement)
- [Disconfirmation Requirement](#disconfirmation-requirement)
- [Common Calibration Mistakes](#common-calibration-mistakes)
- [Tracker Presets Reference](#tracker-presets-reference)

---

## Bayesian Update Rule

```
P(H|E) = P(E|H) · P(H) / P(E)
       = LR · P(H) / [LR · P(H) + (1 - P(H))]

where LR = P(E|H) / P(E|¬H)
```

- LR > 1 → evidence confirms H
- LR = 1 → evidence is neutral
- LR < 1 → evidence disconfirms H
- LR = 0 → evidence falsifies H (posterior → 0)

### Bayes Factor (Model Comparison)

K = P(D|M₁) / P(D|M₂)

| log₁₀(K) | Interpretation |
|-----------|---------------|
| > 2 | Decisive |
| 1–2 | Strong |
| 0.5–1 | Substantial |
| 0–0.5 | Barely worth mentioning |

---

## Likelihood Ratio Scale

| Evidence Strength | LR Range | Preset Name | Approx Effect on Prior |
|-------------------|----------|-------------|----------------------|
| Strong confirm | 3.0–5.0 | `strong_confirm` | posterior ≈ prior × 2–3 |
| Moderate confirm | 1.5–3.0 | `moderate_confirm` | posterior ≈ prior × 1.3–2 |
| Weak confirm | 1.1–1.5 | `weak_confirm` | posterior ≈ prior × 1.05–1.3 |
| Neutral | 1.0 | `neutral` | no change |
| Weak disconfirm | 0.5–0.9 | `weak_disconfirm` | posterior × 0.7–0.95 |
| Moderate disconfirm | 0.2–0.5 | `moderate_disconfirm` | posterior × 0.3–0.7 |
| Strong disconfirm | 0.05–0.2 | `strong_disconfirm` | posterior × 0.1–0.3 |
| Falsify | 0.0 | `falsify` | posterior → 0 |

---

## LR Cap Rules

| Phase | Max LR | Rationale |
|-------|--------|-----------|
| 0 (Setup) | 3.0 | No empirical data yet; only prior knowledge and framing |
| 1 (Boundary) | 5.0 | I/O probes are indirect observations |
| 2 (Causal) | 10.0 | Causal tests can be diagnostic |
| 3–5 (Parametric+) | 10.0 | Parametric, model, and validation evidence |

### When LR > 5 Is Justified

ALL three conditions must be met:
1. Evidence is independently verifiable (not self-reported or consensus-only)
2. Evidence eliminates a plausible alternative explanation
3. Evidence was not known or predicted before the analysis started

If any condition is not met, cap at LR = 5.0 and explain in `SESSION_DIR/decisions.md`.

### Common Evidence Types and Appropriate LRs

| Evidence Type | Appropriate LR | Why |
|---------------|---------------|-----|
| Forecaster consensus | ≤ 2.5 | Forecasters miss turning points routinely |
| Institutional reports (IMF, EC) | ≤ 3.0 | Better than consensus but still systematic bias |
| Structural data (known pre-analysis) | ≤ 3.0 | Not new information; validates framing |
| Direct experimental probe result | 3.0–10.0 | Depends on test specificity |
| Controlled A/B test | 5.0–10.0 | High diagnostic value |
| Contradiction found in target's claims | 5.0–10.0 | Direct falsification evidence |

---

## Anti-Bundling Rule

Each `bayesian_tracker.py update` call must correspond to ONE observable fact or data point.

**Rule**: If you have N distinct pieces of evidence, make N separate update calls.

**Wrong**:
```bash
bayesian_tracker.py update H1 "GDP growth strong, fiscal surplus, low NPLs, tourism resilient" --preset strong_confirm
```

**Right**:
```bash
bayesian_tracker.py update H1 "EC projects 2.6% GDP growth 2026" --lr 2.0
bayesian_tracker.py update H1 "Government fiscal surplus 3% of GDP" --lr 1.5
bayesian_tracker.py update H1 "Bank NPL ratio 3.2%, below EU average" --lr 1.5
bayesian_tracker.py update H1 "Tourist arrivals at record levels" --lr 1.3
```

**Flush rule**: If >3 evidence updates accumulate in a single conversation turn, PAUSE and write new findings to `SESSION_DIR/observations/` before continuing with more updates.

---

## Prior Discipline

### Mutually Exclusive Hypotheses
Priors MUST sum to 1.0 (±0.01). These represent exhaustive, non-overlapping outcomes.

Example: "System uses REST" (0.6) vs "System uses GraphQL" (0.3) vs "System uses custom protocol" (0.1) = 1.0

### Non-Exclusive Hypotheses
Priors may sum to more than 1.0, but you MUST:
1. Document why the hypotheses overlap in `SESSION_DIR/decisions.md`
2. Note that posteriors cannot be directly compared as outcome probabilities
3. Use Bayes factor comparison (`bayesian_tracker.py compare H1 H2`) for relative assessment

### Prior Assignment Guidelines

| Knowledge Level | Appropriate Prior Range |
|-----------------|----------------------|
| Strong prior evidence | 0.6–0.8 |
| Moderate prior evidence | 0.3–0.6 |
| Weak/speculative | 0.1–0.3 |
| Adversarial/tail risk | 0.05–0.15 |

Do not assign priors > 0.8 at Phase 0. You haven't done any analysis yet.

---

## Adversarial Hypothesis Requirement

At least one hypothesis MUST test whether the data, system, or information sources are unreliable. Examples:

- "The system is actively concealing its true behavior"
- "The published data systematically understates the true risk"
- "The institutional consensus is wrong because of shared blind spots"
- "The subject is deliberately presenting a false persona" (PSYCH tier)

The adversarial hypothesis:
- Gets a prior of 0.05–0.15 (proportional to information asymmetry)
- Must receive evidence updates like any other hypothesis
- Can be KILLED only if specific falsification evidence is applied (not by default)
- If it reaches posterior > 0.30, escalate analysis to address data reliability concerns

---

## Disconfirmation Requirement

Before any hypothesis exceeds 0.80 posterior:
1. Review the evidence trail: are ALL updates confirmatory?
2. If yes, you MUST actively seek and apply ≥1 disconfirming evidence item
3. This means searching for evidence that would WEAKEN the hypothesis, not just noticing that no counter-evidence exists
4. Log the disconfirmation search in `SESSION_DIR/observations.md`

This rule exists because confirmation bias is the #1 systematic error in Bayesian tracking. An evidence trail with only confirming updates is a red flag, not a sign of a strong hypothesis.

---

## Common Calibration Mistakes

| Mistake | What Happens | Fix |
|---------|-------------|-----|
| LR=10 for forecaster consensus | Hypothesis jumps to >0.85 in one step | Cap at LR=2.5 |
| Bundling 5 facts into one update | Single LR=10 combines evidence that would individually be LR=1.5 each | Split into 5 separate updates |
| No adversarial hypothesis | Analysis captures by consensus narrative | Add H_adversarial |
| Priors > 0.8 at Phase 0 | No room for evidence to move posteriors | Cap Phase 0 priors at 0.6 |
| Only confirming evidence | Posterior inflates to >0.90 without testing | Apply disconfirmation rule |
| Same LR for all evidence | Lazy calibration; no discrimination | Vary LRs based on evidence quality |

---

## Tracker Presets Reference

### bayesian_tracker.py (System Analysis)

| Preset | LR | Use When |
|--------|-----|---------|
| `strong_confirm` | 10.0 | Direct experimental falsification of alternative (Phase 2+ only) |
| `moderate_confirm` | 3.0 | Specific, verifiable data point supports H |
| `weak_confirm` | 1.5 | Indirect or circumstantial support |
| `neutral` | 1.0 | Evidence doesn't discriminate |
| `weak_disconfirm` | 0.67 | Indirect counter-evidence |
| `moderate_disconfirm` | 0.33 | Specific data point contradicts H |
| `strong_disconfirm` | 0.1 | Direct test result contradicts H |
| `falsify` | 0.0 | Logical impossibility proven |

### belief_tracker.py (PSYCH Tier)

| Preset | LR | Use When |
|--------|-----|---------|
| `smoking_gun` | 10.0 | Definitive behavioral indicator |
| `strong_indicator` | 5.0 | Clear, repeated behavioral pattern |
| `indicator` | 3.0 | Observable behavioral data point |
| `weak_indicator` | 1.5 | Ambiguous or single-instance signal |
| `neutral` | 1.0 | No diagnostic value |
| `weak_counter` | 0.67 | Behavior inconsistent with trait |
| `counter_indicator` | 0.33 | Clear counter-evidence |
| `strong_counter` | 0.1 | Definitive counter-indicator |
| `disconfirm` | 0.01 | Behavior impossible under trait |

---

## Cross-References

- Bayesian tracker CLI: CLAUDE.md (full command reference)
- Cognitive traps affecting calibration: `references/cognitive-traps.md`
- Domain-specific plausibility bounds: `references/domain-calibration.md`
- Red flags indicating bad evidence: `references/red-flags.md`
