# Multi-Pass Protocol

Mandatory reopen triggers evaluated at every EXIT GATE (Gate Check step 6). If any trigger fires, `$SM reopen` instead of advancing.

## Table of Contents

- [Universal Triggers](#universal-triggers-every-exit-gate)
- [Phase-Specific Triggers](#phase-specific-triggers)
- [Trigger Override](#trigger-override-when-not-to-reopen)
- [Command](#command)
- [Constraints](#constraints)
- [Workflow](#workflow)
- [Anti-Patterns](#anti-patterns)
- [Example](#example)

---

## Universal Triggers (Every EXIT GATE)

These are checked at **every** phase gate. If any condition is true, reopen the current phase.

| # | Trigger | Threshold | Action |
|---|---------|-----------|--------|
| U1 | **Weak lead** | Lead hypothesis posterior < 0.65 | Reopen same phase |
| U2 | **Stale hypotheses** | 0 Bayesian updates applied this phase | Reopen same phase |
| U3 | **One-sided evidence** | All updates this phase confirm OR all disconfirm (≥3 updates) | Flag cognitive bias in `decisions.md`; reopen if no disconfirming test was attempted |
| U4 | **Adversarial hypothesis neglected** | Adversarial/deceptive H has 0 updates across 2+ phases | Reopen same phase; gather evidence for/against adversarial H |

**Evaluation procedure**: After running `bayesian_tracker.py report` (Gate Check step 5), check U1-U4 against the report output. If firing, log the trigger ID and value in `decisions.md`, then run `$SM reopen`.

---

## Phase-Specific Triggers

These are checked only at the gate of the specified phase. They override the advance-to-next-phase decision.

### Phase 1 → Reopen Phase 1

| # | Trigger | Threshold |
|---|---------|-----------|
| P1.1 | I/O coverage below minimum | < 60% channels characterized (LITE: < 40%) |
| P1.2 | Stimulus-response database thin | < 10 entries (LITE: < 3) |

### Phase 2 → Reopen Phase 2 or Phase 1

| # | Trigger | Threshold | Target |
|---|---------|-----------|--------|
| P2.1 | Causal coverage gap | < 50% behaviors explained | Reopen P2 |
| P2.2 | No falsification achieved | 0 hypotheses refuted or weakened (posterior drop < 0.10) | Reopen P2 |
| P2.3 | Insufficient observations for causal claims | < 2 new observation files this phase | Reopen P1 (need more data) |

### Phase 3 → Reopen Phase 3 or Phase 2

| # | Trigger | Threshold | Target |
|---|---------|-----------|--------|
| P3.1 | Model fit inadequate | Walk-forward R² < 0.65 | Reopen P3 (change model family) |
| P3.2 | Forecast adds no value | FVA ≤ 0% (time-series only) | Reopen P3 |
| P3.3 | Residual structure | Residuals fail whiteness AND show periodic pattern | Reopen P2 (missing causal mechanism) |

### Phase 4 → Reopen Phase 3 or Phase 4

| # | Trigger | Threshold | Target |
|---|---------|-----------|--------|
| P4.1 | Severe emergence gap | Mismatch > 30% | Reopen P3 (improve components) |
| P4.2 | Composition uncertainty blowup | Propagated uncertainty > 2× largest sub-model uncertainty | Reopen P4 (simplify composition) |

### Phase 5 → Reopen Earlier Phase

| # | Trigger | Threshold | Target |
|---|---------|-----------|--------|
| P5.1 | Extrapolation failure | Extrapolation R² < 0.65 | Reopen P3 |
| P5.2 | Conformal coverage miss | Interval coverage < 0.80 at stated confidence | Reopen P3 |
| P5.3 | Domain calibration fail | Key metric outside plausibility bounds | Reopen phase that produced the metric |
| P5.4 | Wrong question | Fidelity target cannot be met with current approach | Reopen P0 (outer loop) |

### PSYCH Tier Triggers

| # | Trigger | Threshold | Target |
|---|---------|-----------|--------|
| PP.1 | Thin baseline | < 5 baseline observations at P1-P gate | Reopen P1-P |
| PP.2 | No deviations recorded | 0 deviation entries at P2-P gate | Reopen P2-P |
| PP.3 | Archetype unconverged | Lead trait posterior < 0.60 at P3-P gate | Reopen P2-P |
| PP.4 | Prediction miss | Behavioral prediction accuracy < 60% at P5-P gate | Reopen P3-P |

---

## Trigger Override: When NOT to Reopen

A trigger firing does not **always** force a reopen. Override is permitted (advance despite trigger) if:

1. **Max reopens reached** — Phase already reopened 3 times. Escalate tier instead.
2. **Data access impossible** — The trigger requires data that cannot be obtained (black-box, no more access). Log as constraint.

Override MUST be logged: `decisions.md` entry with trigger ID, value, and override rationale.

---

## Command

```bash
$SM reopen <phase> "reason for reopening"
```

**What it does:**
1. Archives current `phase_outputs/phase_N.md` → `phase_outputs/phase_N_passK.md`
2. Updates `state.md`: phase number, transition history
3. Prints reminder to update `progress.md` and `decisions.md`

**What it does NOT do:**
- Does not reset hypotheses — carry forward all evidence from previous passes
- Does not delete observations — previous findings remain
- Does not update `progress.md` or `decisions.md` — you must do this manually

---

## Constraints

| Rule | Value | Rationale |
|------|-------|-----------|
| Max reopens per phase | 3 (4 total passes) | Diminishing returns; escalate tier instead |
| Reason required | Always | Must log why in transition history |
| Phase must be completed | Yes | Cannot reopen an in-progress phase |
| Evidence carries forward | Yes | Bayesian posteriors persist; don't re-add old evidence |

---

## Workflow

1. **Detect**: Gate Check step 6 identifies a firing trigger (U1-U4 or phase-specific)
2. **Log**: Write trigger ID, measured value, and threshold to `decisions.md`
3. **Reopen**: `$SM reopen <phase> "trigger: <ID>, value: <X>, threshold: <Y>"`
4. **Re-execute**: Run phase activities with lessons from prior pass; reference `phase_N_passK.md`
5. **Re-gate**: Pass the EXIT GATE again — triggers are re-evaluated
6. **Downstream**: If reopened phase changes inputs to later phases, consider reopening those too

---

## Anti-Patterns

- **Grinding**: Reopening the same phase 3+ times without changing approach → escalate tier
- **Skipping the decision log**: Every reopen MUST log trigger ID and measured value
- **Resetting hypotheses**: Don't wipe posteriors. Add new evidence, don't replay old evidence
- **Ignoring archived passes**: Previous attempts contain valuable negative results

---

## Example

```bash
# Gate Check step 6 at Phase 2 exit:
# bayesian_tracker report shows lead H2 at 0.61 (< 0.65 threshold)
# Trigger U1 fires: weak lead

$SM reopen 2 "trigger: U1, value: lead posterior 0.61, threshold: 0.65"

# Archives phase_outputs/phase_2.md → phase_outputs/phase_2_pass1.md
# Re-execute Phase 2: design new falsification tests, gather disconfirming evidence
# Reference phase_outputs/phase_2_pass1.md for what was already tried

# After pass 2, gate check again — if U1 clears (posterior ≥ 0.65), advance to Phase 3
```

---

## Cross-References

- Iteration loops: `modeling-epistemology.md` (Inner/Middle/Outer)
- Recursive decomposition: `decision-trees.md` (COMPREHENSIVE tier)
- Stopping criteria: `decision-trees.md` ("When to Stop?")
- Evidence rules: `evidence-calibration.md` (don't replay evidence)
