# Engineering Laws for Analytical Rigor

Curated engineering design laws (primarily from Akin's Laws of Spacecraft Design) mapped to the Epistemic Deconstruction Protocol. These are not decorative epigrams — each law addresses a specific failure mode in analytical work.

## Table of Contents

- [The Quantification Imperative](#the-quantification-imperative)
- [The Estimation Hierarchy](#the-estimation-hierarchy)
- [Graceful Degradation Axiom](#graceful-degradation-axiom)
- [The Iteration Infinity Principle](#the-iteration-infinity-principle)
- [Interface Primacy (Shea's Law)](#interface-primacy-sheas-law)
- [The Extremum Distrust Principle](#the-extremum-distrust-principle)
- [The False Linearity Trap (Mar's Law)](#the-false-linearity-trap-mars-law)
- [The Authority of Print Fallacy](#the-authority-of-print-fallacy)
- [Post-Hoc Rationalization (Bowden's Law)](#post-hoc-rationalization-bowdens-law)
- [The Parsimony Razor (de Saint-Exupery's Law)](#the-parsimony-razor-de-saint-exuperys-law)
- [The Multiple Solutions Principle](#the-multiple-solutions-principle)
- [The Sunk Cost Immunity Principle](#the-sunk-cost-immunity-principle)
- [The Partial Credit Fallacy](#the-partial-credit-fallacy)
- [The Sanity Check (Terminal Velocity Test)](#the-sanity-check-terminal-velocity-test)
- [Minimum Viable Data (Miller's Law)](#minimum-viable-data-millers-law)
- [The Start-Now Principle](#the-start-now-principle)
- [The Make-It-Work-First Principle (McBryan's Law)](#the-make-it-work-first-principle-mcbryans-law)
- [Fidelity Sufficiency Principle](#fidelity-sufficiency-principle)
- [Phase Integration Map](#phase-integration-map)
- [Cross-References](#cross-references)

---

## The Quantification Imperative

> "Engineering is done with numbers. Analysis without numbers is only an opinion."
> — Akin's Law #1

**Protocol application**: Every phase output must contain quantitative evidence. Qualitative observations are hypotheses, not findings.

| Phase | Quantification Required |
|-------|------------------------|
| 0 | Prior probabilities assigned numerically; state space estimated in bits |
| 1 | I/O measurements with units; response times with precision |
| 2 | Effect sizes calculated; causal strengths estimated |
| 3 | Parameters with uncertainty bounds; information criteria scores |
| 4 | Emergence gap quantified (|predicted-actual|/|actual|); coupling factors |
| 5 | R², MASE, coverage probability; validation scores |

**Anti-pattern**: "The system seems fast" → No. "Response time: 23ms ± 4ms (n=50)" → Yes.

**Operational rule**: If a claim in `observations.md` lacks a number, it is an assumption to be logged in the Assumption & Bias Log, not an observation.

---

## The Estimation Hierarchy

> "When in doubt, estimate. In an emergency, guess. But be sure to go back and clean up the mess when the real numbers come along."
> — Akin's Law #10

Not all numbers are created equal. Track the provenance of every quantitative claim:

| Level | Source | Confidence | Action Required |
|-------|--------|------------|-----------------|
| **Measured** | Direct observation/experiment | High | Document method, precision, sample size |
| **Estimated** | Derived from related data | Medium | Document derivation, state assumptions |
| **Guessed** | Order-of-magnitude reasoning | Low | Flag explicitly; schedule validation |
| **Assumed** | No empirical basis | Minimal | Log in Assumption & Bias Log; test first opportunity |

**Operational rule**: Every number in a session file should be tagged with its provenance level. When a Guessed or Assumed value is later Measured, update the value AND note the delta — systematic estimation errors reveal calibration bias.

**Protocol integration**: The `evidence-calibration.md` LR caps already implement this — Phase 0 caps at LR=3.0 because early evidence is estimation, not measurement.

---

## Graceful Degradation Axiom

> "To design a spacecraft right takes an infinite amount of effort. This is why it's a good idea to design them to operate when some things are wrong."
> — Akin's Law #2

**Protocol application**: Your model WILL be wrong in some dimension. Design the analysis to produce useful output even when sub-models fail.

**Implementation**:
1. **Phase 3**: When fitting parametric models, document which parameters the model is most sensitive to (via `simulator.py sensitivity`). These are the points of failure.
2. **Phase 4**: When composing sub-models, the Weakest Link Rule (`C_system <= min(C_components) * prod(coupling_factors)`) already operationalizes this — low-confidence components dominate system confidence.
3. **Phase 5**: Report model validity domains explicitly. "Model valid for X in [a,b]; degrades gracefully outside; fails catastrophically at X > c."

**Decision rule**: If a sub-model has confidence < 0.5, do not compose it into the system model without documenting the degradation mode. What does the system model predict if this sub-model is wrong?

---

## The Iteration Infinity Principle

> "Design is an iterative process. The necessary number of iterations is one more than the number you have currently done."
> — Akin's Law #3

**Protocol application**: The three iteration loops in `modeling-epistemology.md` (inner/middle/outer) are never "done." The question is not "have I iterated enough?" but "is the current iteration good enough for the fidelity target?"

**Operational rules**:
- Never declare a model "final" — declare it "sufficient for L[N] fidelity"
- Budget at least one iteration cycle when planning time allocation in Phase 0
- If Phase 5 validation fails, the outer loop returns to Phase 0 — this is expected, not a failure
- Track iteration count in `progress.md` to detect diminishing returns

**Connection to stopping criteria**: The `decision-trees.md` "When to Stop?" tree should be applied at the END of each iteration, not as a one-time gate.

---

## Interface Primacy (Shea's Law)

> "The ability to improve a design occurs primarily at the interfaces. This is also the prime location for screwing it up."
> — Shea's Law (Akin's Law #15)

**Protocol application**: In compositional synthesis (Phase 4), the interfaces between sub-models are where both the greatest improvements AND the greatest errors occur.

**Implementation**:
1. **Interface specification** is not optional paperwork — it is the primary analytical artifact of Phase 4
2. When emergence gap > 20%, check interfaces FIRST before adding complexity to sub-models
3. Interface mismatches (type, units, timing, valid ranges) are the #1 source of composition errors
4. Every interface should have an explicit error propagation path documented

**Audit checklist** (apply at every composition point):
- [ ] Input domain of M₂ ⊆ Output range of M₁?
- [ ] Units compatible and documented?
- [ ] Error/uncertainty propagation path specified?
- [ ] Boundary behavior tested (what happens at interface limits)?
- [ ] Timing constraints satisfiable under worst-case conditions?

---

## The Extremum Distrust Principle

> "In nature, the optimum is almost always in the middle somewhere. Distrust assertions that the optimum is at an extreme point."
> — Akin's Law #8

**Protocol application**: When a parameter estimate, model selection, or claim lands at an extreme of its plausible range, this is a red flag — not a finding.

**Implementation**:
- **Domain calibration**: Results at the ceiling of the "Excellent" range should be treated as borderline "Suspicious." True optima in complex systems are interior points.
- **Parameter estimation**: If a fitted parameter hits the boundary of its allowed range, the model is likely misspecified or the range is too narrow.
- **Hypothesis tracking**: If all evidence points in one direction with no counter-evidence, you're probably not looking hard enough (connects to the disconfirmation requirement).

**Exception**: Binary outcomes (system is/isn't X) don't have a "middle." This principle applies to continuous quantities and multi-factor trade-offs.

---

## The False Linearity Trap (Mar's Law)

> "Everything is linear if plotted log-log with a fat magic marker."
> — Mar's Law (Akin's Law #6)

**Protocol application**: Apparent linear relationships are often artifacts of scale choice, limited range, or low-resolution observation.

**Cognitive trap integration**: This is a specific instance of Pattern Recognition Overfitting (Trap 7), elevated to a named trap because of its frequency in system identification.

**Detection**:
1. If a relationship "looks linear," test it:
   - Plot on linear, log-linear, and log-log scales
   - Fit both linear and nonlinear models; compare via AIC/BIC
   - Test at the EXTREMES of the input range — nonlinearity reveals itself at boundaries
2. If R² > 0.95 on a log-log plot, check the actual scale. Log-log transformations compress nonlinearity and inflate apparent fit.
3. Three data points CANNOT distinguish linear from nonlinear (see Miller's Law below).

**Phase 1 application**: During boundary mapping, always probe beyond the "comfortable" operating range. Systems that appear linear in the middle often reveal nonlinearity at edges.

---

## The Authority of Print Fallacy

> "The fact that an analysis appears in print has no relationship to the likelihood of its being correct."
> — Akin's Law #17

> "The previous people who did a similar analysis did not have a direct pipeline to the wisdom of the ages."
> — Akin's Law #16

**Protocol application**: Published analyses, institutional reports, and prior work are EVIDENCE, not TRUTH. They deserve the same Bayesian treatment as any other evidence — weighted by quality, not by source prestige.

**Implementation**:
- **LR caps for published sources**: Already enforced — institutional reports capped at LR ≤ 3.0, forecaster consensus at LR ≤ 2.5
- **RAPID tier**: The entire red-flags catalog exists because publication does not guarantee validity
- **Phase 0**: Prior analyses of similar systems are starting points for hypothesis generation, not settled conclusions. Weight them by methodology quality, not publication venue.

**Corollary**: Your own prior analysis is equally fallible. When revisiting a previous session, treat your own `summary.md` with the same skepticism you'd apply to a stranger's paper.

---

## Post-Hoc Rationalization (Bowden's Law)

> "Following a testing failure, it's always possible to refine the analysis to show that you really had negative margins all along."
> — Bowden's Law (Akin's Law #25)

**Protocol application**: When a model fails validation, resist the urge to retroactively explain why the failure was "actually expected." This is confirmation bias wearing a lab coat.

**Detection signs**:
- Re-interpreting validation failures as "edge cases" or "known limitations" after the fact
- Adjusting model boundaries to exclude failed predictions
- Discovering "noise" or "outliers" only in data that contradicts the model
- "The model is actually correct, the test conditions were wrong"

**Countermeasure**:
1. Document predictions BEFORE testing (write to `observations.md` via `$SM write`)
2. When validation fails, log the failure in `decisions.md` with the original prediction
3. If the model needs adjustment after failure, treat it as a new iteration — don't silently modify the old one
4. Apply the pre-registered prediction rule: if you didn't predict it beforehand, you can't claim it afterward

---

## The Parsimony Razor (de Saint-Exupery's Law)

> "A designer knows that he has achieved perfection not when there is nothing left to add, but when there is nothing left to take away."
> — de Saint-Exupery's Law (Akin's Law #34)

**Protocol application**: The best model is not the one that explains the most — it is the one that explains enough with the least structure.

**Implementation**:
- **Phase 3**: Model selection via AIC/BIC already implements this — these criteria penalize unnecessary parameters
- **Phase 4**: Before adding emergence correction terms, verify each existing component is load-bearing. Remove before adding.
- **General**: If removing a model component doesn't significantly degrade prediction accuracy, remove it

**Connection to fidelity levels**: L3 understanding (WHY) is more valuable than L4 (PARAMETERS) if L3 achieves the analytical goal. Don't pursue L5 (REPLICATE) when L2 (HOW) suffices.

**Anti-pattern**: "We added a 15-parameter correction to reduce MAPE from 4.2% to 3.9%." Ask: Does the 0.3% improvement matter for the decision being supported?

---

## The Multiple Solutions Principle

> "There is never a single right solution. There are always multiple wrong ones, though."
> — Akin's Law #12

**Protocol application**: This is why the protocol mandates ≥3 hypotheses, not 1. The goal is not to find THE answer but to eliminate wrong answers until a sufficient one remains.

**Implementation**:
- **Hypothesis tracking**: The ≥3 hypothesis requirement (including adversarial) is a structural enforcement of this principle
- **Model selection**: Phase 3 should compare multiple model structures, not fit one and declare victory
- **Validation**: A model that passes validation is "not yet refuted," not "correct"

**Corollary**: If your analysis converges to a single explanation with no plausible alternatives, you have probably pruned too aggressively. Widen the hypothesis space.

---

## The Sunk Cost Immunity Principle

> "Your best design efforts will inevitably wind up being useless in the final design. Learn to live with the disappointment."
> — Akin's Law #4

> "Sometimes, the fastest way to get to the end is to throw everything out and start over."
> — Akin's Law #11

**Protocol application**: A model that consumed 80% of the time budget but doesn't work is not "almost done" — it is a failed approach. The outer iteration loop (Phase 5 → Phase 0 re-scope) exists for this reason.

**Decision rule for starting over** (see `decision-trees.md`):
```
├─ Validation failure on fundamental assumption? → START OVER (re-scope)
├─ 3+ iterations with <5% improvement each? → START OVER (different approach)
├─ Discovered the system is a different archetype than assumed? → START OVER
└─ Invested significant time but model works? → Continue, don't restart for elegance
```

**Warning**: "Starting over" means re-scoping the problem (outer loop), not abandoning the session. Previous observations and falsified hypotheses are valuable — they constrain the restart.

---

## The Partial Credit Fallacy

> "Space is a completely unforgiving environment. If you screw up the engineering, somebody dies (and there's no partial credit because most of the analysis was right...)"
> — Akin's Law #41

**Protocol application**: A model that is 90% correct is not 90% useful. The 10% that is wrong may contain the only thing that matters.

**Implementation**:
- **Phase 5 validation**: Report model failure modes explicitly, not just aggregate accuracy. WHERE does the model fail? WHEN? Under what conditions?
- **Decision support**: If the model will inform a high-stakes decision, the failure modes matter more than the success metrics
- **Uncertainty reporting**: Confidence intervals are not decoration — they are the difference between "probably safe" and "might kill someone"

**Connection to graceful degradation**: Identify the conditions under which the model transitions from "useful but imperfect" to "dangerously wrong." Document this boundary explicitly in `validation.md`.

---

## The Sanity Check (Terminal Velocity Test)

> "The odds are greatly against you being immensely smarter than everyone else in the field. If your analysis says your terminal velocity is twice the speed of light, you may have invented warp drive, but the chances are a lot better that you've screwed up."
> — Akin's Law #19

**Protocol application**: This is the domain calibration principle expressed as an engineering heuristic. If your results dramatically exceed known benchmarks, check for errors before celebrating.

**Implementation**: Already operationalized via:
- `domain-calibration.md` "Too Good" rule
- `red-flags.md` extraordinary claims filter
- `coherence-checks.md` plausibility check

**Operational rule**: Before any result enters the "Suspicious" range of the domain calibration table, run a full error audit:
1. Check units and conversions
2. Verify data pipeline (no leakage?)
3. Re-run with independent implementation
4. If it still holds — then you might have something genuinely novel. But bet on the screwup.

---

## Minimum Viable Data (Miller's Law)

> "Three points determine a curve."
> — Miller's Law (Akin's Law #5)

**Protocol application**: Three data points can fit a curve — but they cannot validate one. The minimum for pattern identification is not the minimum for pattern confirmation.

**Implementation**:
- **Phase 1**: Boundary mapping requires ≥20 stimulus-response entries (STANDARD) precisely because small samples overfit
- **Phase 3**: Model selection requires enough data to split into train/validation/test. If n < 3p (where p = parameters), the model cannot be validated
- **General rule**: If your conclusion rests on fewer than 5 independent data points, flag it as PRELIMINARY, not ESTABLISHED

**Anti-pattern**: "We observed the system respond three times under different conditions and fit a model." Three points determine A curve, not THE curve. More data determines WHICH curve.

---

## The Start-Now Principle

> "Not having all the information you need is never a satisfactory excuse for not starting the analysis."
> — Akin's Law #9

> "A good plan violently executed now is better than a perfect plan next week."
> — Patton's Law (Akin's Law #32)

> "Do what you can, where you are, with what you have."
> — Roosevelt's Law (Akin's Law #33)

**Protocol application**: Phase 0 should not consume the entire time budget perfecting the analysis plan. The plan is a hypothesis about how to analyze — it will be revised.

**Implementation**:
- **Phase 0 budget**: Capped at 10% of total time. If you're spending more, you're over-planning.
- **Tier selection**: When uncertain between LITE and STANDARD, start with LITE. You can escalate; you can't un-spend time.
- **Information gaps**: Use the Estimation Hierarchy (above) to fill gaps with estimates, then proceed. Waiting for perfect information is itself a decision — usually a bad one.

**Anti-pattern**: "We can't start the analysis because we don't have access to [X]." Start with what you have. What CAN you observe? What CAN you test? Often the constraints themselves are informative.

---

## The Make-It-Work-First Principle (McBryan's Law)

> "You can't make it better until you make it work."
> — McBryan's Law (Akin's Law #40)

**Protocol application**: Build a crude-but-functional model first, then refine. Don't optimize a model that doesn't yet predict.

**Implementation**:
- **Phase 3**: Start with ARX (simplest parametric model). Only escalate to NARMAX or State-Space if ARX fails validation.
- **Forecasting**: Start with naive baseline. If naive works, you may not need anything fancier.
- **Simulation**: Run a minimal simulation first to verify the model produces plausible outputs before tuning parameters.

**Connection to iteration loops**: The inner loop (tune parameters) should only start AFTER the model produces non-trivial predictions. Tuning a fundamentally wrong model is wasted effort.

---

## Fidelity Sufficiency Principle

> "Design is based on requirements. There's no justification for designing something one bit 'better' than the requirements dictate."
> — Akin's Law #13

> "'Better' is the enemy of 'good'."
> — Edison's Law (Akin's Law #14)

**Protocol application**: The fidelity target (L1-L5) set in Phase 0 is the stopping criterion. Pursuing higher fidelity than the analytical goal requires is a waste of budget.

**Implementation**:
- **Phase 0**: Set fidelity target explicitly. Document WHY this level.
- **Phase 3**: If L2 fidelity is sufficient, do not fit parametric models (L4). Describe the mechanism; skip the parameters.
- **Phase 5**: Validate against the fidelity target, not against theoretical perfection.

**Anti-pattern**: "We achieved L3 understanding but decided to pursue L5 because the model was interesting." Unless the user requested L5, this is scope creep.

---

## Phase Integration Map

Quick reference for which laws apply most strongly at each phase:

| Phase | Primary Laws | Application |
|-------|-------------|-------------|
| 0 | Start-Now, Fidelity Sufficiency, Estimation Hierarchy | Don't over-plan; set clear targets; estimate what you can't measure |
| 0.5 | Authority of Print, Sanity Check, Extremum Distrust | Published ≠ true; too-good ≠ good; extremes are suspicious |
| 1 | Quantification Imperative, False Linearity, Minimum Viable Data | Numbers not opinions; test linearity assumptions; need enough data |
| 2 | Post-Hoc Rationalization, Multiple Solutions | Don't explain away failures; maintain alternatives |
| 3 | Make-It-Work-First, Parsimony Razor, Iteration Infinity | Simple model first; remove before adding; expect to iterate |
| 4 | Interface Primacy, Graceful Degradation, Partial Credit Fallacy | Interfaces matter most; design for partial failure; 90% right ≠ safe |
| 5 | Sanity Check, Sunk Cost Immunity, Partial Credit | Extraordinary results = likely error; abandon failed approaches; report failure modes |

---

## Cross-References

- Cognitive traps catalog: `references/cognitive-traps.md` (Traps 16-19 derive from these laws)
- Evidence calibration rules: `references/evidence-calibration.md` (Estimation Hierarchy)
- Domain calibration: `references/domain-calibration.md` (Sanity Check, Extremum Distrust)
- Decision trees: `references/decision-trees.md` (Sunk Cost Immunity, Start Over criteria)
- Compositional synthesis: `references/compositional-synthesis.md` (Interface Primacy)
- Validation checklist: `references/validation-checklist.md` (Quantification Imperative, Partial Credit)
- Modeling epistemology: `references/modeling-epistemology.md` (Iteration Infinity, Parsimony Razor)
- Red flags catalog: `references/red-flags.md` (False Linearity, Authority of Print)
