# Plain-Language Mirror — The "Bottom Line" Register

This reference defines the **Plain-Language Mirror**: a fixed writing register the orchestrator and report agents apply so a curious non-specialist (roughly a general adult reader, no statistics background) can understand *what we found and how sure we are* — **without** weakening any of the analytical rigor beneath it.

The plain layer is a **gloss on top**, never a replacement. The full technical artifact (numbers, tables, posteriors, verdict tokens) stays verbatim directly beneath the gloss. This is progressive disclosure: the plain "Bottom Line" answers the question for the non-expert; the retained technical block answers it for the expert and the audit trail.

## Table of Contents

- [Purpose & Scope](#purpose--scope)
- [The HARD Rigor-Preservation & Scope-Bounding Rule](#the-hard-rigor-preservation--scope-bounding-rule)
- [Target Register](#target-register)
- [Jargon -> Plain Translation Table](#jargon---plain-translation-table)
- [Confidence Phrasing Scale](#confidence-phrasing-scale)
- [The "Bottom Line" Structure](#the-bottom-line-structure)
- [Before / After Worked Examples](#before--after-worked-examples)
- [Application Checklist](#application-checklist)
- [Cross-References](#cross-references)

---

## Purpose & Scope

The convention exists because every user-facing surface today is expert-dense: the State Block, mid-session phase handoffs, the Phase-5 `summary.md`, the PSYCH profile, and the RAPID verdict all surface raw `LR` / posterior / `AIC` / `Ljung-Box` values with no interpretation. The Plain-Language Mirror fixes this at the instruction layer: each sanctioned surface **leads** with a plain-English "Bottom Line", then retains its technical content unchanged immediately below.

**Progressive disclosure principle.** Plain Bottom Line on top; full technical artifact retained directly beneath; the technical artifact is **never deleted, replaced, or summarized away**. A reader who wants only the gist reads the top; a reader who wants the math reads down.

---

## The HARD Rigor-Preservation & Scope-Bounding Rule

> **This is the #1 constraint. Read it before applying any other section.**

1. **The plain layer is a REGISTER applied to ALREADY-SANCTIONED surfaces only.** Those surfaces are, exhaustively: the per-response **State Block**, the **phase-handoff chat** message, the Phase-5 **`summary.md`**, the **PSYCH profile** report, and the **RAPID verdict**. The Mirror adds a gloss to these and nothing else.

2. **It does NOT license ANY new mid-session output surface.** The Plain-Language Mirror MUST NOT be read as permission to emit "inline findings", a "preliminary audit", a "synthesis from parallel agents", or any other monolithic report outside Phase 5. Those remain **forbidden** (`SKILL.md:113`, `SKILL.md:116`). If applying this convention seems to require a new standalone report block mid-session, you have misread it — stop.

3. **It NEVER deletes, replaces, or summarizes-away the technical artifact.** Every number, table, posterior, model coefficient, and pinned verdict token (`CREDIBLE` / `SKEPTICAL` / `DOUBTFUL` / `REJECT`, `Coherence:`, `Red Flags:`, session and abductive tokens) stays **verbatim** beneath the gloss. The Bottom Line is additive prose, not an edit to the technical block.

4. **It is a non-negotiable DEFAULT, never a toggle or a question.** The orchestrator stays forbidden from asking the user about output format or depth (`SKILL.md:113`, `ed-orchestrator.md:106`). The tier IS the depth; `summary.md` at Phase 5 IS the report format; the State Block IS the per-response surface. The Mirror is simply always on. Do not add a format question, and do not offer a "technical vs plain" choice.

---

## Target Register

Concrete writing rules for the Bottom Line and any plain gloss:

- **Lead with the bottom line.** First sentence = what we found and how sure we are. No preamble, no methodology recap before the answer.
- **Interpret EVERY number in words.** Never let a bare figure stand. "78%" becomes "78% — we're fairly confident but not certain". A `p`-value, an `R^2`, an `LR`, a posterior: each gets a plain-words reading right next to it.
- **Define each jargon term in plain words on first use, then you may use the term.** First mention: "the posterior (our updated confidence, 0-100%)". After that, "posterior" alone is fine within the same artifact.
- **Short sentences.** One idea per sentence. Prefer everyday words over technical synonyms.
- **No undefined acronyms.** Spell out or gloss on first use (`FVA`, `CVaR`, `OCEAN`, `MICE` all need a plain expansion the first time).
- **Be honest about uncertainty.** Say plainly when the evidence is thin: "we cannot yet tell X from Y", "this is a best guess, not a settled finding". Never inflate a toss-up into a conclusion.

---

## Jargon -> Plain Translation Table

Tight, correct glosses for the codebase's recurring terms. Do **not** distort the math when phrasing for the user.

| Term | Plain-language meaning | How to phrase it to the user |
|------|------------------------|------------------------------|
| **posterior** | Our updated confidence in a hypothesis after weighing the evidence, 0-100%. | "we're now about X% confident that ..." |
| **prior** | Our starting confidence *before* this evidence came in. | "we started out roughly X% confident" |
| **likelihood ratio (LR)** | How strongly one piece of evidence pushes confidence up or down (>1 supports, <1 argues against). | "this clue makes the explanation about N times more believable" |
| **log-odds / Posterior Ratio** | A math-friendly way to add up evidence; large positive = strong support. | "adding up all the clues, the case is strongly for/against" |
| **R^2** | How well the model fits the data, 0 (no fit) to 1 (perfect fit). | "the model explains about X% of what we see" |
| **RMSE / MAE / MAPE** | Average size of the model's prediction error (MAPE is in percent). | "on average the prediction is off by about X" |
| **AIC / BIC / FPE** | Model-quality scores that reward fit and penalize complexity — **lower is better**. | "by a fair-comparison score, model A is the better-balanced choice" |
| **Ljung-Box (residual whiteness)** | A check on whether the leftover error looks like random noise (good) or hidden pattern (bad). | "the leftover error looks like random noise, so we likely captured the pattern" |
| **stationarity** | Whether the data's behavior stays steady over time vs drifting. | "the pattern holds steady over time" / "the pattern is drifting" |
| **heteroscedasticity** | The error spread changes across the range (e.g. bigger swings at higher values). | "the prediction is less reliable in some ranges than others" |
| **conformal prediction interval** | A range we're fairly sure the real value falls inside, with a stated chance. | "we're 90% sure the true value lands between A and B" |
| **FVA (Forecast Value Added)** | Whether our model actually beat a naive guess (like "tomorrow = today"). | "our model beat / failed to beat a simple guess by X%" |
| **coverage_score** | How much of the observed evidence a candidate explanation actually accounts for. | "this explanation covers about X% of what we saw" |
| **OCEAN traits** | Five personality dimensions: Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism. | "openness to new ideas, self-discipline, outgoingness, cooperativeness, emotional volatility" |
| **Dark Triad** | Three antagonistic traits: narcissism, manipulativeness (Machiavellianism), callousness (psychopathy). | "self-centeredness, willingness to manipulate, and lack of empathy" |
| **MICE** | Four common motivations: Money, Ideology, Coercion, Ego. | "what drives them: money, beliefs, pressure, or ego" |
| **ARX / ARMAX / NARMAX (structural model)** | Equations that explain a system's output from its past and its inputs (NARMAX adds non-linear terms). | "an equation linking the result to its recent history and the inputs that drive it" |
| **eigenvalue / settling time** | How fast the system returns to steady after a nudge (settling time = how long to settle). | "after a disturbance, the system settles back in about X" |
| **VaR / CVaR** | Worst-case loss thresholds: VaR = a loss you won't exceed at a given chance; CVaR = the average loss when it's that bad or worse. | "in a bad case (1-in-20), losses reach about X; if it's worse than that, about Y on average" |

---

## Confidence Phrasing Scale

Map probability bands to words consistently, and **always pair the word with the number**. This reuses the High/Medium/Low + percentage pattern already in `belief_tracker.profile_report`.

| Probability | Plain word | Example phrasing |
|-------------|-----------|------------------|
| 0-20% | very unlikely | "very unlikely (about 12%)" |
| 20-40% | unlikely | "unlikely (around 30%)" |
| 40-60% | uncertain / toss-up | "a genuine toss-up (about 50%)" |
| 60-80% | fairly likely | "fairly likely (78%) — confident but not certain" |
| 80-95% | likely | "likely (about 88%)" |
| 95%+ | highly likely | "highly likely (97%) — but not a certainty" |

Never drop the percentage in favor of the word alone, and never report a percentage without the word.

---

## The "Bottom Line" Structure

**(a) Phase handoff (chat).** One to two sentences: *what just happened* + *what it means* + *the current best guess and how sure*. It precedes (or wraps) the technical handoff content, which is retained beneath. Example shape: "We finished mapping the inputs and outputs. One explanation is now clearly ahead — H2 (a caching delay) at about 78%, fairly likely but not settled."

**(b) Final report (`summary.md`).** A Bottom Line section that OPENS the report, above the retained analytical sections, answering four things in plain words: **what we found** / **how sure we are** (words + %) / **what it means** / **what to do next**. The existing `## Executive Summary`, `## Validation Results`, etc. stay beneath, unchanged.

**State Block.** The State Block gets **ONE** plain status sentence accompanying it. The `[STATE: ...]` block itself stays **verbatim** (machine-readable); the sentence is a separate additive line, e.g. "In plain terms: we're in the causal-analysis phase and fairly confident (78%) in the caching-delay explanation."

The same shape applies to the **PSYCH profile** (Bottom Line atop the OCEAN/Dark Triad/MICE tables) and the **RAPID verdict** (one plain sentence beside the `CREDIBLE/...` token).

---

## Before / After Worked Examples

**Example A — dense Bayesian phase handoff.**

```text
BEFORE (technical, retained beneath the gloss):
  Lead H2 posterior 0.78 (prior 0.45), LR 4.1 on obs O-07 (latency
  bimodality). H1 posterior 0.16, H4 0.06. Coherence: 5/5. Red Flags: 0.

AFTER (Bottom Line on top):
  Bottom Line: One explanation is now clearly ahead. We think the slowdown
  is most likely a caching delay — we're fairly confident (78%, up from a
  starting 45%) because the latest clue (response times splitting into two
  clear groups) made that explanation about 4x more believable. The next two
  ideas are now unlikely (16% and 6%). Nothing failed our sanity checks.
  [technical block above is retained verbatim beneath this gloss]
```

**Example B — RAPID verdict, dense -> plain.**

```text
BEFORE (technical, retained beneath the gloss):
  Coherence: 4/5 PASS | Red Flags: 1 | Verdict: SKEPTICAL

AFTER (Bottom Line on top):
  Bottom Line: Treat this claim with caution. It mostly hangs together
  (4 of 5 coherence checks passed), but one warning sign turned up, so we
  rate it SKEPTICAL — plausible but not yet trustworthy. Worth a closer look
  before relying on it.
  Coherence: 4/5 PASS | Red Flags: 1 | Verdict: SKEPTICAL
```

In both cases the "after" is readable by a non-expert while the "before" technical artifact is shown **retained beneath it**, with all numbers and pinned tokens intact.

---

## Application Checklist

Self-apply before emitting any sanctioned surface:

- [ ] Led with the bottom line (the answer comes first)?
- [ ] Every number interpreted in words, paired with the figure?
- [ ] Each jargon term defined in plain words on first use?
- [ ] Confidence stated as word **and** percentage, per the scale?
- [ ] Technical artifact retained **verbatim** directly beneath the gloss?
- [ ] Pinned tokens (`CREDIBLE/SKEPTICAL/DOUBTFUL/REJECT`, `Coherence:`, `Red Flags:`) unchanged?
- [ ] **No new report surface introduced** — only an already-sanctioned surface was glossed?
- [ ] No format/depth question asked of the user?

---

## Cross-References

- `SKILL.md` — Plain-Language Mandate; State Block spec; the forbidden-surface rules at `SKILL.md:113,116`.
- `agents/ed-orchestrator.md` — Plain-Language Presentation procedure for phase handoffs and the final summary.
- `agents/ed-validator.md` — `summary.md` opens with the Bottom Line section.
- `agents/ed-psych-profiler.md`, `references/profile-synthesis.md` — PSYCH profile Bottom Line.
- `agents/ed-rapid-screener.md` — RAPID verdict plain gloss.
- `references/evidence-calibration.md` — source definitions for posterior / prior / LR / log-odds.
- `references/profile-synthesis.md` — source definitions for OCEAN / Dark Triad / MICE.
