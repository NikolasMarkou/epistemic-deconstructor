# Abductive Reasoning — Phase 1.5 Reference

Phase 1.5 of the Epistemic Deconstruction Protocol (STANDARD, COMPREHENSIVE, LITE; skipped in RAPID). Operates on the premise that **rigorous hypothesis tracking is wasted without rigorous hypothesis generation**. Phase 0.7 interrogates the system boundary (where the frame might be wrong). Phase 1.5 interrogates the interior (what candidate causes could explain the observations we now have).

This reference documents the five abductive operators TI, AA, SA, AR, IC, the coverage-weighted promotion gate, the provenance discipline that bounds LLM-parametric evidence, and three worked examples across distinct domain classes.

## Table of Contents

- [Core Principle](#core-principle)
- [Abduction vs Induction vs Deduction](#abduction-vs-induction-vs-deduction)
- [The Five Operators](#the-five-operators)
  - [TI — Trace Inversion](#ti--trace-inversion)
  - [AA — Absence Audit](#aa--absence-audit)
  - [SA — Surplus Audit](#sa--surplus-audit)
  - [AR — Analogical Retrieval](#ar--analogical-retrieval)
  - [IC — Inference Chains](#ic--inference-chains)
- [Coverage-Weighted Selection](#coverage-weighted-selection)
- [Provenance Discipline](#provenance-discipline)
- [LLM-Parametric LR Caps](#llm-parametric-lr-caps)
- [Failure Modes](#failure-modes)
- [Tier Scaling](#tier-scaling)
- [Phase 1.5 Exit Gate](#phase-15-exit-gate)
- [Worked Example A — Software Service Latency Spike](#worked-example-a--software-service-latency-spike)
- [Worked Example B — Real Estate Price Anomaly](#worked-example-b--real-estate-price-anomaly)
- [Worked Example C — Behavioral Deviation (PSYCH tier)](#worked-example-c--behavioral-deviation-psych-tier)
- [Cross-References](#cross-references)

---

## Core Principle

**Abduction = inference to the best explanation.** Given a set of observations, an abductive reasoner works backwards from effects to candidate causes, then compares candidates by how much of the observation record each explains (coverage) weighted against how complex it needs to be (parsimony).

Previous versions of this protocol tracked and falsified hypotheses rigorously, but the hypothesis set itself was generated informally by the analyst at Phase 0 and (via scope audit) at Phase 0.7. Phase 1.5 makes interior hypothesis *generation* an auditable, tool-mediated step.

**Failure mode prevented**: "right answers to the wrong set of hypotheses." The analyst diligently falsifies H1 vs H2 vs H3, converges on H2, and reports high confidence — but the true cause was H7, a candidate the analyst never proposed because it lived in a causal domain the analyst did not know to look at. Phase 1.5 makes the generation step transparent so the audit trail shows *why* the candidate set is what it is.

---

## Abduction vs Induction vs Deduction

| Form | Direction | Input | Output | Tool role |
|------|-----------|-------|--------|-----------|
| **Deduction** | premise → conclusion | rules + facts | guaranteed conclusion | Phase 2 (causal-analyst applies rules to observed facts) |
| **Induction** | many facts → rule | repeated observations | probabilistic rule | Phase 3 (parametric-id generalizes from data) |
| **Abduction** | observation → cause | one observation + a library | list of candidate causes | **Phase 1.5** (new) |

Abduction is the step scientists use first and acknowledge last: "what would need to be true for this to be the case?" Phase 1.5 formalizes it so that the answer is produced by a tool, with provenance tags, rather than by the analyst's unaided intuition.

---

## The Five Operators

### TI — Trace Inversion

Given an observation, produce candidate causes. Works in two stages:
1. **Catalog lookup**: consult `src/config/trace_catalog.json` keyed on the observation category. Catalog entries are library-sourced (provenance: `library`).
2. **LLM-parametric suggestion** (optional): the analyst or an agent supplies additional candidate causes from LLM parametric knowledge. These carry provenance `llm_parametric` and are hard-capped.

```bash
python3 scripts/abductive_engine.py --file $($SM path abductive_state.json) \
    invert --obs-id O1 --text "p99 latency spike at noon" --category timing
```

Each candidate is staged into `hypothesis_candidates.json` with:
- `cause` — short label
- `mechanism` — one-line mechanism
- `prior` — suggested prior probability
- `source` — provenance tag
- `complexity` — parsimony cost (default 1.0)
- `observations_explained` — list of observation IDs

### AA — Absence Audit

Given a hypothesis, enumerate **what should be observed if it is true**. Each prediction becomes a pending entry in `predictions_pending.json`. When the analyst later confirms observed or absent, the pending entry is closed and the outcome applies evidence to the tracked hypothesis.

```bash
python3 scripts/abductive_engine.py --file $($SM path abductive_state.json) \
    absence-audit --hypothesis H7 \
    --predictions "log entry 'retry storm';metric replayed_count > 0;disk write spike"
```

Purpose: convert the hypothesis from a claim into a falsifiable test plan **before** the analyst goes looking for supporting evidence. This counters the confirmation bias of starting from a hypothesis and searching until something matches.

### SA — Surplus Audit

Diff the observation record against the union of candidate-explained observations. Every observation not claimed by any candidate is **surplus** — an unexplained leftover that may itself be a candidate for a new hypothesis (iterate TI on it). Surplus observations are persisted to `surplus_audit.json`.

```bash
python3 scripts/abductive_engine.py --file $($SM path abductive_state.json) \
    surplus-audit
```

Purpose: no observation is allowed to silently disappear. The protocol explicitly forces the analyst to name every unexplained observation or close the audit with an attestation that none remain.

### AR — Analogical Retrieval

Match a case signature (a short free-text description of the observed symptom pattern) against a library of archetype signatures. Matches suggest which archetype the case belongs to — which in turn brings the archetype's accomplice library (shared with Phase 0.7 scope interrogation) back into consideration as candidate causes.

```bash
python3 scripts/abductive_engine.py --file $($SM path abductive_state.json) \
    analogize --signature "price rising despite flat local demand, foreign buyer dominance"
```

The archetype library now carries a bidirectional index: Phase 0.7 uses `archetype → accomplice domains`, Phase 1.5 uses `signature → archetype` (defined in the `trace_signatures` field of `src/config/archetypes.json`).

### IC — Inference Chains

Structured JSON logs of micro-inference steps, each with a small likelihood ratio, composing via Bayesian odds update into a final posterior. Each step carries its own provenance tag so the auditor can see exactly which steps were library-grounded, analyst-reasoned, or llm_parametric. An auditable chain answers "how did we get to this posterior?" step by step.

```bash
python3 scripts/abductive_engine.py --file $($SM path abductive_state.json) \
    chain start --target H7 --premise "observation O1 (latency) fires"
python3 scripts/abductive_engine.py --file $($SM path abductive_state.json) \
    chain step --id IC1 --claim "latency aligns with GC pause period" --lr 1.8 --source analyst
python3 scripts/abductive_engine.py --file $($SM path abductive_state.json) \
    chain step --id IC1 --claim "GC tuning log confirms periodic pauses" --lr 2.0 --source library
python3 scripts/abductive_engine.py --file $($SM path abductive_state.json) \
    chain close --id IC1 --seed-prior 0.3
```

A separate `chain audit` subcommand checks for gaps (missing references field, closed chain shorter than 2 steps, LR=0 mid-chain without immediate close). Run it before promoting a candidate whose justification depends on the chain.

---

## Coverage-Weighted Selection

**The primary mitigation for hypothesis explosion.** Abductive generation produces more candidates than can be tracked. Something must gate their promotion into the live hypothesis set.

Let `N` = total observations recorded in the session, `E(c)` = number of observations candidate `c` claims to explain, `K(c)` = candidate complexity (default 1.0 for a one-mechanism cause). The candidate's score is:

```
coverage_score(c) = (E(c) / N) / K(c)
```

Candidates are **rejected at promotion** if `coverage_score < threshold` (default 0.30). This is enforced in code, not docs — `abductive_engine.py candidates promote` raises a RuntimeError and refuses to write to the tracker.

### Why coverage over elegance

Coverage is a **functional** measure — it asks "how much of the observation record does this candidate account for?" — rather than an aesthetic one. A narrow cause that explains one data point and is elegant is still narrow. A broader cause that explains many data points and is ugly is still broad. Phase 1.5 prefers breadth by construction.

### Why parsimony weighting

Complexity penalizes candidates that require many moving parts. Without a parsimony term, the engine would prefer any candidate that happened to touch many observations — including "Rube Goldberg" multi-mechanism explanations that are technically complete but lose all explanatory power. The complexity weight is a simple floor against that.

---

## Provenance Discipline

Every candidate carries a `source` field, one of:

| source | Meaning | Trust level |
|--------|---------|-------------|
| `library` | From the trace catalog or archetype library — curated by human analysts | High |
| `llm_parametric` | From an LLM's parametric knowledge without independent check | **Capped** |
| `analyst` | Hand-entered by the analyst running the session | Moderate |
| `chain_derived` | Produced by an inference chain that has been audited | Moderate |

The tool rejects any value outside this set. Every report groups candidates by source so the auditor can see at a glance which portion of the hypothesis set depends on unchecked LLM output.

---

## LLM-Parametric LR Caps

**Hard-coded, not docs-coded.** LLM parametric knowledge is useful for *generating* candidate causes that a human analyst might not have thought of, but it is unreliable as *evidence*. The engine enforces two caps in `abductive_engine.py`:

```
LLM_PARAMETRIC_MAX_PRIOR = 0.30
LLM_PARAMETRIC_MAX_LR    = 2.0
```

- `add_candidate` rejects any `llm_parametric` candidate with prior > 0.30
- `chain_step` rejects any step with `source='llm_parametric'` and LR > 2.0
- `promote` re-checks both caps at promotion time in case of post-hoc source modification

Upgrading the source (to `library`, `analyst`, or `chain_derived`) requires independent evidence. This is the Phase 1.5 equivalent of Evidence Rule 4 ("Consensus ≠ Strong Evidence"): LLM output is a lot like forecaster consensus, and it gets the same treatment.

### Why not refuse LLM output entirely

Because refusing it would mean losing access to the one cheap source of "have you considered X?" suggestions across every domain. The goal is to let LLM suggestions enter the **generator** while keeping them out of the **confirmation loop** until an independent check exists.

---

## Failure Modes

Four failure modes that abductive reasoning is especially prone to:

1. **Retroduction-as-confirmation** — having proposed a cause C, the analyst searches for evidence that C is true and stops there. The AA (Absence Audit) operator counters this: AA forces the analyst to name *all* predictions that C would make, including ones that are likely absent, before collecting evidence.

2. **Narrative fallacy** — the best-sounding story wins, regardless of coverage. Countered by (a) the coverage-weighted selection rule (empirical rather than aesthetic), and (b) the `cognitive-auditor` agent's narrative-fallacy check, which runs specifically against abductive outputs looking for "too clean" framings.

3. **Just-so stories** — plausible but untestable causes. Countered by the AA operator requiring explicit predictions, and by the IC operator requiring structured steps with references.

4. **Hypothesis explosion** — the generator produces more candidates than can be tracked. Countered by the coverage-weighted promotion gate (threshold 0.30 by default) and by LLM-parametric caps which keep the generator from running on unchecked autopilot.

---

## Tier Scaling

Phase 1.5 scales by tier:

| Tier | Phase 1.5 scope |
|------|-----------------|
| **RAPID** | Skipped entirely (the whole point of RAPID is to avoid building a candidate library) |
| **LITE** | **SA + AA only** — surplus audit on observations, absence audit on the one or two hypotheses in play. No TI catalog lookup, no AR archetype matching, no inference chains. |
| **STANDARD** | All five operators run. Catalog lookup for every observation. Absence audit for every active hypothesis. Surplus audit at the end. Analogical retrieval at least once. Inference chains for every candidate being promoted. |
| **COMPREHENSIVE** | All five operators run plus **multiple passes** — iterate TI on surplus observations, re-run AR with a different signature, stage additional chains for cross-candidate comparison. |

The orchestrator selects operator invocation per tier automatically at Phase 1.5 entry.

---

## Phase 1.5 Exit Gate

Required to leave Phase 1.5:

- [ ] `abductive_engine.py invert` run on **≥3 observations**
- [ ] `abductive_engine.py surplus-audit` produced a non-empty diff OR `decisions.md` logs an explicit "no unexplained observations" attestation
- [ ] ≥1 new hypothesis promoted through the staging flow OR `decisions.md` logs an explicit "no promotion warranted" justification
- [ ] ≥1 inference chain logged **per promoted hypothesis**, with **≥2 chain steps each**
- [ ] `predictions_pending.json`, `inference_chains.json`, `hypothesis_candidates.json`, `surplus_audit.json` all exist in the session directory
- [ ] `state.md`, `progress.md`, `phase_outputs/phase_1_5.md` updated

Tier adjustments:
- LITE: skip the `invert ≥3` requirement (LITE may run SA only)
- COMPREHENSIVE: exit gate is the same, but the triggers for reopening Phase 1.5 for a second pass are more permissive

---

## Worked Example A — Software Service Latency Spike

**Phase 1 observations** (abbreviated):
- O1: p99 latency spikes every hour on the hour (category: timing)
- O2: memory usage climbs monotonically between restarts (category: resource)
- O3: requests from US-West region show 3x higher error rate (category: output_anomaly)
- O4: error bursts are correlated with deploy times but not content (category: failure)

**TI** produces (abbreviated):
- From catalog `timing`: queueing/contention, cache effect, clock drift, GC pause
- From catalog `resource`: memory leak, unbounded collection growth, hidden retry storm
- From catalog `output_anomaly`: edge-case input handling, state-dependent branch, integer overflow
- From catalog `failure`: resource exhaustion, invalid precondition, dependency outage
- LLM-parametric (capped): "hourly batch job in a colocated service contends with the request path" → prior 0.25 (at cap), source: llm_parametric

**AA** for the GC-pause candidate: predicts "pause log entries", "GC count metric spike", "young gen promotion rate". The analyst confirms pause log entries (observed) and GC count (observed) but the promotion rate is absent — the chain refuses to promote the GC candidate above 0.6.

**SA** finds that O3 (US-West region skew) is not claimed by any of the top candidates. It becomes a new TI target with category output_anomaly, producing candidates "region-specific deploy skew" and "edge network path difference."

**AR** against signature `"latency spike hourly, memory climb, region skew"` matches the `feedback_control_system` archetype trace_signature "persistent steady-state error under constant setpoint" only weakly, and `api_backed_software_service` trace_signature "load pattern shift from bot or abuse traffic" more strongly. The analyst considers the accomplice list from the `api_backed_software_service` archetype.

**IC** for the "hourly batch job" candidate: starts from prior 0.25 (capped), step 1 (analyst, LR 1.5, "timing correlation is exact"), step 2 (library, LR 2.0, "batch jobs are a known pattern in this archetype"). Final posterior 0.58. Because the candidate's coverage score is 3/4 = 0.75 (explains O1 latency, O2 resource pressure, and O4 deploy correlation), it passes the 0.30 gate and is promoted.

**Outcome**: the staged hypothesis is promoted to `hypotheses.json` via `candidates promote --id CANDn --tracker-path`. The tracked hypothesis id is `H_ABDUCT_CANDn`, visible to Phase 2 and beyond.

---

## Worked Example B — Real Estate Price Anomaly

**Context**: Phase 0.7 scope audit already surfaced cross-border capital flows, illicit finance, immigration, tourism as exogeneity candidates for the Cyprus real estate market. Phase 1 collected observations including "price/sqm rising 2x above the regional fundamentals-based forecast."

**TI** on the price observation (category: generic → fallback):
- From catalog generic: unobserved upstream input, measurement artifact
- LLM-parametric (capped at prior 0.30): "non-resident investor demand floor", "residency-by-investment program demand", "laundering vehicle premium"

**AR** on signature `"price rising despite flat local demand, foreign buyer dominance"` matches the `speculative_asset_market` archetype with high similarity (exact match on two seeded signatures). The archetype's accomplice list gives back the Phase 0.7 candidates as now **interior** hypothesis targets rather than scope expansion candidates.

**AA** on the "non-resident investor demand floor" candidate: predicts "foreign-buyer transaction share > 30%", "price does not track local income", "price tracks exchange-rate or sanction events". All three are confirmed from public sources → hypothesis enters with posterior ~0.7.

**SA**: the "rent yields trending down while prices rise" observation is not explained by the investor-demand candidate (investor candidates typically rent back into the market). Surplus entry is promoted to a new TI iteration in COMPREHENSIVE mode.

**IC**: steps 1-3 compose the investor-demand hypothesis from source library (archetype match, LR 2.0) + analyst (transaction share, LR 2.0) + library (regulator announcement, LR 1.5). Coverage 5/6 observations, score 0.83 — promoted.

**Outcome**: the hypothesis promoted at P1.5 is that the exogenous investor-demand floor is a primary driver. Phase 2 then tests it against the causal graph.

---

## Worked Example C — Behavioral Deviation (PSYCH tier)

**Phase 1-P observation**: subject's baseline linguistic pattern is "we-centric, future-focused, detail-heavy". In the target session, subject shifted to "I-centric, past-focused, abstract" — a clear deviation.

**TI** on the deviation observation (category: behavioral_deviation):
- From catalog: stress or time pressure, role conflict, audience effect
- LLM-parametric (capped): "unspoken conflict with a second party present", "undisclosed deadline", "health event"

**AA** on the "stress or time pressure" candidate: predicts "shorter response latencies", "interruption rate increases", "increased use of filler words". Analyst reviews the recording — two out of three are confirmed, one absent. Posterior updates to ~0.55.

**AR** on signature `"I-centric pronoun shift with past-focused framing under time pressure"` matches the `individual_persona` archetype's trace_signature "decision pattern change under time pressure" with moderate similarity. The analyst folds in the archetype's accomplice list (family, employer, financial pressure, health, peer network) as additional TI targets.

**IC**: a chain argues from "pronoun shift + past-focus" (step 1, analyst, LR 1.8) to "consistent with acute stressor profile" (step 2, library, LR 1.5) to "no evidence of sustained stress in baseline" (step 3, analyst, LR 1.5). Final posterior ~0.6.

**Outcome**: the candidate "acute unresolved stressor — likely family or health, not work" is promoted into `beliefs.json` (PSYCH tier uses `belief_tracker.py`) and Phase 2-P tests it via controlled elicitation probes.

---

## Cross-References

- Protocol: `references/scope-interrogation.md` (Phase 0.7, complementary boundary-level operator set)
- Protocol: `references/evidence-calibration.md` (LR caps, disconfirm-before-confirm)
- Protocol: `references/multi-pass-protocol.md` (Phase 1.5 may fire U3 one-sided evidence trigger)
- Library: `src/config/trace_catalog.json` (TI catalog)
- Library: `src/config/archetypes.json` (trace_signatures for AR)
- Library: `references/archetype-accomplices.md` (Phase 0.7 accomplice narrative; shared with AR in P1.5)
- Tool: `src/scripts/abductive_engine.py`
- Tool: `src/scripts/bayesian_tracker.py` (promotion target)
- Agent: `src/agents/abductive-engine.md` (Phase 1.5 sub-agent)
- Related traps: `references/cognitive-traps.md` — narrative fallacy, confirmation bias, anchoring. Phase 1.5 is specifically audited for narrative fallacy by `src/agents/cognitive-auditor.md`.
- Related reasoning: `references/modeling-epistemology.md` (abduction as reasoning form)
