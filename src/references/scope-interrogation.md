# Scope Interrogation Protocol

Phase 0.7 of the Epistemic Deconstruction Protocol (STANDARD and COMPREHENSIVE tiers). Operates on the premise that **the most dangerous analytical errors are not wrong answers but right answers to the wrong question**: drivers that live outside the initially-framed scope and therefore never receive a hypothesis.

This reference documents the four generalized mechanisms M1-M4 and provides worked examples across three distinct domain classes.

## Table of Contents

- [Core Principle](#core-principle)
- [The H_S Standing Pair](#the-h_s-standing-pair)
- [Mechanism M1 — Flow Tracing](#mechanism-m1--flow-tracing)
- [Mechanism M2 — Archetype Accomplice Library](#mechanism-m2--archetype-accomplice-library)
- [Mechanism M3 — Residual-Signature Matching](#mechanism-m3--residual-signature-matching)
- [Mechanism M4 — Adversarial Scoping (Steelman Prompt)](#mechanism-m4--adversarial-scoping-steelman-prompt)
- [Phase 0.7 Exit Gate](#phase-07-exit-gate)
- [Worked Example A — Cyprus Real Estate Market](#worked-example-a--cyprus-real-estate-market)
- [Worked Example B — API-Backed Software Service](#worked-example-b--api-backed-software-service)
- [Worked Example C — Organizational Actor](#worked-example-c--organizational-actor)
- [Phase 5 Scope Completeness Check](#phase-5-scope-completeness-check)
- [Cross-References](#cross-references)

---

## Core Principle

**The system boundary is a hypothesis, not a premise.** Phase 0 frames the analysis around "the target system." That frame is itself a claim about which causal domains matter — a claim that can be wrong. Scope interrogation makes that claim explicit and tests it like any other hypothesis.

Failure mode prevented: the Cyprus real estate analysis that framed the system as "the housing market" and then had to be re-run four times after the analyst manually discovered that illicit fund flows, investment legislation, immigration, and digital-nomad flows all materially drove prices. Each of those domains should have surfaced as a candidate hypothesis in the first pass.

---

## The H_S Standing Pair

At Phase 0, every STANDARD / COMPREHENSIVE analysis seeds two **standing** hypotheses alongside the normal H1..HN:

- **[H_S]** — "The drivers of the target live within the initially-framed scope S."
- **[H_S_prime]** — "Material drivers of the target exist outside S."

These are tracked via `bayesian_tracker.py` using the canonical `[H_S]` and `[H_S_prime]` **statement prefixes** (IDs are auto-assigned). They are non-exclusive (both can be true — some drivers inside, some outside — but the pair is treated as a Bayesian test of "is our frame sufficient?"). `H_S_prime` satisfies the existing Phase 0 "≥1 adversarial hypothesis" requirement.

Both posteriors are tracked for the entire session. Phase 0.7 updates them based on M1-M4 findings. Phase 5 fails validation if `H_S_prime` posterior exceeds 0.40 and no scope-expansion reopen has been logged.

```bash
# At Phase 0, seed both:
$SM write hypotheses.json  # or directly via tracker:
python3 scripts/bayesian_tracker.py --file $($SM path hypotheses.json) \
    add "[H_S] Drivers of <target> live within initial scope <S>" --prior 0.6 --phase P0
python3 scripts/bayesian_tracker.py --file $($SM path hypotheses.json) \
    add "[H_S_prime] Material drivers exist outside scope <S>" --prior 0.4 --phase P0
```

---

## Mechanism M1 — Flow Tracing

**Universal basis**: every system has inputs and outputs. Every input has a generator upstream; every output has a consumer downstream. If a trace crosses into a domain not currently in scope, that domain is an **exogeneity candidate**.

### Procedure

| Step | Action | Output |
|------|--------|--------|
| 1 | Enumerate every input channel to the target (explicit + implicit) | Input list |
| 2 | For each input, name its immediate upstream generator — one level only | Input→generator map |
| 3 | Check: is the generator inside scope S? | Boundary flag per input |
| 4 | Enumerate every output channel from the target | Output list |
| 5 | For each output, name its immediate downstream consumer — one level only | Output→consumer map |
| 6 | Check: is the consumer inside scope S? | Boundary flag per output |
| 7 | Every flagged generator or consumer is an exogeneity candidate | Candidate list |

One level upstream/downstream is sufficient. Going further collapses into "trace the whole economy" which is not useful. The goal is to catch the **immediate neighbors** that are outside the frame.

### CLI

```bash
python3 scripts/scope_auditor.py trace \
    --inputs "<comma,separated>" \
    --outputs "<comma,separated>" \
    --file $($SM path scope_audit.json)
```

Prints a checklist of generators and consumers to name, and records them to `scope_audit.json`.

---

## Mechanism M2 — Archetype Accomplice Library

**Universal basis**: systems cluster into archetypes (markets, controllers, supply chains, organizations, individuals). Each archetype has **known accomplices** — the typical co-driving domains that analysts of that archetype have historically needed to consider. A library of accomplices per archetype encodes pattern knowledge without prescribing hardcoded domain categories.

### Procedure

| Step | Action | Output |
|------|--------|--------|
| 1 | Classify the target into one or more archetypes from `archetype-accomplices.md` | Archetype ID list |
| 2 | Query `scope_auditor.py enumerate --archetype <id>` for each | Accomplice list per archetype |
| 3 | Union the lists; dedupe by domain | Candidate domain set |
| 4 | For each candidate, ask: is this domain already in scope S? | Boundary flag per candidate |
| 5 | Every flagged candidate becomes a hypothesis seed with the suggested prior | Candidate hypotheses |

### CLI

```bash
python3 scripts/scope_auditor.py enumerate \
    --archetype speculative_asset_market \
    --file $($SM path scope_audit.json)
```

Archetypes are **orthogonal**. A system may match multiple — enumerate all that apply, then dedupe.

### Library Extension

The library lives in `src/config/archetypes.json` with narrative in `references/archetype-accomplices.md`. Users add archetypes by editing the JSON and the narrative. Design rules:

- Archetypes must be **empirically grounded**, not speculative categories
- Each accomplice needs a one-line **mechanism**, not just a label
- Archetypes must **generalize** — if it applies to one instance only, it is too narrow

---

## Mechanism M3 — Residual-Signature Matching

**Universal basis**: when a model explains part of a system's behavior, the residuals are "what's left over." Unexplained residuals have a signature — spectral content, regime shifts, correlation with external indices. Matching residual signatures against external reference series can reveal an omitted driver domain.

### Procedure

| Step | Action | Output |
|------|--------|--------|
| 1 | Compute residuals from the current model (Phase 3 output or Phase 1 mismatches) | Residual series |
| 2 | Extract signature: dominant frequency (via `fourier_analyst.py`), regime shifts (via `ts_reviewer.py`), trend | Signature features |
| 3 | Compare residual series against a library of external indices (policy-change timelines, macro series, event calendars) | Correlation per index |
| 4 | Any index with \|r\| ≥ 0.3 and p < 0.05 is a candidate omitted driver | Candidate list |

### CLI

```bash
python3 scripts/scope_auditor.py residual-match \
    --file $($SM path scope_audit.json) \
    --residuals residuals.csv \
    --indices-dir ./external_indices/
```

The `external_indices/` directory contains one CSV per candidate index (e.g. `interest_rate.csv`, `sanctions_events.csv`, `tourism_arrivals.csv`). Users maintain this directory per analysis.

### When to Run M3

M3 is stateful: it requires an initial model to compute residuals from. Run at:

- **Phase 0.7 initial pass** — if a preliminary baseline model exists from a quick RAPID screening
- **After Phase 3** — on the fitted-model residuals (re-check scope completeness)
- **Phase 5** — as the scope-completeness check

---

## Mechanism M4 — Adversarial Scoping (Steelman Prompt)

**Universal basis**: the analyst's blind spots cluster around their role, training, and viewpoint. Explicit role-play as three distinct **outsider personas** forces generation of critiques the analyst would not naturally produce.

### The Three Personas

| Persona | Perspective | Typical Critique |
|---------|-------------|------------------|
| **Domain outsider** | Someone who has never worked in the target's field | "You are assuming the system is closed — what happens outside your frame affects it how?" |
| **Investigative journalist** | Looks for hidden interests and unnamed beneficiaries | "Who is not on your stakeholder list but would materially care about this system?" |
| **Regulator** | Concerned with compliance, externalities, systemic risk | "What externality does this system produce that could force a policy response?" |

### Procedure

| Step | Action | Output |
|------|--------|--------|
| 1 | Write a one-paragraph summary of the current scope S | Target frame |
| 2 | For each persona, produce a written steelman critique naming one excluded domain **and** one mechanism | 3 critiques |
| 3 | For each critique, log as hypothesis with prior ≥ 0.15 | Candidate hypotheses |
| 4 | Log all three critiques in `scope_audit.md` under "Steelman Critiques" | Permanent record |

The steelman must be specific (not "consider geopolitics" but "Russian sanctions enforcement in Cyprus drives demand for bearer-asset equivalents — real estate is one"). Generic critiques fail the M4 test.

### Why Three

Single-persona critiques tend to miss what that persona does not naturally see. Three orthogonal perspectives produce a more complete sweep. Stop at three — more than three produces diminishing returns and critique-fatigue.

---

## Phase 0.7 Exit Gate

Phase 0.7 cannot complete without:

- [ ] `scope_audit.md` written with the four mechanism outputs: M1 flow trace, M2 archetype enumeration, M3 residual match (or "deferred" if no baseline model yet), M4 three steelman critiques
- [ ] At least **3 exogeneity candidates** logged as hypotheses in `hypotheses.json` (each with prior ≥ 0.05). These candidates are counted separately from H1..HN and `[H_S]`/`[H_S_prime]`.
- [ ] Both `[H_S]` and `[H_S_prime]` hypotheses present in `hypotheses.json` with non-trivial priors
- [ ] `state.md` updated with Phase 0.7 completion
- [ ] `progress.md` updated
- [ ] `decisions.md` logs any scope-expansion decisions (added new domains to S)

Less than 3 candidates → not done. "None found" is not a valid output — if the analyst cannot produce 3 candidates, the archetype classification (M2) or steelman procedure (M4) was too shallow.

---

## Worked Example A — Cyprus Real Estate Market

**Target as framed by user**: "The Cyprus real estate market."

**Phase 0 seed**: H1 supply-demand imbalance, H2 interest-rate sensitivity, H3 demographic trends. Plus `[H_S]` (drivers within housing market) and `[H_S_prime]` (drivers outside housing market).

### M1 Flow Trace

Inputs to the housing market:
- *Buyer capital* → upstream: foreign transfers, local mortgages, investment funds. **Foreign transfers cross into cross-border capital flows — not in scope.**
- *Construction materials* → upstream: import logistics, EU construction market. **Crosses into supply chain — not in scope.**

Outputs from the housing market:
- *Housing units* → downstream: residents, tourists, non-residents, short-term renters. **Non-residents and tourists cross into immigration and tourism domains — not in scope.**
- *Price signals* → downstream: tax authorities, macroprudential regulators. **Crosses into regulatory regime — not in scope.**

M1 produces 4 exogeneity candidates.

### M2 Archetype Enumeration

Classification: **speculative_asset_market** (open-economy thin-pool store-of-value market in a tax-favorable jurisdiction).

```bash
python3 scripts/scope_auditor.py enumerate --archetype speculative_asset_market
```

Accomplices returned:
1. cross-border capital flows (prior 0.25)
2. **illicit finance / AML regime** (prior 0.20) — **missed in original analysis**
3. tax & residency regime (prior 0.20)
4. macroprudential & monetary policy (prior 0.25)
5. **investment legislation changes** (prior 0.20) — **missed in original analysis**
6. **immigration / nomad flows** (prior 0.20) — **missed in original analysis**
7. tourism / rental-yield consumers (prior 0.15)
8. geopolitical shocks (prior 0.15)

All four originally-missed domains — illicit flows, legislation, immigration, digital nomads — are surfaced by the archetype library **on the first pass** before Phase 1 begins.

### M3 Residual-Signature Matching

Deferred until Phase 3 fits a baseline price model.

### M4 Steelman Critiques

- **Domain outsider** (an African commodities trader): "You are treating Cyprus as a closed housing market. My industry sends capital to Cyprus specifically to park outside sanctions — your frame doesn't include the sanctions-arbitrage mechanism that drives my colleagues' purchases. Domain: **sanctions arbitrage / illicit finance**. Mechanism: price floor decoupled from local demand."
- **Investigative journalist**: "You haven't named golden-visa holders as a buyer class. The citizenship-by-investment program has been a major demand driver and a political scandal. Domain: **residency-by-investment policy**. Mechanism: legal-status demand not responsive to mortgage rates."
- **Regulator**: "The ECB and FATF have been watching Cyprus real estate for AML risk. If they act, transaction volumes stop. Domain: **AML enforcement / macroprudential policy**. Mechanism: regulatory action cuts buyer pool by fiat."

M4 produces 3 more critiques, all independently confirming the M2 accomplices.

### Exit Gate

- Exogeneity candidates logged: 8 (from M2) + 4 (from M1) + 3 (from M4) = 15 (many deduplicated → ~8 unique)
- `scope_audit.md` written
- `[H_S]` posterior dropped from 0.6 → 0.35 as evidence of external drivers accumulated
- `[H_S_prime]` posterior rose from 0.4 → 0.65
- Phase 0 outputs updated: the scope is **expanded** to include illicit flows, legislation, immigration, and tourism domains. Phase 1 onward operates on the expanded scope.

Result: on the FIRST analytical pass, the Cyprus analysis now has all four originally-missed domains flagged and budgeted for. No manual re-run needed.

---

## Worked Example B — API-Backed Software Service

**Target**: "An AI coding assistant called CodeBuddy that wraps a third-party LLM API and stores user code in a vector database."

**Phase 0 seed**: H1 model quality drives usage, H2 UX drives retention, H3 pricing drives churn. Plus `[H_S]` and `[H_S_prime]`.

### M1 Flow Trace
- Inputs: user code → upstream: developer IDEs, git repos. Crosses into **IDE plugin ecosystem** — not in scope.
- Inputs: LLM completions → upstream: upstream LLM provider API. Crosses into **upstream provider regime** — not in scope.
- Outputs: generated code → downstream: compilers, CI pipelines, production deploy. Crosses into **CI/deploy toolchain** — not in scope.

### M2 Archetype Enumeration

Classification: **api_backed_software_service** (primary) + **platform_ecosystem** (secondary, if integrators exist).

Accomplices: upstream API provider, hosting / cloud provider, integrator ecosystem, authentication / identity provider, data-privacy regulation, adversarial users.

Missed if M2 not run: upstream provider SLA and rate limits are the single biggest driver of observed "quality variance" — usually labeled as model quality bugs.

### M4 Steelman

- **Domain outsider** (a security auditor): "Your scope doesn't include the prompt-injection threat surface. Adversarial users coming in via malicious code snippets drive incidents you'd call bugs. Domain: **adversarial user inputs**. Mechanism: prompt injection."
- **Investigative journalist**: "Who pays for the LLM calls when users exceed quota? That question reveals a hidden cost driver — the upstream provider's billing model. Domain: **upstream billing regime**. Mechanism: cost structure drives hard availability limits."
- **Regulator**: "Where does user code go after ingest? GDPR and export-control regimes care. Domain: **data-privacy regulation**. Mechanism: legal forces limit which users can access the service."

### Exit Gate

- 6 exogeneity candidates from M2 + 3 from M1 + 3 from M4 = ~10 unique
- `[H_S_prime]` posterior: 0.4 → 0.55 (upstream API regime + adversarial users identified as material)

---

## Worked Example C — Organizational Actor

**Target**: "An environmental NGO campaigning against plastic packaging."

**Phase 0 seed**: H1 public awareness drives behavior, H2 industry response drives policy, H3 media coverage amplifies. Plus `[H_S]` and `[H_S_prime]`.

### M1 Flow Trace
- Inputs: funding → upstream: foundation grants, member donations, EU program money. **Foundation grant regime** is outside scope.
- Inputs: volunteers → upstream: universities, partner NGOs. **Partner NGO network** is outside scope.
- Outputs: policy advocacy → downstream: parliamentary committees, regulators, industry associations. All outside scope.
- Outputs: media campaigns → downstream: journalist networks, social media platforms. **Platform algorithms** are outside scope.

### M2 Archetype Enumeration

Classification: **organizational_actor**.

Accomplices: competitor / rival org, regulatory / oversight body, capital / funding source, labor / talent market, supplier / counterparty network, macro policy regime.

### M4 Steelman

- **Domain outsider** (a major donor to the NGO): "Our decision to renew funding next year is the single largest determinant of your next campaign. You don't name us. Domain: **funding source regime**."
- **Investigative journalist**: "What about plastic-industry counter-campaigns and astroturf? They materially shape public response. Domain: **adversarial corporate communications**."
- **Regulator**: "The NGO's effectiveness depends entirely on whether the proposed directive passes. That's an exogenous political process. Domain: **EU legislative calendar**."

### Exit Gate

9+ candidates logged. Scope expanded to include funding regime, adversarial corporate counter-campaigns, and political calendar.

---

## Phase 5 Scope Completeness Check

At Phase 5 validation, one additional activity:

| Check | Method | Action on Fail |
|-------|--------|----------------|
| `[H_S_prime]` posterior ≤ 0.40 | Read `hypotheses.json`, find `[H_S_prime]` entry | If > 0.40, validation fails unless decisions.md logs a completed scope-expansion reopen |
| `scope_audit.md` exists | File check | If missing, Phase 0.7 was skipped improperly — re-run |
| Re-run M3 on Phase 3 residuals | `scope_auditor.py residual-match` | If any new high-correlation index surfaces, log trigger S1 and reopen Phase 0 |

A passing scope-completeness check either confirms `[H_S]` is well-founded (posterior ≥ 0.80) or **explicitly documents an open scope gap** that is out of budget for the current analysis.

---

## Cross-References

- Archetype library: `archetype-accomplices.md` + `config/archetypes.json`
- Agent: `agents/scope-auditor.md`
- Tool: `scripts/scope_auditor.py`
- Multi-pass trigger: `multi-pass-protocol.md` (trigger S1 Scope Gap)
- Related cognitive traps: `cognitive-traps.md` (Trap 20 Framing Effect, Trap 21 Streetlight Effect, Trap 22 Omitted-Variable Bias, Trap 23 Premature Closure)
- Residual diagnostics: `timeseries-review.md`, `spectral-analysis.md`
- Evidence rules: `evidence-calibration.md`
