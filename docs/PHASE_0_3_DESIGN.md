# Phase 0.3 — Domain Orientation: Design Document

> Draft design for a conditional pre-analysis phase that constructs an auditable domain vocabulary, metrics catalog, and canonical-source ledger before the analyst commits to hypotheses expressed in unfamiliar jargon.

**Status**: DRAFT — design only, no implementation.
**Target version**: v7.15.0 (tentative).
**Depends on**: existing provenance machinery from `abductive_engine.py` (Phase 1.5).

---

## Table of Contents

- [1. Executive Summary](#1-executive-summary)
- [2. Problem Statement](#2-problem-statement)
- [3. Scope and Non-Goals](#3-scope-and-non-goals)
- [4. Design Principles](#4-design-principles)
- [5. Placement in the FSM](#5-placement-in-the-fsm)
- [6. Trigger Conditions](#6-trigger-conditions)
- [7. Tier Matrix](#7-tier-matrix)
- [8. Phase Specification](#8-phase-specification)
  - [8.1 Operators](#81-operators)
  - [8.2 Exit Gate](#82-exit-gate)
  - [8.3 File Write Matrix Addendum](#83-file-write-matrix-addendum)
- [9. Tool Specification: `domain_orienter.py`](#9-tool-specification-domain_orienterpy)
- [10. Artifact Schemas](#10-artifact-schemas)
- [11. Integration Points](#11-integration-points)
- [12. Provenance and Evidence Discipline](#12-provenance-and-evidence-discipline)
- [13. Worked Example](#13-worked-example)
- [14. Tradeoffs and Risks](#14-tradeoffs-and-risks)
- [15. Alternatives Considered](#15-alternatives-considered)
- [16. Open Questions](#16-open-questions)
- [17. Implementation Checklist (for later)](#17-implementation-checklist-for-later)

---

## 1. Executive Summary

The Epistemic Deconstructor presumes **domain literacy**: the analyst already knows what metrics matter in the target field, what technical terms mean, what plausible ranges look like, and which external references to consult. When that presumption fails — for example, an analyst asked to reverse-engineer a credit derivatives engine without knowing what a CDS spread is, or to profile a biomedical assay without knowing the conventional units — the protocol silently misfires at Phase 0. Every downstream phase inherits the framing error, and Trap 20 (Framing) becomes invisible because the analyst lacks the vocabulary to notice the gap.

**Phase 0.3 Domain Orientation** is a conditional phase that runs between Phase 0 (Setup) and Phase 0.5 / 0.7. Its purpose is to construct three auditable artifacts before the analyst commits to hypotheses:

1. `domain_glossary.md` — grounded definitions for the field's technical terms.
2. `domain_metrics.json` — canonical metrics, units, and plausibility ranges (patches `src/config/domains.json`).
3. `domain_sources.md` — the field's authoritative references (textbooks, regulators, standards bodies, seminal papers).

The phase reuses the provenance discipline proven in Phase 1.5: every term, metric, and source carries `source ∈ {library, llm_parametric, analyst, chain_derived}`. LLM-parametric entries are hard-capped until independently grounded. A new `domain_orienter.py` enforces the caps in code, not just documentation.

**Key claim**: Phase 0.3 is cheap (operator-style, stdlib-only, fits in < 1 hour for most domains) and pays for itself by eliminating the single largest class of silent failures — Phase 0 framing errors that are unreachable by downstream falsification because the wrong vocabulary shapes the wrong hypotheses.

---

## 2. Problem Statement

### 2.1 Hidden Domain-Literacy Assumptions

The current protocol encodes domain literacy as an implicit precondition at multiple points:

| Location | Assumption | Failure mode when violated |
|---|---|---|
| P0 Question Pyramid L3-L5 | Analyst knows WHY the field cares about specific parameters | L4/L5 questions collapse into generic prose |
| P0 Hypothesis seeding | Analyst can state H1/H2/H3 in the field's native idiom | Hypotheses are unfalsifiable because they don't map to measurable quantities |
| P0.7 M2 Archetype Accomplices | Analyst can classify the target into a domain-native archetype | Archetype match is generic ("it's a feedback system") instead of domain-specific ("it's a term-structure model") |
| P0.7 M4 Steelman personas | The "domain outsider / journalist / regulator" personas can actually speak the language | Critiques are surface-level; real concerns missed |
| P0.7 M3 Residual-match | Analyst knows which external indices exist for the field | The check is silently skipped; scope gaps persist |
| P1.5 TI Trace Inversion | `trace_catalog.json` has entries matching the field's observable vocabulary | Trace inversion falls back to generic categories and produces low-coverage candidates |
| Evidence calibration (`domains.json`) | Domain is already identified and keyed | `domain_calibration.check_metric` returns "UNKNOWN" silently |
| P3 plausibility bounds | Analyst knows native parameter ranges | Bootstrapped parameter CIs lack a sanity baseline |
| P5 validation | Analyst knows field-native acceptance criteria | Validation passes technically but fails on the criterion the domain actually cares about |

### 2.2 Why Existing Mechanisms Don't Cover This

Three existing mechanisms partially overlap but do not solve this problem:

1. **Multi-pass trigger S1 (scope expansion)** is **reactive**: it fires after Phase 5 flags `[H_S_prime] > 0.40`. By then, the analyst has spent the whole protocol with wrong vocabulary. Fixing framing errors post-hoc is expensive.
2. **Phase 0.7 Scope Interrogation** finds drivers *outside* the scope, but assumes the vocabulary *inside* the scope is correct. Scope correctness and vocabulary correctness are orthogonal concerns.
3. **`domains.json`** presumes domain identification is a solved sub-problem. There's no tool to populate it from scratch when the analyst faces an unfamiliar field.

### 2.3 Concrete Failure Scenarios

- **Scenario A — Credit derivatives without finance fluency**: Analyst frames H1 as "the pricing function is a linear model of inputs." Native vocabulary would frame H1 as "the model implements a reduced-form intensity process with stochastic recovery." Phases 1-4 proceed on the wrong structure; Phase 5 validation shows R² > 0.8 on in-sample data but the model is useless for the actual question (how does the engine price tail risk?).
- **Scenario B — Biomedical assay without lab-science fluency**: Analyst measures "output intensity" instead of the field's native `OD600` or `CFU/mL`. Plausibility bounds in `domains.json` don't trigger because the metric name is wrong. The entire analysis is unmoored from calibration.
- **Scenario C — PSYCH tier across cultures**: Subject uses a regional idiom for an emotional state. Analyst maps it to an English OCEAN trait based on an LLM-parametric guess. `belief_tracker.py` updates fire on a mistranslation. The profile is fluent but wrong.

---

## 3. Scope and Non-Goals

### 3.1 In Scope

- A new **conditional phase** (Phase 0.3) in the FSM.
- A new **CLI tool** (`src/scripts/domain_orienter.py`) with stdlib-only core and optional web fetch.
- Three new **artifacts** (`domain_glossary.md`, `domain_metrics.json`, `domain_sources.md`) plus runtime state (`domain_orientation.json`).
- Updates to the **File Write Matrix** and **Exit Gate procedure** in `SKILL.md`.
- Integration hooks into P0 (re-statement of hypotheses), P0.7 (archetype and persona augmentation), P1.5 (trace catalog patch), P3 (plausibility bounds), and P5 (validation criteria).
- A new reference document `src/references/domain-orientation.md`.
- A new sub-agent definition `src/agents/domain-orienter.md` (follows the `scope-auditor.md` template).

### 3.2 Non-Goals

- **Full ontology learning**: we are building a *sufficient* vocabulary for analysis, not a domain ontology. Ten grounded terms beats a hundred ungrounded ones.
- **Cross-domain translation**: if a system spans two domains (e.g., fintech that's half finance, half software), run Phase 0.3 twice with two separate scopes. Cross-mapping is not in scope for v1.
- **Automatic domain detection**: the analyst declares `domain_familiarity` in `analysis_plan.md`. A future version might add heuristic detection, but not v1.
- **Replacement of `references/domain-calibration.md`**: Phase 0.3 *populates* domain calibration for new domains, it does not replace the static reference.
- **Subject-matter expert surrogate**: Phase 0.3 produces an auditable glossary, not expertise. It lowers the risk of framing errors; it doesn't make the analyst a domain expert.
- **RAPID tier coverage**: RAPID's whole value proposition is speed. Adding a vocabulary phase defeats that.

---

## 4. Design Principles

1. **Provenance first**: every term, metric, source carries `source ∈ {library, llm_parametric, analyst, chain_derived}`. This mirrors Phase 1.5 exactly. Same machinery, reused.
2. **Hard caps in code, not prose**: LLM-parametric entries cannot seed metrics or become canonical sources without grounding. `domain_orienter.py` enforces this by raising `RuntimeError` on violations (pattern copied from `abductive_engine.py`).
3. **Cheap by default**: the exit gate is intentionally low (≥10 terms, ≥3 metrics, ≥2 sources, alias map). This is a floor, not a ceiling. Akin's Law 15 applies.
4. **Conditional, not mandatory**: RAPID skips. LITE skips unless flagged. STANDARD/COMPREHENSIVE/PSYCH mandatory only when `domain_familiarity ∈ {low, unknown}`.
5. **Operator-shaped**: the phase has named operators (TE, TG, MM, AM, CS) matching the style of Phase 0.7 (M1-M4) and Phase 1.5 (TI/AA/SA/AR/IC). This is the protocol's convention for auditable sub-activities.
6. **Artifact > conversation**: the deliverable is three artifacts the analyst can re-read at any later phase. The phase is not valuable unless the artifacts persist.
7. **Reversible**: if Phase 0.3 produces a bad glossary, re-running via `$SM reopen 0.3 "reason"` is supported. Artifacts are versioned as `phase_0_3_passK.md`.
8. **Tool-complementary**: `domain_orienter.py` patches `domains.json` additively. Existing calibration data is never overwritten; it's namespaced by session.

---

## 5. Placement in the FSM

### 5.1 FSM Diagram Update

```
INIT → P0 → [P0.3 if domain_familiarity ∈ {low, unknown}] → P0.5 / P0.7 → P1 → P1.5 → P2 → P3 → P4 → P5
```

For RAPID: `INIT → P0.5 → P5_R` (unchanged).
For LITE: `INIT → P0 → [P0.3 if flagged] → P1_L → P1.5_L → P5_L`.
For PSYCH: `INIT → P0P → [P0.3P if flagged] → P0P.7 → P1P → ...` (Phase 0.3 for PSYCH frames "domain" as the subject's cultural and situational vocabulary).

### 5.2 FSM Transition Rule Addendum

Phase 0.3 passes its exit gate OR the analyst invokes `$SM skip 0.3 "domain_familiarity=high justification"` with a logged rationale in `decisions.md`. The skip command does not currently exist and must be added as part of this phase's implementation.

---

## 6. Trigger Conditions

The trigger is a new field in `analysis_plan.md`:

```yaml
domain_familiarity: high | medium | low | unknown
```

| Value | Meaning | Phase 0.3 action |
|---|---|---|
| `high` | Analyst is a domain expert | Skipped (logged in `decisions.md`) |
| `medium` | Analyst knows the field's basic vocabulary but not edge-case idioms | Optional; analyst decides |
| `low` | Analyst is literate in analysis methodology but not the target domain | MANDATORY |
| `unknown` | Analyst has not assessed their own domain fluency | MANDATORY (default for ambiguous cases) |

**Default value when field is missing**: `unknown` (fail-safe toward more orientation, not less).

**How the analyst assesses `domain_familiarity`**: a short self-assessment checklist in `references/domain-orientation.md`:

- Can I name the top-3 metrics the field uses? (yes/no)
- Can I name the field's canonical textbook or seminal paper? (yes/no)
- Can I define the target system's native archetype in one sentence? (yes/no)
- Would a domain expert recognize my Phase 0 hypotheses as well-framed? (yes/no)

Three or four "yes" → `high`. Two "yes" → `medium`. One or zero "yes" → `low`. Cannot answer honestly → `unknown`.

---

## 7. Tier Matrix

| Tier | Phase 0.3 | Notes |
|---|---|---|
| RAPID | Never | Conflicts with RAPID's speed objective |
| LITE | Conditional on `domain_familiarity` | If triggered, run TE + TG + CS only (skip MM, AM) |
| STANDARD | Conditional on `domain_familiarity` | Full five operators |
| COMPREHENSIVE | Mandatory | Full five operators, multi-pass permitted |
| PSYCH | Conditional on `domain_familiarity` | Framed as cultural and situational vocabulary |

The LITE tier gets a reduced operator set because LITE already skips Phases 2-4; spending effort on MM (metrics mapping) and AM (alias map) is disproportionate.

---

## 8. Phase Specification

### 8.1 Operators

Phase 0.3 has five operators, each producing auditable output. All artifacts must be written via `$SM write`.

#### TE — Term Extraction

**Purpose**: extract candidate technical terms from initial materials. These are the terms the glossary will later ground.

**Inputs**: user's system description, initial documents, interface specs, log samples, anything the analyst has in hand at Phase 0.

**Method**:
- Heuristic extraction over source text: capitalized phrases, acronyms, domain-signature tokens (e.g., tokens matching `[A-Z]{2,}`, multi-word noun phrases, tokens containing digits).
- Stdlib-only (no external term-extraction library dependency in v1).
- Output: a ranked list of candidate terms with frequency and source span.

**CLI**: `domain_orienter.py extract --input <path-or-text>` appends candidates to `domain_orientation.json` under `candidate_terms`.

**Invariants**:
- Each candidate has `{text, frequency, source_ref}`.
- Duplicates deduplicated case-insensitively; canonical form preserves original casing from first occurrence.

#### TG — Term Grounding

**Purpose**: assign each candidate term a definition, source, and confidence. This is the operator where provenance discipline applies hardest.

**Method**:
- For each candidate: the analyst either (a) cites a `library` source (URL, textbook, regulator), (b) provides an analyst-grounded definition from personal expertise, (c) accepts an LLM-parametric definition (capped), or (d) marks the term as "not relevant to the analysis" and discards it.
- Web fetches are supported via the existing `WebFetch` tool wiring used elsewhere in the protocol.

**CLI**: `domain_orienter.py ground --term <text> --definition "<def>" --source <library|analyst|llm_parametric> --url <optional>`.

**Caps (enforced in code)**:
- `llm_parametric` terms: `confidence ≤ 0.60`. Cannot be referenced by a metric or canonical source until upgraded.
- `analyst` terms: `confidence ≤ 0.80` unless the analyst also logs a short inference chain (mirrors P1.5 `chain_derived` pattern).
- `library` terms: `confidence = 1.00` by construction.

**Exit condition for TG**: `grounded_terms ≥ 10` (or tier-specific floor) AND `library_sourced_fraction ≥ 0.3`.

#### MM — Metrics Mapping

**Purpose**: identify the field's canonical metrics, their units, and their plausibility ranges. This directly patches `src/config/domains.json`.

**Method**:
- For each metric: `{name, symbol, units, higher_is_better, plausibility: [suspicious, plausible_low, plausible_high, excellent], source, confidence}`.
- Plausibility-range format matches `domains.json` exactly.

**CLI**:
```
domain_orienter.py add-metric --name "sharpe_ratio" --units "dimensionless" \
  --higher-is-better true --plausibility "3.0,0.5,1.5,2.5" \
  --source library --url "..." --domain finance
```

**Caps**:
- `llm_parametric` metrics **cannot** be promoted to `domain_metrics.json` without analyst countersignature.
- Each metric must have `source_url` or `source_citation` unless `source=analyst`.

**Exit condition for MM**: `metrics_with_plausibility ≥ 3` (LITE: skipped).

#### AM — Alias Map

**Purpose**: capture synonyms, competing terminologies, and regional variations. Example: "Basel III" (global) ↔ "CRR/CRD IV" (EU); "accuracy" ↔ "hit rate" (ML vs. classical stats).

**Method**:
- For each canonical term, list known aliases with source and region/school tag.

**CLI**: `domain_orienter.py alias --canonical "<term>" --aliases "a1,a2,a3" --source <...> --region <optional>`.

**Exit condition for AM**: alias map present OR explicit attestation "no aliases identified" logged in `decisions.md` (LITE: skipped).

#### CS — Canonical Sources

**Purpose**: identify the field's authoritative references. These become the `library` source for future provenance claims in P1.5 and evidence calibration.

**Method**:
- For each source: `{title, authors_or_publisher, year, url_or_doi, category ∈ {textbook, regulator, standard, seminal_paper, benchmark_dataset}, coverage_notes}`.

**CLI**: `domain_orienter.py source --title "..." --publisher "..." --category textbook --url "..."`.

**Caps**:
- `llm_parametric` sources (the model "knows" a reference exists): MUST be verified before promotion. Minimum verification is a successful WebFetch returning status 200. Unverified references stay in `candidate_sources` and cannot be cited downstream.

**Exit condition for CS**: `verified_sources ≥ 2`.

### 8.2 Exit Gate

All items below must be satisfied before Phase 0.3 can be closed. Write operations via `$SM write` (or tool CLI that wraps it).

- [ ] `domain_orientation.json` persisted (runtime file, created on first mutation).
- [ ] `domain_orienter.py extract` run at least once; `candidate_terms` non-empty.
- [ ] `grounded_terms ≥ 10` (STANDARD/COMPREHENSIVE) or `≥ 5` (LITE). `library_sourced_fraction ≥ 0.3`.
- [ ] `metrics_with_plausibility ≥ 3` (STANDARD/COMPREHENSIVE). LITE: skipped.
- [ ] Alias map present or explicit attestation logged. LITE: skipped.
- [ ] `verified_sources ≥ 2`.
- [ ] `domain_glossary.md` written (human-readable).
- [ ] `domain_metrics.json` written.
- [ ] `domain_sources.md` written.
- [ ] `analysis_plan.md` updated: `domain_familiarity` re-evaluated post-orientation (expected: `low` → `medium` or `high`).
- [ ] `decisions.md`: log trade-offs (e.g., "used LLM-parametric definition for term X because no library source found within time budget").
- [ ] Phase 0 hypotheses **re-read** with the new glossary; if any hypothesis requires re-statement in native idiom, invoke `bayesian_tracker.py rename` (new subcommand — see §17) or log the re-statement explicitly.
- [ ] `state.md`, `progress.md`, `phase_outputs/phase_0_3.md` updated.

**"None found" is not a valid Phase 0.3 output.** If extraction returns fewer than 10 candidate terms, the source materials are insufficient — the analyst must gather more or log an explicit attestation that the domain is trivially small (with justification in `decisions.md`).

### 8.3 File Write Matrix Addendum

Extend the `SKILL.md` matrix with:

| File | P0 | **P0.3** | P0.5 | P0.7 | P1 | P1.5 | P2 | P3 | P4 | P5 |
|------|-----|------|------|------|-----|------|-----|-----|-----|-----|
| `domain_orientation.json` | — | **R+W** | — | R | R | R | R | R | R | R |
| `domain_glossary.md` | — | **W** | — | R | R | R | R | R | R | R |
| `domain_metrics.json` | — | **W** | — | R | R | R | R | R | R | R |
| `domain_sources.md` | — | **W** | — | R | R | R | R | R | R | R |

`analysis_plan.md` becomes R+W in P0.3 (analyst re-assesses `domain_familiarity` at exit).

---

## 9. Tool Specification: `domain_orienter.py`

### 9.1 Location and Dependencies

- Path: `src/scripts/domain_orienter.py`
- Dependencies: Python stdlib only in core. Optional: the same `common.py` helpers used by `abductive_engine.py` (JSON I/O with locking, run-id generation).
- Web fetch: routed through the same pattern as `session_manager.py` external calls (no new HTTP dependency).

### 9.2 Command Surface

```
domain_orienter.py --file <path> <subcommand> [args]

Subcommands:
  start                                    # initialize domain_orientation.json
  extract  --input <path|text>             # TE — extract candidate terms
  ground   --term <t> --definition <d> \    # TG — ground a term
           --source <library|analyst|llm_parametric> \
           [--url <u>] [--confidence <c>]
  add-metric --name <n> --units <u> \       # MM — add a metric
           --higher-is-better <bool> \
           --plausibility "sus,pl_lo,pl_hi,exc" \
           --domain <d> --source <...>
  alias    --canonical <t> --aliases <a1,a2,...> \  # AM — register aliases
           [--region <r>] --source <...>
  source   --title <t> --category <c> \     # CS — register canonical source
           --url <u> [--authors <...>] [--year <y>]
  verify   --source-id <sid>                # HTTP-fetch a source, mark verified
  candidates list                           # list candidates (terms / sources)
  candidates promote --id <id>              # promote candidate to grounded
  glossary render                           # write domain_glossary.md
  metrics  render                           # write domain_metrics.json
  sources  render                           # write domain_sources.md
  gate                                      # check exit gate, return PASS/FAIL
  report [--verbose]                        # human-readable status
  skip --reason <r>                         # record phase skip (writes decisions.md)
```

### 9.3 Enforcement Semantics

The CLI must **raise RuntimeError** (exit non-zero with clear stderr) on the following violations. These are hard rules, not warnings:

| Violation | Command | Error |
|---|---|---|
| Promote LLM-parametric metric | `candidates promote` | "Metric source=llm_parametric cannot be promoted; ground it first" |
| Canonical source unverified | `metrics render` / `glossary render` | "Source SID-N referenced but never verified; run `verify` first" |
| Term confidence > cap | `ground` | "source=llm_parametric cap is 0.60; requested 0.80" |
| Gate check below floor | `gate` | "grounded_terms=8 < floor=10 (STANDARD)"; returns exit code 2 |

Mirrors `abductive_engine.py` style exactly. Analyst muscle-memory transfers.

### 9.4 State File Format

`domain_orientation.json` (runtime, created on first mutation):

```json
{
  "run_id": "domain-<uuid>",
  "created_at": "2026-04-16T12:00:00Z",
  "tier": "STANDARD",
  "domain_declared": "credit_derivatives",
  "candidate_terms": [
    {"id": "TERM-001", "text": "CDS spread", "frequency": 12, "source_ref": "input.md:23"}
  ],
  "grounded_terms": [
    {"id": "TERM-001", "text": "CDS spread",
     "definition": "Annual premium (bps) a protection buyer pays...",
     "source": "library", "source_id": "SID-002",
     "confidence": 1.0, "grounded_at": "..."}
  ],
  "metrics": [
    {"id": "MET-001", "name": "cs01", "units": "USD/bp",
     "higher_is_better": false, "plausibility": [null, 100, 10000, null],
     "source": "library", "source_id": "SID-001", "promoted": true}
  ],
  "aliases": [
    {"canonical_id": "TERM-001", "aliases": ["credit default swap spread", "CDS premium"],
     "region": "global", "source": "library", "source_id": "SID-002"}
  ],
  "sources": [
    {"id": "SID-001", "title": "ISDA 2014 Credit Derivatives Definitions",
     "category": "standard", "url": "https://...", "verified": true,
     "verified_at": "...", "http_status": 200}
  ],
  "gate_last_checked": "...",
  "gate_status": "PASS"
}
```

---

## 10. Artifact Schemas

### 10.1 `domain_glossary.md`

Rendered by `domain_orienter.py glossary render`. Grouped by category; each entry:

```markdown
### CDS spread

**Definition**: Annual premium (basis points) a protection buyer pays on a credit default swap.

**Source**: ISDA 2014 Credit Derivatives Definitions [library, verified]
**Aliases**: credit default swap spread, CDS premium
**Confidence**: 1.0
**Term ID**: TERM-001
```

### 10.2 `domain_metrics.json`

Structured. Consumable by a patch function applied to `src/config/domains.json` on session close or via `domain-calibration.check_metric(... session=...)` overload. **Does not overwrite `domains.json` globally**; lives in session state.

```json
{
  "credit_derivatives": {
    "cs01": [null, 100, 10000, null],
    "dv01": [null, 50, 5000, null]
  }
}
```

`null` = no bound in that direction.

### 10.3 `domain_sources.md`

```markdown
## Standards

### ISDA 2014 Credit Derivatives Definitions

- **Category**: Standard
- **Publisher**: International Swaps and Derivatives Association
- **URL**: https://...
- **Verified**: 2026-04-16 (HTTP 200)
- **Coverage**: CDS mechanics, settlement, events of default
- **Source ID**: SID-001
```

---

## 11. Integration Points

### 11.1 Phase 0 (Setup & Frame) — feedback loop

After Phase 0.3, the analyst re-reads Phase 0 artifacts with the new glossary. If any hypothesis is now poorly framed in native idiom, invoke either:
- `bayesian_tracker.py rename <HID> "new statement"` (new subcommand), OR
- Log the re-statement in `decisions.md` with both old and new text.

This is enforced by the Phase 0.3 exit gate.

### 11.2 Phase 0.7 M2 Archetype Accomplices

`scope_auditor.py enumerate` gains an optional `--glossary <path>` flag that reads `domain_orientation.json` and biases archetype classification toward domain-native archetypes over generic ones.

### 11.3 Phase 0.7 M4 Steelman

The "domain outsider / journalist / regulator" personas are prompted with the glossary context. A future enhancement: require each persona critique to reference at least one `library`-sourced term from the glossary.

### 11.4 Phase 1.5 TI Trace Inversion

`trace_catalog.json` is augmented at runtime with domain-specific entries derived from the glossary. `abductive_engine.py invert` consults both the global catalog and the session-local patch.

### 11.5 Phase 3 Parametric Identification

Plausibility bounds for parameter estimates consult `domain_metrics.json` first, then `domains.json`. A parameter CI that falls outside `[plausible_low, plausible_high]` raises a warning logged to `observations/`.

### 11.6 Phase 5 Validation

Validation hierarchy thresholds (R², FVA, coverage_80) consult `domain_metrics.json` for field-native acceptance criteria. If domain-specific thresholds differ from defaults, the divergence is logged in `validation.md`.

### 11.7 PSYCH tier integration

For PSYCH, Phase 0.3 frames "domain" as the subject's cultural, professional, and situational vocabulary. The glossary captures idioms, slang, and context-specific term usage. OCEAN trait elicitation in Phase 2-P consults the glossary to avoid mistranslating cultural expressions.

---

## 12. Provenance and Evidence Discipline

### 12.1 Source Taxonomy (mirrors Phase 1.5)

- **`library`**: cited external reference with URL or DOI. Ideally verified (HTTP 200). Confidence 1.0.
- **`analyst`**: analyst's own domain expertise. Confidence ≤ 0.80. May be upgraded to `chain_derived` with a logged inference chain.
- **`llm_parametric`**: derived from the LLM's parametric knowledge. Hard-capped at confidence 0.60. **Cannot seed metrics. Cannot be promoted to canonical source without verification. Cannot be cited as evidence in P1.5 chains.**
- **`chain_derived`**: result of a documented reasoning chain over other sources (mirrors `abductive_engine.py chain start / step / close`).

### 12.2 Caps Summary (enforced in code)

| Entity | `library` | `analyst` | `llm_parametric` | `chain_derived` |
|---|---|---|---|---|
| Term confidence | 1.00 | ≤ 0.80 | ≤ 0.60 | ≤ 0.90 |
| Metric promotion | Allowed | Allowed w/ countersign | **Blocked** | Allowed |
| Canonical source | Allowed | Not applicable | **Blocked unless verified** | Not applicable |
| Cite as P1.5 evidence | LR ≤ 5.0 | LR ≤ 2.5 | **LR ≤ 2.0** (inherits P1.5 rule) | LR ≤ 3.0 |

### 12.3 Interaction with Existing Evidence Rules

Evidence Rule 8 (LLM-parametric caps) in `SKILL.md` already enforces `prior ≤ 0.30` and `LR ≤ 2.0` for LLM-parametric *hypotheses*. Phase 0.3 extends this pattern to **terminology**: LLM-parametric *definitions*, *metrics*, and *sources* are similarly capped. The existing rule's prose should be updated to reference Phase 0.3 explicitly.

---

## 13. Worked Example

### 13.1 Scenario

Analyst is asked to reverse-engineer a pricing engine used by a credit-derivatives desk. Analyst has general software reverse-engineering skills but no finance background. `domain_familiarity: low`. Tier: STANDARD.

### 13.2 Phase 0.3 Walkthrough

```bash
SM="python3 <skill>/scripts/session_manager.py --base-dir <project>"
DO="python3 <skill>/scripts/domain_orienter.py --file $($SM path domain_orientation.json)"

# Start
$DO start

# TE — Term Extraction from initial materials
$DO extract --input $($SM path analysis_plan.md)
# → candidate terms: CDS, ISDA, CS01, DV01, recovery rate, term structure, ...

# TG — Term Grounding (library sources preferred)
$DO ground --term "CDS spread" --source library \
  --url "https://isda.org/definitions/" \
  --definition "Annual premium (bps) paid by protection buyer..."

$DO ground --term "CS01" --source library --url "..." \
  --definition "Change in CDS value per 1bp widening of the spread."

$DO ground --term "reduced-form model" --source analyst \
  --definition "Intensity-based default model where default time is first jump of a Cox process." \
  --confidence 0.7

# MM — Metrics Mapping
$DO add-metric --name "cs01" --units "USD/bp" --higher-is-better false \
  --plausibility "null,100,10000,null" --domain credit_derivatives \
  --source library --url "..."

# CS — Canonical Sources
$DO source --title "ISDA 2014 Credit Derivatives Definitions" --category standard \
  --url "https://..." --year 2014
$DO verify --source-id SID-001

$DO source --title "Credit Risk Modeling" --category textbook \
  --authors "Lando" --year 2004

# AM — Alias Map
$DO alias --canonical "CDS spread" --aliases "credit default swap spread,CDS premium" \
  --source library

# Render artifacts
$DO glossary render
$DO metrics render
$DO sources render

# Gate check
$DO gate
# → PASS: 14 grounded terms, 4 metrics, 3 verified sources

# Update state
$SM write phase_outputs/phase_0_3.md <<'EOF'
Phase 0.3 Domain Orientation — complete
...
EOF
```

### 13.3 Downstream Impact

Phase 0 hypotheses (originally stated in generic software-RE idiom) get re-read:

- **Before P0.3** — `H1: "the pricing function is a linear model of inputs"`.
- **After P0.3** — `H1: "the pricing function implements a reduced-form intensity process with piecewise-constant hazard and stochastic recovery"`.

Phase 0.7 M2 archetype classification proposes `credit_pricing_engine` archetype (domain-native) instead of `generic_function_approximator`. Phase 1.5 TI has domain-specific trace entries. Phase 3 parameter CIs are sanity-checked against `domain_metrics.json`. Phase 5 validation uses field-native acceptance (e.g., "calibration to market CDS spreads within 2 bps" instead of "R² > 0.95").

### 13.4 Without Phase 0.3 (Counterfactual)

Analyst fits a linear model, achieves R² > 0.95 in-sample on the training data, declares success in Phase 5. The desk rejects the model because it fails on tail-risk scenarios the linear model structurally cannot represent. The analysis is technically complete but practically useless. The framing error originated in Phase 0; Phase 0.3 is the only phase where it could have been caught cheaply.

---

## 14. Tradeoffs and Risks

### 14.1 Tradeoffs

- **Cost**: adds one phase, one tool, three artifacts, a reference document, and a sub-agent. Upfront implementation cost and ongoing maintenance burden.
- **Friction**: another gate in the FSM. Analyst must produce ≥10 grounded terms before analysis begins proper. Mitigated by making it conditional on `domain_familiarity`.
- **Scope creep potential**: a poorly-disciplined analyst could spend days building a glossary. Mitigated by operator-shaped exit gate and the Akin's-Law-15 principle.
- **Duplicate effort**: for analysts who already know the domain, Phase 0.3 is pure overhead. Mitigated by the `high` / `medium` skip path.

### 14.2 Risks

| Risk | Severity | Mitigation |
|---|---|---|
| LLM hallucinates plausible-looking but wrong definitions | **High** (compound errors) | Hard caps in code. LLM-parametric definitions cannot seed metrics or sources. Analyst must ground against `library` before promoting. |
| Analyst uses `analyst` source to disguise LLM-parametric content | Medium | Social; reference document warns against it. Could add auto-detection heuristic (definitions flagged if they match LLM stylistic signatures) — deferred. |
| Wrong canonical source (regulator obsoleted, textbook outdated) | Medium | Same as P1.5: sources are auditable and revisable. Verification timestamps visible in glossary. |
| Phase 0.3 becomes a bottleneck on a time-critical analysis | Medium | `skip` command with logged rationale. Trigger matrix makes it conditional by default. |
| Integration breaks existing sessions | Low | Phase 0.3 is additive. Sessions without `domain_orientation.json` fall through to current behavior. |
| Inflated confidence after 0.3 → Trap 23 (Premature Closure) | Medium | Reference document explicitly warns: a glossary is not expertise. Phase 5 still validates against domain plausibility. |

### 14.3 Rejected Simplifications

- **"Just add a few activities to Phase 0"**: tried in design iteration; Phase 0 becomes unwieldy and the exit gate conflates vocabulary with framing. Separate phase cleaner.
- **"Make it a static checklist, not a tool"**: loses enforcement. Without code-level caps, LLM-parametric content leaks silently. Unacceptable given severity.
- **"Run it only when Phase 5 catches a framing error"**: too late. Analysis is sunk cost by then.

---

## 15. Alternatives Considered

1. **Status quo + better documentation**: add a warning to `SKILL.md` telling analysts to be careful about domain literacy. Rejected: documentation-only interventions do not survive context pressure.
2. **Phase 0.3 as an always-available toolkit (no FSM state)**: the analyst can call `domain_orienter.py` whenever they want, but there's no gate. Rejected: without a gate, the default becomes skipping, which reproduces the current failure mode.
3. **Fold into Phase 1.5 as a sixth operator**: Phase 1.5 runs after Phase 1 boundary mapping. Too late — Phase 0 hypotheses are already committed. Rejected.
4. **Run Phase 0.3 automatically by default in all tiers**: rejected because it breaks RAPID's value proposition and creates friction for analysts who already know the domain.
5. **Make `domain_familiarity` auto-detected via a heuristic**: deferred to a future version. The v1 self-assessment is simpler and avoids false-negative auto-detection.

---

## 16. Open Questions

These decisions are deferred to implementation:

1. **`domain_familiarity` field placement**: in `analysis_plan.md` top-level, or in a new `domain_orientation` section? **Recommendation**: top-level, next to `tier` and `fidelity`.
2. **Multi-domain systems**: a fintech system is half finance, half software. Run Phase 0.3 once per domain, or once with unioned scope? **Recommendation**: once per domain, with sessions namespaced (`domain_orientation_finance.json`, `domain_orientation_software.json`). Needs `--namespace` flag on tool.
3. **`domains.json` patch policy**: should successful Phase 0.3 runs contribute back to the shared `src/config/domains.json`? **Recommendation**: NO in v1. Session-local only. Opt-in promotion in a later version to avoid contamination.
4. **Bayesian tracker rename command**: do we add `bayesian_tracker.py rename <HID>` in this release, or just mutate via JSON edit? **Recommendation**: add the command — aligns with the "tool for every mutation" invariant.
5. **Sub-agent `domain-orienter.md`**: should it run in the background like `scope-auditor.md`, or synchronously? **Recommendation**: synchronous, because its output feeds Phase 0.7 immediately.
6. **Reference document `domain-orientation.md` size**: one file, or split into `domain-orientation-protocol.md` + `domain-orientation-examples.md`? **Recommendation**: one file in v1; split only if it exceeds 400 lines.
7. **Web fetch pattern**: reuse `session_manager.py` wiring, or direct WebFetch tool calls from the sub-agent? **Recommendation**: direct WebFetch from sub-agent, persistence via tool CLI. Matches existing pattern.
8. **Handling of proprietary sources** (paywalled journals, internal docs): how do we verify without access? **Recommendation**: allow `verified_by: citation` (cite a DOI/ISBN even if not fetchable) with lower confidence than `verified_by: http_200`.

---

## 17. Implementation Checklist (for later)

Ordered by dependency. Each item is implementable as an isolated commit.

### 17.1 Config and Reference (no code)

- [ ] Add `src/references/domain-orientation.md` (self-assessment checklist, operator guide, worked example).
- [ ] Update `CLAUDE.md` to mention Phase 0.3 in the phase integration flow.
- [ ] Update `src/config/domains.json` schema documentation.

### 17.2 Core Tool

- [ ] `src/scripts/domain_orienter.py` — full CLI per §9. Stdlib-only. Pattern-match `abductive_engine.py`.
- [ ] `tests/test_domain_orienter.py` — unit tests for: start, extract, ground caps, metric promotion caps, gate thresholds, rendering.

### 17.3 Session Manager Extensions

- [ ] `session_manager.py` — add `domain_orientation.json`, `domain_glossary.md`, `domain_metrics.json`, `domain_sources.md` to the managed file list.
- [ ] Add `session_manager.py skip <phase> --reason` subcommand.

### 17.4 Tracker Extension

- [ ] `bayesian_tracker.py rename <HID> "new statement"` subcommand.
- [ ] `belief_tracker.py rename <BID> "new statement"` subcommand.

### 17.5 Protocol Integration

- [ ] `src/SKILL.md` — add Phase 0.3 section between Phase 0 and Phase 0.5.
- [ ] `src/SKILL.md` — update FSM diagram, File Write Matrix, Evidence Rule 8 note.
- [ ] `src/agents/domain-orienter.md` — new sub-agent definition.
- [ ] `src/agents/scope-auditor.md` — update to consult `domain_orientation.json` for M2 archetype classification.
- [ ] `src/agents/epistemic-orchestrator.md` — FSM updates, trigger logic.
- [ ] `src/agents/hypothesis-engine.md` — add rename flow, glossary-aware hypothesis statement handling.

### 17.6 Downstream Hooks

- [ ] `scope_auditor.py` — `--glossary` flag on `enumerate`.
- [ ] `abductive_engine.py` — session-local `trace_catalog.json` patch loader.
- [ ] `parametric_identifier.py` — plausibility check against session `domain_metrics.json`.
- [ ] `forecast_modeler.py` — same.

### 17.7 Validation and Docs

- [ ] `tests/` — integration test: end-to-end STANDARD tier run with `domain_familiarity: low` proceeds through Phase 0.3 correctly.
- [ ] `CHANGELOG.md` — v7.15.0 entry.
- [ ] Update `README.md` phase list.
- [ ] Update `docs/subagents.md` with `domain-orienter` entry.

### 17.8 Release Gate

- [ ] All 543 existing tests pass unchanged.
- [ ] New `test_domain_orienter.py` passes (target ≥ 40 test cases).
- [ ] Worked-example session runs end-to-end on the credit-derivatives scenario from §13.
- [ ] Decision anchors added for any DECISION-tier guards in `domain_orienter.py` (mirroring `abductive_engine.py` D-003/D-004/D-005 pattern).

---

## Appendix A — Comparison to Phase 0.7 and Phase 1.5

| Dimension | Phase 0.3 Domain Orientation | Phase 0.7 Scope Interrogation | Phase 1.5 Abductive Expansion |
|---|---|---|---|
| Asks | "What do things *mean*?" | "What's *outside* the scope?" | "What *causes* are consistent with observations?" |
| Inputs | Initial materials | Phase 0 scope S | Phase 1 observations |
| Outputs | Glossary, metrics, sources | Exogeneity candidates | Promoted hypotheses |
| Operators | TE, TG, MM, AM, CS | M1, M2, M3, M4 | TI, AA, SA, AR, IC |
| Provenance | library / analyst / llm_parametric / chain_derived | implicit (mechanism-sourced) | library / analyst / llm_parametric / chain_derived |
| Gate | ≥10 grounded terms, ≥3 metrics, ≥2 sources | ≥3 unique candidates | ≥3 inverted observations |
| Tool | `domain_orienter.py` | `scope_auditor.py` | `abductive_engine.py` |
| Tier gating | Conditional on `domain_familiarity` | All but RAPID / LITE | All but RAPID |
| Feeds | P0 (re-frame), P0.7, P1.5, P3, P5 | P1, P1.5, P2, P3, P5 | P2, P3, P4, P5 |

The three phases are **orthogonal**: vocabulary, scope, causes. A correct analysis needs all three. The current protocol has the second and third but not the first.

---

## Appendix B — Out-of-Scope Extensions (Deferred to v2+)

- Auto-detection of `domain_familiarity` from analyst's prior session history.
- Cross-session domain knowledge reuse (shared glossary cache per analyst).
- Promotion of Phase-0.3 outputs back to `src/config/domains.json` via a curation workflow.
- Ontology-learning-grade relation extraction (term-to-term is-a, part-of relationships).
- Multi-language glossaries for PSYCH tier cross-cultural analysis.
- Visual glossary browser (web UI reading `domain_orientation.json`).

---

**End of design document.**
