# Domain Orientation Protocol

Phase 0.3 of the Epistemic Deconstruction Protocol (conditional: STANDARD / COMPREHENSIVE / PSYCH, and LITE when flagged). Operates on the premise that **the analyst cannot frame what the analyst cannot name**: without the field's native vocabulary, Phase 0 hypotheses silently misframe the target and every downstream phase inherits the error.

Phase 0.3 builds three auditable artifacts — glossary, metrics, canonical sources — before the analyst commits to hypotheses expressed in unfamiliar jargon. It is deliberately cheap (target ≤ 1 hour wall-clock for most domains) and operates via the same provenance discipline as Phase 1.5.

## Table of Contents

- [Core Principle](#core-principle)
- [Self-Assessment: `domain_familiarity`](#self-assessment-domain_familiarity)
- [Operator TE — Term Extraction](#operator-te--term-extraction)
- [Operator TG — Term Grounding](#operator-tg--term-grounding)
- [Operator MM — Metrics Mapping](#operator-mm--metrics-mapping)
- [Operator AM — Alias Map](#operator-am--alias-map)
- [Operator CS — Canonical Sources](#operator-cs--canonical-sources)
- [Phase 0.3 Exit Gate](#phase-03-exit-gate)
- [Worked Example — Credit Derivatives](#worked-example--credit-derivatives)
- [Downstream Consumption](#downstream-consumption)
- [Provenance and Hard Caps](#provenance-and-hard-caps)
- [Cross-References](#cross-references)

---

## Core Principle

**Vocabulary is a prerequisite, not a deliverable.** Phase 0 asks the analyst to state hypotheses in the field's idiom. If the analyst doesn't speak the idiom, the hypotheses are shaped by the analyst's own vocabulary — and the framing error becomes unreachable by downstream falsification because the wrong words shape the wrong tests.

Failure mode prevented: an analyst asked to reverse-engineer a credit-derivatives pricing engine frames H1 as "the pricing function is a linear model of inputs." The native framing is "the engine implements a reduced-form intensity process with stochastic recovery." The linear model fits R² > 0.95 in-sample, passes Phase 5 validation on its own criteria, and fails catastrophically on the question the desk actually cares about (tail risk). The framing error could only have been caught at Phase 0 — Phase 0.3 is the cheapest place to catch it.

---

## Self-Assessment: `domain_familiarity`

At Phase 0, declare `domain_familiarity: high | medium | low | unknown` in `analysis_plan.md`. Use this checklist:

- [ ] Can I name the top-3 metrics the field uses (with units and plausibility ranges)?
- [ ] Can I name the field's canonical textbook, seminal paper, or regulator?
- [ ] Can I define the target system's native archetype in one sentence?
- [ ] Would a domain expert recognize my Phase 0 hypotheses as well-framed?

| Yes count | `domain_familiarity` | Phase 0.3 |
|-----------|----------------------|-----------|
| 3–4 | `high` | Skipped (log in `decisions.md`) |
| 2 | `medium` | Optional — analyst decides |
| 0–1 | `low` | MANDATORY |
| Cannot answer honestly | `unknown` | MANDATORY (default) |

**If the field is missing from `analysis_plan.md`**, the orchestrator treats it as `unknown`. Fail-safe toward more orientation.

**Warning — do not inflate.** It is tempting to claim `high` to skip the phase. The cost of a false `high` is a protocol-wide framing error; the cost of a false `low` is ~1 hour of orientation work. Bias toward lower.

---

## Operator TE — Term Extraction

**Purpose**: surface candidate technical terms from initial materials so they can be grounded in TG.

**Inputs**: system description, initial documents, interface specs, log samples, observations of the target in use — anything available at Phase 0.

**Method (stdlib only)**:
- Tokenize source text. Extract candidates matching:
  - Acronyms: `[A-Z]{2,}` (e.g., CDS, DV01, ISDA, MTBF).
  - Multi-word capitalized phrases (proper nouns).
  - Tokens combining letters and digits (`BaseL3`, `OD600`, `IPv6`).
  - Distinctive morphology for the likely domain (e.g., suffixed units).
- Deduplicate case-insensitively; preserve original casing from first occurrence.
- Rank by frequency.

**CLI**:
```bash
domain_orienter.py --file $($SM path domain_orientation.json) extract --input <path-or-text>
```

**Invariants**: each candidate has `{id, text, frequency, source_ref}`. Candidates without grounding are NOT citable downstream.

**"None found" is not a valid TE output.** If fewer than 10 candidates emerge from initial materials, the materials are insufficient — gather more inputs, or log an explicit attestation in `decisions.md` that the domain is trivially small.

---

## Operator TG — Term Grounding

**Purpose**: assign each candidate term a definition, source, and confidence. This is where provenance discipline bites hardest.

**Method**: for each term, the analyst picks ONE:

| Source | When | Confidence cap |
|--------|------|----------------|
| `library` | cited external reference with URL/DOI | 1.00 |
| `analyst` | analyst's own domain expertise | ≤ 0.80 |
| `llm_parametric` | derived from LLM's parametric knowledge | ≤ 0.60 |
| `chain_derived` | documented reasoning chain over other sources | ≤ 0.90 |
| — discard — | term not relevant to the analysis | — |

**CLI**:
```bash
domain_orienter.py --file <state.json> ground \
  --term "CDS spread" \
  --definition "Annual premium (bps) paid by the protection buyer on a credit default swap." \
  --source library \
  --url "https://isda.org/definitions/"
```

**Hard caps (enforced in code — `ValueError` on violation, exit code 2)**:
- `llm_parametric` terms: `confidence ≤ 0.60`. Cannot be referenced by a metric or canonical source.
- `analyst` terms: `confidence ≤ 0.80` unless the analyst also logs a short inference chain (mirrors Phase 1.5 `chain_derived` pattern).
- `library` terms: `confidence = 1.00` by construction.

**TG exit condition**: `grounded_terms ≥ 10` (STANDARD/COMPREHENSIVE/PSYCH) or `≥ 5` (LITE) AND `library_sourced_fraction ≥ 0.3`.

---

## Operator MM — Metrics Mapping

**Purpose**: identify the field's canonical metrics, their units, and their plausibility ranges. Populates session-local `domain_metrics.json` consumed by Phase 3 and Phase 5.

**Method**: for each metric, record `{name, symbol, units, higher_is_better, plausibility, source, confidence}`. The plausibility tuple matches `src/config/domains.json` exactly: `[suspicious, plausible_low, plausible_high, excellent]` with `null` permitted for unbounded sides.

**CLI**:
```bash
domain_orienter.py --file <state.json> add-metric \
  --name "sharpe_ratio" \
  --units "dimensionless" \
  --higher-is-better true \
  --plausibility "3.0,0.5,1.5,2.5" \
  --source library --url "..." \
  --domain finance
```

**Hard caps**:
- `llm_parametric` metrics CANNOT be promoted to `domain_metrics.json`. `candidates promote --id <MET-NNN>` raises `RuntimeError` if source is `llm_parametric`.
- Every promoted metric must have either `source_url` (for library/chain_derived) or `source_citation` (for offline references) — except `source=analyst` which stands on analyst expertise.

**MM exit condition**: `metrics_with_plausibility ≥ 3` (STANDARD/COMPREHENSIVE). LITE: this operator is skipped; attest in `decisions.md`.

---

## Operator AM — Alias Map

**Purpose**: capture synonyms, competing terminologies, regional variants. Prevents the analyst from mistranslating aliases during Phase 1 observations or Phase 0.7 steelman critiques.

**Examples**:
- "Basel III" (global) ↔ "CRR/CRD IV" (EU)
- "accuracy" (ML) ↔ "hit rate" (classical statistics)
- "CDS spread" ↔ "credit default swap spread" ↔ "CDS premium"

**CLI**:
```bash
domain_orienter.py --file <state.json> alias \
  --canonical "CDS spread" \
  --aliases "credit default swap spread,CDS premium" \
  --region global --source library
```

**AM exit condition**: alias map present OR explicit "no aliases identified" attestation in `decisions.md`. LITE: skipped.

---

## Operator CS — Canonical Sources

**Purpose**: identify the field's authoritative references. These become the `library` source for future provenance claims (Phase 1.5 chains, Phase 3 calibration, Phase 5 validation).

**Method**: for each source, record `{title, authors_or_publisher, year, url_or_doi, category, coverage_notes}` where category ∈ {textbook, regulator, standard, seminal_paper, benchmark_dataset}.

**CLI**:
```bash
# 1. Register a candidate source
domain_orienter.py --file <state.json> source \
  --title "ISDA 2014 Credit Derivatives Definitions" \
  --category standard \
  --url "https://..." --year 2014

# 2. Verify via WebFetch (sub-agent fetches, passes status to tool)
domain_orienter.py --file <state.json> verify --source-id SID-001 --http-status 200
```

**Hard caps**:
- `llm_parametric` sources MUST be verified before promotion. Verification is `--http-status 200` (from a real WebFetch) OR `--verified-by citation` (DOI/ISBN the analyst confirms exists, lower confidence).
- Unverified sources stay in `candidate_sources` and cannot be cited downstream.

**CS exit condition**: `verified_sources ≥ 2`.

---

## Phase 0.3 Exit Gate

All items satisfied before transitioning to Phase 0.5/0.7. Write operations via `$SM write` or `domain_orienter.py` CLI.

- [ ] `domain_orientation.json` persisted (lazy creation on first mutation).
- [ ] `extract` run ≥1 time; `candidate_terms` non-empty.
- [ ] `grounded_terms ≥ 10` (STANDARD/COMPREHENSIVE/PSYCH) or `≥ 5` (LITE). `library_sourced_fraction ≥ 0.3`.
- [ ] `metrics_with_plausibility ≥ 3` (STANDARD/COMPREHENSIVE). LITE: skipped.
- [ ] Alias map present OR attestation logged. LITE: skipped.
- [ ] `verified_sources ≥ 2`.
- [ ] `domain_glossary.md`, `domain_metrics.json`, `domain_sources.md` all rendered.
- [ ] `analysis_plan.md` updated: `domain_familiarity` re-evaluated post-orientation (expected: `low` → `medium` or `high`).
- [ ] `decisions.md`: log any trade-off (e.g., "used LLM-parametric definition for term X because no library source was findable inside the 1-hour time budget").
- [ ] Phase 0 hypotheses re-read with the new glossary. If any requires re-statement in native idiom, invoke `bayesian_tracker.py rename <HID> "new statement"` OR log the re-statement explicitly in `decisions.md`.
- [ ] `state.md`, `progress.md`, `phase_outputs/phase_0_3.md` updated.

**Gate check**: `domain_orienter.py gate` returns:
- exit code **0** — PASS; proceed to Phase 0.5 or 0.7.
- exit code **1** — FAIL; exit gate not met.
- exit code **2** — usage or runtime error.

**Skip path**: `session_manager.py skip 0.3 --reason "domain_familiarity=high: <justification>"` writes the attestation to `decisions.md` and flips state without running the operators. Available only when `domain_familiarity=high`.

---

## Worked Example — Credit Derivatives

**Scenario**: analyst asked to reverse-engineer a credit-derivatives pricing engine. General software reverse-engineering skills but no finance background. `domain_familiarity: low`. Tier: STANDARD.

```bash
SM="python3 <skill>/scripts/session_manager.py --base-dir <project>"
DO="python3 <skill>/scripts/domain_orienter.py --file $($SM path domain_orientation.json)"

# 1. Bootstrap state
$DO start --tier STANDARD --domain credit_derivatives

# 2. TE — Term Extraction from analysis_plan.md and any docs
$DO extract --input $($SM path analysis_plan.md)
# Output: TERM-001 "CDS", TERM-002 "ISDA", TERM-003 "CS01", TERM-004 "DV01",
#         TERM-005 "recovery rate", TERM-006 "term structure", ...

# 3. TG — Term Grounding (library preferred)
$DO ground --term "CDS spread" --source library \
  --url "https://isda.org/definitions/" \
  --definition "Annual premium (bps) paid by the protection buyer."

$DO ground --term "CS01" --source library --url "..." \
  --definition "Dollar change in CDS value per 1bp widening of the spread."

$DO ground --term "reduced-form model" --source analyst --confidence 0.7 \
  --definition "Intensity-based default model where default time is the first jump of a Cox process."

# ... continue until grounded_terms >= 10 AND library_fraction >= 0.3

# 4. MM — Metrics Mapping
$DO add-metric --name "cs01" --units "USD/bp" --higher-is-better false \
  --plausibility "null,100,10000,null" --domain credit_derivatives \
  --source library --url "..."

$DO add-metric --name "recovery_rate" --units "fraction" --higher-is-better true \
  --plausibility "0.99,0.2,0.6,0.8" --domain credit_derivatives \
  --source library --url "..."

# 5. CS — Canonical Sources
$DO source --title "ISDA 2014 Credit Derivatives Definitions" --category standard \
  --url "https://www.isda.org/..." --year 2014
$DO verify --source-id SID-001 --http-status 200

$DO source --title "Credit Risk Modeling" --category textbook \
  --authors "Lando" --year 2004
$DO verify --source-id SID-002 --verified-by citation

# 6. AM — Alias Map
$DO alias --canonical "CDS spread" \
  --aliases "credit default swap spread,CDS premium" \
  --source library

# 7. Render artifacts
$DO glossary render
$DO metrics render
$DO sources render

# 8. Gate check
$DO gate
# → Phase 0.3 Exit Gate: PASS
#   - grounded_terms: 14 (>= 10) ✓
#   - library_sourced_fraction: 0.64 (>= 0.30) ✓
#   - metrics_with_plausibility: 4 (>= 3) ✓
#   - verified_sources: 3 (>= 2) ✓
#   - alias_map_present: yes ✓

# 9. Re-frame Phase 0 hypotheses with the new glossary
$($SM path hypotheses.json) && python3 <skill>/scripts/bayesian_tracker.py \
  --file $($SM path hypotheses.json) \
  rename H1 "The pricing function implements a reduced-form intensity process with piecewise-constant hazard and stochastic recovery."

# 10. Update state, progress, write phase output
$SM write phase_outputs/phase_0_3.md <<'EOF'
# Phase 0.3 — Domain Orientation (complete)
... summary ...
EOF
```

### Downstream impact

Phase 0 hypotheses get re-read with the new glossary. Example diff:

| Before P0.3 | After P0.3 |
|-------------|------------|
| `H1: "the pricing function is a linear model of inputs"` | `H1: "the pricing function implements a reduced-form intensity process with piecewise-constant hazard and stochastic recovery"` |
| `H2: "the engine uses a lookup table"` | `H2: "the engine performs copula-based joint default simulation over a reference portfolio"` |

Phase 0.7 M2 archetype classification proposes `credit_pricing_engine` (domain-native) instead of `generic_function_approximator`. Phase 1.5 TI merges domain-specific trace entries into `trace_catalog.json`. Phase 3 parameter CIs are sanity-checked against `domain_metrics.json`. Phase 5 validation uses field-native acceptance (e.g., "calibration to market CDS spreads within 2 bps") instead of generic R² > 0.95.

### Counterfactual (without Phase 0.3)

Analyst fits a linear model, achieves R² > 0.95 in-sample, declares success in Phase 5. The desk rejects the model because it fails on the tail-risk scenarios a linear structure cannot represent. The framing error originated at Phase 0; Phase 0.3 was the only phase where it could have been caught cheaply.

---

## Downstream Consumption

Artifacts created in Phase 0.3 are consumed by later phases:

| Artifact | Consumer | Behavior |
|----------|----------|----------|
| `domain_glossary.md` | Phase 0.7 M2 Archetype Accomplices, M4 Steelman | Orchestrator passes glossary context to the sub-agents. `scope_auditor.py enumerate --glossary <path>` biases archetype ranking toward domain-native archetypes. |
| `domain_glossary.md` | Phase 1.5 TI Trace Inversion | `abductive_engine.py invert` merges session-local trace entries derived from glossary terms (entries carry `source: "analyst"` to bypass the LLM-parametric prior cap). |
| `domain_metrics.json` | Phase 3 parametric identification | Plausibility bounds for parameter estimates consult session metrics first, then `src/config/domains.json`. Out-of-bounds CIs emit warnings to `observations/`. *(v7.15.1 — deferred from v7.15.0.)* |
| `domain_metrics.json` | Phase 5 validation | Validation thresholds (R², FVA, coverage) consult session metrics for field-native acceptance criteria. Divergence from defaults logged in `validation.md`. *(v7.15.1 — deferred from v7.15.0.)* |
| `domain_sources.md` | Phase 1.5 chains, evidence calibration | Library sources become `source: "library"` in `bayesian_tracker.py` updates (LR ≤ 5.0) vs `analyst` (LR ≤ 2.5) vs `llm_parametric` (LR ≤ 2.0). |

---

## Provenance and Hard Caps

Phase 0.3 reuses the provenance taxonomy from Phase 1.5 exactly. Every term, metric, and source carries `source ∈ {library, analyst, llm_parametric, chain_derived}`.

**Caps summary** (enforced in code, raise `ValueError` or `RuntimeError` → caught in `main()` → exit code 2):

| Entity | `library` | `analyst` | `llm_parametric` | `chain_derived` |
|--------|-----------|-----------|------------------|-----------------|
| Term confidence | 1.00 | ≤ 0.80 | ≤ 0.60 | ≤ 0.90 |
| Metric promotion | Allowed | Allowed w/ countersign | **Blocked** | Allowed |
| Canonical source | Allowed | N/A | **Blocked unless verified** | N/A |
| Cite as P1.5 evidence | LR ≤ 5.0 | LR ≤ 2.5 | **LR ≤ 2.0** (inherits P1.5 rule) | LR ≤ 3.0 |

**Evidence Rule 8 extension**: the existing SKILL.md Evidence Rule 8 governing LLM-parametric hypotheses extends to Phase 0.3 terminology. LLM-parametric definitions, metrics, and sources are all hard-capped until grounded against a verified `library` reference.

**Why this matters**: an ungrounded LLM-parametric definition, once accepted, shapes every downstream hypothesis expressed in that term. The cost of a wrong definition compounds through all subsequent phases. The caps are the only defense.

---

## Cross-References

- `SKILL.md` Phase 0.3 section (protocol details and gate checklist).
- `SKILL.md` Evidence Rule 8 (LLM-parametric caps — extended to Phase 0.3 terminology).
- `references/scope-interrogation.md` (Phase 0.7 — downstream consumer of glossary via M2, M4).
- `references/abductive-reasoning.md` (Phase 1.5 — provenance taxonomy and caps that Phase 0.3 mirrors).
- `references/evidence-calibration.md` (LR caps by source, which Phase 0.3 sources participate in).
- `references/cognitive-traps.md` Trap 20 (Framing — the primary trap Phase 0.3 defends against).
- `references/domain-calibration.md` (static domain calibration data; Phase 0.3 populates session-local extensions without overwriting).
