# Phase Protocols

Per-phase procedural recipes (Activities + EXIT GATE checklists), the File Write Matrix, and the Gate Check Procedure. Relocated from `src/SKILL.md` at v7.15.7 (plan `plan_2026-05-19_618156aa`) so SKILL.md keeps only identity + invariants. The FSM diagram, Tier Selection, Evidence Rules, Refusal Protocol, Intake & Reframe, and Protocol Inviolability remain in SKILL.md. Per-phase agents and the orchestrator's Gate Check Procedure read THIS file for procedural detail.

## Table of Contents
- [File Write Matrix](#file-write-matrix)
- [Gate Check Procedure](#gate-check-procedure)
- [Phase 0: Setup & Frame](#phase-0-setup--frame)
- [Phase 0.3: Domain Orientation](#phase-0-3-domain-orientation)
- [Phase 0.5: Coherence Screening](#phase-0-5-coherence-screening)
- [Phase 0.7: Scope Interrogation](#phase-0-7-scope-interrogation)
- [Phase 1: Boundary Mapping](#phase-1-boundary-mapping)
- [Phase 1.5: Abductive Expansion](#phase-1-5-abductive-expansion)
- [Phase 2: Causal Analysis](#phase-2-causal-analysis)
- [Phase 3: Parametric Identification](#phase-3-parametric-identification)
- [Phase 4: Model Synthesis](#phase-4-model-synthesis)
- [Phase 5: Validation & Report](#phase-5-validation--report)
- [PSYCH Tier](#psych-tier-psychological-profiling)

---

## File Write Matrix

R = must read before starting. W = must write before leaving. W? = write if applicable. — = don't touch. **Use `$SM read`/`$SM write` for all operations.**

| File | P0 | P0.3 | P0.5 | P0.7 | P1 | P1.5 | P2 | P3 | P4 | P5 |
|------|-----|------|------|------|-----|------|-----|-----|-----|-----|
| `state.md` | W | R+W | W | R+W | R+W | R+W | R+W | R+W | R+W | R+W |
| `analysis_plan.md` | W | R+W | — | R+W | R | R | R | — | — | R |
| `hypotheses.json` | W | R+W | — | R+W | R+W | R+W | R+W | R+W | R+W | R+W |
| `rapid_assessment.json` | — | — | W | — | — | — | — | — | — | R |
| `domain_orientation.json` | — | R+W | — | R | R | R | R | R | R | R |
| `domain_glossary.md` | — | W | — | R | R | R | R | R | R | R |
| `domain_metrics.json` | — | W | — | R | R | R | R | R | R | R |
| `domain_sources.md` | — | W | — | R | R | R | R | R | R | R |
| `scope_audit.md` | — | — | — | W | R | R | R | R | R | R |
| `scope_audit.json` | — | — | — | R+W | R | R | R | R+W | R | R+W |
| `observations.md` | — | — | — | — | W | R | W | W | W? | R |
| `observations/` | — | — | — | — | W | R | W | W? | W? | R |
| `abductive_state.json` | — | — | — | — | — | R+W | R | — | — | R |
| `hypothesis_candidates.json` | — | — | — | — | — | R+W | R | — | — | R |
| `predictions_pending.json` | — | — | — | — | — | R+W | R | — | — | R |
| `inference_chains.json` | — | — | — | — | — | R+W | R | — | — | R |
| `surplus_audit.json` | — | — | — | — | — | R+W | R | — | — | R |
| `decisions.md` | W | W? | W? | W? | W? | W? | W? | W? | W? | W? |
| `progress.md` | W | W | W | W | W | W | W | W | W | W |
| `phase_outputs/` | W | W | W | W | W | W | W | W | W | W |
| `validation.md` | — | — | — | — | — | — | — | — | — | W |
| `summary.md` | — | — | — | — | — | — | — | — | — | W |

Phase 0.3 column applies only when triggered (`domain_familiarity ∈ {low, unknown}` in `analysis_plan.md`, or COMPREHENSIVE tier mandatory). When skipped via `$SM skip 0.3 "<reason>"`, the four `domain_*` files are absent and downstream phases proceed without them.

**`state.md` `## Phase:` field is updated by `$SM advance` only.** Free-writing it via `$SM write state.md` is refused (see SKILL.md Refusal Protocol).

---

## Gate Check Procedure

BEFORE moving from Phase N to Phase N+1, execute ALL steps using `$SM write`/`$SM read`:

1. `$SM write phase_outputs/phase_N.md <<'EOF' ... EOF` — phase deliverables
2. `$SM write state.md <<'EOF' ... EOF` — phase number, hypothesis count, lead hypothesis + posterior, confidence
3. `$SM write progress.md <<'EOF' ... EOF` — mark Phase N completed, set Phase N+1 in progress, list remaining
4. `$SM write decisions.md <<'EOF' ... EOF` — log analytical decisions (format: "X at the cost of Y")
5. Run `bayesian_tracker.py --file $($SM path hypotheses.json) report` and verify posteriors are current
6. **Multi-pass eval**: Check triggers in `references/multi-pass-protocol.md` — if any fire, `$SM reopen` instead of advancing
7. End response with state block matching `state.md`

**CRITICAL**: No monolithic reports outside Phase 5. Build evidence phase by phase.

---

## Phase 0: Setup & Frame

**GATE IN**: Session created via `$SM new`. Verify `$SM read state.md` works.

**Activities:**
1. Define position (insider/outsider), access, constraints, system type
2. Build Question Pyramid (L1-L5: DO → HOW → WHY → PARAMETERS → REPLICATE)
3. Seed ≥3 hypotheses via `bayesian_tracker.py --file $($SM path hypotheses.json) add` (H1: likely, H2: alternative, H3: adversarial/deceptive)
4. **Seed the H_S standing pair** (STANDARD/COMPREHENSIVE/PSYCH): add `"[H_S] Drivers of <target> live within initial scope <S>"` and `"[H_S_prime] Material drivers exist outside <S>"` via `bayesian_tracker.py add`. `[H_S_prime]` satisfies Evidence Rule 3's adversarial requirement.
5. Adversarial pre-check (high entropy? anti-debug? information asymmetry?)
6. Acknowledge cognitive vulnerabilities (see `references/cognitive-traps.md` — pay attention to Trap 20 Framing, Trap 23 Premature Closure)

**Fidelity Levels:**
| Level | Question | Goal | Test |
|-------|----------|------|------|
| L1 | DO | Trigger a response? | Any output from input |
| L2 | HOW | What transforms I→O? | Explain processing steps |
| L3 | WHY | What drives mechanism? | Predict design choices |
| L4 | PARAMETERS | What values control it? | <5% error on measurables |
| L5 | REPLICATE | Can I rebuild it? | Replica indistinguishable |

**EXIT GATE — write each via `$SM write <filename>`:**
- [ ] `analysis_plan.md`: ALL fields filled (system, access, adversary, tier, fidelity, pyramid, hypotheses, pre-check, cognitive traps, scope S definition). No placeholder text. LITE: may omit adversarial pre-check if no adversary indicated.
- [ ] `hypotheses.json`: ≥3 hypotheses via CLI, including ≥1 adversarial. STANDARD/COMPREHENSIVE/PSYCH MUST also contain `[H_S]` and `[H_S_prime]` statements (grep-verifiable via `bayesian_tracker.py report | grep "\[H_S"`)
- [ ] `decisions.md`: tier selection logged with trade-off rationale
- [ ] `state.md`: updated (phase=0 complete, tier, fidelity, hypothesis count, lead H)
- [ ] `progress.md`: Phase 0 complete, remaining phases listed
- [ ] `phase_outputs/phase_0.md`: setup deliverables written

**Reference**: `references/setup-techniques.md`, `references/cognitive-traps.md`, `references/scope-interrogation.md` (H_S standing pair), `references/modeling-epistemology.md` (foundational reasoning principles)

---

## Phase 0.3: Domain Orientation

**(CONDITIONAL — STANDARD / COMPREHENSIVE / PSYCH; LITE if flagged)**

**Runs after Phase 0, before Phase 0.5/0.7.** Conditional on `domain_familiarity ∈ {low, unknown}` declared in `analysis_plan.md`. **MANDATORY in COMPREHENSIVE.** Skipped in RAPID. Skipped via `$SM skip 0.3 "<justification>"` when `domain_familiarity=high`. Purpose: build an auditable glossary, metrics catalog, and canonical-source ledger before Phase 0 hypotheses harden into the analyst's own (potentially wrong) idiom.

**GATE IN**: `$SM read state.md`, `$SM read analysis_plan.md`. Confirm `domain_familiarity` field present and triggers Phase 0.3. If `high`, run `$SM skip 0.3 "<reason>"` and proceed to Phase 0.5/0.7.

**Activities** (five operators — full set for STANDARD/COMPREHENSIVE/PSYCH; LITE runs TE + TG + CS only):

1. Start the orientation session: `python3 <skill-dir>/scripts/domain_orienter.py --file $($SM path domain_orientation.json) start --tier <tier> --domain <declared>`
   **Flag-order rule**: `--file` is a parent-parser option and MUST come before the subcommand (between `domain_orienter.py` and `start|extract|ground|...`). Placing `--file` after the subcommand silently defaults to `./domain_orientation.json` (cwd-relative) and your session writes will land in the wrong file.
2. **TE Term Extraction**: tokenize initial materials and surface candidate technical terms.
   `domain_orienter.py --file $($SM path domain_orientation.json) extract --input $($SM path analysis_plan.md)` (rerun for additional input paths)
3. **TG Term Grounding**: assign each candidate `{definition, source, confidence}` with provenance discipline. Library sources preferred; LLM-parametric capped at confidence 0.60.
   `domain_orienter.py --file $($SM path domain_orientation.json) ground --term "<text>" --definition "<def>" --source <library|analyst|llm_parametric> [--url <url>]`
4. **MM Metrics Mapping** (skip in LITE): identify the field's canonical metrics and their plausibility ranges.
   `domain_orienter.py --file $($SM path domain_orientation.json) add-metric --name <n> --units <u> --higher-is-better <bool> --plausibility <sus,lo,hi,exc> --source <...> --domain <d>`
5. **AM Alias Map** (skip in LITE): capture synonyms, regional variants, competing schools.
   `domain_orienter.py --file $($SM path domain_orientation.json) alias --canonical "<term>" --aliases "a1,a2,..." --source <...>`
6. **CS Canonical Sources**: identify and verify the field's authoritative references via WebFetch.
   `domain_orienter.py --file $($SM path domain_orientation.json) source --title "..." --category <textbook|regulator|standard|seminal_paper|benchmark_dataset> --url "..."`
   `domain_orienter.py --file $($SM path domain_orientation.json) verify --source-id SID-N --http-status 200`
7. Render artifacts:
   - `domain_orienter.py --file $($SM path domain_orientation.json) glossary render --output $($SM path domain_glossary.md)`
   - `domain_orienter.py --file $($SM path domain_orientation.json) metrics render --output $($SM path domain_metrics.json)`
   - `domain_orienter.py --file $($SM path domain_orientation.json) sources render --output $($SM path domain_sources.md)`
8. Run gate: `domain_orienter.py --file $($SM path domain_orientation.json) gate` — exit code 0 PASS, 1 FAIL, 2 ERROR. PASS requires `grounded_terms>=10` (STANDARD/COMPREHENSIVE/PSYCH; LITE>=5), `library_sourced_fraction>=0.30`, `verified_sources>=2`; MM/AM gates apply outside LITE.
9. Re-read Phase 0 hypotheses with the new glossary. For each that reads better in native idiom, rename via `bayesian_tracker.py rename <HID> "<native statement>"` (PSYCH: `belief_tracker.py rename <TID> "<native trait>"`). Renaming preserves prior, posterior, and evidence trail.

**Tier scaling**:
- **LITE**: TE + TG + CS only (skip MM, AM); grounded-term floor relaxed to 5.
- **STANDARD**: all five operators; grounded-term floor 10.
- **COMPREHENSIVE**: all five operators; multi-pass permitted via `$SM reopen 0.3`.
- **PSYCH**: all five operators with "domain" framed as the subject's cultural/situational vocabulary.

**EXIT GATE — write each via `$SM write <filename>` or tool CLI:**
- [ ] `domain_orientation.json` persisted (lazy creation on first mutation; runtime file — no repo template)
- [ ] `extract` run ≥1 time; `candidate_terms` non-empty
- [ ] `grounded_terms ≥ 10` (STANDARD/COMPREHENSIVE/PSYCH) or `≥ 5` (LITE); `library_sourced_fraction ≥ 0.30`
- [ ] `metrics_with_plausibility ≥ 3` (STANDARD/COMPREHENSIVE; LITE skips)
- [ ] Alias map present OR `decisions.md` attestation "no aliases identified" (LITE skips)
- [ ] `verified_sources ≥ 2`
- [ ] `domain_glossary.md`, `domain_metrics.json`, `domain_sources.md` rendered
- [ ] `analysis_plan.md` updated: `domain_familiarity` re-evaluated post-orientation
- [ ] `decisions.md` logs trade-offs (e.g., "used LLM-parametric for term X because no library source within time budget")
- [ ] Phase 0 hypotheses re-read; renames applied via `bayesian_tracker.py rename` OR explicit "no rename warranted" attestation in `decisions.md`
- [ ] `domain_orienter.py gate` returns PASS (exit 0)
- [ ] `state.md` updated | `progress.md` updated | `phase_outputs/phase_0_3.md` written

**"None found" is NOT a valid Phase 0.3 output.** If extraction returns fewer than 10 candidate terms, the source materials are insufficient — gather more or attest in `decisions.md` that the domain is trivially small. Evidence Rule 8 hard caps apply throughout (LLM-parametric definitions/metrics/sources blocked from promotion until grounded).

**Reference**: `references/domain-orientation.md` (full protocol, self-assessment checklist, worked example), `references/cognitive-traps.md` (Trap 20 Framing — the primary trap this phase defends against)

---

## Phase 0.5: Coherence Screening

**(RAPID Entry)**

**GATE IN**: `$SM read state.md`, review claim or system description

**Activities:**
1. Claim-task alignment (data matches task? metrics appropriate?)
2. Instant reject conditions (impossibility, contamination, incoherence)
3. Red flag scan (missing baseline? tool worship? documentation gaps?)
4. Domain calibration check

| Verdict | Criteria | Action |
|---------|----------|--------|
| **CREDIBLE** | 0 rejects, 0-1 flags, coherent | DONE (or proceed to full analysis) |
| **SKEPTICAL** | 2+ flags, minor concerns | Request info or escalate to STANDARD |
| **DOUBTFUL** | 4+ flags or 3+ categories | Escalate to STANDARD with caution |
| **REJECT** | Reject condition OR critical flags | Analysis stops; log rationale |

**EXIT GATE — write each via `$SM write <filename>`:**
- [ ] `state.md`: updated with verdict
- [ ] `progress.md`: Phase 0.5 checked off
- [ ] `phase_outputs/phase_0_5.md`: coherence report written
- [ ] If REJECT: analysis stops. If escalating tier: log in `decisions.md`

**Reference**: `references/rapid-assessment.md`, `references/coherence-checks.md`, `references/red-flags.md`, `references/domain-calibration.md`

---

## Phase 0.7: Scope Interrogation

**(STANDARD / COMPREHENSIVE / PSYCH)**

**Runs after Phase 0, before Phase 1.** Purpose: promote the system boundary from a premise to a hypothesis. Surface drivers that live outside the initially-framed scope S so they enter the analysis as hypotheses in the FIRST pass, not after re-runs. **Skipped in RAPID and LITE tiers.**

**GATE IN**: `$SM read state.md`, `$SM read analysis_plan.md`, `$SM read hypotheses.json`. Confirm `[H_S]` and `[H_S_prime]` are already seeded from Phase 0.

**Activities:**
1. Start a scope audit session: `python3 <skill-dir>/scripts/scope_auditor.py --file $($SM path scope_audit.json) start "<target>"`
   **Flag-order rule**: `--file` is a parent-parser option and MUST come before the subcommand (between `scope_auditor.py` and `trace|enumerate|...`). Placing `--file` after the subcommand makes argparse exit 2 ("unrecognized arguments") on most subcommands, or silently default to `./scope_audit.json` on others. Always: `scope_auditor.py --file <path> <subcommand> [args]`.
   **Tip**: run `scope_auditor.py --file $($SM path scope_audit.json) list-archetypes` once at the top of Phase 0.7 to learn the valid archetype IDs accepted by step 3 (`enumerate --archetype <id>`).
2. **M1 Flow Tracing**: enumerate input and output channels from `analysis_plan.md`. For each, name the immediate upstream generator (inputs) or downstream consumer (outputs). Any neighbor outside scope S → exogeneity candidate.
   `scope_auditor.py --file $($SM path scope_audit.json) trace --inputs "c1,c2" --outputs "c3,c4"`
3. **M2 Archetype Accomplices**: classify the target into 1-3 archetypes from `references/archetype-accomplices.md`. First confirm valid IDs via `scope_auditor.py --file $($SM path scope_audit.json) list-archetypes`. For each:
   `scope_auditor.py --file $($SM path scope_audit.json) enumerate --archetype <id> [--glossary $($SM path domain_glossary.md)]`
   The optional `--glossary` flag (added v7.15.0) consumes the Phase 0.3 glossary and prints an advisory that aligns accomplice vocabulary with the grounded terms from domain orientation — skip this flag if Phase 0.3 was skipped.
4. **M3 Residual-Signature Matching** (deferred if no baseline model exists yet): if a preliminary model is available, compare residuals against external indices:
   `scope_auditor.py --file $($SM path scope_audit.json) residual-match --residuals residuals.csv --indices-dir ./indices/`
5. **M4 Adversarial Scoping (Steelman)**: produce three critiques from distinct personas — domain outsider, investigative journalist, regulator. Each must name one excluded domain AND one mechanism. Log each:
   `scope_auditor.py --file $($SM path scope_audit.json) steelman --persona outsider|journalist|regulator --domain "..." --mechanism "..."`
6. Dedupe candidates: `scope_auditor.py --file $($SM path scope_audit.json) dedupe`
7. Check the Phase 0.7 gate: `scope_auditor.py --file $($SM path scope_audit.json) gate` — PASS requires `candidates_unique>=3` AND `has_archetype_query=True` (M2 was run at least once). `has_traces`/`has_steelman` are RECOMMENDED but not gated. Exit 0 PASS, 1 FAIL.
8. For each final candidate, seed an exogeneity hypothesis in `hypotheses.json` via `bayesian_tracker.py add` with the suggested prior. Use a distinctive statement prefix like `[H_SCOPE_<domain>]` for traceability.
9. Write `scope_audit.md` via `$SM write scope_audit.md` — human-readable summary of M1-M4 outputs and the final candidate list. Use `scope_auditor.py report --verbose` as the body.

**EXIT GATE — write each via `$SM write <filename>`:**
- [ ] `scope_audit.md`: written with all four mechanism outputs (M1, M2, M3 or "deferred", M4)
- [ ] `scope_audit.json`: persisted, `scope_auditor.py --file <path> gate` returns PASS (`candidates_unique>=3` AND `has_archetype_query`)
- [ ] `hypotheses.json`: exogeneity candidates seeded as additional hypotheses with priors ≥ 0.05
- [ ] `analysis_plan.md`: updated scope S (if expanded) and cross-references to new hypotheses
- [ ] `decisions.md`: log scope-expansion decisions with trade-off (what was added, at the cost of what depth elsewhere)
- [ ] `state.md`: updated (Phase 0.7 complete)
- [ ] `progress.md`: updated
- [ ] `phase_outputs/phase_0_7.md`: summary written

**CRITICAL**: "None found" is NOT a valid Phase 0.7 output. If you cannot produce 3 unique candidates, the archetype classification (M2) or steelman procedure (M4) was too shallow — rerun with a different archetype or different persona lens. The PSYCH tier uses the same protocol but frames S as "which life domains inform the subject's behavior."

**Reference**: `references/scope-interrogation.md` (full protocol + worked examples), `references/archetype-accomplices.md` (library), `references/cognitive-traps.md` Traps 20-23 (Framing, Streetlight, OVB, Premature Closure)

---

## Phase 1: Boundary Mapping

**GATE IN**: `$SM read state.md`, `$SM read analysis_plan.md`, `$SM read hypotheses.json`

**Activities:**
1. Enumerate I/O channels (explicit, implicit, side-channel, feedback)
2. Apply probe signals (step, impulse, PRBS, edge cases)
3. Assess data quality (coherence γ² ≈ 1.0 = good)
4. Build stimulus-response database
5. Write each finding via `$SM write observations/obs_NNN_topic.md`
6. Update `observations.md` via `$SM write observations.md` after every 2 findings
7. Update hypotheses via `bayesian_tracker.py update --file $($SM path hypotheses.json)` — one update per data point

**EXIT GATE — write each via `$SM write <filename>`:**
- [ ] `observations/`: ≥3 observation files (LITE: ≥1)
- [ ] `observations.md`: index updated with all files
- [ ] `hypotheses.json`: evidence applied (≥1 update per active hypothesis)
- [ ] ≥80% I/O channels characterized; stimulus-response database ≥20 entries (LITE: ≥5)
- [ ] `state.md` updated | `progress.md` updated | `phase_outputs/phase_1.md` written

**Reference**: `references/boundary-probing.md`, `references/spectral-analysis.md` (frequency-domain profiling via `fourier_analyst.py`)

---

## Phase 1.5: Abductive Expansion

**Runs after Phase 1, before Phase 2.** MANDATORY for LITE / STANDARD / COMPREHENSIVE / PSYCH. SKIPPED for RAPID. Purpose: formalize backward inference from observations to candidate causes, so the interior hypothesis set is generated by an auditable tool rather than analyst intuition.

**GATE IN**: `$SM read state.md`, `$SM read observations.md`, `$SM read hypotheses.json`.

**Activities:**
1. Start an abductive session:
   `python3 <skill-dir>/scripts/abductive_engine.py --file $($SM path abductive_state.json) start`
   **Flag-order rule**: `--file` is a parent-parser option and MUST come before the subcommand (between `abductive_engine.py` and `invert|absence-audit|...`). Placing `--file` after the subcommand silently defaults to `./abductive_state.json` (cwd-relative); your mutations land in the wrong file.
2. **TI Trace Inversion**: for each observation in Phase 1, run `abductive_engine.py --file $($SM path abductive_state.json) invert` with the observation id, text, and category. The CLI consults `src/config/trace_catalog.json` keyed on category (`timing`, `resource`, `output_anomaly`, `failure`, `behavioral_deviation`, `generic`) to produce library-sourced candidates. The analyst may supply additional LLM-parametric candidates — these are hard-capped at prior 0.30 by the engine.
3. **AA Absence Audit**: for each active hypothesis, enumerate "what should be observed if true" predictions:
   `abductive_engine.py --file $($SM path abductive_state.json) absence-audit --hypothesis H1 --predictions "A;B;C"`
   As predictions resolve, close each with `abductive_engine.py --file $($SM path abductive_state.json) close-prediction --id PPN --outcome observed|absent`.
4. **SA Surplus Audit**: after TI on all observations, diff the observation record against the union of candidate coverages:
   `abductive_engine.py --file $($SM path abductive_state.json) surplus-audit`
   Every unexplained observation is a surplus candidate. Either iterate TI on it or log an explicit "no unexplained observations" attestation in `decisions.md`.
5. **AR Analogical Retrieval**: match a short case signature describing the symptom pattern against archetype `trace_signatures`:
   `abductive_engine.py --file $($SM path abductive_state.json) analogize --signature "<one-line symptom description>"`
   High-similarity matches bring the archetype's accomplice library back as interior hypothesis targets.
6. **IC Inference Chains**: for every candidate you recommend promoting, log a structured chain:
   `abductive_engine.py --file $($SM path abductive_state.json) chain start --target CANDn --premise "..."`
   `abductive_engine.py --file $($SM path abductive_state.json) chain step --id ICk --claim "..." --lr 1.5 --source analyst`
   `abductive_engine.py --file $($SM path abductive_state.json) chain close --id ICk --seed-prior 0.3`
   Each chain must have ≥2 steps and pass `chain audit --id ICk` (no gaps).
7. **Coverage-weighted promotion** (the primary mitigation against hypothesis explosion): `abductive_engine.py --file $($SM path abductive_state.json) candidates list` shows staged candidates sorted by `coverage_score = (observations_explained / total_observations) / complexity`. Candidates with `coverage_score < 0.30` (default threshold) are rejected at promotion. For each promotable candidate:
   `abductive_engine.py --file $($SM path abductive_state.json) candidates promote --id CANDn --tracker-path $($SM path hypotheses.json)`
   (or delegate promotion to `hypothesis-engine` in the agent workflow).
8. Report: `abductive_engine.py --file $($SM path abductive_state.json) report --verbose` → write the human-readable summary to `phase_outputs/phase_1_5.md`.

**Tier scaling:**
- **LITE**: SA + AA only (surplus audit + absence audit per hypothesis). Skip TI, AR, IC.
- **STANDARD**: all five operators.
- **COMPREHENSIVE**: STANDARD + iterate TI on surplus observations in a second pass; AR with multiple signatures; cross-candidate chains.

**EXIT GATE — write each via `$SM write <filename>`:**
- [ ] `abductive_state.json` persisted (runtime file — no repo template)
- [ ] `abductive_engine.py --file <path> invert` run on ≥3 observations (LITE may skip this)
- [ ] `abductive_engine.py --file <path> surplus-audit` produced a non-empty diff OR `decisions.md` logs explicit "no unexplained observations" attestation
- [ ] ≥1 new hypothesis promoted via the staging flow OR `decisions.md` logs explicit "no promotion warranted" attestation
- [ ] ≥1 inference chain logged per promoted hypothesis, each with ≥2 chain steps
- [ ] `hypothesis_candidates.json`, `predictions_pending.json`, `inference_chains.json`, `surplus_audit.json` exist in the session directory (all created by the engine when first mutated)
- [ ] `state.md` updated | `progress.md` updated | `phase_outputs/phase_1_5.md` written
- [ ] `hypotheses.json` now contains any promoted candidates with statement prefix `[H_ABDUCT_CANDn]`

**Evidence discipline reminder**: Evidence Rule 8 applies — LLM-parametric candidates are hard-capped at prior 0.30 and LR 2.0. Coverage-weighted promotion is enforced in code (`abductive_engine.py promote` raises RuntimeError below threshold).

**Reference**: `references/abductive-reasoning.md` (full protocol with TI/AA/SA/AR/IC procedures, coverage-weighted selection, provenance discipline, three worked examples), `references/cognitive-traps.md` (narrative fallacy specifically — `cognitive-auditor` runs a targeted check on abductive outputs)

---

## Phase 2: Causal Analysis

**GATE IN**: `$SM read state.md`, `$SM read observations.md`, `$SM read hypotheses.json`, `$SM read decisions.md`

**Activities:**
1. Static analysis (if visible): disassembly, decompilation, data flow
2. Dynamic analysis: tracer injection, differential analysis
3. Sensitivity analysis (Morris screening, Sobol' indices)
4. Construct causal graph (nodes, edges, feedback loops R/B)
5. Falsification loop: for each H, design test to break it, run, update tracker
6. Write new findings via `$SM write observations/...`; log causal model decisions via `$SM write decisions.md`

**EXIT GATE — write each via `$SM write <filename>`:**
- [ ] ≥70% behaviors have causal explanation
- [ ] ≥1 hypothesis refuted or significantly weakened
- [ ] `observations/` and `observations.md` updated with causal findings
- [ ] `hypotheses.json`: falsification evidence applied
- [ ] `decisions.md`: causal model choices logged
- [ ] `state.md` updated | `progress.md` updated | `phase_outputs/phase_2.md` written

**Reference**: `references/causal-techniques.md`, `references/tools-sensitivity.md`

---

## Phase 3: Parametric Identification

**GATE IN**: `$SM read state.md`, `$SM read phase_outputs/phase_2.md`, `$SM read hypotheses.json`

**Activities:**
1. **Identifiability gate**: `scripts/parametric_identifier.py assess` — data length, SNR, coherence. Returns GO / MARGINAL / NO-GO.
2. **Select model structure (ARX → ARMAX → NARMAX → State-Space)**: `scripts/parametric_identifier.py compare --families arx,armax,narmax` ranks candidates by BIC with whiteness gate. State-space still requires fourier_analyst FRF or manual fit.
3. **Estimate parameters (OLS, structure selection)**: `scripts/parametric_identifier.py fit --family arx --grid --criterion bic` (or specify --na/--nb/--nk). ARMAX uses SARIMAX backend; NARMAX uses polynomial basis + FROLS term selection.
4. **Quantify uncertainty (bootstrap)**: `parametric_identifier.py fit --bootstrap 500` — residual bootstrap for parameter CIs (temporally safe). Analytic CIs from `cov_params` as cheap fallback. Bayesian priors out of scope in current tooling.
5. **Residual diagnostics**: Ljung-Box whiteness is auto-run and reported in every `parametric_identifier.py` fit. Further diagnostics via `scripts/ts_reviewer.py` on residuals.
6. **Spectral complement**: `scripts/fourier_analyst.py` for transfer function estimation and frequency-domain system ID.
7. **Forecasting (if deliverable is prediction, not structure)**: `scripts/forecast_modeler.py fit` to fit ARIMA/ETS/CatBoost; use `assess` for forecastability gate.
8. **Pipe to Phase 4**: Fitted ARX output converts to simulator format via `FitResult.to_simulator_format()` → drops into `scripts/simulator.py` ARX/MC modes.
9. Update hypotheses with model-derived evidence

**EXIT GATE — write each via `$SM write <filename>`:**
- [ ] Model selected via information criterion; parameters documented with uncertainty bounds
- [ ] Residuals pass whiteness test (if applicable)
- [ ] Cross-validation R² > 0.8 (walk-forward); FVA > 0% for time-series
- [ ] `hypotheses.json` updated | `decisions.md` updated (model choice + trade-off)
- [ ] `state.md` updated | `progress.md` updated | `phase_outputs/phase_3.md` written

**Reference**: `references/system-identification.md`, `references/timeseries-review.md`, `references/forecasting-science.md`, `references/forecasting-tools.md`, `references/spectral-analysis.md`, `references/financial-validation.md`, `references/modeling-epistemology.md` (tradeoff navigation, assumption audit)

---

## Phase 4: Model Synthesis

**GATE IN**: `$SM read state.md`, `$SM read phase_outputs/phase_3.md`, `$SM read phase_outputs/phase_1.md`

**Activities:**
1. Compose sub-models (serial H₁·H₂, parallel H₁+H₂, feedback G/(1+GH))
2. Propagate uncertainty through composition
3. Test for emergence: `mismatch = |predicted - actual| / |actual|`; if > 20%, emergence present
4. Classify archetype (see `references/simulation-guide.md` archetype table)
5. Run simulation if applicable: `scripts/simulator.py`

**EXIT GATE — write each via `$SM write <filename>`:**
- [ ] All sub-models composed with explicit semantics; uncertainty propagated
- [ ] Emergence test performed and documented
- [ ] Archetype identified with vulnerability assessment
- [ ] `hypotheses.json` updated | `observations.md` updated if simulation produced findings
- [ ] `state.md` updated | `progress.md` updated | `phase_outputs/phase_4.md` written

**Reference**: `references/compositional-synthesis.md`, `references/simulation-guide.md`, `references/distributions-guide.md`

---

## Phase 5: Validation & Report

**GATE IN**: `$SM read state.md`, `$SM read` all `phase_outputs/`, `$SM read observations.md`, `$SM read hypotheses.json`

**Tier-specific scope:**
- **RAPID**: Domain calibration + verdict documentation + summary only (activities 4, 8). Skip residuals/FVA/simulation.
- **LITE**: Validation hierarchy + domain calibration + summary (activities 1, 4, 8). Skip simulation bridge.
- **STANDARD/COMPREHENSIVE**: All activities.

**Activities:**
1. Validation hierarchy (interpolation R²>0.95, extrapolation R²>0.80, counterfactual)
2. Residual diagnostics: `ts_reviewer.py` phases 7-10 (if applicable)
3. Baseline comparison: FVA > 0% required for time-series
4. Domain calibration against plausibility bounds
5. Uncertainty quantification: `forecast_modeler.py` conformal prediction, or `conformal_intervals()` / `cqr_intervals()` from ts_reviewer
6. If simulator ran: `scripts/simulator.py bridge` to validate predictions
7. Adversarial posture classification (if applicable)
8. **Scope completeness check** (STANDARD/COMPREHENSIVE/PSYCH): verify `[H_S]` posterior ≥ 0.80 OR that `[H_S_prime]` > 0.40 has been resolved via a scope-expansion pass (trigger S1 in `decisions.md`). Re-run `scope_auditor.py residual-match` on Phase 3 residuals against any external index set. Validation FAILS if `[H_S_prime]` > 0.40 and no scope-expansion reopen has been completed.
9. **`$SM write summary.md`** — final report referencing observations, evidence trail, and state block.

**EXIT GATE — write each via `$SM write <filename>`:**
- [ ] `validation.md`: fully populated (validation hierarchy table, verdict)
- [ ] `summary.md`: final report written, references session files, includes state block
- [ ] `hypotheses.json`: final posteriors recorded
- [ ] `state.md`: Phase 5 complete, final confidence
- [ ] `progress.md`: all phases marked complete
- [ ] `phase_outputs/phase_5.md` written

**Reference**: `references/validation-checklist.md`, `references/adversarial-heuristics.md`, `references/timeseries-review.md`

---

## PSYCH Tier: Psychological Profiling

For analyzing human behavior, personas, and profiles. See `references/psych-tier-protocol.md` for the complete protocol.

**Phases:** 0-P (Context) → 0-P.7 (Scope Interrogation — see Phase 0.7, with scope S framed as "which life-context domains inform the subject's behavior") → 1-P (Baseline) → 2-P (Stimulus-Response) → 3-P (Structural ID) → 4-P (Motive) → 5-P (Validation)

**Same FSM rules apply**: EXIT GATE must be passed at each phase. File writes are mandatory. Use `scripts/belief_tracker.py` instead of `bayesian_tracker.py`.

**PSYCH File Write Matrix**: Same as system analysis matrix, replacing `hypotheses.json` with `beliefs.json`. See `references/psych-tier-protocol.md` for phase-specific exit gates.

**Ethical Constraints**: No clinical diagnosis. Cultural calibration required. Document consent. Defensive use only.

**Key Outputs:** OCEAN profile, Dark Triad assessment, MICE/RASP driver ranking, behavioral predictions, interaction strategy.

**State Block:** `[STATE: Phase X-P | Tier: PSYCH | Archetype: Y | Rapport: L/M/H | Stress: L/M/H]`

**Reference**: `references/psych-tier-protocol.md`, `references/archetype-mapping.md`, `references/motive-analysis.md`, `references/elicitation-techniques.md`, `references/linguistic-markers.md`, `references/profile-synthesis.md`
