# Changelog

All notable changes to the Epistemic Deconstructor project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [7.14.0] - 2026-04-15

### Added — Phase 1.5 Abductive Expansion

A new mandatory sub-phase between Phase 1 (Boundary Mapping) and Phase 2 (Causal Analysis) that formalizes backward inference from observations to candidate causes. Phase 0.7 made the system boundary a hypothesis instead of a premise; Phase 1.5 makes interior hypothesis *generation* an auditable, tool-mediated step rather than analyst intuition. Closes the epistemological gap where rigorous tracking sat on top of informally-generated hypothesis sets.

- **`src/scripts/abductive_engine.py`** (new, ~1150 lines) — stdlib CLI implementing five operators:
  - **TI Trace Inversion** (`invert`): consults `trace_catalog.json` keyed on observation category; produces library-sourced candidate causes plus optional LLM-parametric suggestions.
  - **AA Absence Audit** (`absence-audit`, `close-prediction`): enumerates "what should be observed if hypothesis is true" predictions, queues them to `predictions_pending.json`, applies evidence on close.
  - **SA Surplus Audit** (`surplus-audit`): diffs observation record against candidate coverage union; unexplained observations become candidates for new hypotheses.
  - **AR Analogical Retrieval** (`analogize`): matches case signatures against archetype `trace_signatures` for bidirectional archetype <-> signature mapping.
  - **IC Inference Chains** (`chain start|step|close|audit`): structured JSON micro-inference logs that compose via Bayesian odds update. Auditable for gaps via the dedicated `chain audit` subcommand.
  - **Coverage-weighted promotion** (`candidates list|promote`): the primary mitigation for hypothesis explosion. Promotion is **rejected in code** if `coverage_score = (observations_explained / total_observations) / complexity` falls below threshold (default 0.30). Promoted candidates are written to `hypotheses.json` via subprocess to `bayesian_tracker.py` with statement prefix `[H_ABDUCT_CANDn]`.
  - **Catalog bootstrap** (`catalog bootstrap|review`): emits an LLM prompt template + JSON schema for offline catalog extension; the resulting JSON is loaded via `catalog review` and staged with `pending_review` provenance. Stays stdlib-only with no network dependency.
  - **Provenance discipline**: every candidate carries `source` ∈ `{library, llm_parametric, analyst, chain_derived}` (validated in code; invalid values raise `ValueError`).
  - **LLM-parametric hard caps** (Evidence Rule 8): `add_candidate` rejects `llm_parametric` candidates with prior > 0.30; `chain_step` rejects `llm_parametric` steps with LR > 2.0; `promote` re-checks both at promotion time.
- **`src/config/trace_catalog.json`** (new) — seed trace catalog with 6 categories (`generic`, `timing`, `resource`, `output_anomaly`, `failure`, `behavioral_deviation`) and 25 candidate causes spanning software, infrastructure, and behavioral-analysis domains. Each entry has `cause`, `mechanism`, `prior`, `source`, `complexity`. Schema documented inline.
- **`src/config/archetypes.json`** (modified) — every archetype gained an optional `trace_signatures` field (free-text signature strings used by the AR operator). The Phase 0.7 scope_auditor reader is unaffected (it ignores the new field), and `tests/test_scope_auditor.py` still passes 34/34.
- **`src/references/abductive-reasoning.md`** (new, 307 lines) — full Phase 1.5 protocol reference with Table of Contents. Covers: abduction vs induction vs deduction, the five operators, coverage-weighted selection, provenance discipline, LLM-parametric LR caps, four failure modes (retroduction-as-confirmation, narrative fallacy, just-so stories, hypothesis explosion), tier scaling, exit gate, and three worked examples (software latency spike, real estate price anomaly, PSYCH-tier behavioral deviation).
- **`src/agents/abductive-engine.md`** (new) — sonnet sub-agent definition (cyan, background-capable). Imitates `scope-auditor.md` structure. Tier-aware: skipped in RAPID, SA+AA only in LITE, full five operators in STANDARD, multi-pass permitted in COMPREHENSIVE. **Does not write to `hypotheses.json` directly** — returns promotion recommendations to the orchestrator, which delegates to `hypothesis-engine`. Preserves the single-writer contract.
- **`tests/test_abductive_engine.py`** (new, 77 tests) — unit + CLI smoke tests across `TestTraceCatalog`, `TestAbductiveState`, `TestCandidateProvenance`, `TestTraceInversion`, `TestAbsenceAudit`, `TestSurplusAudit`, `TestAnalogicalRetrieval`, `TestInferenceChains`, `TestCoverageGate`, `TestCatalogBootstrap`, `TestGateAndReport`, `TestHelpers`, `TestCLISmoke`. **No mocks anywhere** — real file I/O, real JSON round-trips, end-to-end CLI test that runs all five operators on synthetic data and verifies the persisted state. **93% line coverage** on `abductive_engine.py` (well above the 90% target).

### Changed — protocol enforcement

- **`src/SKILL.md`** — version bump 7.13.0 → 7.14.0. Added Phase 1.5 section between Phase 1 and Phase 2 with GATE IN, activities (TI/AA/SA/AR/IC), tier scaling table, and exit gate checklist. Updated FSM diagrams to include `P1_5` state in STANDARD/COMPREHENSIVE, LITE, and PSYCH tier subgraphs. Extended File Write Matrix with a new P1.5 column and five new runtime session files (`abductive_state.json`, `hypothesis_candidates.json`, `predictions_pending.json`, `inference_chains.json`, `surplus_audit.json`). Extended Tier Selection table with "Phase 1.5 scope" column. Added **Evidence Rule 8** (LLM-parametric caps + coverage-weighted promotion gate) to the existing 7-rule list. Cross-references to `references/abductive-reasoning.md`.
- **`src/agents/epistemic-orchestrator.md`** — added `abductive-engine` to the `Agent()` tools list. Added Phase 1.5 to the FSM responsibility (P0 → P0.7 → P1 → P1.5 → P2 → ...). Added "Do NOT run Phase 1.5 abductive expansion → delegate to abductive-engine" rule. Updated the Tier Routing table to include `abductive-engine` in LITE (SA+AA only), STANDARD, COMPREHENSIVE, and PSYCH rows. Added explicit Phase 1.5 exit gate criteria to Step 3 of the orchestrator's gate verification (≥3 inverted, surplus run, ≥1 chain per promoted candidate).
- **`src/agents/cognitive-auditor.md`** — added new "Abductive Generation Traps (Phase 1.5)" section with three new trap definitions: **Trap 17 Narrative Fallacy (abductive output audit)** specifically targeting Phase 1.5 chain composition and promoted-candidate consistency, **Trap 18 Retroduction-as-confirmation** (verifies AA queue exists before confirming evidence is sought), and **Trap 19 Hypothesis explosion** (verifies the coverage gate fired by checking for rejected entries in `hypothesis_candidates.json`). Added Step 7 (Abductive output audit) to the Audit Procedure that reads `phase_outputs/phase_1_5.md`, `inference_chains.json`, `hypothesis_candidates.json`, and `predictions_pending.json`.
- **`CLAUDE.md`** — version bump 7.13.0 → 7.14.0. Added `src/scripts/abductive_engine.py` and `src/config/trace_catalog.json` to the Repository Structure tree. Added `abductive-engine.md` to the agent file list. Added a new "Abductive Engine CLI" section under Key Commands documenting all subcommands with examples (start, invert, absence-audit/close-prediction, surplus-audit, analogize, chain start/step/close/audit, candidates list/promote, catalog bootstrap/review, gate, report). Added P1.5 row to the System Analysis Phases table. Added P1-P.5 row to the PSYCH Tier Phases table. Updated Tier System table to show Phase 1.5 inclusion per tier. Added `abductive-engine` row to the Agent Roles table. Updated File Modification Guidelines to describe `abductive_engine.py` and the new tool-integration flow including Phase 1.5. Test count updated 466 → 543.
- **`Makefile`** — `VERSION := 7.14.0`.

### Verified

- `make validate`: skill structure passes.
- `make lint`: all 12 scripts compile cleanly (11 existing + 1 new).
- `pytest tests/test_abductive_engine.py`: **77 tests passing**.
- `pytest tests/`: **543 tests passing** (466 baseline + 77 new), 0 regressions.
- Coverage on `abductive_engine.py`: **93%** (target was 90%).
- End-to-end CLI dry-run executes all five operators (TI on 3 observations, AA with 3 predictions, SA, AR with 5 archetype matches, IC chain with 2 steps + close + audit) and produces a valid persisted state file with 3 observations, 10 staged candidates, 1 closed inference chain, 3 pending predictions, 1 analogy. Phase 1.5 exit gate returns PASS for STANDARD-tier dry-run.
- No emojis introduced in any new or modified file (verified by Unicode emoji-block grep across `abductive_engine.py`, `trace_catalog.json`, `abductive-reasoning.md`, `abductive-engine.md`, `cognitive-auditor.md`, `epistemic-orchestrator.md`, `SKILL.md`, `CLAUDE.md`, `test_abductive_engine.py`).
- No `.claude/` directory created in the project. The pre-existing `.claude/` is in `.gitignore` (user state only).

### Rationale

The v7.12 (Scope Interrogation) release made Phase 0.7 promote the system boundary from premise to hypothesis. v7.14 makes the corresponding move for the interior: hypothesis *generation*. Previously, the protocol tracked and falsified hypotheses rigorously, but the candidate set itself was generated informally by the analyst at Phase 0 and (via scope audit) at Phase 0.7. Phase 1.5 closes that loop: every candidate cause now has a provenance tag, every promotion goes through a coverage gate enforced in code, every inference chain is structured JSON that an auditor can diff and replay. The cognitive-auditor extension means narrative-fallacy checks now run specifically against the Phase 1.5 outputs that are most prone to it.

The five-operator naming (TI/AA/SA/AR/IC) mirrors the Phase 0.7 four-mechanism naming (M1/M2/M3/M4) so the protocol vocabulary stays consistent. The coverage-weighted promotion gate is the structural mitigation against hypothesis explosion — the gate is enforced in `abductive_engine.py promote()` and re-checked in tests (`test_low_coverage_candidate_rejected_at_promotion` is the primary regression guard).

## [7.13.0] - 2026-04-15

### Changed
- **Agent wiring of v7.12 protocol features** — v7.12.0 added the Scope Interrogation capability (Phase 0.7, H_S standing pair, multi-pass trigger S1) to `SKILL.md` and the reference layer, and added `scope_auditor.py` + `scope-auditor` agent. However, the remaining 12 sub-agent definitions in `src/agents/` were not updated to *enforce* those features at their own exit gates. The orchestrator relied on human memory of the protocol to check triggers, and phase agents did not surface structured trigger evaluations. This release wires the enforcement layer into all affected agents so the protocol is self-enforcing.
- **`src/agents/validator.md`** — added two new mandatory Phase 5 activities:
  - **Activity 8 Scope Completeness Check** — runs `grep -E "\[H_S(_prime)?\]"` against `bayesian_tracker.py report --verbose` output and applies a decision table: PASS if `[H_S_prime]` ≤ 0.40; CONDITIONAL PASS if > 0.40 but a prior `S1 Scope Gap` reopen is logged in `decisions.md`; FAIL otherwise. On FAIL, the validator halts and refuses to write `summary.md` until the scope-expansion pass has been executed, preventing "successful" session closure on structurally incomplete analyses.
  - **Activity 9 Multi-Pass Trigger Evaluation (P5.1-P5.4)** — evaluates extrapolation failure, conformal coverage miss, domain calibration fail, and "wrong question" triggers from `references/multi-pass-protocol.md`. Any firing trigger halts finalization and emits a `$SM reopen` recommendation to the orchestrator.
  - Output format extended with **Scope Completeness** and **Multi-Pass Triggers** sections; exit gate checklist explicitly requires both to be PASS before `summary.md` is written.
- **`src/agents/epistemic-orchestrator.md`** — replaced the two-line exit-gate verification step in the Phase Execution Pattern with an explicit **7-step Gate Check Procedure**: (1) file completeness via session-clerk, (2) content validation, (3) hypothesis state review, (4) H_S pair check (STANDARD/COMPREHENSIVE/PSYCH Phase 0 exit), (5) multi-pass trigger evaluation covering U1-U4 universal triggers + S1 scope trigger + phase-specific P1.1-P5.4 triggers from `multi-pass-protocol.md`, (6) reopen-or-advance decision with concrete `$SM reopen` commands, (7) cognitive-auditor background launch for independent scope hygiene audit. Orchestrator is now a proper state-machine enforcer rather than a delegation hub.
- **`src/agents/hypothesis-engine.md`** — added two concrete enforcement procedures:
  - **Disconfirm-before-confirm verification**: before any hypothesis can cross the 0.80 posterior threshold, the engine runs `report --verbose`, inspects the evidence trail for that H, and counts entries with `lr < 1.0` or `*_disconfirm` / `falsify` presets. Zero disconfirming evidence → UPDATE BLOCKED with a structured response naming the hypothesis, current posterior, and required corrective action.
  - **H_S standing pair guard**: before any Phase 1 evidence update is accepted in STANDARD/COMPREHENSIVE/PSYCH tiers, the engine greps the report output for `[H_S]` and `[H_S_prime]` statements. Missing → H_S PAIR MISSING response with the exact `bayesian_tracker.py add` commands to seed the pair. This turns Evidence Rule #7 from a human-memory constraint into a machine-enforced invariant.
- **`src/agents/psych-profiler.md`** — PSYCH tier previously had no scope interrogation phase despite SKILL.md requiring it. Added:
  - **Phase 0-P.7** row to the Phase Flow table and a new dedicated section framing scope S as "life-context domains" (financial pressures, unseen relationships, medication/substance effects, cultural/religious commitments, professional stressors). Delegates M1-M4 to the scope-auditor agent rather than running them in-place.
  - **PP.1-PP.4 + U1-U4 + S1 trigger evaluation** in the per-phase output format.
  - **Phase 5-P Scope Completeness Check** — same decision table as the system validator, but operating on `beliefs.json` via `belief_tracker.py report --verbose`. Final validated profile is not produced if the check FAILs.
- **`src/agents/boundary-mapper.md`** — added `ts_reviewer.py quick` and `fourier_analyst.py quick` / `analyze` as mandatory Phase 1 steps when I/O data is numeric. SKILL.md's Phase 1 reference section has called for this since v7.0 but the agent did not actually invoke either tool, leaving spectral profiling to hope. Added **P1.1 I/O coverage + P1.2 stimulus-response database + U1/U2 universal** trigger evaluation to the output format.
- **`src/agents/causal-analyst.md`** — added **P2.1/P2.2/P2.3 + U1/U3/U4/S1** trigger evaluation section. P2.3 (insufficient observations for causal claims) correctly reopens Phase 1 rather than Phase 2 per `multi-pass-protocol.md`.
- **`src/agents/parametric-id.md`** — added **P3.1/P3.2/P3.3 + S1** trigger evaluation and a mandatory **post-fit `scope_auditor.py residual-match`** step that runs the fitted residuals against an external-index directory. Any |r| ≥ 0.30 with p < 0.05 fires S1 directly at the Phase 3 gate, catching omitted drivers at the earliest point they become detectable (residual structure) rather than waiting for Phase 5.
- **`src/agents/model-synthesizer.md`** — Archetype classification step now cross-checks against `references/archetype-accomplices.md` (not just `simulation-guide.md`). If the identified archetype implies co-driver domains not present in the causal graph, the agent reports this as a potential S1 Scope Gap signal. Added distribution family guidance via `references/distributions-guide.md` for MC/ABM/DES work to prevent default-normal-distribution errors on heavy-tailed data. Added **P4.1/P4.2 + U1/S1** trigger evaluation.

### Rationale
The v7.12 audit surfaced that SKILL.md documented the protocol correctly but the agent layer was still running a pre-v7.12 workflow. This is a structural gap: the protocol is only as good as the agent that executes it, and an un-enforced evidence rule is indistinguishable from no evidence rule. This release brings all agents up to spec so the orchestrator can actually rely on delegation, and so a full multi-agent session (`claude --agent epistemic-orchestrator`) produces a result that matches SKILL.md's written guarantees. No `scripts/`, `references/`, `config/`, or `tests/` changes — this is a pure agent-definition release.

### Unchanged (intentionally)
- `session-clerk.md`, `research-scout.md`, `rapid-screener.md`, `scope-auditor.md`, `cognitive-auditor.md` — already aligned with v7.12 at the v7.12.0 release. Reviewed and confirmed no drift.

### Verified
- `make validate`: skill structure passes.
- Frontmatter integrity: all 13 agents parse correctly (name + description + tools + model fields intact).
- 8 files modified, 260 insertions, 10 deletions. No `scripts/` or `tests/` changes, so existing 466 pytest suite is unaffected.

## [7.12.1] - 2026-04-15

### Fixed
- **`build.ps1` `sync-skill` dropped `archetypes.json`** — `Invoke-SyncSkill` hardcoded `Copy-Item "src/config/domains.json"` instead of `src/config/*.json`. Windows users installing via `.\build.ps1 sync-skill` got an install missing the scope interrogation archetype library, breaking `scope_auditor.py enumerate` on first use. Makefile already used the correct glob. Fixed to match.
- **"conformal Phase 5" nonsense phrase** in `src/SKILL.md:425` and `CLAUDE.md:644`. Replaced with "conformal prediction" in both. Same corruption in both files suggested a copy-paste at edit time.
- **Stale script docstrings** — `bayesian_tracker.py` and `rapid_checker.py` still declared `v7.10.0` two releases after the fact. All stamped scripts now at `v7.12.1`.
- **`SKILL.md` preset list unlabeled** — the preset list at line 479 documented only `bayesian_tracker.py` / `rapid_checker.py` presets, but sat below a line naming all three trackers. PSYCH-tier readers trying `strong_confirm` on `belief_tracker.py` hit `ValueError: Unknown preset`. Now split into two explicit lists (one per tracker family).
- **`common.py` `save_json` had no lock around the write** — atomic at the bytes level (tempfile + `os.replace`), but two concurrent `save_json` calls could each complete their rename and the earlier writer's data was silently lost. Added sidecar `<path>.lock` file with exclusive POSIX/Windows lock for the duration of the write. Verified with a 4-process × 20-iteration stress test: no file corruption, JSON parses cleanly under contention. **Known limit**: the lock does not make a `load_json` → modify → `save_json` sequence atomic across processes — callers that perform read-modify-write from multiple processes must still coordinate externally. Documented in the `save_json` docstring.

### Documentation
- **Reference navigability** — added `## See Also` cross-link sections to the 4 reference files that had zero outbound sibling links: `boundary-probing.md`, `causal-techniques.md`, `coherence-checks.md`, `tools-sensitivity.md`. Improves discoverability of the knowledge graph for Phase 1–2 topics and the RAPID tier. (Initial audit flagged 15 orphans; direct grep with a proper pattern found only 4.)

### Audit notes
- Comprehensive v7.12.0 codebase audit applied the Epistemic Deconstruction protocol to itself (COMPREHENSIVE tier, L3 fidelity). Final hypothesis posteriors: H1 doc-drift 0.98 CONFIRMED, H2 script-bugs 0.40, H3 cross-ref 0.14 WEAKENED, H4 agent-misalignment 0.20 WEAKENED.
- **7 false positives rejected** with line-level verification and documented in the audit's `observations/obs_006_verified_false_positives.md`, so they do not resurface in a future audit: (1) `ts_reviewer.py` Ljung-Box message "inverted" — context is forecastability of the series, not residual whiteness, message is correct; (2) `rapid_checker.py` calibration bounds "inverted for lower-is-better" — traced with real `domains.json` tuples, direction sentinel is correct at every boundary; (3) `parametric_identifier.py` bootstrap `ci_method` "not reset on failure" — actually reset correctly at line 500 and line 521; (4) threshold band "inconsistency" between trackers — bands match `SKILL.md` docs exactly, PSYCH wider-band design is intentional; (5) "38 silently skipped tests" — all run when numpy is present, pytest reports 466 passed / 0 skipped; (6) saturation-warning 5% buffer "off by 0.05" — intentional early-warning before hard threshold; (7) "reference files > 100 lines missing TOC" — all 35 have TOCs in the first 30 lines.

### Verified
- Full pytest suite: `466 passed / 0 skipped / 0 failed` after all fixes.
- `make lint`: all 11 scripts compile.
- `--help` smoke test: all 11 scripts respond correctly.
- Concurrent stress test on `save_json`: 4 processes × 20 iterations, no byte-level corruption.

## [7.12.0] - 2026-04-15

### Added
- **Scope Interrogation capability (Phase 0.7)** — promotes the system boundary from a premise to a hypothesis. The protocol now surfaces drivers that live outside the initially-framed scope on the FIRST analytical pass instead of requiring manual re-runs after the omission is discovered.
  - **H_S standing hypothesis pair**: Every STANDARD/COMPREHENSIVE/PSYCH analysis must seed `[H_S]` ("drivers within initial scope") and `[H_S_prime]` ("material drivers exist outside S") at Phase 0 and track them for the entire session. `[H_S_prime]` satisfies the existing ≥1 adversarial hypothesis rule. Enforced via new **Evidence Rule #7** in SKILL.md.
  - **Four generalized mechanisms (M1-M4)**:
    - **M1 Flow Tracing** — for every input, trace one level upstream to its generator; for every output, one level downstream to its consumer. Neighbors outside scope are exogeneity candidates.
    - **M2 Archetype Accomplice Library** — classify the target into archetypes from the library; each archetype ships a list of typical co-driver domains with mechanisms and priors.
    - **M3 Residual-Signature Matching** — correlate model residuals against external index series to detect omitted drivers.
    - **M4 Adversarial Scoping (Steelman Prompt)** — three outsider critiques (domain outsider, investigative journalist, regulator), each naming one excluded domain AND one mechanism.
- **`src/scripts/scope_auditor.py`** (new, ~600 lines) — CLI implementing M1-M4. Subcommands: `start`, `list-archetypes`, `enumerate`, `trace`, `steelman`, `residual-match`, `report`, `gate`, `dedupe`, `add-candidate`. Stdlib + numpy-free (pure-Python Pearson correlation + Student-t p-value via regularized incomplete beta continued fraction). Optional scipy for p-value accuracy.
- **`tests/test_scope_auditor.py`** (new, 34 tests) — archetype library schema, Cyprus acceptance test (speculative_asset_market must surface illicit-flows/legislation/immigration/nomad), state persistence, residual matching with known correlations, CLI subprocess smoke tests, gate status. Total suite: **466 tests passing** (432 baseline + 34 new).
- **`src/config/archetypes.json`** (new) — machine-readable archetype library with 10 seed archetypes spanning: open-economy speculative asset market, regulated infrastructure service, API-backed software service, supply-chain network, speculative information market, organizational actor, individual persona, platform ecosystem, research/knowledge producer, feedback control system.
- **`src/references/archetype-accomplices.md`** (new) — human-readable narrative companion to `archetypes.json` with per-archetype accomplice tables and extensibility rules.
- **`src/references/scope-interrogation.md`** (new) — full Phase 0.7 protocol: M1-M4 procedures, exit gate, three worked examples (Cyprus real estate, API-backed software, environmental NGO). The Cyprus example is an acceptance test — running M2 against `speculative_asset_market` + M4 steelman critiques deterministically surfaces the four originally-missed domains.
- **`src/agents/scope-auditor.md`** (new) — sonnet sub-agent definition, background-capable. Runs M1-M4 at Phase 0.7 and post-Phase 3.
- **`src/references/cognitive-traps.md`** — 4 new traps in a new "Scope-Related Traps" section: **Trap 20 Framing Effect**, **Trap 21 Streetlight Effect**, **Trap 22 Omitted-Variable Bias**, **Trap 23 Premature Closure**. All cross-reference `scope-interrogation.md` and the scope-auditor agent.
- **`src/references/multi-pass-protocol.md`** — new **trigger S1 Scope Gap** in a new "Scope Triggers" section. Fires when residual-signature correlation, accomplice-hypothesis posterior > 0.40, or cognitive-auditor Out-of-Frame Report occurs. Reopens Phase 0 for scope expansion (distinct from P5.4 "Wrong question").

### Changed
- **`src/SKILL.md` Phase 0**: adds H_S/H_S_prime standing pair seeding to Activities; Phase 0 exit gate now requires both in `hypotheses.json`.
- **`src/SKILL.md` FSM**: inserts `P0_7` node between P0 and P1 in STANDARD/COMPREHENSIVE and between P0P and P1P in PSYCH. RAPID and LITE tiers unchanged.
- **`src/SKILL.md` File Write Matrix**: new `P0.7` column + `scope_audit.md` and `scope_audit.json` rows.
- **`src/SKILL.md` Evidence Rules**: new rule #7 "SCOPE HYPOTHESIS STANDING PAIR".
- **`src/SKILL.md` Phase 0.7 section**: new full phase documentation between Phase 0.5 and Phase 1.
- **`src/SKILL.md` Phase 5 Activities**: new "Scope completeness check" — validation fails if `[H_S_prime]` > 0.40 and unresolved.
- **`src/agents/epistemic-orchestrator.md`**: adds `scope-auditor` to Agent() tools list; FSM docs updated for P0 → P0.7 → P1 path; Tier Routing table updated to show scope-auditor in STANDARD/COMPREHENSIVE/PSYCH rows; Evidence Rules extended with H_S standing pair enforcement.
- **`src/agents/cognitive-auditor.md`**: adds Scope Omission Audit (4 new traps) to What You Audit; new Audit Procedure step 6 (scope check); new Out-of-Frame Report output format that fires trigger S1.
- **`Makefile`**: `VERSION := 7.12.0`. Config copy step updated to copy all `src/config/*.json` (now includes `archetypes.json`). `sync-skill` target updated.
- **`build.ps1`**: `$Version = "7.12.0"` with equivalent copy.
- **`README.md`**: version badge → v7.12.0, test count → 466.
- **`CLAUDE.md`**: repo structure diagram updated with new files; new "Scope Auditor CLI" section; version references bumped.

---

## [7.11.0] - 2026-04-15

### Added
- **`src/scripts/parametric_identifier.py`** (1884 lines) — New CLI for structural system identification. Closes the Phase 3 tooling gap: the protocol prescribed ARX → ARMAX → NARMAX → State-Space but only forecasting models (ARIMA/ETS/CatBoost) were previously available. Features:
  - **ARX fitting**: OLS via QR decomposition, singular-matrix detection, `(na, nb, nk)` grid search
  - **ARMAX fitting**: SARIMAX backend with graceful fallback when statsmodels missing
  - **NARMAX fitting**: polynomial basis expansion + FROLS (Forward Regression Orthogonal Least Squares) term selection via Error Reduction Ratio
  - **Unified structure selection**: AIC / BIC / AICc / FPE ranked across all three families with whiteness-gated winner rule
  - **Uncertainty quantification**: residual bootstrap (temporally safe — forward-regenerates y through fitted model) plus analytic CIs from `(ΦᵀΦ)⁻¹σ²` / `cov_params()` fallback. No heavy deps (no PyMC).
  - **Integrated Ljung-Box whiteness**: auto-run post-fit, statsmodels backend with numpy + Wilson–Hilferty fallback
  - **Walk-forward cross-validation**: expanding-window forward chaining with per-fold R²/RMSE, satisfies Phase 3 exit gate (R² > 0.8)
  - **Identifiability gate**: `assess` subcommand (data length + SNR + coherence) — Phase 3 analogue of forecast_modeler's forecastability gate
  - **Simulator bridge**: `FitResult.to_simulator_format()` produces `{type: 'arx', a, b, nk}` dicts that drop directly into `simulator.py` ARX/MC modes, closing the Phase 3 → Phase 4 → Phase 5 loop
  - **CLI**: `assess`, `fit`, `compare`, `demo` subcommands matching existing tool conventions
- **`tests/test_parametric_identifier.py`** (482 lines, 31 tests) — known-answer ARX recovery, grid-search selection, bootstrap CI coverage, FROLS term selection, Ljung-Box correctness, walk-forward CV, NARMAX polynomial basis shape, simulator format round-trip, CLI subprocess smoke test. Total suite: **432 tests passing**.

### Changed
- **`src/SKILL.md` Phase 3 Activities**: rewritten to invoke `parametric_identifier.py` for structural identification instead of deferring to the forecasting tool. New explicit 9-step workflow: identifiability → structure selection → estimation → uncertainty → residual diagnostics → spectral complement → forecasting (if applicable) → simulator pipe → hypothesis update.
- **`src/agents/parametric-id.md`**: agent now has `$PID` binding; procedure updated with forecasting-vs-sysid decision rule; tool table expanded.
- **`CLAUDE.md`**: new "Parametric Identifier CLI" section with full command reference; tool-flow paragraph updated with sysid-vs-forecasting semantic distinction; tests registry updated.
- **`src/references/system-identification.md`**: header note promoting pseudo-code reference to working CLI; Practical Workflow section rewritten to reference CLI invocations instead of describing steps abstractly.
- Version bump to 7.11.0 across all files; test count updated from 401 → 432 in README.md and CLAUDE.md.

---

## [7.10.0] - 2026-04-06

### Fixed
- **`common.py`**: Fix file locking race condition — `_unlock_file()` no longer called when lock acquisition fails
- **`belief_tracker.py`**: Fix saturation warning boundary (`0.10 <` instead of `0.10 <=`) so warning doesn't fire at exact REFUTED threshold
- **`session_manager.py`**: Add symlink resolution and path escape detection to `cmd_read_file` and `cmd_path` (security hardening)
- **`session_manager.py`**: `read_pointer()` now warns to stderr when active session directory no longer exists instead of failing silently
- **`README.md`**: Fix reference count (32→33), add missing `engineering-laws.md` to Knowledge Base table
- **`CLAUDE.md`**: Fix orphaned `docs/FINDINGS.md` reference, add `tests/` and `docs/` to repository structure diagram
- **`bayesian_tracker.py`**, **`rapid_checker.py`**: Update stale docstring version strings (v7.7.0→v7.10.0)

### Added
- **`bayesian_tracker.py`**: Warning when updating a CONFIRMED hypothesis (symmetric with REFUTED guard)
- **`belief_tracker.py`**: Warning when updating a CONFIRMED trait (symmetric with REFUTED guard)
- **`rapid-screener.md`**: Added skill reference for discoverability

### Changed
- Version bump to 7.10.0 across all files

---

## [7.9.0] - 2026-04-06

### Added
- **Sub-agent architecture**: 12 specialized Claude Code agent definitions in `src/agents/` that decompose the monolithic protocol into coordinated specialists:
  - `epistemic-orchestrator` (opus): Phase FSM, tier selection, delegation — the main agent
  - `session-clerk` (haiku, background): All filesystem I/O for session management
  - `hypothesis-engine` (sonnet): Centralized Bayesian tracking with evidence rule enforcement (LR caps, anti-bundling, prior discipline, disconfirm-before-confirm)
  - `cognitive-auditor` (sonnet, background): Independent bias and cognitive trap detection
  - `rapid-screener` (sonnet): Phase 0.5 RAPID coherence screening
  - `boundary-mapper` (sonnet): Phase 1 I/O boundary mapping
  - `causal-analyst` (opus): Phase 2 causal graphs and falsification loops
  - `parametric-id` (sonnet): Phase 3 model fitting (ts_reviewer, forecast_modeler, fourier_analyst)
  - `model-synthesizer` (sonnet): Phase 4 composition, emergence testing, simulation
  - `validator` (opus): Phase 5 validation hierarchy and final report generation
  - `psych-profiler` (opus): PSYCH tier behavioral analysis (all 6 phases)
  - `research-scout` (haiku, background): Web research and external information gathering
- **`docs/SUBAGENT_REDESIGN.md`**: Full architectural design document covering agent topology, state ownership model, communication patterns, implementation strategy, and agent teams for competing hypotheses
- **`docs/subagents.md`**: Claude Code subagents reference documentation

### Changed
- **Makefile**: Added `AGENT_FILES` variable, `agents/` directory creation in `build` and `sync-skill` targets
- **`build.ps1`**: Added `agents/` directory creation and copy in `Invoke-Build` and `Invoke-SyncSkill` functions
- **`CLAUDE.md`**: Added `src/agents/` to repository structure, installation instructions section, agent roles table, multi-agent activation mode
- Version bump to 7.9.0 across all files

---

## [7.8.0] - 2026-04-02

### Added
- **`engineering-laws.md`**: New reference doc integrating Akin's Laws of Spacecraft Design into protocol. 18 curated engineering laws mapped to protocol phases with operational rules, anti-patterns, and phase integration map.
- **`cognitive-traps.md`**: 4 new engineering-derived traps: Post-Hoc Rationalization (Bowden's Law, Trap 16), False Linearity (Mar's Law, Trap 17), Authority of Print (Trap 18), Extremum Bias (Trap 19).
- **`modeling-epistemology.md`**: New "Engineering Design Axioms" section with Graceful Degradation, Interface Primacy (Shea's Law), Estimation Hierarchy, McBryan's Law, and Parsimony as Subtraction (de Saint-Exupery's Law).
- **`red-flags.md`**: New "Engineering Sanity Check Red Flags" table with 10 engineering-derived indicators.
- **`decision-trees.md`**: New "When to Start Over?" decision tree with sunk cost immunity criteria.
- **`validation-checklist.md`**: New "Engineering Sanity Checks" section (Quantification Check, Extremum Check, Partial Credit Check, Interface Audit).
- **`evidence-calibration.md`**: New "Estimation Hierarchy" section (Measured > Estimated > Guessed > Assumed) with LR cap implications per provenance level.
- **`setup-techniques.md`**: New "Start-Now Principle" section (Roosevelt's Law, Akin's Law #9).
- **`SKILL.md`**: 3 new critical reminders: numbers-not-opinions, design-for-partial-failure, don't-over-model.

### Changed
- **`compositional-synthesis.md`**: Interface Specification section enriched with Shea's Law quote and degradation mode property.
- **`domain-calibration.md`**: Added "Extremum Distrust Principle" section before "Too Good" rule, with Akin's Law #8 and #19 quotes.
- **`decision-trees.md`**: "When to Stop?" tree enriched with Edison's Law fidelity sufficiency check.
- **`CLAUDE.md`**: Added `engineering-laws.md` to repository structure listing.

---

## [7.7.0] - 2026-03-20

### Fixed
- **`SKILL.md`**: Unify RAPID verdict terminology from GO/CONDITIONAL/NO-GO to CREDIBLE/SKEPTICAL/DOUBTFUL/REJECT, matching `rapid-assessment.md` and decision trees.
- **`SKILL.md`**: Add LITE/RAPID tier-specific Phase 5 scope (RAPID: calibration + verdict only; LITE: skip simulation bridge).
- **`SKILL.md`**: Add Phase 3 tool selection guidance (ts_reviewer vs forecast_modeler vs fourier_analyst).
- **`SKILL.md`**: Add `spectral-analysis.md` reference to Phase 1 and Phase 3.
- **`simulator.py`**: Fix operator precedence bug in settling time calculation (line 188) — boolean expression now uses explicit parentheses.
- **`belief_tracker.py`**: Fix saturation warning boundary from `0.10 <` to `0.10 <=` so warning fires at REFUTED threshold, not after.
- **`rapid_checker.py`**: Replace hardcoded "lower is better" metric list with dynamic detection from bounds structure (`suspicious < excellent`). Removes sync risk with `domains.json`.
- **`rapid_checker.py`**: Remove unreachable `plaus_low is None` branch in calibration logic.
- **`decision-trees.md`**: Define EFSM (Extended Finite State Machine) in model structure tree. Add `compose()` and `augment_emergence()` to recursive decomposition definitions.
- **`compositional-synthesis.md`**: Define emergence terms concretely (interaction potentials, feedback loops, nonlinear coupling, time delays) and add black-box gradient guidance (MC perturbation / finite differences).
- **`validation-checklist.md`**: Add FVA <= 0% decision rule — reject model and revert to baseline or revisit Phase 3.
- **`simulation-guide.md`**: Add validation bridge failure handling table with thresholds and actions for all validation types.
- **`financial-validation.md`**: Reference `domains.json` as source of truth instead of hardcoding threshold values.
- **`setup-techniques.md`**: Add prior assignment methods (reference class frequency, maximum entropy, domain base rate, expert elicitation).
- **`forecasting-science.md`**: Expand CQR specification (symmetric Q correction) and add default method recommendation (ICP → CQR → EnbPI).
- **`distributions-guide.md`**: Add mixture distributions, correlated parameters, and time-varying distributions sections.
- **`README.md`**: Add `rapid_checker.py domains` command to RAPID examples.

### Changed
- **`SKILL.md`**: Remove phase budget percentages and time estimates from all phases and tier selection table. Trim redundant explanatory text (488 → 474 lines).
- **Version**: Bumped from 7.6.1 to 7.7.0 in Makefile, build.ps1, SKILL.md, README.md, and CLAUDE.md.

---

## [7.6.1] - 2026-03-20

### Fixed
- **`build.ps1`**: Fix `$sourcePath.Path` crash in `package` command — `Resolve-Path` result now extracted as string, preventing null dereference on `.Path.Length`.
- **`common.py`**: `bayesian_update()` now rejects priors at 0.0 and 1.0 with `ValueError` instead of silently clamping to epsilon (callers already validate, but the math layer now enforces correctness).
- **`Makefile`**: Add `|| exit 1` after `unittest discover` so test failures stop the build instead of falling through to smoke tests.
- **`bayesian_tracker.py`**: `compare()` now returns finite caps (`1e6`/`1e-6`) instead of `inf`/`-inf`, making Bayes factor output JSON-serializable.
- **`bayesian_tracker.py`**: `add_coherence()` now warns on stderr when overwriting an existing coherence check of the same type.
- **`bayesian_tracker.py`**: `report()` returns clean "No hypotheses tracked." message instead of header-only markdown table when empty.
- **`belief_tracker.py`**: `trait_report()` returns clean "No traits tracked." message when empty.
- **`belief_tracker.py`**: Saturation warning now says "approaching REFUTED threshold (0.10)" instead of misleading "approaching refutation" (WEAKENED fires first at 0.30).
- **`simulator.py`**: Remove unused `import warnings`.
- **`simulator.py`**: Replace Unicode arrow `→` with ASCII `->` in print output, fixing `UnicodeEncodeError` on Windows CP1252 terminals.
- **`README.md`**: Fix footer version from v7.4.0 to v7.6.1.
- **`test_common.py`**: Update boundary prior tests to expect `ValueError` (matching new validation).

### Changed
- **Version**: Bumped from 7.6.0 to 7.6.1 in Makefile, build.ps1, SKILL.md, and CLAUDE.md.

---

## [7.6.0] - 2026-03-20

### Added
- **`modeling-epistemology.md`**: New "Depth Heuristic: The Iceberg Model" section — 4-layer analytical depth framework (Events/Patterns/Structures/Mental Models) mapped to protocol phases with failure modes and operational implications.
- **`modeling-epistemology.md`**: Maxim 13 (Epistemological) — "Analyze at depth, not just surface" with cross-reference to Iceberg Model section.
- **`modeling-epistemology.md`**: PSYCH tier cross-reference added to Cross-References section.

### Changed
- **Version**: Bumped from 7.5.0 to 7.6.0 in Makefile, build.ps1, README, SKILL.md, and CLAUDE.md.

---

## [7.5.0] - 2026-03-19

### Added
- **`simulator.py`**: 5 new distributions in `_sample_distribution()`: `gamma` (shape, scale), `poisson` (lam), `weibull` (a, scale), `binomial` (n, p), `chisquare` (df). Total supported distributions: 12.
- **`distributions-guide.md`**: New reference doc with distribution selection decision tree, JSON spec examples, relationship map, and phase integration guidance.
- **`test_simulator.py`**: 5 new test cases for added distributions (gamma, poisson, weibull, binomial, chisquare). Total tests: 390.

### Changed
- **`simulation-guide.md`**: Updated supported distributions list from 7 to 12 with cross-reference to distributions-guide.md.
- **Version**: Bumped from 7.4.3 to 7.5.0 in Makefile, build.ps1, README, SKILL.md, and CLAUDE.md.

---

## [7.4.3] - 2026-03-16

### Fixed
- **`bayesian_tracker.py`**: Fixed stale comment referencing `KILLED` instead of `REFUTED` in saturation warning (line 283).
- **`belief_tracker.py`**: Added `UNASSESSED` value to `TraitStatus` enum — profile methods (`get_ocean_profile`, `get_dark_triad_profile`, `get_mice_profile`, `calculate_dt_risk`) now use `TraitStatus.UNASSESSED.value` instead of raw `'UNASSESSED'` strings.
- **`evidence-calibration.md`**: Added PSYCH tier row to LR Cap Rules table documenting the `smoking_gun` preset (LR=20.0) exception.
- **`release-notes.md`**: Updated from stale v7.3.0 to current v7.4.3.

### Changed
- **Version**: Bumped from 7.4.2 to 7.4.3 in Makefile, build.ps1, README, SKILL.md, and CLAUDE.md.

---

## [7.4.2] - 2026-03-16

### Fixed
- **`bayesian_tracker.py`**: Renamed `Status.KILLED` enum to `Status.REFUTED` for consistency with `belief_tracker.py`'s `TraitStatus.REFUTED`. All internal references, error messages, and tests updated.
- **`bayesian_tracker.py`**: `strong_confirm` preset LR reduced from 10.0 to 5.0 to match documented LR scale (3.0–5.0) and SKILL.md evidence rules (MAX LR = 5.0 for Phases 0-1).
- **`bayesian_tracker.py`**: Saturation warning boundary changed from `<= 0.10` to `< 0.10` to correctly warn when approaching REFUTED threshold (0.05), not at WEAKENED boundary.
- **`belief_tracker.py`**: Saturation warning boundary changed from `<= 0.15` to `< 0.15` for consistency with bayesian_tracker pattern.
- **`fourier_analyst.py`**: Replaced `raise ImportError` with `sys.exit(1)` on missing numpy, matching error handling pattern of other scripts.
- **`CLAUDE.md`**: Added missing `Lead: HN (PP%)` field to state block template, matching SKILL.md format.
- **`CLAUDE.md`**: Renamed "KILLED/REFUTED" to "REFUTED" in threshold bands table to match actual script enum names.
- **`evidence-calibration.md`**: Corrected `strong_confirm` preset LR from 10.0 to 5.0 in tracker presets reference table.
- **`evidence-calibration.md`**: Added missing "Status Transition Thresholds" section with CONFIRMED/REFUTED/WEAKENED/ACTIVE thresholds for both trackers.
- **`psych-tier-protocol.md`**: Added "Belief Tracker Presets" table and "Status Transition Thresholds" section so PSYCH tier users don't need to reference CLAUDE.md for this information.

### Changed
- **Version**: Bumped from 7.4.1 to 7.4.2 in Makefile, build.ps1, README, SKILL.md, and CLAUDE.md.

---

## [7.4.1] - 2026-03-16

### Fixed
- **`evidence-calibration.md`**: Corrected 5 mismatched belief_tracker preset LR values (smoking_gun 10→20, indicator 3→2, counter_indicator 0.33→0.5, strong_counter 0.1→0.2, disconfirm 0.01→0.1) and added missing `falsify` preset.
- **`rapid_checker.py`**: Fixed `r2_prices` hardcoded fallback bounds from `(0.95, None, None, None)` to `(0.95, 0.30, 0.70, 0.85)` to match `domains.json`.
- **`rapid_checker.py`**: Added missing `time_series` domain to hardcoded fallback calibration, matching `domains.json`.

### Changed
- **Version**: Bumped from 7.4.0 to 7.4.1 in Makefile, build.ps1, README, SKILL.md, and CLAUDE.md.

---

## [7.4.0] - 2026-03-16

### Fixed
- **`bayesian_tracker.py`**: Saturation warnings now fire *before* CONFIRMED/KILLED thresholds (0.85–0.90 and 0.05–0.10) instead of after (was 0.95/0.05), making them useful as early alerts.
- **`belief_tracker.py`**: Saturation warnings now fire *before* CONFIRMED/REFUTED thresholds (0.85–0.90 and 0.10–0.15) instead of after, with PSYCH-appropriate wider lower band.
- **`belief_tracker.py`**: Replaced raw string literals for status values with `TraitStatus` enum, matching `bayesian_tracker.py`'s `Status` enum pattern for consistency and type safety.
- **`README.md`**: Fixed stale test count comment (191 → 385) in Unix build commands section.

### Changed
- **Version**: Bumped from 7.3.0 to 7.4.0 in Makefile, build.ps1, README, SKILL.md, and CLAUDE.md.

---

## [7.3.0] - 2026-02-24

### Fixed
- **Version strings**: Updated SKILL.md and CLAUDE.md from stale v6.9 to match actual release version.
- **`CLAUDE.md`**: Added missing `falsify` preset to belief_tracker presets documentation.

### Added
- **`README.md`**: Added Forecast Modeler CLI Tools section with usage examples, phase descriptions, and programmatic API reference. Added `forecast_modeler.py` to project structure tree.

### Changed
- **Version**: Bumped from 7.2.0 to 7.3.0 in Makefile, build.ps1, README, SKILL.md, and CLAUDE.md.

---

## [7.2.0] - 2026-02-24

### Added
- **Modeling epistemology reference** (`src/references/modeling-epistemology.md`): 179-line domain-agnostic reference distilling foundational modeling reasoning principles — epistemic incompleteness as root cause, seven modeling principles, fidelity-generalizability tradeoff table across 7 analytical traditions, probability-optimization duality table, three iteration loops (inner/middle/outer), 12 practical maxims, known blind spots catalog, and explicit vs implicit assumption framework. Generalized from ML Mindset research; complements operational references with unified conceptual grounding.

### Changed
- **`src/references/system-identification.md`**: Added Assumption Audit section — table mapping each model structure (ARX, ARMAX, State-Space, NARMAX, Neural Network) to its explicit and implicit assumptions, with cross-reference to `modeling-epistemology.md`.
- **`src/references/forecasting-science.md`**: Added Dataset Characteristics → Model Constraints decision matrix — maps data properties (short series, non-stationary, periodic, high noise, intermittent) to recommended modeling constraints and preferred models. Added `modeling-epistemology.md` cross-reference.
- **`src/SKILL.md`**: Added `modeling-epistemology.md` to Phase 0 and Phase 3 reference lists (482 lines, within 500 limit).
- **`CLAUDE.md`**: Added `modeling-epistemology.md` to repository structure listing.
- **`README.md`**: Added `modeling-epistemology.md` to Knowledge Base table, updated reference count (29→30), updated project structure reference count and SKILL.md line count.
- **Version**: Bumped from 7.1.0 to 7.2.0 in Makefile and build.ps1.

---

## [7.1.0] - 2026-02-24

### Fixed
- **`rapid_checker.py`**: Fixed KeyError in `verdict` CLI command — `v['total_flags']` changed to `v['flags']['total']` (verdict computation was correct, only display formatting was broken).
- **`README.md`**: Corrected reference document count from 28 to 29 (missing `forecasting-tools.md`), updated version badge from v6.9.0, updated test count badge from 191 to 385.

---

## [7.0.0] - 2026-02-20

### Added
- **Forecast Modeler** (`src/scripts/forecast_modeler.py`): 1,950-line forecasting model fitting & selection tool with 7-phase pipeline — Forecastability Gate (PE, naive baselines, go/no-go), Classical Fitting (Auto-ARIMA, Auto-ETS via statsmodels/pmdarima), ML Fitting (CatBoost with feature engineering: lag, rolling, calendar, Fourier), Model Comparison (MASE, RMSSE, WAPE, ME bias, FVA), Conformal Prediction (ICP, CQR), Forecast Generation, Report Summary. CLI with `fit`, `assess`, `compare`, `demo` subcommands. All dependencies optional — stdlib-only mode provides forecastability assessment + naive baselines + conformal intervals.
- **Forecasting tools reference** (`src/references/forecasting-tools.md`): 249-line usage guide with phase mapping, CLI reference, graceful degradation table, utility functions, workflow examples, cross-references.
- **103 unit tests** (`tests/test_forecast_modeler.py`): Data structures (12), numeric helpers (8), metric helpers (20), naive forecasts (8), conformal prediction (5), feature engineering (13), phase tests (6), full pipeline (5), convenience functions (3), graceful degradation (3), CLI smoke tests (2), report output (2). All passing.

### Changed
- **`src/references/forecasting-science.md`**: Enriched from 318 to 452 lines with ARIMA identification guide (ACF/PACF signature table, Box-Jenkins quick reference, SARIMA configurations), ETS model selection (taxonomy, additive vs multiplicative decision rules, damped trend), CatBoost feature engineering (checklist, leakage prevention, lag selection heuristics). Updated protocol phase integration to reference `forecast_modeler.py`.
- **`src/SKILL.md`**: Added `forecast_modeler.py fit` to Phase 3 activities and conformal prediction to Phase 5 (481→482 lines, within 500 limit).
- **`CLAUDE.md`**: Added Forecast Modeler CLI section, updated repo structure tree, script descriptions, and tool integration flow.
- **`src/references/tool-catalog.md`**: Added `forecast_modeler.py` to Tool Integration Summary table and Phase 3/5 recommendations.

---

## [6.9.0] - 2026-02-20

### Added
- **Fourier / Spectral Analyst** (`src/scripts/fourier_analyst.py`): ~1,000-line frequency-domain analysis tool with 9-phase diagnostic framework — Spectral Profile (FFT, PSD, dominant frequencies), Harmonic Analysis (THD, sidebands), Windowing Quality (leakage detection), Noise Floor (SNR, noise color), Bandwidth Analysis (rolloff, centroid), System Identification (transfer function, coherence), Spectral Anomaly Detection (baseline comparison), Time-Frequency (STFT, stationarity), System Health (vibration diagnostics, bearing fault frequencies). CLI with `analyze`, `quick`, `compare`, `demo` subcommands. Requires numpy; scipy optional for advanced features.
- **Spectral analysis reference** (`src/references/spectral-analysis.md`): Usage guide covering when to use, phase mapping, CLI reference, utility functions, domain-specific guidance (vibration, electrical, acoustic, digital), verdict interpretation.

### Changed
- **`tool-catalog.md`**: Added `fourier_analyst.py` to Tool Integration Summary table and Phase 1/3/5 recommendations. Replaced raw `scipy.signal` entry with proper tool reference.
- **`CLAUDE.md`**: Added Fourier Analyst CLI section, updated repo structure tree and tool integration flow description.
- **`README.md`**: Added Fourier / Spectral Analyst to CLI Tools section. Updated reference count (27→28), script count (7→8), and project structure tree.

---

## [6.8.0] - 2026-02-20

### Added
- **Session file I/O routing** (`session_manager.py`): New `write`, `read`, and `path` subcommands that resolve absolute paths internally. Agents pass filenames only — no path construction needed. Eliminates the class of Write tool errors caused by relative paths.
- **Evidence calibration reference** (`src/references/evidence-calibration.md`): New 220-line reference covering LR scale, LR cap rules by phase, anti-bundling enforcement, prior discipline, adversarial hypothesis requirement, disconfirmation requirement, common calibration mistakes, and tracker preset reference tables.
- **Evidence Rules section** in SKILL.md: Six enforceable rules (LR caps, anti-bundling, adversarial hypothesis, consensus cap, disconfirm-before-confirm, prior discipline) with WRONG/RIGHT examples.

### Changed
- **`src/SKILL.md`**: Rewritten to use `$SM write`/`$SM read` pattern for all session file operations. All GATE IN statements, EXIT GATE checklists, and phase activities now route through `session_manager.py` instead of Write/Read tools. Added `--base-dir` flag documentation to Session Bootstrap.
- **`session_manager.py`**: `read_pointer()` now returns absolute paths (was directory name). `new` and `resume` output `SESSION_DIR=` line for path capture. Added `.session_dir` breadcrumb file management. Path traversal protection on `write`/`read`/`path` commands (rejects `..` and absolute path components).
- **`session-memory.md`**: Updated recovery procedures and persistence rules to use `SESSION_DIR/` prefix and `$SM read` pattern.
- **`bayesian_tracker.py`**, **`belief_tracker.py`**, **`rapid_checker.py`**: Help text updated to emphasize absolute path requirement for `--file` flag.
- **`CLAUDE.md`**: Rewritten Session Manager CLI section with `$SM` shorthand and new write/read/path commands.

### Fixed
- **Relative path Write errors** (Critical): Agents using the skill consistently produced `Write(analyses/analysis_.../file.md)` errors because LLMs don't store variables — they reconstruct paths from context. Fixed by routing all file I/O through `session_manager.py write`/`read` subcommands that resolve paths internally.
- **Session manager tests**: Updated `test_valid_pointer` and `test_resume_outputs_state` to match new `read_pointer()` return type (absolute path) and new resume output format.
- **`--base-dir` flag**: Added to session_manager.py to decouple skill installation directory from analysis output directory. Prevents analyses/ from being created inside the skill directory.

---

## [6.7.0] - 2026-02-20

### Added
- **Simulation engine** (`src/scripts/simulator.py`): 1,400+ line simulation engine supporting five paradigms — System Dynamics (SD), Monte Carlo (MC), Agent-Based Modeling (ABM), Discrete-Event Simulation (DES), and Sensitivity Analysis (Morris/Sobol/OAT). Includes validation bridge for feeding results back to Phase 5. Adapted from standalone epistemic-simulator skill with common.py integration and CLI error handling. Requires numpy, scipy, matplotlib.
- **Simulation guide reference** (`src/references/simulation-guide.md`): 347-line combined reference covering domain fit gate, paradigm descriptions, archetype-to-mode mapping, model conversion recipes (ARX/state-space/NARMAX), validation bridge protocol, convergence diagnostics, visualization outputs, and state block extension format.
- **41 new tests** (`tests/test_simulator.py`): Distribution sampling (8), input functions (6), JSON serialization (4), SD linear simulation (4), MC single run (3), topology building (4), validation bridge (3), CLI parser (6), sensitivity OAT fallback (1). All guarded with `@unittest.skipUnless(HAS_NUMPY)`.
- **Cross-references** added to 6 existing reference files: `tool-catalog.md`, `system-identification.md`, `compositional-synthesis.md`, `validation-checklist.md`, `timeseries-review.md`, `forecasting-science.md`.

### Changed
- **`src/SKILL.md`**: Added `Simulation | simulator.py (SD, MC, ABM, DES, sensitivity)` row to Tool Integration table (496→497 lines, within 500 limit)
- **`CLAUDE.md`**: Added simulator.py to repo structure tree and scripts listing; added Simulator CLI section with usage examples
- **`README.md`**: Added Simulator section to CLI Tools; added simulation-guide.md to Knowledge Base; updated reference count (24→25), script count (6→7), and project structure tree
- **`tool-catalog.md`**: Updated Phase 4 Simulation tool from `MATLAB/Simulink, custom` to `simulator.py`

---

## [6.6.3] - 2026-02-19

### Added
- **Forecasting science reference** (`src/references/forecasting-science.md`): 294-line distilled reference covering forecastability assessment (CoV critique, Permutation Entropy, FVA, naive benchmarks), model selection hierarchy (decision tree, ranked table, ARIMA vs ETS comparison), forecast error metrics framework (MAPE/sMAPE avoidance rationale, metric selection decision framework, minimum recommended set), conformal prediction overview (ICP/CQR/EnbPI with when-to-use table, CQR algorithm), and common pitfalls table
- **Permutation Entropy in Phase 4** (`ts_reviewer.py`): Proper forecastability measure using ordinal-pattern complexity. Adaptive order (D=3/4/5 based on series length). Thresholds: >0.95 WARN HIGH (effectively random), >0.85 WARN MEDIUM, <0.5 PASS (strong structure). Pure stdlib implementation
- **Forecast Value Added (FVA) in Phase 6** (`ts_reviewer.py`): Reports percentage improvement over best baseline when model predictions supplied. FVA <0% FAIL, 0-10% WARN, >10% PASS
- **New metric helper functions** (`ts_reviewer.py`): `_rmsse` (M5 competition metric), `_wape` (volume-weighted percentage error), `_me_bias` (forecast bias), `_pinball_loss` (quantile loss), `_fva` (forecast value added), `_permutation_entropy` (ordinal complexity). All pure stdlib
- **CQR intervals** (`ts_reviewer.py`): `cqr_intervals()` for Conformalized Quantile Regression — adaptive-width prediction intervals from pre-computed quantile predictions
- **Extended `compare_models()`** (`ts_reviewer.py`): Now includes `wape` and `me_bias` keys in output
- **27 new tests** (`tests/test_ts_reviewer.py`): `TestPermutationEntropy` (5), `TestNewMetrics` (14), `TestCQRIntervals` (3), `TestPhase4PermutationEntropy` (2), `TestPhase6FVA` (3), `TestCompareModelsExtended` (1). Total: 61 tests

### Changed
- **Version**: Bumped from 6.6.2 to 6.6.3 in Makefile and build.ps1
- **`timeseries-review.md`**: Updated Phase 4/6 table rows, added New Capabilities section documenting PE, FVA, new metrics, and CQR; added `forecasting-science.md` to Cross-References
- **`CLAUDE.md`**: Added `forecasting-science.md` to repo structure tree; added brief note about new metric functions and CQR in ts_reviewer CLI section
- **`README.md`**: Added `forecasting-science.md` to project structure; updated reference count to 24; bumped version to v6.6.3

---

## [6.6.2] - 2026-02-19

### Added
- **New test files**:
  - `tests/test_session_manager.py`: 17 tests covering all CLI commands (new, resume, status, close, list), pointer file handling, gitignore management
  - `tests/test_ts_reviewer.py`: 24 tests covering constructor, `_to_floats`, phases 1-6, helper functions (`_mean`, `_std`, `_diff`, `_r_squared`, `_mae`, `_rmse`), `full_review()`, `walk_forward_split()`, `quick_review()`, `ReviewReport.to_dict()`
- **New tests in existing files** (21 tests):
  - `test_common.py`: 3 tests for `load_json()` on empty, malformed, and whitespace-only files
  - `test_bayesian_tracker.py`: 9 tests — KILLED hypothesis guard, SKEPTICAL/DOUBTFUL/REJECT/CREDIBLE verdict paths with varying flag counts and category distributions
  - `test_belief_tracker.py`: 11 tests — REFUTED trait guard, OCEAN/Dark Triad/MICE empty and populated profiles, DT risk calculation (no traits, high-all, low-polarity inversion)
- **Threshold Bands documentation table** in CLAUDE.md explaining intentional threshold differences between `bayesian_tracker` (system analysis) and `belief_tracker` (PSYCH tier)

### Fixed
- **`load_json()` crash on malformed files** (Bug): `common.py` now reads file content then parses with `json.loads()`, catching `JSONDecodeError`/`ValueError`. Empty, whitespace-only, and malformed JSON files return `None` instead of raising an exception. All callers already handle `None`.
- **KILLED/REFUTED hypothesis resurrection** (Bug): `bayesian_tracker.py` and `belief_tracker.py` now raise `ValueError` with a descriptive message when attempting to update a KILLED or REFUTED hypothesis. Both CLIs already catch `ValueError` in their `main()` functions, so users see a clean error.

### Changed
- **Version**: Bumped from 6.6.0 to 6.6.2 in Makefile and build.ps1
- Added inline threshold comments in `bayesian_tracker.py` and `belief_tracker.py` pointing to CLAUDE.md cross-reference

---

## [6.6.1] - 2026-02-18

### Added
- **Shared utilities module** (`scripts/common.py`):
  - `bayesian_update()` with division-by-zero protection via epsilon clamping
  - `load_json()` / `save_json()` with platform-aware file locking (`fcntl`/`msvcrt`)
  - `clamp_probability()` and `POSTERIOR_EPSILON` constant
- **RAPID Assessment Reference** (`references/rapid-assessment.md`):
  - Consolidated 5-step RAPID workflow, verdict logic, CLI quick reference
  - Cross-reference to SKILL.md Phase 0.5
- **Unit test suite** (`tests/`):
  - `test_common.py`: clamp, Bayesian math edge cases, JSON I/O roundtrip (13 tests)
  - `test_bayesian_tracker.py`: add/update/remove, monotonic IDs, verdict, 50x confirm (11 tests)
  - `test_belief_tracker.py`: traits, baselines, deviations, 50x indicator (9 tests)
  - `test_rapid_checker.py`: session, coherence, flags, calibration, verdict (14 tests)
- **build.ps1 parity**: Added `all`, `install`, `package-tar`, and `test` commands

### Fixed
- **Division-by-zero in Bayesian update** (Critical): Repeated strong confirms/disconfirms
  could push posterior to exactly 0.0 or 1.0, causing `ZeroDivisionError` on next update.
  Now uses epsilon-clamped math via `common.bayesian_update()`.
- **No CLI error handling**: Raw tracebacks on invalid input (e.g., bad hypothesis ID).
  All three tracker scripts now catch `KeyError`/`ValueError`/`RuntimeError` and print
  clean error messages to stderr with exit code 1.
- **No concurrency safety**: Concurrent processes could corrupt JSON state files.
  All load/save operations now use platform-aware file locking.

### Changed
- `bayesian_tracker.py`, `belief_tracker.py`, `rapid_checker.py`: Refactored to use
  shared `common.py` for Bayesian math (~170 lines deduplication) and locked JSON I/O
- `Makefile` `test` target: Now runs `python -m unittest discover` before smoke tests
- `SKILL.md`: Added `references/rapid-assessment.md` cross-reference in Phase 0.5
- `CLAUDE.md`: Added `common.py` and `rapid-assessment.md` to repository structure

---

## [6.6.0] - 2026-02-18

### Added
- **Time-Series Signal Review Tool** (`scripts/ts_reviewer.py`):
  - 10-phase systematic evaluation: coherence, data quality, stationarity (ADF/KPSS),
    forecastability (ACF, entropy, SNR), decomposition (STL), baseline benchmarks,
    overfitting screen, residual diagnostics, uncertainty calibration, regime analysis
  - CLI with `review`, `quick`, and `demo` subcommands
  - Graceful degradation: works with pure stdlib, enhanced with numpy/scipy/statsmodels
  - Verdicts: PASS/WARN/FAIL/REJECT with severity levels
  - Convenience functions: `quick_review()`, `compare_models()`, `walk_forward_split()`, `conformal_intervals()`
- **Time-Series Review Reference** (`references/timeseries-review.md`):
  - Phase mapping between protocol phases and ts_reviewer phases
  - Verdict interpretation guide and escalation rules
  - CLI quick reference and programmatic usage examples
- **New Calibration Domain** `time_series` in `domains.json`:
  - MASE, R2, coverage_80, MAPE bounds

### Changed
- **SKILL.md**: Added ts_reviewer.py to Phase 3 activities, Tool Integration table, and Tracker Commands
- **Makefile**: `lint` and `test` targets now loop over all scripts via `$(SCRIPT_FILES)` wildcard
- **build.ps1**: `Invoke-Lint` now loops over all `src/scripts/*.py` files
- **Version**: Bumped to 6.6.0 across all build scripts and SKILL.md

### Fixed
- 4 syntax errors in ts_reviewer.py source (clipping check, phase 7 comment, normality branch, regime recommendation)

---

## [6.5.0] - 2026-02-18

### Added
- **Financial Forecasting Validation Framework** (`references/financial-validation.md`):
  - Financial disqualifiers (Tier 0): Martingale baseline, price-level metrics, reconstructed R² trick
  - Stationarity requirements: ADF/KPSS testing, differencing rules, level-based metrics prohibition
  - Forecastability assessment: autocorrelation analysis, entropy, noise floor, predictability decay
  - Class imbalance in financial context: resampling delusion, "always predict up" test, MCC, threshold calibration
  - Regime testing: bull/bear/high-vol/low-vol/structural break decomposition
  - Economic significance: transaction cost framework, capacity constraints, backtesting illusion
  - Financial statistical tests: Diebold-Mariano, Reality Check/SPA, multiple testing correction
  - The 5% problem: component weight breakdown for financial claim validation
  - Consolidated financial validation checklist (Tier 0–4)
- **New Calibration Metrics** for `financial_prediction` domain:
  - `annual_alpha`: [0.30, 0.02, 0.08, 0.15] — after risk adjustment
  - `mcc`: [0.60, 0.02, 0.10, 0.25] — direction prediction
  - `max_drawdown`: [0.05, 0.50, 0.20, 0.10] — lower is better
  - `r2_returns_train`: [0.05, 0.80, 0.50, 0.30] — lower is better; high = overfit

### Changed
- `rapid_checker.py`: Extended lower-is-better metric list with `max_drawdown` and `r2_returns_train`
- `domain-calibration.md`: Added 4 new rows to Financial Prediction table with cross-reference
- `domains.json`: Added 4 new metrics to `financial_prediction` domain

---

## [6.4.0] - 2026-01-29

### Added
- **SKILL.md Clarity Improvements**:
  - Fidelity Levels table (L1-L5: DO → HOW → WHY → PARAMETERS → REPLICATE) with test criteria
  - Tier Selection from Questionnaire mapping table
  - Archetype Signatures table with vulnerabilities and examples
  - Emergence mismatch formula definition
  - LITE tier specifics (abbreviated setup, ≥5 probes)
  - Falsification loop structure with iteration count and stopping criterion
  - Executable recursive decomposition with defined constants
  - RAPID → Next Tier decision tree
  - RASP framework mention (Revenge, Addiction, Sex, Power)
- **New Decision Trees**:
  - "When to Stop?" with quantifiable criteria
  - "RAPID → Next Tier?" escalation guidance
- **OSINT Tools** in setup-techniques.md: nmap, whois, Shodan, theHarvester, SpiderFoot, Amass
- **Domain Calibration Config**: Externalized to `config/domains.json` for user customization
- **Hypothesis Removal**: `bayesian_tracker.py remove` command for deleting hypotheses
- **Table of Contents**: Added to remaining 8 reference files >100 lines

### Changed
- **CONFIRMED Threshold**: Unified to 0.90 across all scripts (was 0.95 in bayesian_tracker)
- **Tool Catalog Cleanup**: Removed commercial/niche tools for accessibility
  - Removed: IDA Pro, Binary Ninja, MATLAB SI Toolbox, Intel PIN, DynamoRIO, KLEE, AFLNet, Cuckoo, Any.Run, binwalk
  - Kept: Ghidra, Frida, Unicorn, angr, SysIdentPy, SIPPY, Netzob, Wireshark, Scapy, AFL++, libFuzzer, SALib
- **Exit Codes**: All scripts now return proper exit codes (sys.exit(1)) on CLI errors

### Fixed
- Undefined terms in SKILL.md (Fidelity Target, Question Pyramid levels, emergence mismatch)
- Missing questionnaire → tier mapping
- Incomplete decision trees (stopping criteria, recursive decomposition)
- LITE tier had no phase-specific guidance

---

## [6.3.0] - 2026-01-28

### Changed
- **SKILL.md Refactored**: Reduced from 1,304 to 417 lines (68% reduction) using progressive disclosure
  - Procedural instructions remain in SKILL.md
  - Detailed reference material extracted to `references/` files
  - All decision trees and stop conditions preserved

### Added
- **New Reference Documents**:
  - `references/psych-tier-protocol.md`: Complete PSYCH tier protocol (extracted from SKILL.md)
  - `references/tool-catalog.md`: Tool recommendations by phase and domain
  - `references/adversarial-heuristics.md`: Adversarial posture levels L0-L4, anti-analysis bypass techniques
- **Table of Contents**: Added to 8 reference files >100 lines for navigation
  - cognitive-traps.md, profile-synthesis.md, motive-analysis.md, elicitation-techniques.md
  - causal-techniques.md, system-identification.md, linguistic-markers.md, archetype-mapping.md
- **CLAUDE.md**: Added v6.3 refactoring notes with known limitations

### Notes
- PSYCH tier users should load `references/psych-tier-protocol.md` for complete protocol
- Tool-specific guidance now in `references/tool-catalog.md`
- Adversarial bypass details now in `references/adversarial-heuristics.md`

---

## [6.2.0] - 2026-01-28

### Added
- **PSYCH Tier**: New tier for psychological profiling and behavioral analysis (1-4h)
  - Phase 0-P: Context & Frame (relationship dynamics, objectives)
  - Phase 1-P: Baseline Calibration (linguistic, emotional, timing patterns)
  - Phase 2-P: Stimulus-Response Mapping (elicitation probes, stress testing)
  - Phase 3-P: Structural Identification (OCEAN, Dark Triad, cognitive distortions)
  - Phase 4-P: Motive Synthesis (MICE/RASP, drive matrix, archetype)
  - Phase 5-P: Validation & Prediction (behavioral predictions, interaction strategy)
  - PSYCH-specific state block format with Archetype, Rapport, and Stress indicators
  - PSYCH auto-pilot questionnaire for persona analysis
  - Psychological axioms (Baseline is God, Rational Actor Fallacy, Projection Trap, etc.)
  - Cross-domain integration for combined system + human analysis
- **New Reference Documents**:
  - `references/archetype-mapping.md`: Big Five (OCEAN), Dark Triad, MICE/RASP frameworks with behavioral indicators
  - `references/linguistic-markers.md`: Text analysis patterns, distancing language, deception markers, pronoun analytics
  - `references/elicitation-techniques.md`: Probing methods (Columbo, Bracketing, Silence, Challenge, etc.)
  - `references/motive-analysis.md`: MICE/RASP motivation frameworks with detailed indicators and predictions
  - `references/profile-synthesis.md`: Trait composition, confidence propagation, profile templates, common archetypes
- **Enhanced Cognitive Traps** (`references/cognitive-traps.md`):
  - Trap 9: Counter-Transference (analyst projects feelings onto subject)
  - Trap 10: Fundamental Attribution Error (character vs. situation)
  - Trap 11: Mirror-Imaging (Psychological) (assuming shared values/rationality)
  - Trap 12: Halo/Horn Effect (one trait colors all assessment)
  - Trap 13: Barnum Effect (accepting vague descriptions as accurate)
  - Trap 14: Narrative Fallacy (smoothing over contradictions)
  - Trap 15: Projection (attributing own traits to subject)
  - Psychological Analysis Debiasing Checklist
- **Belief Tracker Tool** (`scripts/belief_tracker.py`):
  - Psychological trait tracking with Bayesian inference
  - Big Five (OCEAN), Dark Triad, and MICE motivation tracking
  - Baseline observation management
  - Deviation recording from baseline
  - Unified psychological profile generation
  - Dark Triad risk score calculation
  - CLI with `subject`, `add`, `update`, `baseline`, `deviation`, `profile`, `report` commands
  - Likelihood ratio presets for behavioral evidence

### Changed
- Tier table now includes PSYCH tier with 0-P→5-P phases
- CLAUDE.md updated to v6.2 with PSYCH tier documentation
- README.md updated with PSYCH tier and belief_tracker documentation

### Ethical Considerations
- Added ethical constraints section in SKILL.md:
  - No clinical diagnosis (observable traits only)
  - Cultural calibration requirement
  - Consent awareness documentation
  - Defensive use emphasis

## [6.1.0] - 2026-01-28

### Added
- **RAPID Tier**: New tier for quick claim validation (<30min)
  - Phase 0.5: Coherence Screening with instant reject conditions
  - RAPID workflow: 5-step assessment (coherence, verifiability, red flags, calibration, verdict)
  - Extended state block format for validation phases
- **New Reference Documents**:
  - `references/red-flags.md`: Comprehensive red flag catalog for invalid claims
  - `references/coherence-checks.md`: Quick coherence validation (60-second filter)
  - `references/domain-calibration.md`: Plausibility bounds by domain
  - `references/validation-checklist.md`: Consolidated validation requirements
- **Enhanced Phase 5: Validation**:
  - Model validity checks (overfitting, leakage detection)
  - Baseline comparison requirements
  - Domain calibration
  - Practical significance
  - Uncertainty quantification
  - Reproducibility assessment
- **Cognitive Trap**: Added Trap 8 "Tool Worship (Cargo-Cult)" to cognitive-traps.md
- **Bayesian Tracker Extensions** (`scripts/bayesian_tracker.py`):
  - Red flag tracking (`flag add`, `flag report`, `flag count`)
  - Coherence check tracking (`coherence`, `coherence-report`)
  - Verdict generation (`verdict`, `verdict --full`)
- **RAPID Checker Tool** (`scripts/rapid_checker.py`):
  - Standalone 10-minute assessment CLI
  - Built-in domain calibration for ML, finance, engineering, medical
  - Automatic verdict computation

### Changed
- Tier table now includes RAPID tier with 0.5→5 phases
- Phase 5 expanded with subsections 5.3-5.10 for comprehensive validation
- Output artifacts list updated to include Coherence Screening Report

## [6.0.0] - 2025-01-28

### Added
- **Core Protocol (SKILL.md)**: Complete 6-phase Epistemic Deconstruction Protocol
  - Phase 0: Setup & Frame (context definition, question pyramid, hypothesis seeding)
  - Phase 1: Boundary Mapping (I/O enumeration, probe signals, stimulus-response database)
  - Phase 2: Causal Analysis (static/dynamic analysis, sensitivity analysis, causal graphs)
  - Phase 3: Parametric Identification (model structure selection, parameter estimation)
  - Phase 4: Model Synthesis & Emergence (compositional synthesis, uncertainty propagation)
  - Phase 5: Validation & Adversarial (validation hierarchy, residual diagnostics, attack surface)
- **Tier System**: LITE, STANDARD, and COMPREHENSIVE analysis tiers with clear triggers
- **State Block Protocol**: Mandatory state tracking for conversation continuity
- **Auto-Pilot Mode**: Guided questionnaire for new users
- **Bayesian Hypothesis Tracker** (`scripts/bayesian_tracker.py`): CLI tool for proper Bayesian inference
  - Hypothesis management (add, update, compare)
  - Likelihood ratio presets for common evidence types
  - Report generation with evidence trails
- **Reference Knowledge Base** (`references/`):
  - `boundary-probing.md`: I/O characterization techniques
  - `causal-techniques.md`: Causality establishment methods
  - `cognitive-traps.md`: Analytical bias countermeasures
  - `compositional-synthesis.md`: Sub-model composition mathematics
  - `setup-techniques.md`: Phase 0 framing procedures
  - `system-identification.md`: Parametric estimation algorithms
  - `tools-sensitivity.md`: Binary RE tools and sensitivity analysis
- **CLAUDE.md**: AI assistant guidance for working with the codebase
- **Decision Trees**: Model structure selection, stopping criteria, decomposition triggers
- **Recursive Decomposition**: For COMPREHENSIVE tier with partitioning strategies

### Features
- Proper Bayesian updating with likelihood ratios
- Cognitive trap awareness and countermeasures
- Emergence detection and handling
- Adversarial posture classification (L0-L4)
- Multi-domain system support (software, hardware, biological, organizational)
