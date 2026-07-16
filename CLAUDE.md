# CLAUDE.md

This file provides guidance for Claude (AI) when working with the Epistemic Deconstructor codebase.

## Project Purpose

**Epistemic Deconstructor v7.16.5** is a systematic framework for AI-assisted reverse engineering of unknown systems using scientific methodology. It transforms epistemic uncertainty into predictive control through principled experimentation, compositional modeling, and Bayesian inference.

Use cases include:
- Black-box analysis of unknown systems (software, hardware, biological, organizational)
- Competitive intelligence and system interrogation
- Security analysis and attack surface mapping
- Forensics and root cause analysis
- Building predictive models from observations
- **Psychological profiling and behavioral analysis (PSYCH tier)**

## Repository Structure

```
epistemic-deconstructor/
├── README.md                # User documentation
├── LICENSE                  # GNU GPLv3
├── CHANGELOG.md             # Version history
├── CLAUDE.md                # This file
├── Makefile                 # Unix/Linux build script
├── build.ps1                # Windows PowerShell build script
├── tests/                   # 880 unit tests (pytest)
│   ├── test_common.py
│   ├── test_bayesian_tracker.py
│   ├── test_belief_tracker.py
│   ├── test_rapid_checker.py
│   ├── test_session_manager.py
│   ├── test_ts_reviewer.py
│   ├── test_fourier_analyst.py
│   ├── test_forecast_modeler.py
│   ├── test_parametric_identifier.py
│   ├── test_scope_auditor.py
│   ├── test_abductive_engine.py
│   ├── test_domain_orienter.py
│   ├── test_phase_0_3_integration.py
│   ├── test_phase_0_7_integration.py
│   ├── test_phase_1_5_integration.py
│   ├── test_simulator.py
│   ├── test_mermaid_render.py
│   ├── test_phase_gate.py
│   ├── test_concurrency.py
│   ├── test_agent_authority.py
│   ├── test_doc_consistency.py
│   ├── test_doc_fences.py
│   └── test_test_count.py
├── plans/                   # Iterative-planner state (gitignored)
├── docs/                    # Design documentation
│   └── subagents.md             # Sub-agent reference documentation
└── src/
    ├── SKILL.md                 # Core protocol (6-phase methodology) - the main instruction set
    ├── config/
    │   ├── domains.json         # Domain calibration data
    │   ├── archetypes.json      # Scope-interrogation archetype library (M2) + Phase 1.5 trace_signatures (AR)
    │   └── trace_catalog.json   # Phase 1.5 abductive trace catalog (TI operator)
    ├── scripts/
    │   ├── common.py            # Shared utilities (Bayesian math, JSON I/O with locking)
    │   ├── session_manager.py   # Python CLI for analysis session management
    │   ├── phase_gate.py        # Python CLI for tier-blind phase EXIT-GATE checks (stdlib-only)
    │   ├── bayesian_tracker.py  # Python CLI for Bayesian hypothesis + flag tracking
    │   ├── belief_tracker.py    # Python CLI for PSYCH tier trait tracking
    │   ├── rapid_checker.py     # Python CLI for RAPID tier assessments
    │   ├── ts_reviewer.py       # Python CLI for time-series signal diagnostics
    │   ├── fourier_analyst.py   # Python CLI for frequency-domain spectral analysis
    │   ├── forecast_modeler.py  # Python CLI for forecasting model fitting & selection
    │   ├── parametric_identifier.py # Python CLI for structural system ID (ARX/ARMAX/NARMAX + bootstrap UQ)
    │   ├── domain_orienter.py   # Python CLI for Phase 0.3 domain orientation (TE/TG/MM/AM/CS operators)
    │   ├── scope_auditor.py     # Python CLI for Phase 0.7 scope interrogation (M1-M4 mechanisms)
    │   ├── abductive_engine.py  # Python CLI for Phase 1.5 abductive expansion (TI/AA/SA/AR/IC operators)
    │   ├── simulator.py         # Python CLI for simulation (SD, MC, ABM, DES, sensitivity)
    │   └── mermaid_render.py    # Stdlib-only deterministic Mermaid emitter (FSM / inference chains / coverage / causal / adjacency)
    ├── agents/                  # Sub-agent definitions (Claude Code)
    │   ├── ed-orchestrator.md  # Main orchestrator (opus) — phase FSM, delegation
    │   ├── ed-session-clerk.md           # Filesystem I/O handler (haiku, background)
    │   ├── ed-hypothesis-engine.md       # Bayesian tracking + evidence rules (sonnet)
    │   ├── ed-cognitive-auditor.md       # Bias/trap detection (sonnet, background)
    │   ├── ed-rapid-screener.md          # Phase 0.5 RAPID screening (sonnet)
    │   ├── ed-boundary-mapper.md         # Phase 1 I/O mapping (sonnet)
    │   ├── ed-causal-analyst.md          # Phase 2 causal graphs + falsification (opus)
    │   ├── ed-parametric-id.md           # Phase 3 model fitting (sonnet)
    │   ├── ed-model-synthesizer.md       # Phase 4 composition + simulation (sonnet)
    │   ├── ed-validator.md               # Phase 5 validation + report (opus)
    │   ├── ed-psych-profiler.md          # PSYCH tier behavioral analysis (opus)
    │   ├── ed-domain-orienter.md         # Phase 0.3 domain orientation (sonnet, synchronous; uses WebFetch)
    │   ├── ed-scope-auditor.md           # Phase 0.7 scope interrogation (sonnet, background)
    │   ├── ed-abductive-engine.md        # Phase 1.5 abductive expansion (sonnet, background)
    │   └── ed-research-scout.md          # Background web research (haiku)
    └── references/              # Knowledge base documents
        # System Analysis References
        ├── boundary-probing.md       # I/O characterization techniques
        ├── causal-techniques.md      # Methods for establishing causality
        ├── cognitive-traps.md        # Countermeasures for analytical bias (incl. psychological)
        ├── coherence-checks.md       # Quick coherence validation (60-second filter)
        ├── compositional-synthesis.md # Math for combining sub-models
        ├── domain-calibration.md     # Plausibility bounds by domain
        ├── red-flags.md              # Red flag catalog for invalid claims
        ├── setup-techniques.md       # Phase 0 framing procedures
        ├── system-identification.md  # Parametric estimation algorithms
        ├── mermaid-conventions.md    # Mermaid diagram conventions: default-vs-exception policy + diagram-type map
        ├── tools-sensitivity.md      # Binary tools & sensitivity analysis
        ├── validation-checklist.md   # Consolidated validation requirements
        ├── tool-catalog.md           # Tool recommendations by phase/domain
        ├── abductive-reasoning.md    # Phase 1.5 abductive expansion (TI/AA/SA/AR/IC operators, coverage-gated promotion)
        ├── adversarial-heuristics.md # Anti-analysis bypass, posture levels
        ├── financial-validation.md  # Financial forecasting validation framework
        ├── forecasting-science.md   # Forecasting science: PE, FVA, metrics, conformal prediction
        ├── forecasting-tools.md     # Forecasting tools usage guide (forecast_modeler.py)
        ├── rapid-assessment.md      # Consolidated RAPID tier workflow reference
        ├── timeseries-review.md     # Time-series signal review guide
        ├── session-memory.md        # Filesystem memory protocol for analysis sessions
        ├── simulation-guide.md      # Simulation paradigms, model conversion, validation bridge
        ├── distributions-guide.md   # Distribution selection guide for MC/DES/ABM simulation
        ├── evidence-calibration.md  # LR caps, anti-bundling, prior discipline, calibration rules
        ├── decision-trees.md        # Model selection, stopping, decomposition, tier escalation
        ├── engineering-laws.md      # Engineering design laws (Akin's Laws) mapped to protocol phases
        ├── spectral-analysis.md     # Frequency-domain spectral analysis guide
        ├── modeling-epistemology.md  # Foundational modeling reasoning principles
        ├── multi-pass-protocol.md   # Multi-pass phase reopening rules and workflow
        ├── phase-protocols.md       # Per-phase Activities + EXIT GATE checklists + File Write Matrix + Gate Check Procedure (v7.15.7+)
        ├── domain-orientation.md    # Phase 0.3 domain orientation (TE/TG/MM/AM/CS operators, self-assessment, worked example)
        ├── scope-interrogation.md   # Phase 0.7 scope interrogation (H_S pair, M1-M4 mechanisms)
        ├── archetype-accomplices.md # Archetype-to-accomplice library (M2 mechanism data)
        ├── plain-language-layer.md  # Plain-language "Bottom Line" register for user-facing output
        ├── warning-reception.md     # Observer-side warning-reception dynamics (break-point taxonomy, counterfeit-witness discriminators)
        # PSYCH Tier References
        ├── psych-tier-protocol.md    # Complete PSYCH tier protocol (extracted from SKILL.md)
        ├── archetype-mapping.md      # OCEAN, Dark Triad, MICE/RASP frameworks
        ├── linguistic-markers.md     # Text analysis, deception markers
        ├── elicitation-techniques.md # Probing methods for trait discovery
        ├── motive-analysis.md        # MICE/RASP motivation frameworks
        └── profile-synthesis.md      # Combining traits into unified models
```

## Working with This Codebase

### File Modification Guidelines

- **src/SKILL.md** is the core protocol. Changes here affect all analysis behavior.
- **src/references/** files provide domain-specific knowledge. Add new reference files for new domains.
- **src/scripts/** contains executable tooling:
  - `session_manager.py` for analysis session management. Subcommands: `new`/`resume`/`status`/`close`/`list`/`reopen`/`skip`/`advance`/`gate-check`/`set-phase`/`write`/`read`/`path`/`declare`/`diagram`.
  - `bayesian_tracker.py` for system analysis hypothesis tracking. Subcommands: `add`/`update`/`compare`/`remove`/`rename`/`report`/`flag`/`coherence`/`coherence-report`/`verdict`/`validate-priors`.
  - `belief_tracker.py` for psychological trait tracking
  - `rapid_checker.py` for RAPID tier assessments
  - `ts_reviewer.py` for time-series signal diagnostics, forecasting validation, and conformal prediction intervals. Also provides `compare_models()`, `walk_forward_split()`, `conformal_intervals()`, and `cqr_intervals()` as standalone functions for Phase 3/5.
  - `forecast_modeler.py` for forecasting model fitting (ARIMA, ETS, CatBoost) with conformal prediction intervals. Phase 3 uses `fit`/`compare` for forecasting model selection; Phase 5 uses conformal prediction for calibrated intervals. Provides `auto_forecast()`, `compare_forecasters()`, `conformal_forecast()` as standalone functions.
  - `parametric_identifier.py` for **structural** system identification (ARX, ARMAX, NARMAX) with OLS/QR estimation, FROLS term selection, AIC/BIC/AICc/FPE across families, residual bootstrap parameter CIs, integrated Ljung-Box whiteness, and walk-forward CV. Phase 3 uses `assess`/`fit`/`compare` when the deliverable is structure + parameters (not forecasts). Fitted ARX output feeds `simulator.py` directly via `to_simulator_format()`.
  - `domain_orienter.py` for Phase 0.3 domain orientation (TE/TG/MM/AM/CS operators). Stdlib-only. Conditional on `domain_familiarity ∈ {low, unknown}` declared in `analysis_plan.md`; mandatory in COMPREHENSIVE; skipped via `$SM skip 0.3 "<reason>"` for `high`. Enforces provenance discipline (same taxonomy as Phase 1.5) and hard caps `llm_parametric` definitions at confidence 0.60. Metric promotion blocks `llm_parametric` source. Canonical sources must be verified (HTTP 200 or citation) before promotion. DECISION anchors D-006/D-007/D-008 guard the constants.
  - `scope_auditor.py` for Phase 0.7 scope interrogation (M1-M4 mechanisms). Stdlib-only with optional scipy. New in v7.15.0: `enumerate --glossary <path>` flag prints a glossary-alignment advisory when a Phase 0.3 `domain_orientation.json` is supplied.
  - `abductive_engine.py` for Phase 1.5 abductive expansion (TI/AA/SA/AR/IC operators). Stdlib-only. Enforces provenance discipline (source ∈ {library, llm_parametric, analyst, chain_derived}) and hard caps `llm_parametric` candidates at prior 0.30 and chain LR 2.0. Coverage-weighted promotion gate (`coverage_score ≥ 0.30`) blocks low-coverage candidates from entering `hypotheses.json` — the primary mitigation against hypothesis explosion. Subcommands: `start`/`invert`/`absence-audit`/`close-prediction`/`surplus-audit`/`analogize`/`chain`/`catalog`/`candidates`/`report`/`gate`/`chain-diagram`/`coverage-diagram`.
  - `simulator.py` for forward simulation (SD, MC, ABM, DES, sensitivity). Phase 4 uses archetype-to-paradigm mapping from `simulation-guide.md`. Consumes ARX dicts from `parametric_identifier.py`. Phase 5 uses `bridge` command to validate predictions.
- **Tool integration flow**: **Phase 0.3 (domain_orienter TE/TG/MM/AM/CS — conditional on `domain_familiarity`; produces session glossary/metrics/sources for downstream consumption)** → Phase 0.7 (scope_auditor M1-M4 scope expansion; M2 enumerate consumes Phase 0.3 glossary via `--glossary` flag) → Phase 1 (fourier_analyst spectral profiling + ts_reviewer signal quality) → **Phase 1.5 (abductive_engine TI/AA/SA/AR/IC interior hypothesis generation, coverage-gated promotion to hypotheses.json)** → Phase 3 (ts_reviewer diagnostics + fourier_analyst transfer functions + **parametric_identifier for ARX/ARMAX/NARMAX structural fit** + forecast_modeler for forecasting fit) → Phase 4 (simulator forward projection consuming parametric_identifier output) → Phase 5 (forecast_modeler conformal prediction + ts_reviewer residual validation + fourier_analyst spectral anomaly + simulator bridge + scope_auditor residual-match for scope completeness). See `references/domain-orientation.md`, `references/scope-interrogation.md`, `references/abductive-reasoning.md`, `references/timeseries-review.md`, `references/forecasting-tools.md`, `references/system-identification.md`, and `references/spectral-analysis.md` for utility function usage per phase.

### Agent Install

Per `docs/subagents.md:160-166`, Claude Code's sub-agent loader scans:
`.claude/agents/`, `~/.claude/agents/`, `--agents` CLI JSON, and plugin
directories. **Skill subdirectories (`~/.claude/skills/<name>/agents/`) are NOT
scanned.** `make sync-skill` (Unix) and `build.ps1 sync-skill` (Windows) install
the 15 agents in `src/agents/*.md` to `~/.claude/agents/` so they are
discoverable. The `unsync-agents` target removes them. Never create a
`.claude/` directory inside this repository — agent files live in the global
`~/.claude/agents/` install only. (Claude Code may auto-create a local
`.claude/` for session state; that directory is `.gitignore`d at line 216 and
must never be committed.)

### Tech Stack

- Python 3.8+ (for tracker scripts; runtime guard in `src/scripts/common.py` warns on 3.7 and hard-errors below 3.7 — D-002 of plan_2026-05-25_c0b0049a)
- Markdown documentation
- PowerShell/Make for build scripts

### Build Commands

```bash
# Windows (PowerShell)
.\build.ps1 package          # Create zip package
.\build.ps1 package-combined # Create single-file skill
.\build.ps1 validate         # Validate structure
.\build.ps1 clean            # Clean artifacts

# Unix/Linux/macOS
make package                 # Create zip package
make package-combined        # Create single-file skill
make validate                # Validate structure
make clean                   # Clean artifacts
```

### Recording Findings

Record version results and analysis in `analyses/FINDINGS.md` (created automatically by the session manager).

### Adding New Reference Material

Reference files should follow this pattern:
1. Clear section headers
2. Tables for quick reference
3. Code snippets where applicable
4. Cross-references to other reference files
5. **Files >100 lines MUST have a Table of Contents** at the top
6. **Fenced code blocks MUST be balanced** (even count of triple-backtick fence lines). Mermaid diagrams follow `references/mermaid-conventions.md`, which makes Mermaid the default representation for dynamic systems / graphs / flows / state machines (subject to its EXCEPTION list: transfer-function math, dense numeric tables, stock-flow SD equations, and directory trees keep their current notation).

