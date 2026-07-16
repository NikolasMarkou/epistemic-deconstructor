# Epistemic Deconstructor

[![License](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-v7.16.5-green.svg)](CHANGELOG.md)
[![Tests](https://img.shields.io/badge/Tests-880%20passing-brightgreen.svg)](tests/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](src/scripts/)
[![Sponsored by Electi](https://img.shields.io/badge/Sponsored%20by-Electi-red.svg)](https://www.electiconsulting.com)

**Turn "I don't know how this works" into a quantified, validated model of how it works.**

Epistemic Deconstructor is a [Claude](https://claude.com/claude-code) skill. You hand Claude a black box; it runs a structured scientific investigation — Bayesian hypothesis tracking, falsification tests, parametric model fitting, conformal prediction — and writes every step to disk so context-window loss can't erase progress. The deliverable is a validated predictive model with parameter uncertainty bounds, not a summary document. Because every hypothesis, probe, and update is persisted to a session directory, a context-window reset mid-analysis is a non-event: the next turn resumes from disk, not from memory.

You don't run the tools by hand. You talk to Claude; Claude adopts the orchestrator role, picks a depth tier, and drives the whole protocol for you.

### What it sounds like

**Engineering — black-box reverse engineering.**

> *"Help me figure out how this API's rate limiter works — I only have request/response logs."*

Claude seeds three competing mechanisms (token bucket / sliding window / leaky bucket) with priors, designs differential probes (burst vs. steady traffic), fits a parametric model with capacity and refill-rate uncertainty, and returns allow/reject predictions with conformal intervals plus a head-to-head comparison against a "count requests" baseline.

**Security — side-channel investigation.**

> *"Does this login endpoint leak whether a username exists through response-time variance?"*

Claude seeds H1 timing oracle, H2 rate-limit artifact, H3 network jitter; runs timed enumeration vs. non-enumeration probes; the falsification gate forbids any hypothesis from crossing 0.80 posterior without a disconfirming test, and a standing adversarial hypothesis tracks the possibility that the signal is measurement noise.

**Forecasting — calibration audit.**

> *"I have a sales forecast from our demand planner — can you tell me whether it beats a naive baseline, and where its intervals are honest?"*

Claude runs walk-forward cross-validation, compares against a seasonal naive, produces conformal prediction intervals with guaranteed marginal coverage, and returns a calibrated assessment of where the planner's intervals are over- or under-confident at each horizon.

Every Claude response carries a state line, so progress is visible at a glance:

```
[STATE: Phase 2 | Tier: STANDARD | Active Hypotheses: 3 | Lead: H2 (78%) | Confidence: Medium]
```

### Use this when you need to

- **Reverse-engineer an unknown system** — software, hardware, biological, or organizational black boxes.
- **Run competitive intelligence** — interrogate a system or product whose internals you cannot see.
- **Map an attack surface** — security analysis, side-channel probing, boundary enumeration.
- **Do forensics or root-cause analysis** — reconstruct what a system did and why.
- **Build a predictive model from observations** — fit, validate, and bound-uncertainty on real data.
- **Profile behavior** — psychological and behavioral analysis via the PSYCH tier.

A separate **PSYCH tier** applies the same Bayesian machinery to behavioral analysis — OCEAN, Dark Triad, and MICE frameworks for structured profiling, with its own trait tracker and FSM subgraph. The clinical framing is intentional: the tier exists for HUMINT, negotiation prep, and behavioral due-diligence work, and it carries no worked example here by design. Protocol and ethics: [`src/references/psych-tier-protocol.md`](src/references/psych-tier-protocol.md).

---

## Quickstart

### 1. Install

**Recommended — full install (skill + sub-agents):**

```bash
git clone https://github.com/NikolasMarkou/epistemic-deconstructor.git
cd epistemic-deconstructor
make sync-skill          # Windows: .\build.ps1 sync-skill
```

`make sync-skill` copies `src/` into `~/.claude/skills/epistemic-deconstructor/` **and** installs the 15 per-phase agents into `~/.claude/agents/`. That second step matters: Claude Code does **not** scan a skill's own subdirectories for agents, and the orchestrator needs them in `~/.claude/agents/` to dispatch the per-phase specialists (`ed-causal-analyst`, `ed-validator`, …). Restart the Claude Code session afterward so the new agents load. Re-running `sync-skill` after edits re-syncs both; `make unsync-agents` removes the installed agents cleanly. (Never create a `.claude/` directory inside the repo.)

**Quick try — skill only:**

```bash
git clone https://github.com/NikolasMarkou/epistemic-deconstructor.git ~/.claude/skills/epistemic-deconstructor
```

Claude Code picks the skill up automatically. The orchestrator runs inline from the skill bundle; for the full multi-agent experience, run `make sync-skill` to install the agents too.

**Claude Project / Custom Instructions:** grab `epistemic-deconstructor-combined.md` from [Releases](https://github.com/NikolasMarkou/epistemic-deconstructor/releases) — the protocol with all references inlined into one file. (CLI scripts are not included in this variant; analysis runs in "narrative" mode without the Python enforcement layer.)

### 2. Run an analysis

In Claude Code, invoke the skill and describe your target — that's it:

```
/epistemic-deconstructor

I have 50k request/response log lines from an API. I think there's a rate
limiter but I don't know the algorithm or the limits. Reverse-engineer it.
```

Claude takes over from there. A real session looks like this:

```
You:  /epistemic-deconstructor  Reverse-engineer this API's rate limiter from logs.

Claude:  This is a black-box reverse-engineering task → STANDARD tier.
         Creating session analyses/analysis_2026-05-28_a1b2c3d4/
         Seeding 3 competing mechanisms:
           H1 token bucket    (prior 0.40)
           H2 sliding window  (prior 0.35)
           H3 leaky bucket    (prior 0.25)
         + standing adversarial hypothesis (logs are unreliable / clock skew).

         [STATE: Phase 0 | Tier: STANDARD | Active Hypotheses: 4 | Lead: H1 (40%) | Confidence: Low]

         Phase 1 — I need a burst probe vs. a steady-rate probe to tell the
         three apart. Here's the probe design and what each mechanism predicts...
```

Each phase writes its evidence to a session directory under `analyses/`. When the context window resets mid-analysis, Claude resumes from those files — that's the design assumption, not an emergency fallback. Phase 5 produces a final report with conformal intervals, residual diagnostics, and an explicit head-to-head against a baseline.

> **Not sure how to start?** Say *"Help me start"* or *"Walk me through it"* and the orchestrator runs an Auto-Pilot questionnaire that maps your answers to a tier.

---

## Why trust this?

Most LLM-based analysis stops at "the model thinks about the problem" — plausible narratives with no calibration discipline, no falsification requirement, and no persistence. Epistemic Deconstructor imposes a six-phase protocol that Claude must execute step by step: each phase transition requires specific files to exist on disk, every hypothesis update is logged as a Bayesian likelihood-ratio update with hard phase-keyed caps, and the session outlives any context window.

### The phases

Phase names are plain English; the operator codes in parentheses are the internal mechanisms each phase runs.

| # | Phase | What happens |
|---|-------|--------------|
| **0** | Setup & Frame | Define scope, seed 3+ competing hypotheses, pick a fidelity target |
| **0.3** | Domain Orientation | Ground unfamiliar jargon: glossary, metrics, verified sources (TE/TG/MM/AM/CS) |
| **0.5** | RAPID Screening | Coherence and red-flag check for external claims |
| **0.7** | Scope Interrogation | Enumerate boundary conditions (M1–M4 mechanisms) |
| **1** | Boundary Mapping | Characterize I/O, apply probes, build a stimulus-response database |
| **1.5** | Abductive Expansion | Generate interior hypotheses with coverage-gated promotion (TI/AA/SA/AR/IC) |
| **2** | Causal Analysis | Differential tests, causal graphs, falsification |
| **3** | Parametric ID | Fit models (ARX/ARMAX/NARMAX/ARIMA/ETS), quantify uncertainty |
| **4** | Model Synthesis | Compose sub-models, test emergence, run simulations |
| **5** | Validation | Beat baselines, conformal prediction, adversarial review, final report |

### The tiers

Tier selection determines not just which phases run but which operators within a phase run (e.g., Phase 1.5 runs SA+AA only in LITE vs. all five operators in STANDARD). Tier choice is the first thing the orchestrator decides.

| Tier | When | Route |
|------|------|-------|
| **RAPID** | Papers, vendor pitches, external claims | 0.5 → 5 |
| **LITE** | Known archetype, single function, stable | 0 → 1 → 1.5 → 5 |
| **STANDARD** | Unknown internals, single domain | 0 → 0.7 → 1 → 1.5 → 2 → 3 → 4 → 5 |
| **COMPREHENSIVE** | Multi-domain, adversarial, critical | All + recursive decomposition |
| **PSYCH** | Behavioral analysis of a person or persona | 0-P → 5-P |

Every phase has an **EXIT GATE** — a checklist verified before advancing. Phases 0.3, 0.7, and 1.5 are gated by exit codes from `domain_orienter.py gate`, `scope_auditor.py gate`, and `abductive_engine.py gate`; the structural gate for the remaining phases is enforced by `phase_gate.py`; agent-attested quality criteria (e.g. "≥80% I/O channels characterized", "residual whiteness", "FVA > 0%") layer on top. No skipping.

### Five things that make this different

- **Bayesian hypothesis tracking with phase-keyed LR caps** — likelihood ratios are capped at 3.0 in Phase 0 (no empirical data yet), 5.0 in Phase 1 (indirect probes), and 10.0 in Phases 2+, with the PSYCH `smoking_gun` preset reaching 20.0 only for direct admission or unambiguous behavioral indicators. Caps are enforced in `bayesian_tracker.py` presets, not by convention.
- **File-as-memory** — every step routes through `session_manager.py` to JSON (machine-parseable) or Markdown (human-readable) on disk. Agents are explicitly forbidden from using Claude's native Write/Read tools for session files; mid-analysis context loss is a non-event because the next turn resumes from disk.
- **Abductive expansion with coverage-gated promotion** — five operators (TI/AA/SA/AR/IC) generate candidate hypotheses; promotion is a separate step requiring `coverage_score ≥ 0.30` (observations-explained ÷ total ÷ complexity). LLM-parametric candidates are hard-capped at prior 0.30 and chain LR 2.0. This is the primary defense against hypothesis explosion.
- **Conformal prediction intervals** — Phase 5 emits intervals with guaranteed marginal coverage (and CQR variants for heteroscedastic series) rather than bare point estimates. The validator agent refuses to ship a final report without them.
- **Diagrams as a first-class artifact (v7.16.0)** — the protocol renders its FSM state machines, causal graphs, coverage maps, and inference chains as Mermaid diagrams via the stdlib-only `mermaid_render.py`, so the analysis is auditable visually, not just as JSON.

Decision anchoring complements these: `# DECISION` inline code comments carry the rationale, the plan-document trace, and the override path at the exact line where each load-bearing constant is enforced (e.g. the AST allowlist that closes the `simulator.py` sandbox-escape in v7.15.22). Editors see *why* before they touch.

### Six principles

- **Falsify, don't confirm** — design tests to break hypotheses, not support them. The disconfirmation gate is procedural, not aspirational.
- **Quantify uncertainty** — posteriors, parameter bootstrap CIs, and conformal intervals. Never bare point estimates.
- **Map ≠ territory** — every model documents *how* it is wrong (residual structure, scope omissions, scenarios it cannot represent).
- **Beat naive first** — every model justifies its complexity against a simple baseline; failure to beat naive is a stop condition.
- **Files are truth** — if it isn't written to a session file, it didn't happen. The session files *are* the analysis.
- **Gate checks are non-negotiable** — every phase transition requires specific disk writes, verified by a script-level gate.

Calibration is enforced in code — LR caps by phase, anti-bundling (one data point per `update` call), the adversarial-hypothesis requirement, and the disconfirmation gate. Full rules: [`src/references/evidence-calibration.md`](src/references/evidence-calibration.md).

### What this is not

It is not a general-purpose LLM wrapper, a RAG framework, an agentic code-writer, or a one-shot analysis tool. It is a discipline layer for Claude when the task is "characterize a system you cannot read the source of." If the task is "write me code" or "summarize this document," reach for something else — and the skill will tell you so, then offer to reframe it as a reverse-engineering task or decline.

The Python CLIs are thin protocol-enforcement tools, not a replacement for scipy or statsmodels — they lean on the scientific-Python stack where they need to, and stop there. Session files accumulate across many conversation turns by design; one-shot use is supported but underuses the architecture.

---

## What's in the box?

Three layers of artifact: the CLI tools that enforce the protocol on disk, the sub-agent definitions that wire those tools to Claude Code, and the reference documents that supply per-domain knowledge. All three are user-readable Markdown or stdlib Python — nothing is compiled and there is no service to run.

### Tools

`src/scripts/` holds 15 `.py` files: **14 command-line tools plus the shared `common.py` library** (Bayesian math and JSON I/O with file locking, imported by the rest). The CLIs are grouped by role below. Most are stdlib-only; third-party packages (**numpy / scipy / pandas / statsmodels**, plus **catboost / scikit-learn** for forecasting) are used only by the signal/model-fitting tools — `simulator.py` requires numpy, while `ts_reviewer.py`, `forecast_modeler.py`, `parametric_identifier.py`, and `fourier_analyst.py` guard their imports and degrade gracefully without them. Every CLI follows the same `--file <state.json> <subcommand> [args]` convention, so invocations stay regular across the suite.

#### Session & I/O

| Tool | Role |
|------|------|
| `session_manager` | Creates sessions and routes all file I/O — Claude never fabricates paths |
| `phase_gate` | Structural exit-gate dispatcher for the phases without a dedicated gate script |
| `mermaid_render` | Stdlib-only deterministic Mermaid emitter — FSM / inference chains / coverage / causal / adjacency diagrams (v7.16.0) |

#### Hypothesis tracking

| Tool | Role |
|------|------|
| `bayesian_tracker` | Hypothesis tracking with LR-based updates and calibration rules |
| `belief_tracker` | OCEAN / Dark Triad / MICE trait tracking (PSYCH tier) |
| `rapid_checker` | Quick claim validation for papers, pitches, forecasts |

#### Phase sub-tools

| Tool | Role |
|------|------|
| `domain_orienter` | Phase 0.3 domain orientation (TE/TG/MM/AM/CS operators) — conditional on `domain_familiarity` |
| `scope_auditor` | Phase 0.7 scope interrogation (M1–M4 mechanisms) |
| `abductive_engine` | Phase 1.5 abductive expansion (TI/AA/SA/AR/IC) with coverage-gated promotion |

#### Signal & model fitting

| Tool | Role |
|------|------|
| `ts_reviewer` | Time-series diagnostics with conformal and CQR intervals |
| `fourier_analyst` | Frequency-domain spectral analysis and transfer functions |
| `forecast_modeler` | ARIMA / ETS / CatBoost with conformal prediction |
| `parametric_identifier` | ARX / ARMAX / NARMAX structural system ID with bootstrap uncertainty |
| `simulator` | System Dynamics / Monte Carlo / ABM / DES + sensitivity analysis |

### Sub-agents

The orchestrator dispatches 15 specialized sub-agents (opus×4, sonnet×9, haiku×2), each owning a phase or cross-cutting role. Only the orchestrator holds the `Agent` tool; sub-agents cannot spawn further sub-agents, which keeps the call graph flat and the budget legible. Full specs: [`docs/subagents.md`](docs/subagents.md).

| Agent | Role |
|-------|------|
| `ed-orchestrator` | Main FSM orchestrator; routes tier, verifies exit gates, dispatches all others (opus) |
| `ed-session-clerk` | Filesystem I/O via `session_manager.py`; background (haiku) |
| `ed-hypothesis-engine` | Bayesian tracking via `bayesian_tracker.py` / `belief_tracker.py` (sonnet) |
| `ed-cognitive-auditor` | Bias and analytical-trap detection after each phase; background (sonnet) |
| `ed-rapid-screener` | RAPID tier Phase 0.5 screening via `rapid_checker.py` (sonnet) |
| `ed-domain-orienter` | Phase 0.3 domain orientation; uses WebFetch (sonnet, synchronous) |
| `ed-scope-auditor` | Phase 0.7 scope interrogation; also post-Phase-3; background (sonnet) |
| `ed-boundary-mapper` | Phase 1 input/output mapping (sonnet) |
| `ed-abductive-engine` | Phase 1.5 abductive expansion; background (sonnet) |
| `ed-causal-analyst` | Phase 2 causal graphs and falsification (opus) |
| `ed-parametric-id` | Phase 3 structural ID + forecasting fits (sonnet) |
| `ed-model-synthesizer` | Phase 4 composition and simulation (sonnet) |
| `ed-validator` | Phase 5 validation, conformal prediction, final report (opus) |
| `ed-psych-profiler` | PSYCH tier owner; runs all six phases 0-P through 5-P (opus) |
| `ed-research-scout` | Background web research (WebSearch, WebFetch); any phase (haiku) |

### Knowledge base

41 domain references under [`src/references/`](src/references/), grouped by purpose. Claude pulls the relevant ones in as protocol context during a run; this is not a RAG retrieval store.

- **System analysis** — boundary probing, causal techniques, system identification, compositional synthesis, adversarial heuristics, multi-pass protocol, scope interrogation, domain orientation, archetype accomplices, abductive reasoning, mermaid conventions, warning-reception dynamics
- **Validation & diagnostics** — validation checklist, domain calibration, red flags, cognitive traps, evidence calibration, modeling epistemology, engineering laws, coherence checks, decision trees, phase protocols, session memory, setup techniques, tool catalog, tools & sensitivity, rapid assessment
- **Forecasting & time series** — forecasting science, forecasting tools, timeseries review, spectral analysis, financial validation
- **Simulation** — simulation guide, distributions guide
- **PSYCH tier** — OCEAN / Dark Triad / MICE mapping, linguistic markers, elicitation, motive analysis, profile synthesis, psych-tier protocol

The core protocol itself lives in [`src/SKILL.md`](src/SKILL.md).

### Build & test

```bash
# Unix / Linux / macOS
make package              # distributable zip
make package-combined     # single-file skill with references inlined
make package-tar          # distributable tarball
make validate             # check structure and cross-references
make test                 # run the unit suite (880 tests)
make sync-skill           # install skill + 15 agents to ~/.claude/
make unsync-agents        # remove the installed agents
make clean

# Windows (PowerShell)
.\build.ps1 package
.\build.ps1 package-combined
.\build.ps1 validate
.\build.ps1 test
.\build.ps1 clean
```

The suite runs on Python 3.8+. Optional dependencies are declared in `pyproject.toml` groups: `numeric` (numpy / scipy / pandas / statsmodels), `forecast` (catboost / scikit-learn), `test` (pytest), and `all`.

---

## Going deeper

The master protocol — phase FSM, tier routing, evidence rules, and the three protective layers (Protocol Inviolability / Intake & Reframe / Refusal Protocol) — lives in [`src/SKILL.md`](src/SKILL.md). It is the file Claude reads first when invoked, and it cross-references everything else. The per-phase procedural recipes (Activities + EXIT GATE checklists + File Write Matrix + Gate Check Procedure) live in [`src/references/phase-protocols.md`](src/references/phase-protocols.md) and are consumed by the per-phase agents directly.

New domains are added by writing a new reference file rather than by modifying the protocol; the protocol stays small and the knowledge surface grows.

Sub-agent definitions live in [`src/agents/`](src/agents/) as one Markdown file per agent. Frontmatter declares the model (`opus` / `sonnet` / `haiku`), the tools list, and the background flag; the body declares the contract (inputs, outputs, exit conditions). Adding a sub-agent is a matter of writing one such file and wiring an entry in the orchestrator.

The 880 unit tests under [`tests/`](tests/) cover every CLI end-to-end with no mocks — real file I/O, real JSON round-trips, real subprocess invocations. Adding a feature without a test fails review.

The run-time configuration Claude consults is in [`src/config/`](src/config/): `domains.json` (plausibility bounds for 7 domains), `archetypes.json` (12 system-archetype definitions used by the Phase 0.7 scope auditor), and `trace_catalog.json` (6 trace categories the Phase 1.5 inversion operator searches over). Edit those to extend domain coverage without touching the protocol.

Version history, deferral notes, and per-release decision logs are in [`CHANGELOG.md`](CHANGELOG.md). The repository itself is developed under an iterative-planning workflow (EXPLORE / PLAN / EXECUTE / REFLECT / CLOSE); surviving plan artifacts and lessons live under `plans/`.

---

[GNU General Public License v3.0](LICENSE) · Sponsored by [Electi Consulting](https://www.electiconsulting.com) · See [CHANGELOG.md](CHANGELOG.md) for version history.
