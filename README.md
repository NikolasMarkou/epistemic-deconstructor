# Epistemic Deconstructor

[![License](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-v7.15.2-green.svg)](CHANGELOG.md)
[![Tests](https://img.shields.io/badge/Tests-655%20passing-brightgreen.svg)](tests/)
[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](src/scripts/)
[![Sponsored by Electi](https://img.shields.io/badge/Sponsored%20by-Electi-red.svg)](https://www.electiconsulting.com)

**Turn "I don't know how this works" into a quantified, validated model of how it works.**

You hand Claude a black box; it runs a structured investigation with Bayesian hypothesis tracking, falsification tests, and conformal prediction — writing every step to disk so context-window loss can't erase progress. The deliverable is a validated predictive model with parameter uncertainty bounds, not a summary document.

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

Every Claude response carries a state line so progress is visible at a glance:

```
[STATE: Phase 2 | Tier: STANDARD | Active Hypotheses: 3 | Lead: H2 (78%) | Confidence: Medium]
```

A separate **PSYCH tier** applies the same Bayesian machinery to behavioral analysis — OCEAN, Dark Triad, and MICE frameworks for structured profiling. The clinical framing is intentional: the tier exists for HUMINT, negotiation prep, and behavioral due-diligence work.

---

## Install

### Claude Code (recommended)

```bash
git clone https://github.com/NikolasMarkou/epistemic-deconstructor.git ~/.claude/skills/epistemic-deconstructor
```

Claude Code picks the skill up automatically.

### Single-file bundle

Grab `epistemic-deconstructor-combined.md` from [Releases](https://github.com/NikolasMarkou/epistemic-deconstructor/releases) and paste it into a Claude Project or Custom Instructions. (Scripts are not included in this variant.)

### Full package

Download the release zip and upload `src/` to a Claude Project. Includes the protocol, references, CLI tools, and domain calibration config.

For local development, `make sync-skill` copies the live `src/` into `~/.claude/skills/epistemic-deconstructor` so edits propagate without re-cloning.

### Shape of a session

Claude picks a tier (RAPID / LITE / STANDARD / COMPREHENSIVE / PSYCH) sized to the question, creates a session directory under `analyses/`, then walks through phases in order. Each phase has an EXIT GATE — a checklist of files that must exist on disk before advancing. The session files *are* the analysis: when context resets, Claude resumes from disk rather than from memory, which is the design assumption rather than an emergency fallback. Phase 5 produces a final report with conformal intervals, residual diagnostics, and an explicit head-to-head against a baseline.

A minimal session, on the command line:

```bash
SM="python3 <skill-dir>/scripts/session_manager.py --base-dir ."
$SM new "Rate limiter black-box analysis"

python3 src/scripts/bayesian_tracker.py --file $($SM path hypotheses.json) \
    add "Uses token bucket with capacity 60" --prior 0.4
python3 src/scripts/bayesian_tracker.py --file $($SM path hypotheses.json) \
    update H1 "429 after exactly 60 requests" --preset strong_confirm
$SM status
```

---

## Why trust this?

Most LLM-based analysis tools stop at "the model thinks about the problem" — they produce plausible narratives with no calibration discipline, no falsification requirement, and no persistence. Epistemic Deconstructor imposes a six-phase protocol that Claude must execute step by step, where each phase transition requires specific files to exist on disk, all hypothesis updates are logged as Bayesian likelihood-ratio updates with hard phase-keyed caps, and the session outlives any context window.

### The phases

| # | Phase | What happens |
|---|-------|--------------|
| **0** | Setup & Frame | Define scope, seed 3+ competing hypotheses, pick a fidelity target |
| **0.5** | RAPID Screening | Coherence and red-flag check for external claims |
| **0.7** | Scope Interrogation | Enumerate boundary conditions (M1-M4 mechanisms) |
| **1** | Boundary Mapping | Characterize I/O, apply probes, build a stimulus-response database |
| **1.5** | Abductive Expansion | Generate interior hypotheses with coverage-gated promotion |
| **2** | Causal Analysis | Differential tests, causal graphs, falsification |
| **3** | Parametric ID | Fit models (ARX/ARMAX/NARMAX/ARIMA/ETS), quantify uncertainty |
| **4** | Model Synthesis | Compose sub-models, test emergence, run simulations |
| **5** | Validation | Beat baselines, conformal prediction, adversarial review, final report |

### The tiers

Tier selection determines not just which phases run but which operators within phases run (e.g., Phase 1.5 runs SA+AA only in LITE vs. all five operators in STANDARD). Tier choice is the first thing the orchestrator decides.

| Tier | When | Route |
|------|------|-------|
| **RAPID** | Papers, vendor pitches, external claims | 0.5 → 5 |
| **LITE** | Known system, single function, stable | 0 → 1 → 5 |
| **STANDARD** | Unknown internals, single domain | 0 → 1 → 2 → 3 → 4 → 5 |
| **COMPREHENSIVE** | Multi-domain, adversarial, critical | All + recursive decomposition |
| **PSYCH** | Behavioral analysis of a person or persona | 0-P → 5-P |

Every phase has an **EXIT GATE** — a checklist of files that must exist on disk before advancing. No skipping. The gate is enforced by exit codes from `scope_auditor.py gate`, `domain_orienter.py gate`, and `abductive_engine.py gate`, not by convention.

### Five things that make this different

- **Bayesian hypothesis tracking with phase-keyed LR caps** — likelihood ratios are capped at 3.0 in Phase 0 (no empirical data yet), 5.0 in Phase 1 (indirect probes), 10.0 in Phases 2+, with the PSYCH `smoking_gun` preset reaching 20.0 only for direct admission or unambiguous behavioral indicators. Caps are enforced in code via `bayesian_tracker.py` presets, not convention.
- **File-as-memory** — every step routes through `session_manager.py` to JSON (machine-parseable) or Markdown (human-readable) on disk. Agents are explicitly forbidden from using Claude's native Write/Read tools for session files; mid-analysis context loss is a non-event because the next turn resumes from disk.
- **Abductive expansion with coverage-gated promotion** — five operators (TI/AA/SA/AR/IC) generate candidate hypotheses; promotion is a separate step that requires `coverage_score ≥ 0.30`, where coverage is observations-explained divided by total-observations divided by complexity. LLM-parametric candidates are hard-capped at prior 0.30 and chain LR 2.0. The primary defense against hypothesis explosion.
- **Conformal prediction intervals** — Phase 5 emits intervals with guaranteed marginal coverage (and CQR variants for heteroscedastic series) rather than bare point estimates. The validator agent owns this and refuses to ship a final report without it.
- **Decision anchoring** — `# DECISION D-NNN` inline code comments carry the rationale, the plan-document trace, and the override path at the exact line where each load-bearing constant is enforced. Eight anchors currently live in `abductive_engine.py` and `domain_orienter.py`.

### Six principles

- **Falsify, don't confirm** — design tests to break hypotheses, not support them. The disconfirmation gate is procedural, not aspirational.
- **Quantify uncertainty** — posteriors, parameter bootstrap CIs, and conformal intervals. Never bare point estimates.
- **Map ≠ territory** — every model documents *how* it is wrong (residual structure, scope omissions, scenarios it cannot represent).
- **Beat naive first** — every model justifies its complexity against a simple baseline; failure to beat naive is a stop condition.
- **Files are truth** — if it isn't written to a session file, it didn't happen. The session files *are* the analysis.
- **Gate checks are non-negotiable** — every phase transition requires specific disk writes verified by a script-level gate command.

Calibration is enforced in code — LR caps by phase, anti-bundling (one data point per `update` call), adversarial hypothesis requirement, disconfirmation gate. Full rules: [`src/references/evidence-calibration.md`](src/references/evidence-calibration.md).

### What this is not

It is not a general-purpose LLM wrapper, a RAG framework, an agentic code-writer, or a one-shot analysis tool. It is a discipline layer for Claude when the task is "characterize a system you cannot read the source of." If the task is "write me code" or "summarize this document," reach for something else.

The Python CLIs are thin protocol-enforcement tools, not a replacement for scipy or statsmodels — they call into numpy where they need to, and stop there. The session files accumulate across many conversation turns by design; one-shot use is supported but underuses the architecture.

---

## What's in the box?

The repository ships three layers of artifact: the CLI tools that enforce the protocol on disk, the sub-agent definitions that wire those tools to Claude Code, and the reference documents that supply per-domain knowledge. All three are user-readable Markdown or stdlib Python; nothing is compiled and there is no service to run.

### Tools

12 Python CLIs under [`src/scripts/`](src/scripts/), grouped by role. 8 are stdlib-only; `simulator.py`, `fourier_analyst.py`, and `parametric_identifier.py` require numpy; `ts_reviewer.py` and `forecast_modeler.py` use numpy optionally and degrade gracefully. Every CLI follows the same `--file <state.json> <subcommand> [args]` convention with `--file` on the parent parser, so invocations stay regular across the suite.

#### Session & I/O

| Tool | Role |
|------|------|
| `session_manager` | Creates sessions and routes all file I/O — Claude never fabricates paths |

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
| `scope_auditor` | Phase 0.7 scope interrogation (M1-M4 mechanisms) |
| `abductive_engine` | Phase 1.5 abductive expansion (TI/AA/SA/AR/IC operators) with coverage-gated promotion |

#### Signal & model fitting

| Tool | Role |
|------|------|
| `ts_reviewer` | Time-series diagnostics with conformal and CQR intervals |
| `fourier_analyst` | Frequency-domain spectral analysis and transfer functions |
| `forecast_modeler` | ARIMA / ETS / CatBoost with conformal prediction |
| `parametric_identifier` | ARX / ARMAX / NARMAX structural system ID with bootstrap uncertainty |
| `simulator` | System Dynamics / Monte Carlo / ABM / DES + sensitivity analysis |

### Sub-agents

The orchestrator dispatches 15 specialized sub-agents (opus×4, sonnet×9, haiku×2), each owning a specific phase or cross-cutting role. Only the orchestrator holds the `Agent` tool; sub-agents cannot spawn further sub-agents, which keeps the call graph flat and the budget legible. Full specs: [`docs/subagents.md`](docs/subagents.md).

| Agent | Role |
|-------|------|
| `epistemic-orchestrator` | Main FSM orchestrator; routes tier, verifies exit gates, dispatches all others (opus) |
| `session-clerk` | Filesystem I/O via `session_manager.py`; runs in background (haiku) |
| `hypothesis-engine` | Bayesian tracking via `bayesian_tracker.py` and `belief_tracker.py` (sonnet) |
| `cognitive-auditor` | Bias and analytical-trap detection after each phase; background (sonnet) |
| `rapid-screener` | RAPID tier Phase 0.5 screening via `rapid_checker.py` (sonnet) |
| `domain-orienter` | Phase 0.3 domain orientation; uses WebFetch (sonnet, synchronous) |
| `scope-auditor` | Phase 0.7 scope interrogation; also post-Phase-3; background (sonnet) |
| `boundary-mapper` | Phase 1 input/output mapping (sonnet) |
| `abductive-engine` | Phase 1.5 abductive expansion; background (sonnet) |
| `causal-analyst` | Phase 2 causal graphs and falsification (opus) |
| `parametric-id` | Phase 3 structural ID + forecasting fits (sonnet) |
| `model-synthesizer` | Phase 4 composition and simulation (sonnet) |
| `validator` | Phase 5 validation, conformal prediction, final report (opus) |
| `psych-profiler` | PSYCH tier owner; runs all six phases 0-P through 5-P (opus) |
| `research-scout` | Background web research (WebSearch, WebFetch); any phase (haiku) |

### Knowledge base

37 domain references under [`src/references/`](src/references/), grouped by purpose. Claude pulls the relevant ones in as protocol context during a run; this is not a RAG retrieval store.

- **System analysis** — boundary probing, causal techniques, system identification, compositional synthesis, adversarial heuristics, multi-pass protocol, scope interrogation, domain orientation, archetype accomplices, abductive reasoning
- **Validation & diagnostics** — validation checklist, domain calibration, red flags, cognitive traps, evidence calibration, modeling epistemology, engineering laws, coherence checks
- **Forecasting & time series** — forecasting science, forecasting tools, timeseries review, spectral analysis, financial validation, system identification
- **Simulation** — simulation guide, distributions guide
- **PSYCH tier** — OCEAN / Dark Triad / MICE mapping, linguistic markers, elicitation, motive analysis, profile synthesis, psych-tier-protocol

The core protocol itself lives in [`src/SKILL.md`](src/SKILL.md).

### Build & test

```bash
# Unix / Linux / macOS
make package              # distributable zip
make package-combined     # single-file skill with references inlined
make validate             # check structure and cross-references
make test                 # 655 unit tests
make sync-skill           # install to ~/.claude/skills/epistemic-deconstructor
make clean

# Windows (PowerShell)
.\build.ps1 package
.\build.ps1 package-combined
.\build.ps1 validate
.\build.ps1 test
.\build.ps1 clean
```

---

## Going deeper

The master protocol — phase FSM, tier routing, evidence rules, file-write matrix — lives in [`src/SKILL.md`](src/SKILL.md). It is the file Claude reads first when invoked, and it cross-references everything else.

The 37 reference documents under [`src/references/`](src/references/) carry the per-domain knowledge that Claude pulls in during a run. New domains are added by writing a new reference file rather than by modifying the protocol; the protocol stays small and the knowledge surface grows.

Sub-agent definitions live in [`src/agents/`](src/agents/) as one Markdown file per agent. Frontmatter declares the model (`opus` / `sonnet` / `haiku`), the tools list, and the background flag; the body declares the contract (inputs, outputs, exit conditions). Adding a new sub-agent is a matter of writing one such file and wiring an entry in the orchestrator.

The 655 unit tests under [`tests/`](tests/) cover every CLI end-to-end with no mocks — real file I/O, real JSON round-trips, real subprocess invocations. Adding a new feature without a test fails review.

The Makefile and `build.ps1` build script drive packaging and the test loop; project-internal AI guidance lives in `CLAUDE.md`.

The configuration data Claude consults at run time is in [`src/config/`](src/config/): `domains.json` (plausibility bounds for seven domains), `archetypes.json` (10 system-archetype definitions), and `trace_catalog.json` (six trace categories the Phase 1.5 inversion operator searches over). Edit those to extend domain coverage without touching the protocol.

Version history, deferral notes, and per-release decision logs are in [`CHANGELOG.md`](CHANGELOG.md). The repository follows an iterative-planning workflow with explicit EXPLORE / PLAN / EXECUTE / REFLECT / CLOSE phases; surviving plan artifacts and lessons live under `plans/`.

[GNU General Public License v3.0](LICENSE). See [CHANGELOG.md](CHANGELOG.md) for version history.
