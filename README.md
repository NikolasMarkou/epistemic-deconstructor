# Epistemic Deconstructor

[![License](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-v7.15.2-green.svg)](CHANGELOG.md)
[![Tests](https://img.shields.io/badge/Tests-655%20passing-brightgreen.svg)](tests/)
[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](src/scripts/)
[![Sponsored by Electi](https://img.shields.io/badge/Sponsored%20by-Electi-orange.svg)](https://www.electiconsulting.com)

**Turn "I don't know how this works" into a quantified, validated model of how it works.**

Epistemic Deconstructor is a [Claude Code skill](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/skills) that walks Claude through a rigorous 6-phase protocol for reverse engineering unknown systems — software, hardware, organizations, or people. You hand it a black box; it runs the investigation with Bayesian hypothesis tracking, falsification-driven experiments, and conformal prediction intervals.

> **What's a Claude Code skill?** A reusable instruction set installed into `~/.claude/skills/` that gives Claude a structured workflow it can execute on demand.

---

## What it looks like

You: *"Help me figure out how this API's rate limiter works — I only have request/response logs."*

Claude enters the protocol:

1. **Setup** — writes 3 competing hypotheses (token bucket? sliding window? leaky bucket?) with prior probabilities and a fidelity target.
2. **Boundary mapping** — designs probe experiments to characterize inputs and outputs.
3. **Causal analysis** — differential tests, sensitivity sweeps, falsifies weak hypotheses.
4. **Parametric ID** — fits a mathematical model with parameter uncertainty bounds.
5. **Synthesis & validation** — beats naive baselines, produces conformal intervals, writes a final report.

Every step writes to disk, so context window loss doesn't destroy progress. Every hypothesis update is a Bayesian likelihood-ratio update with hard calibration caps. Every response carries a state line:

```
[STATE: Phase 2 | Tier: STANDARD | Active Hypotheses: 3 | Lead: H2 (78%) | Confidence: Medium]
```

---

## Install

### Claude Code (recommended)

```bash
git clone https://github.com/NikolasMarkou/epistemic-deconstructor.git ~/.claude/skills/epistemic-deconstructor
```

Claude Code picks it up automatically. Say: **"Help me start an epistemic deconstruction."**

### Single-file bundle

Grab `epistemic-deconstructor-combined.md` from [Releases](https://github.com/NikolasMarkou/epistemic-deconstructor/releases) and paste into a Claude Project or Custom Instructions. (Scripts not included in this variant.)

### Full package

Download the release zip, upload `src/` to a Claude Project. Includes protocol, references, CLI tools, and domain calibration config.

---

## The phases

| # | Phase | What happens |
|---|-------|--------------|
| **0** | Setup & Frame | Define scope, seed 3+ competing hypotheses, pick fidelity target |
| **0.5** | RAPID Screening | Coherence and red-flag check for external claims |
| **0.7** | Scope Interrogation | Enumerate boundary conditions (M1-M4 mechanisms) |
| **1** | Boundary Mapping | Characterize I/O, apply probes, build stimulus-response database |
| **1.5** | Abductive Expansion | Generate interior hypotheses with coverage-gated promotion |
| **2** | Causal Analysis | Differential tests, causal graphs, falsification |
| **3** | Parametric ID | Fit models (ARX/ARMAX/NARMAX/ARIMA/ETS), quantify uncertainty |
| **4** | Model Synthesis | Compose sub-models, test emergence, run simulations |
| **5** | Validation | Beat baselines, conformal prediction, adversarial review, final report |

Every phase has an **EXIT GATE** — a checklist of files that must exist on disk before advancing. No skipping.

## The tiers

| Tier | When | Route |
|------|------|-------|
| **RAPID** | Papers, vendor pitches, external claims | 0.5 → 5 |
| **LITE** | Known system, single function, stable | 0 → 1 → 5 |
| **STANDARD** | Unknown internals, single domain | 0 → 1 → 2 → 3 → 4 → 5 |
| **COMPREHENSIVE** | Multi-domain, adversarial, critical | All + recursive decomposition |
| **PSYCH** | Behavioral analysis of a person or persona | 0-P → 5-P |

---

## The tools

Twelve Python CLIs (`src/scripts/`). Most are stdlib-only; `simulator.py`, `fourier_analyst.py`, and `forecast_modeler.py` want numpy/scipy for their advanced features.

| Tool | Role |
|------|------|
| `session_manager` | Creates sessions and routes all file I/O — Claude never fabricates paths |
| `bayesian_tracker` | Hypothesis tracking with LR-based updates and calibration rules |
| `belief_tracker` | OCEAN / Dark Triad / MICE trait tracking (PSYCH tier) |
| `rapid_checker` | Quick claim validation for papers, pitches, forecasts |
| `domain_orienter` | Phase 0.3 domain orientation (TE/TG/MM/AM/CS operators) — conditional on `domain_familiarity` |
| `scope_auditor` | Phase 0.7 scope interrogation (M1-M4 mechanisms) |
| `abductive_engine` | Phase 1.5 abductive expansion (TI/AA/SA/AR/IC operators) |
| `ts_reviewer` | 10-phase time-series diagnostics with conformal intervals |
| `fourier_analyst` | 9-phase frequency-domain analysis |
| `forecast_modeler` | ARIMA / ETS / CatBoost with conformal prediction |
| `parametric_identifier` | ARX / ARMAX / NARMAX structural system ID |
| `simulator` | System Dynamics / Monte Carlo / ABM / DES + sensitivity |

A typical invocation:

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

## Evidence rules

Calibration discipline is enforced to stop Bayesian tracking from drifting toward whatever story sounds good:

- **LR caps** — max 5.0 in Phases 0-1, 10.0 in Phase 2+ (with justification)
- **No batch evidence** — one data point per `update` call
- **Adversarial hypothesis required** — at least one hypothesis must test data reliability or concealment
- **Consensus is not strong evidence** — capped at LR 2.5
- **Disconfirm before confirming strongly** — no hypothesis crosses 0.80 without a disconfirming attempt
- **Priors sum to 1.0** (±0.01) for mutually exclusive sets

Full guide: [`src/references/evidence-calibration.md`](src/references/evidence-calibration.md).

---

## Principles

- **Falsify, don't confirm** — design tests to break hypotheses, not support them
- **Quantify uncertainty** — posteriors and conformal intervals, never bare point estimates
- **Map ≠ territory** — document *how* your model is wrong
- **Beat naive first** — every model justifies its complexity against simple baselines
- **Files are truth** — if it isn't written to a session file, it didn't happen
- **Gate checks are non-negotiable** — every phase transition requires specific disk writes

---

## Knowledge base

35+ domain references under [`src/references/`](src/references/), grouped into:

- **System analysis** — boundary probing, causal techniques, system identification, compositional synthesis, adversarial heuristics, multi-pass protocol
- **Validation & diagnostics** — validation checklist, domain calibration, red flags, cognitive traps, evidence calibration, modeling epistemology, engineering laws
- **Forecasting & time series** — forecasting science, conformal prediction, spectral analysis, financial validation, distribution selection
- **PSYCH tier** — OCEAN / Dark Triad / MICE mapping, linguistic markers, elicitation, motive analysis, profile synthesis

The core protocol itself lives in [`src/SKILL.md`](src/SKILL.md).

---

## Build & test

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

## License

[GNU General Public License v3.0](LICENSE). See [CHANGELOG.md](CHANGELOG.md) for version history.
