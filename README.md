# Epistemic Deconstructor

[![License](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-v6.6.3-green.svg)](CHANGELOG.md)
[![Tests](https://img.shields.io/badge/Tests-150%20passing-brightgreen.svg)](tests/)
[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](src/scripts/)
[![Sponsored by Electi](https://img.shields.io/badge/Sponsored%20by-Electi-orange.svg)](https://www.electiconsulting.com)

A [Claude skill](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/skills) for systematic reverse engineering of unknown systems using scientific methodology. Give it a black box — software, hardware, organizational, or human — and it walks Claude through a rigorous protocol to build a predictive model of how it works.

Bayesian hypothesis tracking, falsification-driven experimentation, compositional modeling, and conformal prediction turn "I don't know how this works" into a validated, quantified understanding.

> **What's a Claude skill?** A reusable instruction set that extends Claude's capabilities for a specific domain. Load it into a Claude Project, Custom Instructions, or Claude Code, and Claude gains a structured workflow it can execute on demand.

---

## Quick Start

**Option A — Single file (simplest):**
Download `epistemic-deconstructor-combined.md` from [Releases](https://github.com/NikolasMarkou/epistemic-deconstructor/releases). Paste into Claude's Custom Instructions or a Project. Bundles SKILL.md + all references into one file (scripts not included).

**Option B — Full package:**
Download the zip from [Releases](https://github.com/NikolasMarkou/epistemic-deconstructor/releases). Upload `src/` to a Claude Project. Includes the protocol, all references, Python CLI tools, and domain calibration config.

**Option C — Claude Code:**
```bash
git clone https://github.com/NikolasMarkou/epistemic-deconstructor.git
```
The `.claude/skills/` directory is pre-configured — Claude Code picks it up automatically.

Then say: **"Help me start"**

---

## What It Does

You describe a system you want to understand. The skill guides you through a phased methodology:

| Phase | Name | What Happens |
|-------|------|-------------|
| **0** | Setup & Frame | Define scope, seed 3+ competing hypotheses, pick fidelity target |
| **0.5** | Screening | Quick coherence/red-flag check for external claims (RAPID tier) |
| **1** | Boundary Mapping | Enumerate I/O channels, apply probe signals, build stimulus-response database |
| **2** | Causal Analysis | Differential analysis, sensitivity testing, build causal graph, falsify hypotheses |
| **3** | Parametric ID | Fit mathematical models, quantify parameter uncertainty |
| **4** | Model Synthesis | Compose sub-models, test for emergence, classify archetype |
| **5** | Validation | Beat naive baselines, conformal prediction, adversarial assessment, document limitations |

Every response tracks state:
```
[STATE: Phase 2 | Tier: STANDARD | Active Hypotheses: 3 | Lead: H2 (78%) | Confidence: Medium]
```

### Use Cases

- **Black-box analysis** of unknown systems (software, hardware, biological, organizational)
- **Competitive intelligence** and system interrogation
- **Security analysis** and attack surface mapping
- **Forensics** and root cause analysis
- **Validating claims** in papers, vendor pitches, or external reports
- **Time-series diagnostics** with forecastability assessment and conformal prediction
- **Psychological profiling** and behavioral analysis (PSYCH tier)

---

## Tiers

The protocol scales to the problem. Pick a tier based on complexity, time, and stakes:

| Tier | When to Use | Phases | Budget |
|------|-------------|--------|--------|
| **RAPID** | Validating a paper, vendor pitch, or external claim | 0.5 &rarr; 5 | <30 min |
| **LITE** | Known system type, single function, stable behavior | 0 &rarr; 1 &rarr; 5 | <2 hr |
| **STANDARD** | Unknown internals, single domain, no adversary | 0 &rarr; 1 &rarr; 2 &rarr; 3 &rarr; 4 &rarr; 5 | 2-20 hr |
| **COMPREHENSIVE** | Multi-domain, adversarial, critical infrastructure | All + recursive decomposition | 20+ hr |
| **PSYCH** | Behavioral analysis of a person or persona | 0-P &rarr; 1-P &rarr; 2-P &rarr; 3-P &rarr; 4-P &rarr; 5-P | 1-4 hr |

---

## CLI Tools

Seven Python scripts persist analysis state across sessions. Most use stdlib only — `simulator.py` requires numpy/scipy/matplotlib.

### Session Manager

```bash
python src/scripts/session_manager.py new "Target system description"
python src/scripts/session_manager.py resume    # Recover full state in a new conversation
python src/scripts/session_manager.py status    # One-line status
python src/scripts/session_manager.py close     # Merge observations, finalize
python src/scripts/session_manager.py list      # All sessions (active + closed)
```

### Bayesian Hypothesis Tracker

```bash
python src/scripts/bayesian_tracker.py add "System uses REST API" --prior 0.6
python src/scripts/bayesian_tracker.py update H1 "Found /api/v1 endpoint" --preset strong_confirm
python src/scripts/bayesian_tracker.py report --verbose
python src/scripts/bayesian_tracker.py verdict   # RAPID tier: CREDIBLE / SKEPTICAL / DOUBTFUL / REJECT
```

### Psychological Trait Tracker (PSYCH tier)

```bash
python src/scripts/belief_tracker.py add "High Neuroticism" --prior 0.5
python src/scripts/belief_tracker.py update T1 "Catastrophizing language" --preset strong_indicator
python src/scripts/belief_tracker.py profile     # Unified OCEAN + Dark Triad + MICE profile
```

### RAPID Claim Validator

```bash
python src/scripts/rapid_checker.py start "Paper: XYZ Claims"
python src/scripts/rapid_checker.py flag methodology "No baseline comparison"
python src/scripts/rapid_checker.py calibrate accuracy 0.99 --domain ml_classification
python src/scripts/rapid_checker.py verdict
```

### Time-Series Signal Reviewer

10-phase systematic diagnostics for any time-ordered numeric signal:

```bash
python src/scripts/ts_reviewer.py review data.csv --column value
python src/scripts/ts_reviewer.py quick data.csv --column temperature --freq 12
python src/scripts/ts_reviewer.py demo
```

**Phases**: Coherence, Data Quality, Stationarity (ADF/KPSS), Forecastability (ACF, Permutation Entropy), Decomposition (STL), Baseline Benchmarks (naive/seasonal/drift + FVA), Overfitting Screen, Residual Diagnostics, Uncertainty Calibration (coverage, Winkler), Regime Analysis.

**Programmatic API**: `TimeSeriesReviewer`, `quick_review()`, `compare_models()`, `walk_forward_split()`, `conformal_intervals()`, `cqr_intervals()`.

### Simulator

Forward simulation engine for identified models (requires numpy, scipy, matplotlib):

```bash
python src/scripts/simulator.py sd --model '{"A": [[0,1],[-2,-3]], "B": [[0],[1]]}' --x0 '[1,0]' --t_end 20 --plot
python src/scripts/simulator.py mc --model '{"a": [-0.5], "b": [1.0]}' --param_distributions '...' --n_runs 1000 --t_end 100
python src/scripts/simulator.py sensitivity --model_func 'k1*x + k2*x**2' --param_ranges '...' --method sobol
```

**Modes**: System Dynamics (SD), Monte Carlo (MC), Agent-Based (ABM), Discrete-Event (DES), Sensitivity Analysis. Validation bridge feeds results back to Phase 5.

---

## Knowledge Base

25 reference documents organized by domain:

### System Analysis
| Reference | Purpose |
|-----------|---------|
| `boundary-probing.md` | I/O characterization techniques |
| `causal-techniques.md` | Methods for establishing causality |
| `system-identification.md` | Parametric estimation algorithms (ARX, ARMAX, state-space) |
| `compositional-synthesis.md` | Math for combining sub-models |
| `setup-techniques.md` | Phase 0 framing procedures |
| `tools-sensitivity.md` | Binary tools and sensitivity analysis |
| `tool-catalog.md` | Tool recommendations by phase/domain |
| `simulation-guide.md` | Simulation paradigms, model conversion, validation bridge |
| `adversarial-heuristics.md` | Anti-analysis bypass, posture levels |

### Validation & Diagnostics
| Reference | Purpose |
|-----------|---------|
| `validation-checklist.md` | Consolidated validation requirements |
| `domain-calibration.md` | Plausibility bounds by domain |
| `red-flags.md` | Red flag catalog for invalid claims |
| `coherence-checks.md` | Quick coherence validation (60-second filter) |
| `cognitive-traps.md` | Countermeasures for 20+ analytical biases |
| `rapid-assessment.md` | RAPID tier workflow reference |

### Forecasting & Time Series
| Reference | Purpose |
|-----------|---------|
| `forecasting-science.md` | Forecastability (PE, FVA), model selection hierarchy, error metrics, conformal prediction |
| `timeseries-review.md` | ts_reviewer usage guide |
| `financial-validation.md` | Finance-specific forecasting validation |
| `session-memory.md` | Filesystem memory protocol for analysis sessions |

### Psychological Analysis (PSYCH tier)
| Reference | Purpose |
|-----------|---------|
| `psych-tier-protocol.md` | Complete PSYCH tier protocol |
| `archetype-mapping.md` | OCEAN, Dark Triad, MICE/RASP frameworks |
| `linguistic-markers.md` | Text analysis, deception markers |
| `elicitation-techniques.md` | Probing methods for trait discovery |
| `motive-analysis.md` | MICE/RASP motivation frameworks |
| `profile-synthesis.md` | Combining traits into unified models |

---

## Project Structure

```
epistemic-deconstructor/
├── README.md
├── CLAUDE.md                # AI assistant instructions
├── CHANGELOG.md
├── LICENSE                  # GNU GPLv3
├── Makefile                 # Unix/Linux/macOS build
├── build.ps1                # Windows PowerShell build
├── docs/
│   └── FORECAST_GUIDE.md    # Source forecasting science guide
├── tests/                   # 150 unit tests
│   ├── test_common.py
│   ├── test_bayesian_tracker.py
│   ├── test_belief_tracker.py
│   ├── test_rapid_checker.py
│   ├── test_session_manager.py
│   └── test_ts_reviewer.py
└── src/
    ├── SKILL.md             # Core protocol (6-phase methodology)
    ├── config/
    │   └── domains.json     # Domain calibration bounds
    ├── scripts/             # 7 Python CLI tools (~6,100 lines)
    │   ├── common.py
    │   ├── session_manager.py
    │   ├── bayesian_tracker.py
    │   ├── belief_tracker.py
    │   ├── rapid_checker.py
    │   ├── ts_reviewer.py
    │   └── simulator.py
    └── references/          # 25 knowledge base documents (~7,900 lines)
```

---

## Building

```bash
# Unix/Linux/macOS
make package                 # Create distributable zip
make package-combined        # Single-file skill with all references inlined
make validate                # Validate structure and cross-references
make test                    # Run test suite
make clean                   # Clean build artifacts

# Windows (PowerShell)
.\build.ps1 package
.\build.ps1 package-combined
.\build.ps1 validate
.\build.ps1 test
.\build.ps1 clean
```

---

## Core Principles

- **Falsify, don't confirm** — Design tests to break hypotheses, not support them
- **Quantify uncertainty** — Never report point estimates alone; use Bayesian posteriors and conformal intervals
- **Map ≠ Territory** — Models are tools, not truth
- **Beat naive first** — Every model must justify its complexity against simple baselines
- **Emergence is real** — Component models may not predict whole-system behavior

---

## License

[GNU General Public License v3.0](LICENSE)

**v6.6.3** — See [CHANGELOG.md](CHANGELOG.md) for full version history.
