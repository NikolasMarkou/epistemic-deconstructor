# Epistemic Deconstructor

[![License](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-v6.9.0-green.svg)](CHANGELOG.md)
[![Tests](https://img.shields.io/badge/Tests-191%20passing-brightgreen.svg)](tests/)
[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](src/scripts/)
[![Sponsored by Electi](https://img.shields.io/badge/Sponsored%20by-Electi-orange.svg)](https://www.electiconsulting.com)

A [Claude Code skill](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/skills) for systematic reverse engineering of unknown systems using scientific methodology. Give it a black box — software, hardware, organizational, or human — and it walks Claude through a rigorous 6-phase protocol to build a predictive model of how it works.

Bayesian hypothesis tracking, falsification-driven experimentation, compositional modeling, and conformal prediction turn "I don't know how this works" into a validated, quantified understanding.

> **What's a Claude Code skill?** A reusable instruction set that extends Claude's capabilities for a specific domain. Install it into Claude Code's skills directory, and Claude gains a structured workflow it can execute on demand.

---

## Quick Start

### Claude Code (recommended)

```bash
git clone https://github.com/NikolasMarkou/epistemic-deconstructor.git ~/.claude/skills/epistemic-deconstructor
```

Claude Code picks it up automatically from `~/.claude/skills/`. Then say: **"Help me start"**

### Single file

Download `epistemic-deconstructor-combined.md` from [Releases](https://github.com/NikolasMarkou/epistemic-deconstructor/releases). Paste into Claude's Custom Instructions or a Project. Bundles SKILL.md + all references into one file (scripts not included).

### Full package

Download the zip from [Releases](https://github.com/NikolasMarkou/epistemic-deconstructor/releases). Upload `src/` to a Claude Project. Includes the protocol, all references, Python CLI tools, and domain calibration config.

---

## What It Does

You describe a system you want to understand. The skill guides Claude through a phased methodology:

| Phase | Name | What Happens |
|-------|------|-------------|
| **0** | Setup & Frame | Define scope, seed 3+ competing hypotheses, pick fidelity target |
| **0.5** | Screening | Quick coherence/red-flag check for external claims (RAPID tier) |
| **1** | Boundary Mapping | Enumerate I/O channels, apply probe signals, build stimulus-response database |
| **2** | Causal Analysis | Differential analysis, sensitivity testing, build causal graph, falsify hypotheses |
| **3** | Parametric ID | Fit mathematical models, quantify parameter uncertainty, forecast validation |
| **4** | Model Synthesis | Compose sub-models, test for emergence, classify archetype, run simulations |
| **5** | Validation | Beat naive baselines, conformal prediction, adversarial assessment, final report |

Every phase has an **EXIT GATE** — a checklist of files that must be written before advancing. No shortcuts, no skipping.

Every response tracks state:
```
[STATE: Phase 2 | Tier: STANDARD | Active Hypotheses: 3 | Lead: H2 (78%) | Confidence: Medium]
```

### Use Cases

- **Black-box analysis** — unknown systems (software, hardware, biological, organizational)
- **Competitive intelligence** — reverse engineering how a system or organization operates
- **Security analysis** — attack surface mapping, vulnerability assessment
- **Forensics** — root cause analysis, incident reconstruction
- **Claim validation** — papers, vendor pitches, forecasts (RAPID tier)
- **Time-series diagnostics** — forecastability assessment, conformal prediction
- **Psychological profiling** — behavioral analysis and persona mapping (PSYCH tier)

---

## Tiers

The protocol scales to the problem:

| Tier | When to Use | Phases | Budget |
|------|-------------|--------|--------|
| **RAPID** | Validating a paper, vendor pitch, or external claim | 0.5 &rarr; 5 | <30 min |
| **LITE** | Known system type, single function, stable behavior | 0 &rarr; 1 &rarr; 5 | <2 hr |
| **STANDARD** | Unknown internals, single domain, no adversary | 0 &rarr; 1 &rarr; 2 &rarr; 3 &rarr; 4 &rarr; 5 | 2-20 hr |
| **COMPREHENSIVE** | Multi-domain, adversarial, critical infrastructure | All + recursive decomposition | 20+ hr |
| **PSYCH** | Behavioral analysis of a person or persona | 0-P &rarr; 1-P &rarr; 2-P &rarr; 3-P &rarr; 4-P &rarr; 5-P | 1-4 hr |

---

## Session Management

Analysis state persists to the filesystem. Context window loss doesn't destroy progress. The session manager handles all file I/O — Claude never constructs file paths directly.

```bash
SM="python3 <skill-dir>/scripts/session_manager.py --base-dir <project-dir>"

# Create a new analysis session
$SM new "Target system description"

# Write a session file (content via heredoc)
$SM write state.md <<'EOF'
# Session State
- **Phase**: 1 (Boundary Mapping)
EOF

# Read a session file
$SM read state.md

# Get absolute path (for tracker --file flags)
$SM path hypotheses.json

# Resume in a new conversation
$SM resume

# One-line status
$SM status

# Close session
$SM close
```

Session directory structure:
```
analyses/analysis_YYYY-MM-DD_XXXXXXXX/
├── state.md              # Current phase, tier, hypotheses summary
├── analysis_plan.md      # Phase 0 output
├── hypotheses.json       # Bayesian hypothesis tracker state
├── decisions.md          # Hypothesis pivots, approach changes
├── observations.md       # Index of observations
├── observations/         # Detailed observation files
├── progress.md           # Phase completion tracking
├── phase_outputs/        # One file per completed phase
├── validation.md         # Phase 5 validation results
└── summary.md            # Final report (Phase 5 only)
```

---

## CLI Tools

Eight Python scripts. Most use stdlib only — `simulator.py` and `fourier_analyst.py` require numpy; scipy optional for advanced features.

### Bayesian Hypothesis Tracker

```bash
python3 src/scripts/bayesian_tracker.py --file $($SM path hypotheses.json) add "System uses REST API" --prior 0.6
python3 src/scripts/bayesian_tracker.py --file $($SM path hypotheses.json) update H1 "Found /api/v1 endpoint" --preset moderate_confirm
python3 src/scripts/bayesian_tracker.py --file $($SM path hypotheses.json) report --verbose
python3 src/scripts/bayesian_tracker.py --file $($SM path hypotheses.json) verdict   # RAPID tier
```

Presets: `strong_confirm`, `moderate_confirm`, `weak_confirm`, `neutral`, `weak_disconfirm`, `moderate_disconfirm`, `strong_disconfirm`, `falsify`.

### Psychological Trait Tracker (PSYCH tier)

```bash
python3 src/scripts/belief_tracker.py --file $($SM path beliefs.json) add "High Neuroticism" --prior 0.5
python3 src/scripts/belief_tracker.py --file $($SM path beliefs.json) update T1 "Catastrophizing language" --preset strong_indicator
python3 src/scripts/belief_tracker.py --file $($SM path beliefs.json) profile     # Unified OCEAN + Dark Triad + MICE profile
```

### RAPID Claim Validator

```bash
python3 src/scripts/rapid_checker.py start "Paper: XYZ Claims"
python3 src/scripts/rapid_checker.py flag methodology "No baseline comparison"
python3 src/scripts/rapid_checker.py calibrate accuracy 0.99 --domain ml_classification
python3 src/scripts/rapid_checker.py verdict
```

### Time-Series Signal Reviewer

10-phase systematic diagnostics for any time-ordered numeric signal:

```bash
python3 src/scripts/ts_reviewer.py review data.csv --column value
python3 src/scripts/ts_reviewer.py quick data.csv --column temperature --freq 12
python3 src/scripts/ts_reviewer.py demo
```

**Phases**: Coherence, Data Quality, Stationarity (ADF/KPSS), Forecastability (ACF, Permutation Entropy), Decomposition (STL), Baseline Benchmarks (naive/seasonal/drift + FVA), Overfitting Screen, Residual Diagnostics, Uncertainty Calibration (coverage, Winkler), Regime Analysis.

**Programmatic API**: `TimeSeriesReviewer`, `quick_review()`, `compare_models()`, `walk_forward_split()`, `conformal_intervals()`, `cqr_intervals()`.

### Fourier / Spectral Analyst

9-phase frequency-domain analysis for signals from physical, mechanical, or digital systems (requires numpy; scipy optional):

```bash
python3 src/scripts/fourier_analyst.py analyze data.csv --column signal --fs 1000
python3 src/scripts/fourier_analyst.py quick data.csv --column voltage --fs 44100
python3 src/scripts/fourier_analyst.py compare sensors.csv --columns ch1,ch2,ch3 --fs 1000
python3 src/scripts/fourier_analyst.py demo
```

**Phases**: Spectral Profile (FFT/PSD), Harmonic Analysis (THD, sidebands), Windowing Quality, Noise Floor (SNR, noise color), Bandwidth, System Identification (transfer function, coherence), Spectral Anomaly Detection, Time-Frequency (STFT), System Health (vibration diagnostics, bearing faults).

**Programmatic API**: `FourierAnalyst`, `quick_spectrum()`, `compare_spectra()`, `transfer_function()`, `spectral_distance()`, `band_energy_profile()`.

### Simulator

Forward simulation engine for identified models (requires numpy, scipy, matplotlib):

```bash
python3 src/scripts/simulator.py sd --model '{"A": [[0,1],[-2,-3]], "B": [[0],[1]]}' --x0 '[1,0]' --t_end 20 --plot
python3 src/scripts/simulator.py mc --model '{"a": [-0.5], "b": [1.0]}' --param_distributions '...' --n_runs 1000
python3 src/scripts/simulator.py sensitivity --model_func 'k1*x + k2*x**2' --param_ranges '...' --method sobol
```

**Modes**: System Dynamics (SD), Monte Carlo (MC), Agent-Based (ABM), Discrete-Event (DES), Sensitivity Analysis (Morris/Sobol/OAT). Validation bridge feeds results back to Phase 5.

---

## Evidence Rules

The protocol enforces calibration discipline to prevent systematic Bayesian tracking errors:

1. **LR caps** — Max LR=5.0 in Phases 0-1, LR=10.0 in Phases 2+ (requires justification)
2. **No batch evidence** — Each data point gets its own `update` call. No bundling.
3. **Adversarial hypothesis** — At least one hypothesis must test data reliability or concealment. Non-negotiable.
4. **Consensus ≠ strong evidence** — Forecaster/institutional consensus capped at LR=2.5
5. **Disconfirm before confirm** — Before any hypothesis exceeds 0.80, apply at least one disconfirming evidence
6. **Prior discipline** — Mutually exclusive priors must sum to 1.0 (±0.01)

See `src/references/evidence-calibration.md` for the full calibration guide.

---

## Knowledge Base

28 reference documents organized by domain:

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
| `decision-trees.md` | Tier escalation, stopping criteria, decomposition triggers |

### Validation & Diagnostics
| Reference | Purpose |
|-----------|---------|
| `validation-checklist.md` | Consolidated validation requirements |
| `domain-calibration.md` | Plausibility bounds by domain |
| `red-flags.md` | Red flag catalog for invalid claims |
| `coherence-checks.md` | Quick coherence validation (60-second filter) |
| `cognitive-traps.md` | Countermeasures for 20+ analytical biases |
| `rapid-assessment.md` | RAPID tier workflow reference |
| `evidence-calibration.md` | LR caps, anti-bundling, prior discipline |

### Forecasting & Time Series
| Reference | Purpose |
|-----------|---------|
| `forecasting-science.md` | Forecastability (PE, FVA), model selection, error metrics, conformal prediction |
| `timeseries-review.md` | ts_reviewer usage guide |
| `spectral-analysis.md` | fourier_analyst usage guide |
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
├── tests/                   # 191 unit tests
│   ├── test_common.py
│   ├── test_bayesian_tracker.py
│   ├── test_belief_tracker.py
│   ├── test_rapid_checker.py
│   ├── test_session_manager.py
│   ├── test_ts_reviewer.py
│   └── test_simulator.py
└── src/
    ├── SKILL.md             # Core protocol (477 lines, 6-phase methodology)
    ├── config/
    │   └── domains.json     # Domain calibration bounds
    ├── scripts/             # 8 Python CLI tools (~7,300 lines)
    │   ├── common.py        # Shared utilities (Bayesian math, JSON I/O with locking)
    │   ├── session_manager.py  # Session management + file I/O routing
    │   ├── bayesian_tracker.py # Hypothesis tracking with Bayesian inference
    │   ├── belief_tracker.py   # PSYCH tier trait tracking
    │   ├── rapid_checker.py    # RAPID tier claim validation
    │   ├── ts_reviewer.py      # Time-series signal diagnostics
    │   ├── fourier_analyst.py  # Frequency-domain spectral analysis
    │   └── simulator.py        # Forward simulation (SD, MC, ABM, DES, sensitivity)
    └── references/          # 28 knowledge base documents (~8,600 lines)
```

---

## Building

```bash
# Unix/Linux/macOS
make package                 # Create distributable zip
make package-combined        # Single-file skill with all references inlined
make validate                # Validate structure and cross-references
make test                    # Run 191 unit tests
make sync-skill              # Sync to ~/.claude/skills/epistemic-deconstructor
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
- **Map ≠ Territory** — Models are tools, not truth. Document HOW your model is wrong
- **Beat naive first** — Every model must justify its complexity against simple baselines
- **Emergence is real** — Component models may not predict whole-system behavior
- **Files are truth** — If it's not written to a session file, it didn't happen
- **Gate checks are non-negotiable** — Every phase transition requires disk writes

---

## License

[GNU General Public License v3.0](LICENSE)

**v6.9.0** — See [CHANGELOG.md](CHANGELOG.md) for full version history.
