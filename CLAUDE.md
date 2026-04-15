# CLAUDE.md

This file provides guidance for Claude (AI) when working with the Epistemic Deconstructor codebase.

## Project Purpose

**Epistemic Deconstructor v7.11.0** is a systematic framework for AI-assisted reverse engineering of unknown systems using scientific methodology. It transforms epistemic uncertainty into predictive control through principled experimentation, compositional modeling, and Bayesian inference.

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
├── tests/                   # 432 unit tests (pytest)
│   ├── test_common.py
│   ├── test_bayesian_tracker.py
│   ├── test_belief_tracker.py
│   ├── test_rapid_checker.py
│   ├── test_session_manager.py
│   ├── test_ts_reviewer.py
│   ├── test_fourier_analyst.py
│   ├── test_forecast_modeler.py
│   ├── test_parametric_identifier.py
│   └── test_simulator.py
├── docs/                    # Design documentation
│   ├── SUBAGENT_REDESIGN.md     # Sub-agent architecture design
│   └── subagents.md             # Sub-agent reference documentation
└── src/
    ├── SKILL.md                 # Core protocol (6-phase methodology) - the main instruction set
    ├── config/
    │   └── domains.json         # Domain calibration data
    ├── scripts/
    │   ├── common.py            # Shared utilities (Bayesian math, JSON I/O with locking)
    │   ├── session_manager.py   # Python CLI for analysis session management
    │   ├── bayesian_tracker.py  # Python CLI for Bayesian hypothesis + flag tracking
    │   ├── belief_tracker.py    # Python CLI for PSYCH tier trait tracking
    │   ├── rapid_checker.py     # Python CLI for RAPID tier assessments
    │   ├── ts_reviewer.py       # Python CLI for time-series signal diagnostics
    │   ├── fourier_analyst.py   # Python CLI for frequency-domain spectral analysis
    │   ├── forecast_modeler.py  # Python CLI for forecasting model fitting & selection
    │   ├── parametric_identifier.py # Python CLI for structural system ID (ARX/ARMAX/NARMAX + bootstrap UQ)
    │   └── simulator.py         # Python CLI for simulation (SD, MC, ABM, DES, sensitivity)
    ├── agents/                  # Sub-agent definitions (Claude Code)
    │   ├── epistemic-orchestrator.md  # Main orchestrator (opus) — phase FSM, delegation
    │   ├── session-clerk.md           # Filesystem I/O handler (haiku, background)
    │   ├── hypothesis-engine.md       # Bayesian tracking + evidence rules (sonnet)
    │   ├── cognitive-auditor.md       # Bias/trap detection (sonnet, background)
    │   ├── rapid-screener.md          # Phase 0.5 RAPID screening (sonnet)
    │   ├── boundary-mapper.md         # Phase 1 I/O mapping (sonnet)
    │   ├── causal-analyst.md          # Phase 2 causal graphs + falsification (opus)
    │   ├── parametric-id.md           # Phase 3 model fitting (sonnet)
    │   ├── model-synthesizer.md       # Phase 4 composition + simulation (sonnet)
    │   ├── validator.md               # Phase 5 validation + report (opus)
    │   ├── psych-profiler.md          # PSYCH tier behavioral analysis (opus)
    │   └── research-scout.md          # Background web research (haiku)
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
        ├── tools-sensitivity.md      # Binary tools & sensitivity analysis
        ├── validation-checklist.md   # Consolidated validation requirements
        ├── tool-catalog.md           # Tool recommendations by phase/domain
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
        # PSYCH Tier References
        ├── psych-tier-protocol.md    # Complete PSYCH tier protocol (extracted from SKILL.md)
        ├── archetype-mapping.md      # OCEAN, Dark Triad, MICE/RASP frameworks
        ├── linguistic-markers.md     # Text analysis, deception markers
        ├── elicitation-techniques.md # Probing methods for trait discovery
        ├── motive-analysis.md        # MICE/RASP motivation frameworks
        └── profile-synthesis.md      # Combining traits into unified models
```

## Key Commands

### Session Manager CLI

The `src/scripts/session_manager.py` tool manages analysis session directories. Sessions persist all analysis state to the filesystem so context window loss doesn't destroy progress.

```bash
# Shorthand used in SKILL.md (resolve <skill-dir> and <project-dir> to absolute paths):
SM="python3 <skill-dir>/scripts/session_manager.py --base-dir <project-dir>"

# Session lifecycle
$SM new "Target system description"     # Create new session
$SM resume                              # Re-entry summary for new conversations
$SM status                              # One-line state summary
$SM close                               # Close session (merges to consolidated files)
$SM new --force "New system"            # Force-close existing and start new
$SM reopen 2 "Weak causal findings"     # Reopen Phase 2 for another pass
$SM reopen 0 "Wrong question framed"   # Outer loop: restart from Phase 0
$SM list                                # Show all sessions (active and closed)

# Session file I/O (resolves absolute paths internally — never need to construct paths)
$SM write state.md <<'EOF'              # Write content to session file
content here
EOF
$SM read state.md                       # Read session file to stdout
$SM read observations/obs_001.md        # Read from subdirectories
$SM path hypotheses.json                # Output absolute path (for --file flags)
$SM path                                # Output absolute session directory path
```

**IMPORTANT**: Use `$SM write`/`$SM read` for ALL session file operations. Do NOT use the Write/Read tools directly — they require absolute paths which cause errors with session files.

Session directory structure:
```
analyses/analysis_YYYY-MM-DD_XXXXXXXX/
├── state.md              # Current phase, tier, hypotheses summary
├── analysis_plan.md      # Phase 0 output
├── decisions.md          # Hypothesis pivots, approach changes
├── observations.md       # Index of observations
├── observations/         # Detailed observation files
├── progress.md           # Phase completion tracking
├── phase_outputs/        # One file per completed phase
├── validation.md         # Phase 5 results
└── summary.md            # Final report (written at close)
```

Redirect tracker scripts to session directory with `--file` (use `$SM path` for absolute paths):
```bash
python3 <skill-dir>/scripts/bayesian_tracker.py --file $($SM path hypotheses.json) add "H1" --prior 0.6
python3 <skill-dir>/scripts/belief_tracker.py --file $($SM path beliefs.json) add "Trait" --prior 0.5
python3 <skill-dir>/scripts/rapid_checker.py --file $($SM path rapid_assessment.json) start "Claim"
```

See `src/references/session-memory.md` for the full filesystem memory protocol (re-read rules, recovery, phase output templates).

### Bayesian Tracker CLI

The `src/scripts/bayesian_tracker.py` tool tracks hypothesis confidence using proper Bayesian inference. Extended with red flag tracking and coherence checking for RAPID tier.

```bash
# Add a hypothesis with prior probability
python3 src/scripts/bayesian_tracker.py add "System uses REST API" --prior 0.6 --phase P0

# Update with evidence using likelihood ratio presets
python3 src/scripts/bayesian_tracker.py update H1 "Found /api/v1 endpoint" --preset strong_confirm

# Available presets: strong_confirm, moderate_confirm, weak_confirm, neutral,
#                    weak_disconfirm, moderate_disconfirm, strong_disconfirm, falsify

# Or use explicit likelihood ratio
python3 src/scripts/bayesian_tracker.py update H1 "Evidence description" --lr 5.0

# Compare two hypotheses (Bayes factor)
python3 src/scripts/bayesian_tracker.py compare H1 H2

# Generate report
python3 src/scripts/bayesian_tracker.py report
python3 src/scripts/bayesian_tracker.py report --verbose  # Include evidence trail

# Red flag tracking
python3 src/scripts/bayesian_tracker.py flag add methodology "No baseline comparison"
python3 src/scripts/bayesian_tracker.py flag report
python3 src/scripts/bayesian_tracker.py flag remove F1    # Remove a flag by ID
python3 src/scripts/bayesian_tracker.py flag count         # Show flag count summary

# Coherence tracking
python3 src/scripts/bayesian_tracker.py coherence "data-task-match" --pass
python3 src/scripts/bayesian_tracker.py coherence "metric-task-match" --fail --notes "Wrong metrics"
python3 src/scripts/bayesian_tracker.py coherence-report   # Generate coherence report

# Verdict (for RAPID tier)
python3 src/scripts/bayesian_tracker.py verdict
python3 src/scripts/bayesian_tracker.py verdict --full
```

### Belief Tracker CLI (PSYCH Tier)

The `src/scripts/belief_tracker.py` tool tracks psychological trait confidence for PSYCH tier behavioral analysis.

```bash
# Set subject info
python3 src/scripts/belief_tracker.py subject "Subject Name" --context "Negotiation counterpart"

# Add trait hypothesis
python3 src/scripts/belief_tracker.py add "High Neuroticism" --category neuroticism --polarity high --prior 0.5

# Update with behavioral evidence
python3 src/scripts/belief_tracker.py update T1 "Catastrophizing language observed" --preset strong_indicator

# Available presets: smoking_gun, strong_indicator, indicator, weak_indicator, neutral,
#                    weak_counter, counter_indicator, strong_counter, disconfirm, falsify

# Track baseline observations
python3 src/scripts/belief_tracker.py baseline add "Uses 'we' frequently" --category linguistic
python3 src/scripts/belief_tracker.py baseline list

# Record deviation from baseline
python3 src/scripts/belief_tracker.py deviation "Switched to passive voice under pressure" --significance moderate

# Generate reports
python3 src/scripts/belief_tracker.py traits           # Trait assessment report
python3 src/scripts/belief_tracker.py baselines        # Baseline observation report
python3 src/scripts/belief_tracker.py profile          # Unified psychological profile
python3 src/scripts/belief_tracker.py report --verbose # Full report with evidence trail
```

### RAPID Checker CLI

The `src/scripts/rapid_checker.py` tool provides standalone 10-minute assessment for claim validation.

```bash
# Start assessment
python3 src/scripts/rapid_checker.py start "Paper: XYZ Claims"

# Record coherence checks
python3 src/scripts/rapid_checker.py coherence data-task-match --pass
python3 src/scripts/rapid_checker.py coherence metric-task-match --fail --notes "Classification metrics for regression"

# Add red flags
python3 src/scripts/rapid_checker.py flag methodology "No baseline comparison"
python3 src/scripts/rapid_checker.py flag results "Test > Train performance" --severity critical

# Check domain calibration
python3 src/scripts/rapid_checker.py calibrate accuracy 0.99 --domain ml_classification

# Get verdict and report
python3 src/scripts/rapid_checker.py verdict
python3 src/scripts/rapid_checker.py report
python3 src/scripts/rapid_checker.py status              # Quick status summary

# List available domains
python3 src/scripts/rapid_checker.py domains
```

### Time-Series Reviewer CLI

The `src/scripts/ts_reviewer.py` tool provides systematic signal diagnostics for time-series outputs from systems under investigation.

```bash
# Full 10-phase review from CSV
python3 src/scripts/ts_reviewer.py review data.csv --column value

# Quick review (phases 1-6 only, no model needed)
python3 src/scripts/ts_reviewer.py quick data.csv --column temperature --freq 12

# Built-in demo with synthetic data
python3 src/scripts/ts_reviewer.py demo
```

**Extended metrics** (available programmatically): `_rmsse`, `_wape`, `_me_bias`, `_pinball_loss`, `_fva`, `_permutation_entropy`. Phase 4 now includes Permutation Entropy; Phase 6 reports Forecast Value Added when predictions supplied. `cqr_intervals()` provides Conformalized Quantile Regression intervals alongside existing `conformal_intervals()`. See `src/references/forecasting-science.md` for methodology.

### Fourier Analyst CLI

The `src/scripts/fourier_analyst.py` tool provides frequency-domain spectral analysis for signals produced by systems under investigation. Complements the time-domain `ts_reviewer.py`. Requires numpy; scipy optional for advanced features (Welch PSD, STFT, system identification).

```bash
# Full 9-phase spectral analysis from CSV
python3 src/scripts/fourier_analyst.py analyze data.csv --column signal --fs 1000

# With known fundamental frequency for harmonic analysis
python3 src/scripts/fourier_analyst.py analyze data.csv --column signal --fs 1000 --fundamental 50

# With shaft speed for vibration diagnostics and JSON output
python3 src/scripts/fourier_analyst.py analyze vibration.csv --column accel_z --fs 10000 \
    --shaft-rpm 3600 --output analysis.json

# Quick review (phases 1-5 only, no model needed)
python3 src/scripts/fourier_analyst.py quick data.csv --column voltage --fs 44100

# Compare multiple signals
python3 src/scripts/fourier_analyst.py compare sensors.csv --columns ch1,ch2,ch3 --fs 1000

# Built-in demo with synthetic data
python3 src/scripts/fourier_analyst.py demo
```

**9-phase analysis**: (1) Spectral Profile — FFT, PSD, dominant frequencies; (2) Harmonic Analysis — fundamental, THD, sidebands; (3) Windowing Quality — leakage, window recommendations; (4) Noise Floor — SNR, dynamic range, noise color; (5) Bandwidth — rolloff, centroid, energy distribution; (6) System Identification — transfer function, coherence (needs I/O pair); (7) Spectral Anomaly — baseline comparison; (8) Time-Frequency — STFT, stationarity; (9) System Health — vibration diagnostics, bearing faults. See `src/references/spectral-analysis.md` for full guide.

**Utility functions** (available programmatically): `quick_spectrum()`, `compare_spectra()`, `transfer_function()`, `spectral_distance()`, `band_energy_profile()`.

### Forecast Modeler CLI

The `src/scripts/forecast_modeler.py` tool fits forecasting models to time-series data and produces predictions with calibrated uncertainty. Complements ts_reviewer.py (diagnostics) by providing the model-fitting step. Optional dependencies: numpy, statsmodels, pmdarima, catboost (all optional — stdlib-only mode provides forecastability assessment + naive baselines + conformal intervals).

```bash
# Full pipeline from CSV
python3 src/scripts/forecast_modeler.py fit data.csv --column value --horizon 12 --freq 12 --coverage 0.95 --output forecast.json

# Forecastability assessment only (Phase 1)
python3 src/scripts/forecast_modeler.py assess data.csv --column value --freq 12

# Multi-model comparison table
python3 src/scripts/forecast_modeler.py compare data.csv --column value --horizon 12 --freq 12

# Built-in demo with synthetic data
python3 src/scripts/forecast_modeler.py demo
```

**7-phase analysis**: (1) Forecastability Gate — PE, naive baselines, go/no-go; (2) Classical Fitting — Auto-ARIMA, Auto-ETS, best by AICc; (3) ML Fitting — CatBoost with feature engineering; (4) Model Comparison — MASE, RMSSE, WAPE, FVA; (5) Conformal Prediction — ICP, CQR; (6) Forecast Generation — point + intervals; (7) Report Summary. See `src/references/forecasting-tools.md` for full guide.

**Utility functions** (available programmatically): `auto_forecast()`, `compare_forecasters()`, `conformal_forecast()`.

### Parametric Identifier CLI

The `src/scripts/parametric_identifier.py` tool performs **structural system identification** — fits ARX, ARMAX, and NARMAX models with automatic structure selection via AIC/BIC, bootstrap parameter confidence intervals, integrated Ljung-Box whiteness testing, and walk-forward cross-validation. This is the Phase 3 primary tool when the deliverable is **structure + parameters + uncertainty** (to feed simulator.py in Phase 4 and validator in Phase 5), as distinct from `forecast_modeler.py` which targets future-value prediction. Requires numpy; statsmodels optional for ARMAX (graceful fallback); scipy optional for improved SNR/coherence in `assess`.

```bash
# Identifiability gate (Phase 3 entry check: data length, SNR, coherence)
python3 src/scripts/parametric_identifier.py assess data.csv --column y --input-column u

# Unified multi-family comparison (ARX vs ARMAX vs NARMAX, ranked by BIC with whiteness gate)
python3 src/scripts/parametric_identifier.py compare data.csv --column y --input-column u \
    --families arx,armax,narmax --cv-folds 5 --output compare.json

# Single-family grid search with bootstrap CIs and walk-forward CV
python3 src/scripts/parametric_identifier.py fit data.csv --column y --input-column u \
    --family arx --grid --criterion bic --bootstrap 500 --cv-folds 5 --output arx_fit.json

# Single-structure fit (user specifies orders)
python3 src/scripts/parametric_identifier.py fit data.csv --column y --input-column u \
    --family arx --na 2 --nb 1 --nk 1 --bootstrap 500

# Autoregressive fit (no input — AR/ARMA/NAR)
python3 src/scripts/parametric_identifier.py fit data.csv --column y --family arx --grid

# Built-in demo with synthetic ARX(2,1,1) data
python3 src/scripts/parametric_identifier.py demo
```

**Model families**:
- **ARX** `A(q)·y = B(q)·u + e` — OLS via QR decomposition, structure grid search over `(na, nb, nk)`
- **ARMAX** `A(q)·y = B(q)·u + C(q)·e` — SARIMAX backend (requires statsmodels), analytic CIs from `cov_params` plus optional bootstrap
- **NARMAX** — polynomial basis expansion with FROLS (Forward Regression Orthogonal Least Squares) term selection via Error Reduction Ratio

**Uncertainty quantification**: Residual bootstrap (temporally safe — regenerates y via forward simulation through fitted model rather than pair-resampling) for all families. Analytic CIs as cheap fallback for ARX (`(Φᵀ Φ)⁻¹ σ²`) and ARMAX (`cov_params()`). **No PyMC or heavy Bayesian deps.**

**Integration**: Fitted ARX output drops directly into `simulator.py` ARX mode via `FitResult.to_simulator_format()` → `{type: 'arx', a, b, nk}`. This closes the Phase 3 → Phase 4 → Phase 5 loop for linear systems (fit → simulate → validate).

**Utility functions** (available programmatically): `fit_arx()`, `fit_arx_grid()`, `fit_armax()`, `fit_armax_grid()`, `fit_narmax()`, `compare_structures()`, `assess_identifiability()`, `walk_forward_cv()`, `residual_bootstrap()`, `ljung_box()`, `compute_criteria()`, `build_arx_regressors()`, `polynomial_basis()`, `frols()`.

### Simulator CLI

The `src/scripts/simulator.py` tool runs forward simulation on identified models. Requires numpy, scipy, matplotlib.

```bash
# System Dynamics (linear state-space)
python3 src/scripts/simulator.py sd \
  --model '{"A": [[0, 1], [-2, -3]], "B": [[0], [1]]}' \
  --x0 '[1.0, 0.0]' --u_func step --t_end 20 --dt 0.01 --plot --output sim_sd.json

# SD additional options: --amplitude, --freq, --t_on (input signal params),
#   --integrator euler|rk4|rk45, --verbose, --report

# Monte Carlo (parameter uncertainty)
python3 src/scripts/simulator.py mc \
  --model '{"type": "arx", "a": [-0.5, 0.3], "b": [1.0]}' \
  --param_distributions '{"a[0]": {"dist": "normal", "mean": -0.5, "std": 0.05}}' \
  --n_runs 10000 --t_end 100 --seed 42 --convergence_check --plot --output sim_mc.json

# MC additional options: --amplitude, --verbose, --report

# Agent-Based Model (topologies: complete, grid, small_world, scale_free)
python3 src/scripts/simulator.py abm \
  --config abm_config.json --n_agents 1000 --t_steps 500 --topology small_world --seed 42 --plot --output sim_abm.json

# Discrete-Event Simulation
python3 src/scripts/simulator.py des --config des_config.json --t_end 10000 --seed 42 --plot --output sim_des.json

# Sensitivity Analysis
python3 src/scripts/simulator.py sensitivity \
  --model_func 'k1 * np.sin(k2 * x) + k3 * x**2' \
  --param_ranges '{"k1": [0.1, 10], "k2": [0, 3.14], "k3": [0, 1], "x": [-5, 5]}' \
  --method sobol --n_samples 4096 --plot --output sens.json

# Validation bridge (feeds simulation output to Phase 5)
python3 src/scripts/simulator.py bridge --sim_output sim_mc.json --output validation_bridge.json

# All modes support: --verbose (detailed output), --report FILE (markdown report to path)
```

See `src/references/simulation-guide.md` for domain fit gate, archetype mapping, model conversion recipes, and convergence diagnostics.

### Sub-Agent Architecture

The protocol can run as a coordinated multi-agent system using Claude Code subagents. Agent definitions are in `src/agents/`.

**Installation** (copy agents to where Claude Code discovers them):
```bash
# Project-level (recommended — version-controlled):
mkdir -p .claude/agents && cp src/agents/*.md .claude/agents/

# User-level (available across all projects):
mkdir -p ~/.claude/agents && cp src/agents/*.md ~/.claude/agents/

# Via build system (syncs skill + agents to ~/.claude/skills/):
make sync-skill
```

**Usage**:
```bash
# Run the full orchestrated system:
claude --agent epistemic-orchestrator

# Or reference agents in conversation:
# "Use the boundary-mapper agent to map this system's I/O"
# "Have the cognitive-auditor check for bias in the analysis"
```

**Agent Roles**:

| Agent | Model | Role |
|-------|-------|------|
| epistemic-orchestrator | opus | Phase FSM, tier selection, delegation (main agent) |
| session-clerk | haiku | Filesystem I/O for session files (background) |
| hypothesis-engine | sonnet | Bayesian tracking + evidence rule enforcement |
| cognitive-auditor | sonnet | Independent bias/trap detection (background) |
| rapid-screener | sonnet | Phase 0.5 coherence screening |
| boundary-mapper | sonnet | Phase 1 I/O probing |
| causal-analyst | opus | Phase 2 causal graphs + falsification |
| parametric-id | sonnet | Phase 3 model fitting (ts_reviewer, forecast_modeler, fourier_analyst) |
| model-synthesizer | sonnet | Phase 4 composition + simulation |
| validator | opus | Phase 5 validation + final report |
| psych-profiler | opus | PSYCH tier behavioral analysis |
| research-scout | haiku | Background web research |

See `docs/SUBAGENT_REDESIGN.md` for the full architectural design.

### Activating the Protocol

Users activate the protocol by:
1. Saying "Help me start" or "Walk me through" (triggers auto-pilot questionnaire mode)
2. Or: "Activate Epistemic Deconstruction Protocol"
3. For PSYCH tier: "Analyze this person" or "Profile this individual"
4. **Multi-agent mode**: `claude --agent epistemic-orchestrator`

## The Phase Methodology

### System Analysis Phases

| Phase | Name | Output |
|-------|------|--------|
| 0.5 | Coherence Screening | Go/No-Go Decision (RAPID tier) |
| 0 | Setup & Frame | Analysis Plan, Question Pyramid, Initial Hypotheses |
| 1 | Boundary Mapping | I/O Surface Map, Transfer Functions |
| 2 | Causal Analysis | Causal Graph, Dependency Matrix |
| 3 | Parametric ID | Mathematical Model, Uncertainty Bounds, FVA (ts_reviewer + forecasting-science) |
| 4 | Model Synthesis | Unified Model, Emergence Report, Simulation Output (simulator.py) |
| 5 | Validation | Validation Report, Conformal Intervals, Baseline/FVA, Simulation Bridge |

### PSYCH Tier Phases

| Phase | Name | Output |
|-------|------|--------|
| 0-P | Context & Frame | Analysis Plan, Initial Hypotheses |
| 1-P | Baseline Calibration | Baseline Profile, Idiosyncrasy Index |
| 2-P | Stimulus-Response Mapping | Deviation Database, Trigger Map |
| 3-P | Structural Identification | OCEAN, Dark Triad, Cognitive Distortions |
| 4-P | Motive Synthesis | MICE Profile, Drive Matrix, Archetype |
| 5-P | Validation & Prediction | Validated Profile, Behavioral Predictions |

### Tier System

| Tier | When to Use | Phases |
|------|-------------|--------|
| RAPID | Quick claim validation, red flag screening | 0.5→5 |
| LITE | Known archetype, stable system, single function | 0→1→5 |
| STANDARD | Unknown internals, single domain, no adversary | 0→1→2→3→4→5 |
| COMPREHENSIVE | Multi-domain, adversarial, critical, recursive | All + decomposition |
| PSYCH | Human persona/behavioral analysis | 0-P→1-P→2-P→3-P→4-P→5-P |

## Important Patterns

### State Block Protocol

Every response during analysis must end with a state block:
```
[STATE: Phase X | Tier: Y | Active Hypotheses: N | Lead: HN (PP%) | Confidence: Low/Med/High]
```

Example:
```
[STATE: Phase 2 | Tier: STANDARD | Active Hypotheses: 3 | Lead: H2 (78%) | Confidence: Medium]
```

PSYCH tier state block:
```
[STATE: Phase 2-P | Tier: PSYCH | Archetype: High-N/Low-A | Rapport: Med | Stress: Low]
```

### Bayesian Hypothesis Tracking

- Maintain 3+ hypotheses at all times (including adversarial/deceptive alternatives)
- Use likelihood ratios for updates: LR > 1 confirms, LR < 1 disconfirms, LR = 0 falsifies
- Track posteriors, not gut feelings

### Threshold Bands (Intentionally Different by Domain)

| Status | bayesian_tracker (System) | belief_tracker (PSYCH) | Rationale |
|--------|--------------------------|----------------------|-----------|
| CONFIRMED | >= 0.90 | >= 0.90 | Unified in v6.4 |
| REFUTED | <= 0.05 | <= 0.10 | Behavioral evidence is noisier |
| WEAKENED | <= 0.20 | <= 0.30 | Wider band for ambiguous signals |
| ACTIVE | otherwise | otherwise | |

### Core Axioms

- **Falsify, don't confirm**: Design tests to break hypotheses
- **Quantify uncertainty**: Never report point estimates alone
- **Map ≠ Territory**: Models are tools, not truth
- **Emergence is real**: Component models may not predict whole-system behavior

### Psychological Axioms (PSYCH Tier)

- **Baseline is God**: Only deviation from baseline is significant
- **Rational Actor Fallacy**: Humans are predictably irrational
- **Projection Trap**: You assume they think like you
- **Mask vs. Face**: Presented self ≠ Actual self

### Cognitive Trap Awareness

Always check for:
- Mirror-imaging ("I would do X, so they did")
- Confirmation bias (only finding supporting evidence)
- Anchoring (first hypothesis dominates)
- Dunning-Kruger (early overconfidence)
- Tool Worship/Cargo-Cult ("We used fancy tool X, so results are valid")

PSYCH tier additional traps:
- Counter-Transference (projecting your feelings onto subject)
- Fundamental Attribution Error (character vs. situation)
- Halo/Horn Effect (one trait colors all assessment)
- Narrative Fallacy (smoothing over contradictions)

See `src/references/cognitive-traps.md` for full catalog.

### RAPID Tier Validation

For quick claim validation:
1. Coherence checks (data-task alignment, metric-task alignment)
2. Red flag scan (methodology, documentation, results, claims)
3. Domain calibration (compare to plausibility bounds)
4. Verdict: CREDIBLE / SKEPTICAL / DOUBTFUL / REJECT

See `src/references/red-flags.md`, `src/references/coherence-checks.md`, `src/references/domain-calibration.md`.

### PSYCH Tier Analysis

For psychological profiling:
1. Context & Frame (relationship dynamics, objectives)
2. Baseline Calibration (linguistic, emotional, timing patterns)
3. Stimulus-Response (elicitation probes, stress testing)
4. Structural ID (OCEAN, Dark Triad, cognitive distortions)
5. Motive Synthesis (MICE/RASP, drive matrix)
6. Validation (predictions, interaction strategy)

See `src/references/archetype-mapping.md`, `src/references/linguistic-markers.md`, `src/references/elicitation-techniques.md`, `src/references/motive-analysis.md`, `src/references/profile-synthesis.md`.

## Working with This Codebase

### File Modification Guidelines

- **src/SKILL.md** is the core protocol. Changes here affect all analysis behavior.
- **src/references/** files provide domain-specific knowledge. Add new reference files for new domains.
- **src/scripts/** contains executable tooling:
  - `bayesian_tracker.py` for system analysis hypothesis tracking
  - `belief_tracker.py` for psychological trait tracking
  - `rapid_checker.py` for RAPID tier assessments
  - `ts_reviewer.py` for time-series signal diagnostics, forecasting validation, and conformal prediction intervals. Also provides `compare_models()`, `walk_forward_split()`, `conformal_intervals()`, and `cqr_intervals()` as standalone functions for Phase 3/5.
  - `forecast_modeler.py` for forecasting model fitting (ARIMA, ETS, CatBoost) with conformal prediction intervals. Phase 3 uses `fit`/`compare` for forecasting model selection; Phase 5 uses conformal Phase 5 for calibrated intervals. Provides `auto_forecast()`, `compare_forecasters()`, `conformal_forecast()` as standalone functions.
  - `parametric_identifier.py` for **structural** system identification (ARX, ARMAX, NARMAX) with OLS/QR estimation, FROLS term selection, AIC/BIC/AICc/FPE across families, residual bootstrap parameter CIs, integrated Ljung-Box whiteness, and walk-forward CV. Phase 3 uses `assess`/`fit`/`compare` when the deliverable is structure + parameters (not forecasts). Fitted ARX output feeds `simulator.py` directly via `to_simulator_format()`.
  - `simulator.py` for forward simulation (SD, MC, ABM, DES, sensitivity). Phase 4 uses archetype-to-paradigm mapping from `simulation-guide.md`. Consumes ARX dicts from `parametric_identifier.py`. Phase 5 uses `bridge` command to validate predictions.
- **Tool integration flow**: Phase 1 (fourier_analyst spectral profiling + ts_reviewer signal quality) → Phase 3 (ts_reviewer diagnostics + fourier_analyst transfer functions + **parametric_identifier for ARX/ARMAX/NARMAX structural fit** + forecast_modeler for forecasting fit) → Phase 4 (simulator forward projection consuming parametric_identifier output) → Phase 5 (forecast_modeler conformal prediction + ts_reviewer residual validation + fourier_analyst spectral anomaly + simulator bridge). See `references/timeseries-review.md`, `references/forecasting-tools.md`, `references/system-identification.md`, and `references/spectral-analysis.md` for utility function usage per phase.

### Tech Stack

- Python 3.x (for tracker scripts)
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

