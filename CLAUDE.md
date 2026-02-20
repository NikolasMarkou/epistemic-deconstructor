# CLAUDE.md

This file provides guidance for Claude (AI) when working with the Epistemic Deconstructor codebase.

## Project Purpose

**Epistemic Deconstructor v6.7** is a systematic framework for AI-assisted reverse engineering of unknown systems using scientific methodology. It transforms epistemic uncertainty into predictive control through principled experimentation, compositional modeling, and Bayesian inference.

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
    │   └── simulator.py         # Python CLI for simulation (SD, MC, ABM, DES, sensitivity)
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
        ├── rapid-assessment.md      # Consolidated RAPID tier workflow reference
        ├── timeseries-review.md     # Time-series signal review guide
        ├── session-memory.md        # Filesystem memory protocol for analysis sessions
        ├── simulation-guide.md      # Simulation paradigms, model conversion, validation bridge
        ├── evidence-calibration.md  # LR caps, anti-bundling, prior discipline, calibration rules
        ├── decision-trees.md        # Model selection, stopping, decomposition, tier escalation
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

# Coherence tracking
python3 src/scripts/bayesian_tracker.py coherence "data-task-match" --pass
python3 src/scripts/bayesian_tracker.py coherence "metric-task-match" --fail --notes "Wrong metrics"

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
#                    weak_counter, counter_indicator, strong_counter, disconfirm

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

### Simulator CLI

The `src/scripts/simulator.py` tool runs forward simulation on identified models. Requires numpy, scipy, matplotlib.

```bash
# System Dynamics (linear state-space)
python3 src/scripts/simulator.py sd \
  --model '{"A": [[0, 1], [-2, -3]], "B": [[0], [1]]}' \
  --x0 '[1.0, 0.0]' --u_func step --t_end 20 --dt 0.01 --plot --output sim_sd.json

# Monte Carlo (parameter uncertainty)
python3 src/scripts/simulator.py mc \
  --model '{"type": "arx", "a": [-0.5, 0.3], "b": [1.0]}' \
  --param_distributions '{"a[0]": {"dist": "normal", "mean": -0.5, "std": 0.05}}' \
  --n_runs 10000 --t_end 100 --seed 42 --convergence_check --plot --output sim_mc.json

# Agent-Based Model
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
```

See `src/references/simulation-guide.md` for domain fit gate, archetype mapping, model conversion recipes, and convergence diagnostics.

### Activating the Protocol

Users activate the protocol by:
1. Saying "Help me start" or "Walk me through" (triggers auto-pilot questionnaire mode)
2. Or: "Activate Epistemic Deconstruction Protocol"
3. For PSYCH tier: "Analyze this person" or "Profile this individual"

## The Phase Methodology

### System Analysis Phases

| Phase | Name | Budget | Output |
|-------|------|--------|--------|
| 0.5 | Coherence Screening | 5-10% | Go/No-Go Decision (RAPID tier) |
| 0 | Setup & Frame | 10% | Analysis Plan, Question Pyramid, Initial Hypotheses |
| 1 | Boundary Mapping | 20% | I/O Surface Map, Transfer Functions |
| 2 | Causal Analysis | 25% | Causal Graph, Dependency Matrix |
| 3 | Parametric ID | 20% | Mathematical Model, Uncertainty Bounds, FVA (ts_reviewer + forecasting-science) |
| 4 | Model Synthesis | 15% | Unified Model, Emergence Report, Simulation Output (simulator.py) |
| 5 | Validation | 10% | Validation Report, Conformal Intervals, Baseline/FVA, Simulation Bridge |

### PSYCH Tier Phases

| Phase | Name | Budget | Output |
|-------|------|--------|--------|
| 0-P | Context & Frame | 10% | Analysis Plan, Initial Hypotheses |
| 1-P | Baseline Calibration | 20% | Baseline Profile, Idiosyncrasy Index |
| 2-P | Stimulus-Response Mapping | 25% | Deviation Database, Trigger Map |
| 3-P | Structural Identification | 20% | OCEAN, Dark Triad, Cognitive Distortions |
| 4-P | Motive Synthesis | 15% | MICE Profile, Drive Matrix, Archetype |
| 5-P | Validation & Prediction | 10% | Validated Profile, Behavioral Predictions |

### Tier System

| Tier | When to Use | Phases | Budget |
|------|-------------|--------|--------|
| RAPID | Quick claim validation, red flag screening | 0.5→5 | <30min |
| LITE | Known archetype, stable system, single function | 0→1→5 | <2h |
| STANDARD | Unknown internals, single domain, no adversary | 0→1→2→3→4→5 | 2-20h |
| COMPREHENSIVE | Multi-domain, adversarial, critical, recursive | All + decomposition | 20h+ |
| PSYCH | Human persona/behavioral analysis | 0-P→1-P→2-P→3-P→4-P→5-P | 1-4h |

## Important Patterns

### State Block Protocol

Every response during analysis must end with a state block:
```
[STATE: Phase X | Tier: Y | Active Hypotheses: N | Confidence: Low/Med/High]
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
| KILLED/REFUTED | <= 0.05 | <= 0.10 | Behavioral evidence is noisier |
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
  - `simulator.py` for forward simulation (SD, MC, ABM, DES, sensitivity). Phase 4 uses archetype-to-paradigm mapping from `simulation-guide.md`. Phase 5 uses `bridge` command to validate predictions.
- **Tool integration flow**: Phase 3 (ts_reviewer diagnostics + forecasting-science model selection) → Phase 4 (simulator forward projection) → Phase 5 (ts_reviewer validation + simulator bridge). See `references/timeseries-review.md` for utility function usage per phase.

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

Record version results and analysis in `docs/FINDINGS.md`.

### Adding New Reference Material

Reference files should follow this pattern:
1. Clear section headers
2. Tables for quick reference
3. Code snippets where applicable
4. Cross-references to other reference files
5. **Files >100 lines MUST have a Table of Contents** at the top

---

## v6.4 Refinement Notes

### What Changed in v6.4

Comprehensive refinement addressing 42 identified issues across unclear concepts, incomplete workflows, and tool inconsistencies.

**SKILL.md Clarity:**
- Added Fidelity Levels table (L1-L5) with test criteria
- Added Tier Selection from Questionnaire mapping
- Added Archetype Signatures table
- Defined emergence mismatch formula
- Added LITE tier specifics
- Added executable recursive decomposition
- Added RAPID → Next Tier decision tree

**Script Improvements:**
- Unified CONFIRMED threshold to 0.90 (was 0.95 in bayesian_tracker)
- Added `remove` command for hypotheses
- Externalized domain calibration to `src/config/domains.json`
- Added proper exit codes (sys.exit(1)) on errors

**Tool Catalog:**
- Removed commercial tools (IDA Pro, Binary Ninja, MATLAB SI Toolbox)
- Removed niche tools (Intel PIN, DynamoRIO, KLEE, AFLNet)
- Focused on free/open-source ecosystem

### Known Limitations

**Content compression trade-offs:**
- Detailed phase tables (e.g., I/O channel types, probe signals) removed from SKILL.md — now only in reference files
- Psychological Axioms table removed from SKILL.md — now only in `src/references/psych-tier-protocol.md`
- Model structure selection heuristics condensed — full detail in `src/references/system-identification.md`
- Adversarial posture details condensed — full detail in `src/references/adversarial-heuristics.md`

**Cross-reference dependencies:**
- SKILL.md now requires loading reference files for full detail
- PSYCH tier users must load `src/references/psych-tier-protocol.md` for complete protocol
- Tool-specific guidance requires `src/references/tool-catalog.md`

**Tool catalog scope:**
- Focused on free/open-source tools (commercial tools removed for accessibility)
- System ID tools limited to Python ecosystem (SysIdentPy, SIPPY)
- Malware-specific tools (Cuckoo, Any.Run, binwalk) removed as out of scope

### Verification Checklist (Passed)

- [x] `.\build.ps1 validate` passes
- [x] SKILL.md <500 lines
- [x] SKILL.md <5,000 words
- [x] All cross-references in SKILL.md point to existing files
- [x] Decision trees preserved in SKILL.md
- [x] Stop conditions preserved for all phases
- [x] All reference files >100 lines have TOCs
- [x] CONFIRMED threshold unified to 0.90 across all scripts
- [x] Domain calibration externalized to src/config/domains.json
