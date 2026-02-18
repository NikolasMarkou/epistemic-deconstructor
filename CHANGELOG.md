# Changelog

All notable changes to the Epistemic Deconstructor project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
