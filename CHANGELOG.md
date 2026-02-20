# Changelog

All notable changes to the Epistemic Deconstructor project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
