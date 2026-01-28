# Changelog

All notable changes to the Epistemic Deconstructor project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
