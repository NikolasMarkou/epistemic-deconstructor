# Changelog

All notable changes to the Epistemic Deconstructor project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
