# CLAUDE.md

This file provides guidance for Claude (AI) when working with the Epistemic Deconstructor codebase.

## Project Purpose

**Epistemic Deconstructor v6.0** is a systematic framework for AI-assisted reverse engineering of unknown systems using scientific methodology. It transforms epistemic uncertainty into predictive control through principled experimentation, compositional modeling, and Bayesian inference.

Use cases include:
- Black-box analysis of unknown systems (software, hardware, biological, organizational)
- Competitive intelligence and system interrogation
- Security analysis and attack surface mapping
- Forensics and root cause analysis
- Building predictive models from observations

## Repository Structure

```
epistemic-deconstructor/
├── SKILL.md                 # Core protocol (6-phase methodology) - the main instruction set
├── README.md                # User documentation
├── LICENSE                  # GNU GPLv3
├── CHANGELOG.md             # Version history
├── CLAUDE.md                # This file
├── Makefile                 # Unix/Linux build script
├── build.ps1                # Windows PowerShell build script
├── scripts/
│   ├── bayesian_tracker.py  # Python CLI for Bayesian hypothesis + flag tracking
│   └── rapid_checker.py     # Python CLI for RAPID tier assessments
└── references/              # Knowledge base documents
    ├── boundary-probing.md       # I/O characterization techniques
    ├── causal-techniques.md      # Methods for establishing causality
    ├── cognitive-traps.md        # Countermeasures for analytical bias
    ├── coherence-checks.md       # Quick coherence validation (60-second filter)
    ├── compositional-synthesis.md # Math for combining sub-models
    ├── domain-calibration.md     # Plausibility bounds by domain
    ├── red-flags.md              # Red flag catalog for invalid claims
    ├── setup-techniques.md       # Phase 0 framing procedures
    ├── system-identification.md  # Parametric estimation algorithms
    ├── tools-sensitivity.md      # Binary tools & sensitivity analysis
    └── validation-checklist.md   # Consolidated validation requirements
```

## Key Commands

### Bayesian Tracker CLI

The `scripts/bayesian_tracker.py` tool tracks hypothesis confidence using proper Bayesian inference. Extended with red flag tracking and coherence checking for RAPID tier.

```bash
# Add a hypothesis with prior probability
python scripts/bayesian_tracker.py add "System uses REST API" --prior 0.6 --phase P0

# Update with evidence using likelihood ratio presets
python scripts/bayesian_tracker.py update H1 "Found /api/v1 endpoint" --preset strong_confirm

# Available presets: strong_confirm, moderate_confirm, weak_confirm, neutral,
#                    weak_disconfirm, moderate_disconfirm, strong_disconfirm, falsify

# Or use explicit likelihood ratio
python scripts/bayesian_tracker.py update H1 "Evidence description" --lr 5.0

# Compare two hypotheses (Bayes factor)
python scripts/bayesian_tracker.py compare H1 H2

# Generate report
python scripts/bayesian_tracker.py report
python scripts/bayesian_tracker.py report --verbose  # Include evidence trail

# Red flag tracking
python scripts/bayesian_tracker.py flag add methodology "No baseline comparison"
python scripts/bayesian_tracker.py flag report

# Coherence tracking
python scripts/bayesian_tracker.py coherence "data-task-match" --pass
python scripts/bayesian_tracker.py coherence "metric-task-match" --fail --notes "Wrong metrics"

# Verdict (for RAPID tier)
python scripts/bayesian_tracker.py verdict
python scripts/bayesian_tracker.py verdict --full
```

### RAPID Checker CLI

The `scripts/rapid_checker.py` tool provides standalone 10-minute assessment for claim validation.

```bash
# Start assessment
python scripts/rapid_checker.py start "Paper: XYZ Claims"

# Record coherence checks
python scripts/rapid_checker.py coherence data-task-match --pass
python scripts/rapid_checker.py coherence metric-task-match --fail --notes "Classification metrics for regression"

# Add red flags
python scripts/rapid_checker.py flag methodology "No baseline comparison"
python scripts/rapid_checker.py flag results "Test > Train performance" --severity critical

# Check domain calibration
python scripts/rapid_checker.py calibrate accuracy 0.99 --domain ml_classification

# Get verdict and report
python scripts/rapid_checker.py verdict
python scripts/rapid_checker.py report

# List available domains
python scripts/rapid_checker.py domains
```

### Activating the Protocol

Users activate the protocol by:
1. Saying "Help me start" or "Walk me through" (triggers auto-pilot questionnaire mode)
2. Or: "Activate Epistemic Deconstruction Protocol"

## The Phase Methodology

| Phase | Name | Budget | Output |
|-------|------|--------|--------|
| 0.5 | Coherence Screening | 5-10% | Go/No-Go Decision (RAPID tier) |
| 0 | Setup & Frame | 10% | Analysis Plan, Question Pyramid, Initial Hypotheses |
| 1 | Boundary Mapping | 20% | I/O Surface Map, Transfer Functions |
| 2 | Causal Analysis | 25% | Causal Graph, Dependency Matrix |
| 3 | Parametric ID | 20% | Mathematical Model, Uncertainty Bounds |
| 4 | Model Synthesis | 15% | Unified Model, Emergence Report |
| 5 | Validation | 10% | Validation Report, Baseline Comparison, Attack Surface Map |

### Tier System

| Tier | When to Use | Phases | Budget |
|------|-------------|--------|--------|
| RAPID | Quick claim validation, red flag screening | 0.5→5 | <30min |
| LITE | Known archetype, stable system, single function | 0→1→5 | <2h |
| STANDARD | Unknown internals, single domain, no adversary | 0→1→2→3→4→5 | 2-20h |
| COMPREHENSIVE | Multi-domain, adversarial, critical, recursive | All + decomposition | 20h+ |

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

### Bayesian Hypothesis Tracking

- Maintain 3+ hypotheses at all times (including adversarial/deceptive alternatives)
- Use likelihood ratios for updates: LR > 1 confirms, LR < 1 disconfirms, LR = 0 falsifies
- Track posteriors, not gut feelings

### Core Axioms

- **Falsify, don't confirm**: Design tests to break hypotheses
- **Quantify uncertainty**: Never report point estimates alone
- **Map ≠ Territory**: Models are tools, not truth
- **Emergence is real**: Component models may not predict whole-system behavior

### Cognitive Trap Awareness

Always check for:
- Mirror-imaging ("I would do X, so they did")
- Confirmation bias (only finding supporting evidence)
- Anchoring (first hypothesis dominates)
- Dunning-Kruger (early overconfidence)
- Tool Worship/Cargo-Cult ("We used fancy tool X, so results are valid")

See `references/cognitive-traps.md` for full catalog.

### RAPID Tier Validation

For quick claim validation:
1. Coherence checks (data-task alignment, metric-task alignment)
2. Red flag scan (methodology, documentation, results, claims)
3. Domain calibration (compare to plausibility bounds)
4. Verdict: CREDIBLE / SKEPTICAL / DOUBTFUL / REJECT

See `references/red-flags.md`, `references/coherence-checks.md`, `references/domain-calibration.md`.

## Working with This Codebase

### File Modification Guidelines

- **SKILL.md** is the core protocol. Changes here affect all analysis behavior.
- **references/** files provide domain-specific knowledge. Add new reference files for new domains.
- **scripts/** contains executable tooling. The bayesian_tracker.py is the primary tool.

### Tech Stack

- Python 3.x (for bayesian_tracker.py)
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

### Adding New Reference Material

Reference files should follow this pattern:
1. Clear section headers
2. Tables for quick reference
3. Code snippets where applicable
4. Cross-references to other reference files
