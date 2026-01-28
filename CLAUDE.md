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
├── CLAUDE.md                # This file
├── scripts/
│   └── bayesian_tracker.py  # Python CLI for Bayesian hypothesis tracking
└── references/              # Knowledge base documents
    ├── boundary-probing.md       # I/O characterization techniques
    ├── causal-techniques.md      # Methods for establishing causality
    ├── cognitive-traps.md        # Countermeasures for analytical bias
    ├── compositional-synthesis.md # Math for combining sub-models
    ├── setup-techniques.md       # Phase 0 framing procedures
    ├── system-identification.md  # Parametric estimation algorithms
    └── tools-sensitivity.md      # Binary tools & sensitivity analysis
```

## Key Commands

### Bayesian Tracker CLI

The `scripts/bayesian_tracker.py` tool tracks hypothesis confidence using proper Bayesian inference.

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
```

### Activating the Protocol

Users activate the protocol by:
1. Saying "Help me start" or "Walk me through" (triggers auto-pilot questionnaire mode)
2. Or: "Activate Epistemic Deconstruction Protocol"

## The 6-Phase Methodology

| Phase | Name | Budget | Output |
|-------|------|--------|--------|
| 0 | Setup & Frame | 10% | Analysis Plan, Question Pyramid, Initial Hypotheses |
| 1 | Boundary Mapping | 20% | I/O Surface Map, Transfer Functions |
| 2 | Causal Analysis | 25% | Causal Graph, Dependency Matrix |
| 3 | Parametric ID | 20% | Mathematical Model, Uncertainty Bounds |
| 4 | Model Synthesis | 15% | Unified Model, Emergence Report |
| 5 | Validation | 10% | Validation Report, Attack Surface Map |

### Tier System

| Tier | When to Use | Phases | Budget |
|------|-------------|--------|--------|
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

See `references/cognitive-traps.md` for full catalog.

## Working with This Codebase

### File Modification Guidelines

- **SKILL.md** is the core protocol. Changes here affect all analysis behavior.
- **references/** files provide domain-specific knowledge. Add new reference files for new domains.
- **scripts/** contains executable tooling. The bayesian_tracker.py is the primary tool.

### Tech Stack

- Python 3.x (for bayesian_tracker.py)
- Markdown documentation
- No build system - this is a documentation-first project

### Adding New Reference Material

Reference files should follow this pattern:
1. Clear section headers
2. Tables for quick reference
3. Code snippets where applicable
4. Cross-references to other reference files
