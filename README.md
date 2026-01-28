# Epistemic Deconstructor

[![License](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Protocol](https://img.shields.io/badge/Protocol-v6.2-green.svg)](CHANGELOG.md)
[![Type](https://img.shields.io/badge/Type-Claude%20Skill-purple.svg)](SKILL.md)

**A systematic framework for reverse engineering unknown systems and analyzing human behavior.**

Transform epistemic uncertainty into predictive control through principled experimentation, Bayesian inference, and compositional modeling.

---

## What It Does

The Epistemic Deconstructor is an AI skill for Claude that provides structured methodologies for:

| Domain | Use Cases |
|--------|-----------|
| **Systems Analysis** | Reverse engineering black-box systems, security analysis, forensics, competitive intelligence |
| **Claim Validation** | Rapid assessment of research papers, vendor claims, technical reports |
| **Psychological Profiling** | Behavioral analysis, negotiation preparation, deception detection |

## Core Principles

| Principle | Description |
|-----------|-------------|
| **Falsify, Don't Confirm** | Design tests to break hypotheses, not prove them |
| **Quantify Uncertainty** | Track confidence with Bayesian updates, not gut feelings |
| **Baseline is God** | Only deviation from established normal is significant |
| **Map ≠ Territory** | Models are tools for prediction, not truth |

---

## Quick Start

### Installation

**Option A: Claude Project (Recommended)**
1. Create a new Project in Claude
2. Upload `SKILL.md` to Custom Instructions
3. Upload `references/` folder to Project Knowledge

**Option B: Direct Attachment**
1. Attach `SKILL.md` to your conversation
2. Say: *"Activate Epistemic Deconstruction Protocol"*

### Activation Commands

| Command | Effect |
|---------|--------|
| *"Help me start"* | Guided questionnaire for system analysis |
| *"Analyze this person"* | Guided questionnaire for psychological profiling |
| *"Activate Epistemic Deconstruction Protocol"* | Manual activation |

---

## Tier System

Choose your analysis depth based on the task:

| Tier | Use Case | Time | Phases |
|------|----------|------|--------|
| **RAPID** | Validate external claims, spot red flags | <30 min | 0.5 → 5 |
| **LITE** | Known system type, single function | <2 hr | 0 → 1 → 5 |
| **STANDARD** | Unknown internals, single domain | 2-20 hr | 0 → 1 → 2 → 3 → 4 → 5 |
| **COMPREHENSIVE** | Multi-domain, adversarial, critical | 20+ hr | All + recursion |
| **PSYCH** | Human behavioral analysis | 1-4 hr | 0-P → 1-P → 2-P → 3-P → 4-P → 5-P |

---

## System Analysis Phases

| Phase | Name | Purpose | Output |
|-------|------|---------|--------|
| 0.5 | Coherence Screening | Quick claim validation | Go/No-Go decision |
| 0 | Setup & Frame | Define scope, hypotheses | Analysis plan |
| 1 | Boundary Mapping | Enumerate I/O, probe responses | Surface map |
| 2 | Causal Analysis | Map dependencies, trace flows | Causal graph |
| 3 | Parametric ID | Fit mathematical models | Equations + uncertainty |
| 4 | Model Synthesis | Combine sub-models, detect emergence | Unified model |
| 5 | Validation | Test predictions, assess adversarial posture | Validation report |

---

## PSYCH Tier Phases

For analyzing human behavior, personas, and psychological profiles:

| Phase | Name | Purpose | Output |
|-------|------|---------|--------|
| 0-P | Context & Frame | Assess relationship, set objectives | Analysis plan |
| 1-P | Baseline Calibration | Establish "normal" patterns | Baseline profile |
| 2-P | Stimulus-Response | Apply probes, observe deviations | Trigger map |
| 3-P | Structural ID | Map OCEAN, Dark Triad, distortions | Trait profile |
| 4-P | Motive Synthesis | Identify MICE drivers, build model | Drive matrix |
| 5-P | Validation | Generate predictions, plan interaction | Strategy document |

### Frameworks Included

- **Big Five (OCEAN)**: Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism
- **Dark Triad**: Narcissism, Machiavellianism, Psychopathy (always assessed together)
- **MICE/RASP**: Money, Ideology, Coercion, Ego motivation drivers
- **Elicitation Techniques**: Columbo, Bracketing, Silence, Challenge, Flattery Bridge
- **Linguistic Analysis**: Pronoun shifts, distancing language, deception markers

---

## CLI Tools

### Bayesian Tracker

Track system analysis hypotheses with proper Bayesian inference.

```bash
# Add hypothesis
python scripts/bayesian_tracker.py add "Uses REST API" --prior 0.6

# Update with evidence
python scripts/bayesian_tracker.py update H1 "Found /api/v1" --preset strong_confirm

# Track red flags
python scripts/bayesian_tracker.py flag add methodology "No baseline"

# Get verdict
python scripts/bayesian_tracker.py verdict --full
```

### Belief Tracker

Track psychological trait assessments for PSYCH tier.

```bash
# Set subject
python scripts/belief_tracker.py subject "John Doe" --context "Negotiation"

# Add trait hypothesis
python scripts/belief_tracker.py add "High Neuroticism" --category neuroticism --prior 0.5

# Update with evidence
python scripts/belief_tracker.py update T1 "Catastrophizing observed" --preset strong_indicator

# Track baseline
python scripts/belief_tracker.py baseline add "Uses 'we' frequently" --category linguistic

# Record deviation
python scripts/belief_tracker.py deviation "Switched to 'I' under stress" --significance major

# Generate profile
python scripts/belief_tracker.py profile
```

### RAPID Checker

Standalone 10-minute claim assessment.

```bash
python scripts/rapid_checker.py start "Paper: XYZ Claims"
python scripts/rapid_checker.py coherence data-task-match --pass
python scripts/rapid_checker.py flag methodology "No baseline"
python scripts/rapid_checker.py verdict
```

---

## Evidence Presets

### System Analysis (bayesian_tracker)

| Preset | LR | Use When |
|--------|---:|----------|
| `strong_confirm` | 10.0 | Very diagnostic positive evidence |
| `moderate_confirm` | 3.0 | Moderately supports hypothesis |
| `weak_confirm` | 1.5 | Slightly favors hypothesis |
| `neutral` | 1.0 | No diagnostic value |
| `weak_disconfirm` | 0.67 | Slightly contradicts |
| `strong_disconfirm` | 0.1 | Strong evidence against |
| `falsify` | 0.0 | Logically incompatible |

### Psychological Analysis (belief_tracker)

| Preset | LR | Use When |
|--------|---:|----------|
| `smoking_gun` | 20.0 | Direct admission, unambiguous |
| `strong_indicator` | 5.0 | Consistent pattern across contexts |
| `indicator` | 2.0 | Single clear occurrence |
| `weak_indicator` | 1.5 | Suggestive but not definitive |
| `counter_indicator` | 0.5 | Single contradiction |
| `strong_counter` | 0.2 | Pattern contradicts |
| `disconfirm` | 0.1 | Strong evidence against |

---

## Reference Documents

### System Analysis
| Document | Content |
|----------|---------|
| `cognitive-traps.md` | 15 cognitive biases with countermeasures |
| `red-flags.md` | Methodology and claims red flag catalog |
| `coherence-checks.md` | 60-second coherence validation |
| `domain-calibration.md` | Plausibility bounds by domain |
| `system-identification.md` | ARX, N4SID, SINDy algorithms |
| `boundary-probing.md` | I/O characterization techniques |

### Psychological Analysis
| Document | Content |
|----------|---------|
| `archetype-mapping.md` | OCEAN, Dark Triad, MICE frameworks |
| `linguistic-markers.md` | Text analysis, deception markers |
| `elicitation-techniques.md` | 10 probing methods |
| `motive-analysis.md` | MICE/RASP deep dive |
| `profile-synthesis.md` | Trait composition, archetypes |

---

## State Blocks

Every response during analysis ends with a state block for continuity:

**System Analysis:**
```
[STATE: Phase 2 | Tier: STANDARD | Active Hypotheses: 3 | Lead: H2 (78%) | Confidence: Medium]
```

**PSYCH Tier:**
```
[STATE: Phase 3-P | Tier: PSYCH | Archetype: High-N/Low-A | Rapport: Med | Stress: Low]
```

**RAPID Tier:**
```
[STATE: Phase 0.5 | Tier: RAPID | Coherence: PASS | Red Flags: 2 | Verdict: SKEPTICAL]
```

---

## Repository Structure

```
epistemic-deconstructor/
├── SKILL.md                  # Core protocol (load as Custom Instructions)
├── CLAUDE.md                 # AI guidance document
├── README.md                 # This file
├── CHANGELOG.md              # Version history
├── LICENSE                   # GPLv3
├── Makefile                  # Unix build script
├── build.ps1                 # Windows build script
├── scripts/
│   ├── bayesian_tracker.py   # System hypothesis tracking
│   ├── belief_tracker.py     # Psychological trait tracking
│   └── rapid_checker.py      # Quick claim validation
└── references/
    ├── archetype-mapping.md
    ├── boundary-probing.md
    ├── causal-techniques.md
    ├── cognitive-traps.md
    ├── coherence-checks.md
    ├── compositional-synthesis.md
    ├── domain-calibration.md
    ├── elicitation-techniques.md
    ├── linguistic-markers.md
    ├── motive-analysis.md
    ├── profile-synthesis.md
    ├── red-flags.md
    ├── setup-techniques.md
    ├── system-identification.md
    ├── tools-sensitivity.md
    └── validation-checklist.md
```

---

## Building & Packaging

```bash
# Windows
.\build.ps1 validate          # Check structure
.\build.ps1 package           # Create zip
.\build.ps1 package-combined  # Single-file version

# Unix/Linux/macOS
make validate
make package
make package-combined
```

---

## Ethical Guidelines

The PSYCH tier includes ethical constraints:

- **No Clinical Diagnosis**: Use observable traits ("high emotional volatility"), not disorders ("Bipolar")
- **Cultural Calibration**: Adjust for cultural norms before profiling
- **Consent Awareness**: Document when subject is unaware of analysis
- **Defensive Use**: Primary purpose is negotiation and defense, not manipulation

---

## License

GNU General Public License v3.0 - See [LICENSE](LICENSE)

---

## Version History

See [CHANGELOG.md](CHANGELOG.md) for detailed release notes.

**Current**: v6.2.0 - Added PSYCH tier for psychological profiling
