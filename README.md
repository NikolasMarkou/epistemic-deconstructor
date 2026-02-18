# Epistemic Deconstructor

[![License](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Skill](https://img.shields.io/badge/Skill-v6.4-green.svg)](CHANGELOG.md)
[![Sponsored by Electi](https://img.shields.io/badge/Sponsored%20by-Electi-orange.svg)](https://www.electiconsulting.com)

A [Claude skill](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/skills) for systematic reverse engineering of unknown systems. Give it a black box — software, hardware, organizational, or human — and it guides Claude through a structured methodology to build a predictive model of how it works.

Uses Bayesian hypothesis tracking, falsification-driven experimentation, and compositional modeling to turn "I don't know how this works" into a validated, quantified understanding.

> **What's a Claude skill?** A skill is a reusable instruction set that extends Claude's capabilities for a specific domain. Load it into a Claude Project, Custom Instructions, or Claude Code, and Claude gains a new structured workflow it can execute on demand.

## Install

**Option A — Single file (simplest):**
Download `epistemic-deconstructor-combined.md` from [Releases](https://github.com/NikolasMarkou/epistemic-deconstructor/releases). Paste into Claude's Custom Instructions or a Project. This bundles SKILL.md and all reference documents into one file. Scripts are not included — use Option B or C if you need the CLI tools.

**Option B — Full package:**
Download the zip from [Releases](https://github.com/NikolasMarkou/epistemic-deconstructor/releases). Upload the entire `src/` folder to a Claude Project. This includes SKILL.md, all reference documents, the Python CLI scripts, and domain calibration config.

**Option C — Claude Code:**
Clone this repo. The `.claude/skills/` directory is pre-configured.

Then say: **"Help me start"**

## What it does

You describe a system you want to understand. The skill walks you through:

| Phase | What happens |
|-------|-------------|
| **0. Setup** | Define scope, seed 3+ competing hypotheses, pick fidelity target |
| **0.5. Screening** | Quick coherence/red-flag check for external claims (RAPID tier) |
| **1. Boundary** | Enumerate I/O channels, apply probe signals, build stimulus-response database |
| **2. Causal** | Differential analysis, sensitivity testing, build causal graph, falsify hypotheses |
| **3. Parametric** | Fit mathematical models, quantify parameter uncertainty |
| **4. Synthesis** | Compose sub-models, test for emergence, classify archetype |
| **5. Validation** | Beat naive baselines, adversarial assessment, document limitations |

Every response tracks state: `[STATE: Phase 2 | Tier: STANDARD | Active Hypotheses: 3 | Lead: H2 (78%) | Confidence: Medium]`

## Tiers

| Tier | When | Time |
|------|------|------|
| **RAPID** | Validating a paper, vendor pitch, or external claim | <30 min |
| **LITE** | Known system type, single function, stable | <2 hr |
| **STANDARD** | Unknown internals, single domain, no adversary | 2-20 hr |
| **COMPREHENSIVE** | Multi-domain, adversarial, critical infrastructure | 20+ hr |
| **PSYCH** | Behavioral analysis of a person or persona | 1-4 hr |

## CLI tools

Three Python scripts for tracking state across sessions:

```bash
# Bayesian hypothesis tracker (system analysis)
python src/scripts/bayesian_tracker.py add "System uses REST API" --prior 0.6
python src/scripts/bayesian_tracker.py update H1 "Found /api/v1 endpoint" --preset strong_confirm
python src/scripts/bayesian_tracker.py report --verbose

# Psychological trait tracker (PSYCH tier)
python src/scripts/belief_tracker.py add "High Neuroticism" --prior 0.5
python src/scripts/belief_tracker.py update T1 "Catastrophizing language" --preset strong_indicator
python src/scripts/belief_tracker.py profile

# RAPID claim validator
python src/scripts/rapid_checker.py start "Paper: XYZ Claims"
python src/scripts/rapid_checker.py flag methodology "No baseline comparison"
python src/scripts/rapid_checker.py verdict
```

## Project structure

```
epistemic-deconstructor/
├── README.md
├── CLAUDE.md              # AI assistant instructions
├── CHANGELOG.md
├── LICENSE
├── Makefile               # Unix build script
├── build.ps1              # Windows build script
└── src/
    ├── SKILL.md           # Core skill definition (the main instruction set)
    ├── config/
    │   └── domains.json   # Domain calibration bounds
    ├── scripts/
    │   ├── bayesian_tracker.py
    │   ├── belief_tracker.py
    │   └── rapid_checker.py
    └── references/        # 19 knowledge base documents
        ├── boundary-probing.md
        ├── causal-techniques.md
        ├── cognitive-traps.md
        ├── coherence-checks.md
        ├── compositional-synthesis.md
        ├── domain-calibration.md
        ├── system-identification.md
        ├── validation-checklist.md
        ├── red-flags.md
        ├── setup-techniques.md
        ├── tools-sensitivity.md
        ├── tool-catalog.md
        ├── adversarial-heuristics.md
        ├── psych-tier-protocol.md
        ├── archetype-mapping.md
        ├── linguistic-markers.md
        ├── elicitation-techniques.md
        ├── motive-analysis.md
        └── profile-synthesis.md
```

## Building

```bash
# Package as zip
make package

# Single-file skill with references inlined
make package-combined

# Validate structure
make validate
```

## License

GPLv3 — See [LICENSE](LICENSE)

**v6.4.0** — See [CHANGELOG.md](CHANGELOG.md)
