# Epistemic Deconstructor

[![License](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Protocol](https://img.shields.io/badge/Protocol-v6.3-green.svg)](CHANGELOG.md)
[![Type](https://img.shields.io/badge/Type-Claude%20Skill-purple.svg)](SKILL.md)
[![Sponsored by Electi](https://img.shields.io/badge/Sponsored%20by-Electi-orange.svg)](https://www.electiconsulting.com)

A systematic framework for reverse engineering unknown systems and analyzing human behavior. Transform epistemic uncertainty into predictive control.

## Quick Start

```
1. Download the latest release
2. Upload SKILL.md to Claude as Custom Instructions
3. Upload references/ folder to Project Knowledge
4. Say: "Help me start"
```

**Single-file alternative:** Use `epistemic-deconstructor-combined.md` (includes all references).

---

## What It Does

| Tier | Purpose | Time |
|------|---------|------|
| **RAPID** | Validate claims, spot red flags | <30 min |
| **LITE** | Analyze known system types | <2 hr |
| **STANDARD** | Full reverse engineering | 2-20 hr |
| **COMPREHENSIVE** | Multi-domain, adversarial systems | 20+ hr |
| **PSYCH** | Human behavioral profiling | 1-4 hr |

---

## System Analysis (Phases 0-5)

```
Phase 0   → Setup & Frame      → Analysis plan, hypotheses
Phase 0.5 → Coherence Screen   → Go/No-Go (RAPID tier)
Phase 1   → Boundary Mapping   → I/O surface, probe database
Phase 2   → Causal Analysis    → Causal graph, dependencies
Phase 3   → Parametric ID      → Mathematical model
Phase 4   → Model Synthesis    → Unified model, emergence
Phase 5   → Validation         → Tested predictions, limitations
```

**Core principle:** Falsify, don't confirm. Design tests to break hypotheses.

---

## Psychological Profiling (PSYCH Tier)

```
Phase 0-P → Context & Frame    → Relationship dynamics
Phase 1-P → Baseline           → "Normal" patterns
Phase 2-P → Stimulus-Response  → Probes, deviations
Phase 3-P → Structural ID      → OCEAN, Dark Triad
Phase 4-P → Motive Synthesis   → MICE drivers
Phase 5-P → Validation         → Predictions, strategy
```

**Frameworks:** Big Five (OCEAN) • Dark Triad • MICE/RASP • Elicitation techniques • Linguistic markers

**Core principle:** Baseline is God. Only deviation from normal is significant.

---

## CLI Tools

### bayesian_tracker.py — System hypothesis tracking
```bash
python scripts/bayesian_tracker.py add "Uses REST API" --prior 0.6
python scripts/bayesian_tracker.py update H1 "Found /api/v1" --preset strong_confirm
python scripts/bayesian_tracker.py verdict
```

### belief_tracker.py — Psychological trait tracking
```bash
python scripts/belief_tracker.py add "High Neuroticism" --prior 0.5
python scripts/belief_tracker.py update T1 "Catastrophizing" --preset strong_indicator
python scripts/belief_tracker.py profile
```

### rapid_checker.py — Quick claim validation
```bash
python scripts/rapid_checker.py start "Paper: XYZ Claims"
python scripts/rapid_checker.py flag methodology "No baseline"
python scripts/rapid_checker.py verdict
```

---

## Reference Documents

| Category | Files |
|----------|-------|
| **Core Protocol** | `psych-tier-protocol.md` `tool-catalog.md` `adversarial-heuristics.md` |
| **System Analysis** | `boundary-probing.md` `causal-techniques.md` `system-identification.md` `compositional-synthesis.md` |
| **Validation** | `red-flags.md` `coherence-checks.md` `domain-calibration.md` `validation-checklist.md` |
| **Psychology** | `archetype-mapping.md` `linguistic-markers.md` `elicitation-techniques.md` `motive-analysis.md` `profile-synthesis.md` |
| **Meta** | `cognitive-traps.md` `setup-techniques.md` `tools-sensitivity.md` |

---

## State Blocks

Every response ends with a state block for context continuity:

```
[STATE: Phase 2 | Tier: STANDARD | Active Hypotheses: 3 | Lead: H2 (78%) | Confidence: Medium]
[STATE: Phase 3-P | Tier: PSYCH | Archetype: High-N/Low-A | Rapport: Med | Stress: Low]
[STATE: Phase 0.5 | Tier: RAPID | Coherence: PASS | Red Flags: 2 | Verdict: SKEPTICAL]
```

---

## Repository Structure

```
epistemic-deconstructor/
├── SKILL.md              # Core protocol (417 lines)
├── CLAUDE.md             # AI guidance
├── scripts/
│   ├── bayesian_tracker.py
│   ├── belief_tracker.py
│   └── rapid_checker.py
└── references/           # 19 knowledge base documents
```

---

## Building

```bash
# Windows
.\build.ps1 package           # Create zip
.\build.ps1 package-combined  # Single-file version

# Unix/macOS
make package
make package-combined
```

---

## Ethics (PSYCH Tier)

- **No clinical diagnosis** — Use observable traits, not disorder labels
- **Cultural calibration** — Adjust for cultural norms
- **Consent awareness** — Document when subject is unaware
- **Defensive use** — For negotiation and protection, not exploitation

---

## License

GNU General Public License v3.0 — See [LICENSE](LICENSE)

**Current version:** v6.3.0 — See [CHANGELOG.md](CHANGELOG.md)
