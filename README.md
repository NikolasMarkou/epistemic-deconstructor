# Epistemic Deconstructor

[![License](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Protocol](https://img.shields.io/badge/Protocol-v6.3-green.svg)](CHANGELOG.md)
[![Sponsored by Electi](https://img.shields.io/badge/Sponsored%20by-Electi-orange.svg)](https://www.electiconsulting.com)

**Systematic reverse engineering of unknown systems using scientific methodology.**

Black-box analysis. Competitive intelligence. Security analysis. Forensics. Building predictive models from observations.

---

## Install

**Option 1:** Download `epistemic-deconstructor-combined.md` from [Releases](https://github.com/NikolasMarkou/epistemic-deconstructor/releases) → Paste into Claude's Custom Instructions

**Option 2:** Download zip → Upload `SKILL.md` + `references/` folder to a Claude Project

Then say: **"Help me start"**

---

## The Protocol

Transform epistemic uncertainty into predictive control through principled experimentation.

### Phase 0: Setup & Frame
Define what you're analyzing. Set scope, constraints, access level. Generate 3+ competing hypotheses including an adversarial one. Lock your fidelity target (L1-L5: behavior → mechanism → structure → parameters → replication).

### Phase 0.5: Coherence Screening (RAPID)
For external claims: instant reject check, red flag scan, domain calibration. Verdict: CREDIBLE / SKEPTICAL / DOUBTFUL / REJECT. Takes 10-30 minutes.

### Phase 1: Boundary Mapping
Enumerate all I/O channels (explicit, implicit, side-channel). Apply probe signals (step, impulse, edge cases). Build stimulus-response database. Characterize the attack surface.

### Phase 2: Causal Analysis
Static analysis if internals visible. Dynamic tracing with marker injection. Differential analysis—vary one input, observe outputs. Build causal graph with feedback loops. **Falsify hypotheses, don't confirm them.**

### Phase 3: Parametric Identification
Fit mathematical models (ARX → ARMAX → NARMAX → State-Space). Use information criteria (AIC/BIC) to avoid overfitting. Quantify parameter uncertainty. Validate with whiteness tests.

### Phase 4: Model Synthesis
Compose sub-models (serial, parallel, feedback). Propagate uncertainty through composition. Test for emergence—does the whole behave differently than predicted from parts?

### Phase 5: Validation & Adversarial
Test predictions on held-out data. Beat naive baselines or your model has no value. Classify adversarial posture (L0-L4). Map attack surface. Document limitations.

---

## Tier Selection

| Tier | When | Time |
|------|------|------|
| **RAPID** | Validating external claims, papers, vendor pitches | <30 min |
| **LITE** | Known system type, single function, stable | <2 hr |
| **STANDARD** | Unknown internals, single domain, no adversary | 2-20 hr |
| **COMPREHENSIVE** | Multi-domain, adversarial, critical infrastructure | 20+ hr |
| **PSYCH** | Human behavioral analysis | 1-4 hr |

Start with RAPID for claims. Default to STANDARD if unsure. Escalate to COMPREHENSIVE if you find >15 components or adversarial indicators.

---

## Core Principles

**Falsify, don't confirm.** Design experiments to break your hypothesis, not prove it. One refutation beats ten confirmations.

**Quantify uncertainty.** Track posteriors with Bayesian updates. Never report point estimates without confidence intervals.

**Maintain parallel hypotheses.** Always keep 3+ competing explanations including "adversarial/deceptive." Kill your darlings when evidence demands.

**Emergence is real.** Component models don't predict whole-system behavior. Test composition explicitly.

**Map ≠ Territory.** Your model is wrong. The question is whether it's useful.

---

## Bayesian Tracking

```bash
# Add hypothesis with prior
python scripts/bayesian_tracker.py add "System uses REST API" --prior 0.6

# Update with evidence
python scripts/bayesian_tracker.py update H1 "Found /api/v1 endpoint" --preset strong_confirm

# Presets: strong_confirm (10x), moderate_confirm (3x), weak_confirm (1.5x)
#          weak_disconfirm (0.67x), strong_disconfirm (0.1x), falsify (0)

# Compare hypotheses
python scripts/bayesian_tracker.py compare H1 H2

# Generate report
python scripts/bayesian_tracker.py report --verbose
```

---

## State Blocks

Every response ends with state for context continuity:

```
[STATE: Phase 2 | Tier: STANDARD | Active Hypotheses: 3 | Lead: H2 (78%) | Confidence: Medium]
```

---

## Reference Documents

| Purpose | Files |
|---------|-------|
| **Probing** | `boundary-probing.md` `tools-sensitivity.md` |
| **Causality** | `causal-techniques.md` `system-identification.md` |
| **Validation** | `red-flags.md` `coherence-checks.md` `domain-calibration.md` `validation-checklist.md` |
| **Adversarial** | `adversarial-heuristics.md` `tool-catalog.md` |
| **Synthesis** | `compositional-synthesis.md` |
| **Bias Control** | `cognitive-traps.md` `setup-techniques.md` |
| **Psychology** | `psych-tier-protocol.md` `archetype-mapping.md` `elicitation-techniques.md` `motive-analysis.md` `linguistic-markers.md` `profile-synthesis.md` |

---

## Cognitive Traps

The protocol includes countermeasures for:

- **Mirror-imaging** — Assuming designers think like you
- **Confirmation bias** — Only finding supporting evidence
- **Anchoring** — First hypothesis dominates thinking
- **Teleological fallacy** — Assuming everything has purpose (dead code exists)
- **Tool worship** — Complex methods don't create signal from noise
- **Dunning-Kruger** — Early overconfidence in partial understanding

---

## License

GPLv3 — See [LICENSE](LICENSE)

**v6.3.0** — See [CHANGELOG.md](CHANGELOG.md)
