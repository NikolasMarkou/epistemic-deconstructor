---
name: epistemic-deconstructor
description: "Systematic reverse engineering of unknown systems (mechanical, software, biological, organizational, social, adversarial). Use when: (1) Black-box analysis of unknown systems, (2) Competitive intelligence or system interrogation, (3) Security analysis or attack surface mapping, (4) Forensics or root cause analysis, (5) Building predictive models from observations. Features consolidated 6-phase protocol, proper Bayesian inference, compositional model synthesis with uncertainty propagation, concrete stop conditions, and executable tooling. Handles emergence, adversarial resistance, temporal drift, and distributed systems."
---

# Epistemic Deconstruction Protocol v6.0

## Core Objective
Transform epistemic uncertainty into predictive control through principled experimentation, compositional modeling, and Bayesian inference.

## State Block Protocol (REQUIRED)

**Every response must end with a State Block:**
```
[STATE: Phase X | Tier: Y | Active Hypotheses: N | Confidence: Low/Med/High]
```

Examples:
```
[STATE: Phase 2 | Tier: STANDARD | Active Hypotheses: 3 | Lead: H2 (78%) | Confidence: Medium]
```

**Extended format for RAPID tier / validation phases:**
```
[STATE: Phase 0.5 | Tier: RAPID | Coherence: PASS | Red Flags: 2 | Verdict: SKEPTICAL]
[STATE: Phase 5 | Tier: STANDARD | Validation: 5/6 | Baseline: BEAT | Verdict: CREDIBLE]
```

This ensures continuity across long conversations and enables context resync.

## Auto-Pilot Mode

If user says **"Help me start"** or **"Walk me through"**, enter questionnaire mode:

1. "What system are you analyzing? (software/hardware/organizational/other)"
2. "What is your access level? (full source/binary only/black-box I/O)"
3. "Is there an adversary? (yes/no/unknown)"
4. "Time budget? (hours)"
5. "What do you want to know? (how it works/parameters/vulnerabilities)"

Then auto-populate Phase 0 and recommend tier.

## Axioms (Operational)

| Axiom | Implication | Checkpoint |
|-------|-------------|------------|
| Finite Rules | All systems have extractable logic | Can you state the governing rules? |
| Scientific Method | Hypothesis → Test → Measure → Refine | Are you falsifying or confirming? |
| Observability Bound | You can only learn what you can measure | Have you enumerated all I/O? |
| Information = Entropy Reduction | Each measurement decreases uncertainty | Quantify bits gained per test |
| Observer Effect | Measurement perturbs system | What did your probing change? |
| Adversarial Resistance | Systems may counter analysis | Have you tested for active defense? |
| Emergence | Whole ≠ Σ(parts) | Does component model predict whole? |
| Temporal Drift | Systems evolve during analysis | Track version/state changes |
| Map ≠ Territory | Models are tools, not truth | Can you articulate model limitations? |

## Tier Selection (REQUIRED FIRST STEP)

| Tier | Trigger | Phases | Budget |
|------|---------|--------|--------|
| **RAPID** | Quick claim validation, red flag screening | 0.5→5 | <30min |
| **LITE** | Known archetype, stable system, single function | 0→1→5 | <2h |
| **STANDARD** | Unknown internals, single domain, no adversary | 0→1→2→3→4→5 | 2-20h |
| **COMPREHENSIVE** | Multi-domain, adversarial, critical, recursive | All + decomposition + coordination | 20h+ |

**Decision**: Use RAPID for external claim validation before investing analysis time. If unsure between other tiers, start STANDARD. Escalate to COMPREHENSIVE if Phase 2 reveals >15 components or adversarial indicators.

---

## Phase 0: Setup & Frame
*Budget: 10% | Output: Analysis Plan*

### 0.1 Context Definition
1. **Position**: Insider/outsider? Adversarial/cooperative?
2. **Access**: What I/O channels exist? What tools available?
3. **Constraints**: Time budget? Ethical limits? Legal bounds?
4. **System Type**: White-box (full internals) / Grey-box (partial physics) / Black-box (I/O only)

### 0.2 Question Pyramid
Convert objectives to falsifiable questions:
```
L1: What does this system DO? (Behavioral)
L2: HOW does it work? (Functional)  
L3: WHY this design? (Structural)
L4: What are the PARAMETERS? (Quantitative)
L5: Can I REPLICATE it? (Generative)
```
Lock fidelity target: L1-L5. Most analyses need L2-L3.

### 0.3 Hypothesis Seed
Generate 3+ initial hypotheses about system behavior.
```
H1: [most likely mechanism]
H2: [alternative mechanism]  
H3: [adversarial/deceptive mechanism]
```

### 0.4 Adversarial Pre-Check
**Before any probing, assess adversarial risk:**
- Is this a known protected system? (Check references/adversarial-heuristics.md)
- High entropy sections? (packed/encrypted)
- Known anti-debug patterns?

If adversarial indicators present → Escalate to COMPREHENSIVE tier.

### 0.5 Cognitive Trap Awareness
Before proceeding, identify which traps you're vulnerable to:
| Trap | Sign | Countermeasure |
|------|------|----------------|
| Mirror-imaging | "I would do X, so they did" | Generate 3 alternatives |
| Confirmation bias | Only finding supporting evidence | Design falsification tests |
| Anchoring | First hypothesis dominates | Maintain parallel hypotheses |
| Teleological | "Everything has a purpose" | Test component removal |
| Dunning-Kruger | Early overconfidence | Track prediction accuracy |

**Reference**: See references/cognitive-traps.md for full catalog and countermeasures.

### 0.6 Stop Condition
Phase 0 complete when:
- [ ] Tier selected
- [ ] Fidelity target locked
- [ ] ≥3 hypotheses documented
- [ ] I/O channels enumerated
- [ ] Adversarial risk assessed
- [ ] Cognitive vulnerabilities acknowledged

**Reference**: See references/setup-techniques.md for detailed procedures.

---

## Phase 0.5: Coherence Screening (RAPID Tier Entry Point)
*Budget: 5-10% | Output: Go/No-Go Decision*

This phase provides rapid assessment of external claims before investing significant analysis time. Required for RAPID tier; optional for other tiers when evaluating third-party claims.

### 0.5.1 Claim-Task Alignment
Verify the claim makes coherent sense:
| Check | Failure Example |
|-------|-----------------|
| **Data matches claimed task** | Claims "system prediction" but uses unrelated data |
| **Methodology matches domain** | Uses techniques inappropriate for domain |
| **Metrics match task type** | Classification metrics for continuous outputs |
| **Internal consistency** | Results contradict across sections |

**Test**: Can you trace: claimed input → claimed method → claimed output → claimed conclusion?

### 0.5.2 Instant Reject Conditions
| Condition | Interpretation | Action |
|-----------|----------------|--------|
| **Impossibility** | Claimed results violate domain limits | REJECT |
| **Memorization** | Perfect fit on training data | REJECT |
| **Contamination** | Test performance > training performance | REJECT |
| **Incoherence** | Internal contradictions in claim | REJECT |
| **Unverifiability** | No way to validate claims | REJECT (or flag) |

### 0.5.3 Red Flag Scan
Quick scan for methodology failures:
- [ ] Missing baseline comparison?
- [ ] Extraordinary claims without extraordinary evidence?
- [ ] Tool worship (complex method, no validation)?
- [ ] Documentation gaps (can't reproduce)?
- [ ] Suspicious patterns (too clean, too perfect)?

**Count red flags**: 3+ from different categories → treat entire claim as suspect.

**Reference**: See references/red-flags.md for full catalog.

### 0.5.4 Go/No-Go Decision Gate
| Result | Criteria | Next Step |
|--------|----------|-----------|
| **GO** | No instant rejects, <3 red flags, internally coherent | Proceed to Phase 1 or deeper validation |
| **CONDITIONAL** | Minor concerns, needs clarification | Request additional information |
| **NO-GO** | Instant reject condition or 3+ red flags | REJECT claim, document reasons |

### 0.5.5 Stop Condition
Phase 0.5 complete when:
- [ ] Claim-task alignment verified (or failed)
- [ ] Instant reject conditions checked
- [ ] Red flag count documented
- [ ] Go/No-Go decision recorded

**Reference**: See references/coherence-checks.md for detailed checks.

---

## Phase 1: Boundary Mapping
*Budget: 20% | Output: I/O Surface, Transfer Functions*

### 1.1 I/O Enumeration
| Channel Type | Examples | Discovery Method |
|--------------|----------|------------------|
| Explicit | APIs, ports, UI | Documentation, scanning |
| Implicit | Environment, time, state | Observation, isolation |
| Side-channel | Timing, power, EM, acoustic | Precision measurement |
| Feedback | Loops, callbacks | Trace propagation |

### 1.2 Probe Signals
| Signal | What It Reveals | When to Use |
|--------|-----------------|-------------|
| Step input | Time constants, settling | First probe |
| Impulse | Frequency response | Dynamic systems |
| PRBS | Broadband linear response | System ID |
| Chirp | Frequency sweep | Resonances |
| Edge cases | Nonlinearity, saturation | After linear baseline |

### 1.3 Data Quality Assessment
**Coherence Function**: γ²(f) = |Gxy|² / (Gxx·Gyy)
- γ² ≈ 1.0 → Good linear relationship, proceed
- γ² < 0.8 → Noise, nonlinearity, or unmeasured inputs — improve data before identification

### 1.4 Stimulus-Response Database
Record all probes:
```
| Probe_ID | Input | Output | Latency | Anomaly |
|----------|-------|--------|---------|---------|
| P001     | ...   | ...    | ...     | ...     |
```

### 1.5 Stop Condition
Phase 1 complete when:
- [ ] ≥80% of I/O channels characterized (measure: probe coverage)
- [ ] Response variance < 10% on repeated probes
- [ ] Edge cases tested (min/max/null/overflow)
- [ ] Stimulus-response database has ≥20 entries (LITE: ≥5)

**Reference**: See references/boundary-probing.md for signal generation code.

---

## Phase 2: Causal Analysis
*Budget: 25% | Output: Causal Graph, Dependency Matrix*

### 2.1 Static Analysis (if code/structure visible)
- **Disassembly**: Linear sweep + recursive descent (Ghidra, IDA Pro)
- **Decompilation**: Lift to higher abstraction (control flow structuring, type recovery)
- **Data flow**: Reaching definitions, live variables, taint analysis

### 2.2 Dynamic Analysis
**Dynamic Binary Instrumentation**: Frida, PIN, DynamoRIO for runtime observation.

**Tracer Technique**: Inject markers, observe propagation path.
```
Input: marker_X → [SYSTEM] → Output: f(marker_X)
Conclusion: Path exists from input to output via f()
```

**Differential Analysis**: Vary one input, hold others constant.
```
Δy/Δxᵢ ≠ 0 → xᵢ influences y
Δy/Δxᵢ = 0 → xᵢ does not influence y (or is masked)
```

### 2.3 Sensitivity Analysis
**Morris Screening** (fast preliminary): r(k+1) evaluations
- μ* = importance, σ = nonlinearity/interaction

**Sobol' Indices** (precise): N(2k+2) evaluations
- Si (first-order), STi (total including interactions)
- STi ≈ 0 → parameter negligible, fix it

**Reference**: See references/tools-sensitivity.md for implementation.

### 2.3 Causal Graph Construction
Nodes = variables/components. Edges = causal influence.
- Reinforcing loop: A↑→B↑→A↑ (mark as R)
- Balancing loop: A↑→B↑→A↓ (mark as B)
- Delay: Use ‖ symbol

### 2.4 Falsification Loop
```
FOR each hypothesis H:
  design_test(H) → most likely to REFUTE H
  execute_test()
  IF result contradicts H:
    H.status = REFUTED
    generate_alternative(H)
  ELSE:
    H.confidence += Bayesian_update(evidence)
```

### 2.5 Stop Condition
Phase 2 complete when:
- [ ] ≥70% of observed behaviors have causal explanation
- [ ] All major feedback loops identified
- [ ] ≥1 hypothesis refuted (if not, tests aren't rigorous)
- [ ] Causal graph validated by ≥3 differential tests

**Reference**: See references/causal-techniques.md for tracer injection methods.

---

## Phase 3: Parametric Identification
*Budget: 20% | Output: Mathematical Model, Uncertainty Bounds*

### 3.1 Model Structure Selection

| System Behavior | Model Class | Equation Form |
|-----------------|-------------|---------------|
| Linear, single I/O | ARX | A(q)y = B(q)u + e |
| Linear + colored noise | ARMAX | A(q)y = B(q)u + C(q)e |
| Linear, max flexibility | Box-Jenkins | y = B/F·u + C/D·e |
| Nonlinear | NARMAX | y = f(y_{t-1},...,u_{t-1},...) + e |
| Multi-variable | State-Space | x' = Ax + Bu; y = Cx + Du |
| Discrete modes | EFSM | (S, s₀, I, O, V, T) |
| Nonlinear → linear lift | Koopman/DMD | Linear in observable space |
| Discover equations | SINDy | Sparse regression on library |

**Selection Heuristic**:
1. Start with ARX (simplest)
2. Check residual whiteness → if colored, use ARMAX
3. Check linearity → if nonlinear, try NARMAX or Koopman
4. If ≥3 outputs coupled → use State-Space or N4SID (subspace)
5. Want interpretable equations? → SINDy

### 3.2 Parameter Estimation
```python
# OLS estimate (Python pseudocode)
theta_hat = np.linalg.lstsq(Phi, y)[0]
residuals = y - Phi @ theta_hat
```

### 3.3 Structure Selection (Information Criteria)
| Criterion | Formula | Use |
|-----------|---------|-----|
| AIC | n·ln(RSS/n) + 2k | Balance fit/complexity |
| BIC | n·ln(RSS/n) + k·ln(n) | Prefer simpler models |

Lower = better. If AIC and BIC disagree, prefer BIC.

### 3.4 Uncertainty Quantification
**Bootstrap**: Resample data, re-estimate, compute variance.
**Bayesian**: P(θ|D) ∝ P(D|θ)·P(θ)

Report: θ̂ ± 2σ (95% CI)

### 3.5 Stop Condition
Phase 3 complete when:
- [ ] Model selected via information criterion
- [ ] Residuals pass whiteness test (autocorr < 2/√N at all lags)
- [ ] Cross-validation R² > 0.8 (or task-appropriate threshold)
- [ ] Parameter uncertainty bounds computed

**Reference**: See references/system-identification.md for estimation algorithms.

---

## Phase 4: Model Synthesis & Emergence
*Budget: 15% | Output: Unified Model, Emergence Report*

### 4.1 Compositional Synthesis
Combine sub-models into whole-system model:

| Composition | Operator | Transfer Function |
|-------------|----------|-------------------|
| Serial | M₁ → M₂ | H = H₁ · H₂ |
| Parallel | M₁ ‖ M₂ | H = H₁ + H₂ |
| Feedback | M₁ ↔ M₂ | H = G/(1+GH) |
| Hierarchical | M₁ ⊃ M₂ | Nested evaluation |

### 4.2 Uncertainty Propagation
```
Serial: σ²_total = σ²₁ + σ²₂ (if independent)
Confidence: C_composed ≤ min(C₁, C₂) × coupling_factor

Coupling factors:
  Tight (shared state): 0.7-0.9
  Moderate (message passing): 0.8-0.95
  Loose (event-driven): 0.9-1.0
```

### 4.3 Emergence Detection
Test: Does composed model predict whole-system behavior?

| Emergence Type | Detection | Implication |
|----------------|-----------|-------------|
| Weak | Simulation mismatch > 20% | Add interaction terms |
| Strong | Qualitative new behaviors | Model at higher level |
| Downward causation | Macro constrains micro | Add top-down effects |

**Multi-Scale Test**:
1. Run model at component level → predict aggregate
2. Measure aggregate directly
3. Compare: if gap > 20%, emergence present

### 4.4 Archetype Classification
| Archetype | Indicators | Common Vulnerabilities |
|-----------|------------|------------------------|
| State Machine | Discrete modes | Race conditions |
| Pipeline | Sequential flow | Bottlenecks |
| Feedback Controller | Error correction | Instability at limits |
| Pub/Sub | Event-driven | Message loss, ordering |
| Network | Peer topology | Cascading failure |
| Adaptive | Learning/tuning | Poisoning, drift |

### 4.5 Stop Condition
Phase 4 complete when:
- [ ] All sub-models composed with explicit semantics
- [ ] Uncertainty propagated through composition
- [ ] Emergence test performed (mismatch quantified)
- [ ] Archetype identified with vulnerability assessment

**Reference**: See references/compositional-synthesis.md for detailed operators.

---

## Phase 5: Validation & Adversarial
*Budget: 10% | Output: Validation Report, Attack Surface Map*

### 5.1 Validation Hierarchy
| Level | Test | Threshold |
|-------|------|-----------|
| Interpolation | Predict within training range | R² > 0.95 |
| Extrapolation | Predict outside training range | R² > 0.80 |
| Counterfactual | Predict under intervention | Correct direction |

### 5.2 Residual Diagnostics
- **Whiteness**: Autocorrelation ≈ δ(τ) at all lags
- **Independence**: Cross-correlation with input ≈ 0
- **Normality**: Q-Q plot linear, Jarque-Bera p > 0.05

### 5.3 Model Validity Checks

#### Overfitting Assessment
| Indicator | Interpretation |
|-----------|----------------|
| Train fit = Perfect (100%) | Model memorized data; zero generalization |
| Train >> Test performance | Classic overfitting; gap >20% is severe |
| Test > Train performance | Statistical impossibility; contamination |
| Perfect fit on noisy data | Impossible without cheating |

#### Leakage Detection
| Type | Description | Check |
|------|-------------|-------|
| **Future leakage** | Features computed using future data | All features use only past information |
| **Target leakage** | Target information in features | Document feature-target independence |
| **Train-test contamination** | Overlap between sets | Explicit split with clear boundaries |
| **Look-ahead bias** | Decisions based on future info | Walk-forward with realistic lags |

### 5.4 Baseline Comparison
Every model must beat appropriate baselines:

| Baseline Type | Description | When Required |
|---------------|-------------|---------------|
| **Null hypothesis** | "No effect" or "random" | Always |
| **Naive persistence** | "Tomorrow = Today" | Time series |
| **Domain simple** | Simplest domain-appropriate method | Always |
| **Random baseline** | Random predictions | Classification |

**Rule**: If you can't beat naive, your model has no predictive value.

### 5.5 Domain Calibration
Compare claimed results against domain plausibility bounds:

| Domain | Metric | Suspicious | Plausible | Excellent |
|--------|--------|------------|-----------|-----------|
| *Template* | *Accuracy* | *>X%* | *Y-Z%* | *Z-W%* |

**Reference**: See references/domain-calibration.md for specific domains.

### 5.6 Practical Significance
| Check | Requirement |
|-------|-------------|
| **Deployment feasibility** | Can model be deployed in practice? |
| **Operational costs** | Are costs justified by improvement? |
| **Maintenance burden** | Sustainable long-term? |
| **Edge case handling** | Graceful degradation? |

### 5.7 Uncertainty Quantification
| Requirement | Method |
|-------------|--------|
| **Confidence intervals** | On all predictions and metrics |
| **Bootstrap analysis** | Robustness of results |
| **Significance tests** | vs baseline (p < 0.05) |
| **Multiple testing correction** | If >3 comparisons |

**Red flag**: No confidence intervals = can't assess reliability.

### 5.8 Reproducibility Assessment
| Requirement | Description |
|-------------|-------------|
| **Data specification** | Exact sources, dates, preprocessing |
| **Method specification** | All parameters and choices documented |
| **Code availability** | Implementation accessible |
| **Environment specification** | Hardware/software documented |

**Test**: Could an independent party reproduce these results using only the documentation?

**Reference**: See references/validation-checklist.md for complete requirements.

### 5.9 Adversarial Assessment
| Posture Level | Indicators | Response |
|---------------|------------|----------|
| L0: None | No anti-analysis | Standard protocol |
| L1: Passive | Obfuscation, encryption | Deobfuscate, decrypt |
| L2: Active Detection | Debugger checks, logging | Stealth, evasion |
| L3: Active Response | Terminate, corrupt, counter | Isolated analysis |
| L4: Adaptive | Changes behavior, patches | Parallel approaches |

**Anti-Analysis Detection Techniques** (software):
| Technique | Bypass |
|-----------|--------|
| PEB.BeingDebugged | Patch PEB in memory |
| Timing (RDTSC) | Hardware breakpoints, emulation |
| IsDebuggerPresent | Hook API, return false |
| TLS callbacks | Analyze TLS directory first |
| Self-modifying code | Dump after decryption |

**Deception Indicators**:
- Too-clean data → honeypot
- Inconsistent complexity → hidden subsystem
- Perfect documentation → synthetic environment

### 5.10 Attack Surface Mapping (if applicable)
```
Attack_Surface = Σ(Entry_Points × Exposure × Privilege)
```
| Category | Discovery |
|----------|-----------|
| Network | Port scan, traffic analysis |
| File System | Enumeration, permission audit |
| User Input | Interface fuzzing |
| APIs | Schema extraction, fuzzing |

### 5.11 Stop Condition
Phase 5 complete when:
- [ ] Model passes validation hierarchy (interpolation + extrapolation)
- [ ] Residual diagnostics pass (whiteness + independence)
- [ ] Adversarial posture classified (L0-L4)
- [ ] Known model limitations documented

## Tool Integration

| Domain | Tools | Use |
|--------|-------|-----|
| Binary RE | Ghidra, IDA Pro, Binary Ninja | Disassembly, decompilation |
| Dynamic | Frida, PIN, DynamoRIO | Runtime instrumentation |
| Symbolic | angr, KLEE | Path exploration, constraint solving |
| System ID | MATLAB SI Toolbox, SysIdentPy, SIPPY | Parameter estimation |
| Protocol | Netzob, Wireshark, Scapy | State machine extraction |
| Fuzzing | AFL++, libFuzzer, AFLNet | Edge case discovery |
| Sensitivity | SALib | Sobol', Morris screening |

**Web search triggers**: Unknown component, unexpected behavior, CVE lookup, library documentation.

**References**:
- Tool usage: references/tool-catalog.md
- Sensitivity algorithms: scripts/epistemic_lib.py
- Adversarial bypass: references/adversarial-heuristics.md
- Worked example: examples/thermostat-case-study.md

---

## Recursive Decomposition (COMPREHENSIVE tier only)

When system exceeds single-pass capacity:

```
DECOMPOSE(system, depth):
  IF depth > MAX_DEPTH OR complexity < THRESHOLD:
    RETURN analyze_atomic(system)
  
  components = partition(system)  # See partitioning strategies below
  sub_models = [DECOMPOSE(c, depth+1) for c in components]
  interactions = identify_coupling(components)
  composed = compose(sub_models, interactions)
  
  IF emergence_gap(composed, system) > 0.2:
    composed = augment_emergence(composed)
  
  RETURN composed
```

**Partitioning Strategies**:
- Functional: Split by what components DO
- Structural: Split by physical/logical boundaries
- Data flow: Follow transformations
- Temporal: Split by time scale

**Termination**: depth > 5 OR components < 3 OR marginal_info < 5%

---

## Bayesian Hypothesis Tracking

### Proper Update Rule
```
P(H|E) = P(E|H) · P(H) / P(E)

Where:
  P(E|H) = likelihood of evidence given hypothesis
  P(H) = prior belief in hypothesis  
  P(E) = Σᵢ P(E|Hᵢ)P(Hᵢ) = normalizing constant
```

### Practical Shortcuts
| Evidence Type | Likelihood Ratio | Update |
|---------------|------------------|--------|
| Strong confirm | P(E|H)/P(E|¬H) ≈ 10 | posterior ≈ prior × 3 |
| Weak confirm | ratio ≈ 2 | posterior ≈ prior × 1.3 |
| Neutral | ratio ≈ 1 | no change |
| Weak disconfirm | ratio ≈ 0.5 | posterior ≈ prior × 0.7 |
| Strong disconfirm | ratio ≈ 0.1 | posterior ≈ prior × 0.3 |
| Falsify | ratio = 0 | posterior = 0 |

### Model Comparison
Bayes Factor: K = P(D|M₁) / P(D|M₂)
- log₁₀(K) > 2 → decisive evidence for M₁
- log₁₀(K) > 1 → strong evidence
- log₁₀(K) > 0.5 → substantial evidence

**Tool**: Use `scripts/bayesian_tracker.py` for automated tracking.

---

## Decision Trees

### "Which Model Structure?"
```
START
├─ Single output? 
│  ├─ Yes → Linear relationship?
│  │         ├─ Yes → ARX (or ARMAX if noise colored)
│  │         └─ No → NARMAX
│  └─ No → State-Space
└─ Discrete modes visible?
   └─ Yes → EFSM (state machine)
```

### "When to Stop Analysis?"
```
START
├─ Time budget exhausted? → STOP, document uncertainty
├─ Fidelity target met?
│  ├─ Yes → STOP, deliver model
│  └─ No → Continue
├─ Diminishing returns? (info gain < 5%/hour)
│  ├─ Yes → STOP or escalate tier
│  └─ No → Continue
└─ Adversarial detection triggered?
   └─ Yes → Pause, reassess approach
```

### "Recursive Decompose?"
```
START
├─ Components > 15? → Yes → DECOMPOSE
├─ Interactions > 50? → Yes → DECOMPOSE
├─ Cognitive overload? → Yes → DECOMPOSE
└─ Fidelity plateau? (<5% improvement) → DECOMPOSE or STOP
```

---

## RAPID Tier Workflow

For quick claim validation before investing analysis time. Complete in <30 minutes.

### Step 1: Coherence Check (2 minutes)
```
[ ] Claimed data matches claimed task?
[ ] Metrics appropriate for task type?
[ ] Results consistent across claim?
[ ] No obvious AI-slop indicators?
[ ] Claims within physical possibility?
```

### Step 2: Verifiability Check (2 minutes)
```
[ ] Data source specified?
[ ] Methodology documented?
[ ] Parameters disclosed?
[ ] Could be independently reproduced?
```

### Step 3: Red Flag Scan (3 minutes)
```
[ ] Check instant reject conditions
[ ] Count methodology red flags
[ ] Count documentation red flags
[ ] Count results red flags
[ ] Count claims red flags
```

### Step 4: Domain Calibration (3 minutes)
Compare claimed results to domain plausibility bounds:
```
| Claimed Metric | Value | Domain Bound | Assessment |
|----------------|-------|--------------|------------|
| [metric]       | [X]   | [Y-Z]        | PLAUSIBLE/SUSPICIOUS |
```

### Step 5: Verdict
| Verdict | Criteria |
|---------|----------|
| **CREDIBLE** | 0 instant rejects, 0-1 red flags, within bounds |
| **SKEPTICAL** | 0 instant rejects, 2-3 red flags, near bounds |
| **DOUBTFUL** | 4+ red flags OR at bounds |
| **REJECT** | Any instant reject OR >5 red flags OR beyond bounds |

### RAPID State Block Format
```
[STATE: Phase 0.5 | Tier: RAPID | Coherence: PASS/FAIL | Red Flags: N | Verdict: X]
```

**Tool**: Use `scripts/rapid_checker.py` for automated RAPID workflow.

---

## Output Artifacts

1. **Coherence Screening Report** (Phase 0.5, RAPID): Red flags, verdict, recommendation
2. **Analysis Plan** (Phase 0): Tier, questions, hypotheses
3. **I/O Surface Map** (Phase 1): Channels, probe database
4. **Causal Graph** (Phase 2): Nodes, edges, loops
5. **Parametric Model** (Phase 3): Equations, parameters, uncertainty
6. **Composed Model** (Phase 4): Synthesis, emergence report
7. **Validation Report** (Phase 5): Metrics, baselines, uncertainty, limitations
8. **Hypothesis Registry**: All hypotheses with posteriors
9. **Attack Surface Map** (if adversarial): Entry points, risk scores
10. **State Block**: End every response with current state

## Tracker Commands

```bash
# Add hypothesis
python scripts/bayesian_tracker.py add "System uses REST API" --prior 0.6

# Update with evidence
python scripts/bayesian_tracker.py update H1 "Found /api/v1" --preset strong_confirm

# Visualize
python scripts/bayesian_tracker.py viz

# Export for context sync
python scripts/bayesian_tracker.py export

# Check for biases
python scripts/bayesian_tracker.py lint
```

---

## Critical Reminders

- **Tier first**: Don't over-engineer. LITE if possible.
- **Falsify, don't confirm**: Design tests to break hypotheses.
- **Quantify uncertainty**: Never report point estimates alone.
- **Time-box ruthlessly**: Diminishing returns are real.
- **Emergence is real**: Component models ≠ system model.
- **Adversarial systems fight back**: Assume resistance.
- **Map ≠ Territory**: Your model is wrong. Is it useful?
- **Know your traps**: Mirror-imaging, confirmation bias, anchoring kill analyses. See references/cognitive-traps.md.