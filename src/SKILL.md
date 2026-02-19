---
name: epistemic-deconstructor
description: "Systematic reverse engineering of unknown systems using scientific methodology. Use when: (1) Black-box analysis, (2) Competitive intelligence, (3) Security analysis, (4) Forensics, (5) Building predictive models. Features 6-phase protocol, Bayesian inference, compositional synthesis, and psychological profiling (PSYCH tier)."
---

# Epistemic Deconstruction Protocol v6.6

## Core Objective

Transform epistemic uncertainty into predictive control through principled experimentation, compositional modeling, and Bayesian inference.

**Session Memory**: Run `scripts/session_manager.py new "system"` at analysis start. All phase outputs, observations, and decisions persist to disk. See `references/session-memory.md` for full protocol.

---

## State Block Protocol (REQUIRED)

**Every response must end with a State Block:**
```
[STATE: Phase X | Tier: Y | Active Hypotheses: N | Confidence: Low/Med/High]
```

Extended formats:
```
[STATE: Phase 2 | Tier: STANDARD | Active Hypotheses: 3 | Lead: H2 (78%) | Confidence: Medium]
[STATE: Phase 0.5 | Tier: RAPID | Coherence: PASS | Red Flags: 2 | Verdict: SKEPTICAL]
[STATE: Phase 2-P | Tier: PSYCH | Archetype: High-N/Low-A | Rapport: Med | Stress: Low]
```

---

## Auto-Pilot Mode

**"Help me start"** or **"Walk me through"** triggers questionnaire mode:

**System Analysis:**
1. What system? (software/hardware/organizational/other)
2. Access level? (full source/binary only/black-box I/O)
3. Adversary present? (yes/no/unknown)
4. Time budget? (hours)
5. Goal? (how it works/parameters/vulnerabilities)

**PSYCH Analysis** ("Analyze this person" / "Profile this individual"):
1. Subject type? (Real/Fictional/Online persona/Historical)
2. Source material? (Text/Video/Audio/Documents/Mixed)
3. Relationship? (Peer/Superior/Subordinate/Adversary/Observer)
4. Goal? (Predict behavior/Detect deception/Negotiation prep/Build rapport)
5. Time budget? (Brief/Extended/Ongoing)

**Tier Selection from Questionnaire:**
| Access | Adversary | Time | Components | → Tier |
|--------|-----------|------|------------|--------|
| Any | No | <30min | - | RAPID (claim validation only) |
| Any | No/Unknown | <2h | <5 | LITE |
| Any | No | 2-20h | 5-15 | STANDARD |
| Any | Yes | >2h | Any | STANDARD + adversarial |
| Any | Yes/Unknown | >20h | >15 | COMPREHENSIVE |
| Human target | - | 1-4h | - | PSYCH |

---

## Axioms

| Axiom | Checkpoint |
|-------|------------|
| Finite Rules | Can you state the governing rules? |
| Scientific Method | Are you falsifying or confirming? |
| Observability Bound | Have you enumerated all I/O? |
| Observer Effect | What did your probing change? |
| Adversarial Resistance | Have you tested for active defense? |
| Emergence | Does component model predict whole? |
| Map ≠ Territory | Can you articulate model limitations? |

---

## Tier Selection (REQUIRED FIRST STEP)

| Tier | Trigger | Phases | Budget |
|------|---------|--------|--------|
| **RAPID** | Quick claim validation | 0.5→5 | <30min |
| **LITE** | Known archetype, stable system | 0→1→5 | <2h |
| **STANDARD** | Unknown internals, single domain | 0→1→2→3→4→5 | 2-20h |
| **COMPREHENSIVE** | Multi-domain, adversarial, critical | All + decomposition | 20h+ |
| **PSYCH** | Human behavior analysis | 0-P→1-P→2-P→3-P→4-P→5-P | 1-4h |

**LITE Tier Specifics** (for known archetypes):
- Phase 0: Abbreviated setup, archetype-specific hypotheses
- Phase 1: Focus on archetype-relevant I/O (≥5 probes sufficient)
- Phase 5: Validate against archetype baseline, document deviations

**Decision**: Use RAPID for claim validation first. If unsure, start STANDARD. Escalate to COMPREHENSIVE if >15 components or adversarial indicators.

---

## Phase 0: Setup & Frame
*Budget: 10% | Output: Analysis Plan*

**Activities:**
1. Define position (insider/outsider), access, constraints, system type
2. Build Question Pyramid (L1-L5: DO → HOW → WHY → PARAMETERS → REPLICATE)
3. Seed 3+ hypotheses (H1: likely, H2: alternative, H3: adversarial/deceptive)
4. Adversarial pre-check (high entropy? anti-debug patterns?)
5. Acknowledge cognitive vulnerabilities

**Fidelity Levels (Question Pyramid):**
| Level | Question | Goal | Test |
|-------|----------|------|------|
| L1 | DO | Can I trigger a response? | Any output from input |
| L2 | HOW | What transforms input to output? | Explain processing steps |
| L3 | WHY | What drives the mechanism? | Predict design choices |
| L4 | PARAMETERS | What values control behavior? | <5% error on measurables |
| L5 | REPLICATE | Can I rebuild it? | Replica indistinguishable |

**Stop Condition:**
- [ ] Tier selected
- [ ] Fidelity target locked (L1-L5)
- [ ] ≥3 hypotheses documented
- [ ] I/O channels enumerated
- [ ] Adversarial risk assessed

**Reference**: `references/setup-techniques.md`, `references/cognitive-traps.md`

---

## Phase 0.5: Coherence Screening (RAPID Entry)
*Budget: 5-10% | Output: Go/No-Go Decision*

**Activities:**
1. Verify claim-task alignment (data matches task? metrics appropriate?)
2. Check instant reject conditions (impossibility, contamination, incoherence)
3. Red flag scan (missing baseline? tool worship? documentation gaps?)
4. Domain calibration check

**Stop Condition:**
- [ ] Claim-task alignment verified
- [ ] Instant reject conditions checked
- [ ] Red flag count documented
- [ ] Go/No-Go decision recorded

| Result | Criteria | Action |
|--------|----------|--------|
| **GO** | 0 rejects, <3 flags, coherent | Proceed |
| **CONDITIONAL** | Minor concerns | Request info |
| **NO-GO** | Reject condition OR 3+ flags | REJECT |

**Reference**: `references/rapid-assessment.md`, `references/coherence-checks.md`, `references/red-flags.md`, `references/domain-calibration.md`

---

## Phase 1: Boundary Mapping
*Budget: 20% | Output: I/O Surface, Transfer Functions*

**Activities:**
1. Enumerate I/O channels (explicit, implicit, side-channel, feedback)
2. Apply probe signals (step, impulse, PRBS, edge cases)
3. Assess data quality (coherence γ² ≈ 1.0 = good)
4. Build stimulus-response database

**Stop Condition:**
- [ ] ≥80% I/O channels characterized
- [ ] Response variance < 10% on repeated probes
- [ ] Edge cases tested (min/max/null/overflow)
- [ ] Stimulus-response database ≥20 entries (LITE: ≥5)

**Reference**: `references/boundary-probing.md`

---

## Phase 2: Causal Analysis
*Budget: 25% | Output: Causal Graph, Dependency Matrix*

**Activities:**
1. Static analysis (if visible): disassembly, decompilation, data flow
2. Dynamic analysis: tracer injection, differential analysis
3. Sensitivity analysis (Morris screening, Sobol' indices)
4. Construct causal graph (nodes, edges, feedback loops R/B)
5. Run falsification loop for each hypothesis:
   ```
   FOR each hypothesis H:
     design_test(H)  # Test that would falsify if H is false
     run_test()
     IF result contradicts H: update posterior with LR ≤ 0.1
     IF all H falsified: generate new hypotheses, restart
   UNTIL lead hypothesis > 0.8 OR iteration > 5
   ```

**Stop Condition:**
- [ ] ≥70% behaviors have causal explanation
- [ ] All major feedback loops identified
- [ ] ≥1 hypothesis refuted
- [ ] Causal graph validated by ≥3 differential tests

**Reference**: `references/causal-techniques.md`, `references/tools-sensitivity.md`

---

## Phase 3: Parametric Identification
*Budget: 20% | Output: Mathematical Model, Uncertainty Bounds*

**Activities:**
1. Select model structure (ARX → ARMAX → NARMAX → State-Space)
2. Estimate parameters (OLS, subspace methods)
3. Apply information criteria (AIC/BIC) for structure selection
4. Quantify uncertainty (bootstrap, Bayesian)
5. For time-series outputs: run `scripts/ts_reviewer.py` (stationarity, forecastability, decomposition, baselines)

**Stop Condition:**
- [ ] Model selected via information criterion
- [ ] Residuals pass whiteness test
- [ ] Cross-validation R² > 0.8
- [ ] Parameter uncertainty bounds computed

**Reference**: `references/system-identification.md`

---

## Phase 4: Model Synthesis
*Budget: 15% | Output: Unified Model, Emergence Report*

**Activities:**
1. Compose sub-models (serial H₁·H₂, parallel H₁+H₂, feedback G/(1+GH))
2. Propagate uncertainty through composition
3. Test for emergence: `mismatch = |predicted - actual| / |actual|`; if > 20%, emergence present
4. Classify archetype (State Machine, Pipeline, Controller, Pub/Sub, Network, Adaptive)

**Archetype Signatures:**
| Archetype | Signature | Vulnerabilities | Examples |
|-----------|-----------|-----------------|----------|
| State Machine | Discrete modes, transitions | State explosion, race conditions | Protocols, UI flows |
| Pipeline | Sequential transforms | Single point of failure, bottlenecks | ETL, compilers |
| Controller | Feedback loops, setpoints | Instability, windup | PID, thermostats |
| Pub/Sub | Event-driven, decoupled | Message loss, ordering | Message queues |
| Network | Graph topology, routing | Cascade failure, partition | Social, distributed |
| Adaptive | Learning, parameter drift | Concept drift, adversarial | ML models |

**Stop Condition:**
- [ ] All sub-models composed with explicit semantics
- [ ] Uncertainty propagated
- [ ] Emergence test performed
- [ ] Archetype identified with vulnerability assessment

**Reference**: `references/compositional-synthesis.md`

---

## Phase 5: Validation & Adversarial
*Budget: 10% | Output: Validation Report, Attack Surface Map*

**Activities:**
1. Validation hierarchy (interpolation R²>0.95, extrapolation R²>0.80, counterfactual)
2. Residual diagnostics (whiteness, independence, normality)
3. Baseline comparison (must beat naive baseline)
4. Domain calibration (compare to plausibility bounds)
5. Adversarial posture classification (L0-L4)
6. Attack surface mapping (if applicable)

**Stop Condition:**
- [ ] Model passes validation hierarchy
- [ ] Residual diagnostics pass
- [ ] Adversarial posture classified
- [ ] Known limitations documented

**Reference**: `references/validation-checklist.md`, `references/adversarial-heuristics.md`

---

## PSYCH Tier: Psychological Profiling

For analyzing human behavior, personas, and profiles. See `references/psych-tier-protocol.md` for complete protocol.

**Phases:** 0-P (Context) → 1-P (Baseline) → 2-P (Stimulus-Response) → 3-P (Structural ID) → 4-P (Motive) → 5-P (Validation)

**Ethical Constraints (REQUIRED):**
- No clinical diagnosis (use observable traits, not disorder labels)
- Cultural calibration required
- Document consent status
- Defensive use only

**State Block:** `[STATE: Phase X-P | Tier: PSYCH | Archetype: Y | Rapport: L/M/H | Stress: L/M/H]`

**Key Outputs:**
- OCEAN profile with evidence
- Dark Triad assessment (always assess all three)
- MICE/RASP driver ranking (Money, Ideology, Coercion, Ego / Revenge, Addiction, Sex, Power)
- Behavioral predictions
- Interaction strategy (Do/Don't/Watch For)

**Tool**: `scripts/belief_tracker.py`

**Reference**: `references/psych-tier-protocol.md`, `references/archetype-mapping.md`, `references/motive-analysis.md`

---

## Recursive Decomposition (COMPREHENSIVE only)

```
RECURSIVE_DECOMPOSE(system, depth=0):
  # Constants
  MAX_DEPTH = 3              # Prevent infinite recursion
  COMPLEXITY_THRESHOLD = 15  # Component count

  # Base cases
  IF depth > MAX_DEPTH: RETURN shallow_model(system)
  IF count_components(system) < COMPLEXITY_THRESHOLD:
    RETURN analyze_atomic(system)  # Run STANDARD tier on subsystem

  # Recursive case
  partitions = partition_by_coupling(system)  # Group tightly-coupled components
  sub_models = [RECURSIVE_DECOMPOSE(p, depth+1) FOR p in partitions]
  composed = compose(sub_models)  # Phase 4 operators: serial, parallel, feedback

  # Emergence check
  IF emergence_gap(composed, system) > 0.2:
    augment_emergence(composed)  # Add correction term
  RETURN composed
```

**Definitions:**
- `analyze_atomic(s)` = Run STANDARD tier phases 0-5 on subsystem s
- `count_components(s)` = Number of distinct functional units
- `partition_by_coupling(s)` = Group components by interaction strength
- `emergence_gap(model, actual)` = |model_prediction - actual| / |actual|

**Partitioning strategies**: Functional, Structural, Data flow, Temporal

---

## Decision Trees

### "Which Model Structure?"
```
START
├─ Single output?
│  ├─ Linear? → ARX (ARMAX if colored noise)
│  └─ Nonlinear? → NARMAX
└─ Multiple outputs? → State-Space
   Discrete modes? → EFSM
```

### "When to Stop?"
```
START
├─ Time budget exhausted? → STOP, document uncertainty
├─ Fidelity target met? → STOP, deliver model
├─ Diminishing returns? → STOP or escalate tier
│   (Measure: <5% improvement in lead hypothesis posterior per hour)
└─ Adversarial detection triggered? → Pause, reassess
    (Indicators: anti-debug, high entropy, behavioral changes)
```

### "Recursive Decompose?"
```
START
├─ Components > 15? → DECOMPOSE
├─ Interactions > 50? → DECOMPOSE (N×(N-1)/2 > 50 for N>10)
├─ Cognitive overload? → DECOMPOSE (can't hold model in working memory)
└─ Fidelity plateau? → DECOMPOSE or STOP (no progress after 3 iterations)
```

### "RAPID → Next Tier?"
```
After RAPID verdict:
├─ CREDIBLE + no follow-up needed? → DONE
├─ SKEPTICAL/DOUBTFUL + need more? → STANDARD tier
├─ REJECT + must investigate? → STANDARD tier (find root cause)
└─ Complex system revealed? → COMPREHENSIVE tier
```

---

## RAPID Tier Workflow

Complete in <30 minutes:

**Step 1: Coherence (2 min)**
- [ ] Data matches task? Metrics appropriate? Results consistent? No AI-slop?

**Step 2: Verifiability (2 min)**
- [ ] Data source specified? Method documented? Reproducible?

**Step 3: Red Flags (3 min)**
- [ ] Check instant rejects, methodology flags, documentation flags, result flags

**Step 4: Domain Calibration (3 min)**
- Compare claimed results to domain plausibility bounds

**Step 5: Verdict**
| Verdict | Criteria |
|---------|----------|
| **CREDIBLE** | 0 rejects, 0-1 flags, within bounds |
| **SKEPTICAL** | 0 rejects, 2-3 flags, near bounds |
| **DOUBTFUL** | 4+ flags OR at bounds |
| **REJECT** | Any reject OR >5 flags OR beyond bounds |

**Tool**: `scripts/rapid_checker.py`

---

## Bayesian Tracking

### Update Rule
```
P(H|E) = P(E|H) · P(H) / P(E)
```

### Likelihood Ratio Shortcuts
| Evidence | LR | Update |
|----------|-----|--------|
| Strong confirm | ~10 | posterior ≈ prior × 3 |
| Weak confirm | ~2 | posterior ≈ prior × 1.3 |
| Weak disconfirm | ~0.5 | posterior ≈ prior × 0.7 |
| Strong disconfirm | ~0.1 | posterior ≈ prior × 0.3 |
| Falsify | 0 | posterior = 0 |

### Bayes Factor (Model Comparison)
K = P(D|M₁) / P(D|M₂)
- log₁₀(K) > 2 → decisive for M₁
- log₁₀(K) > 1 → strong evidence

**Tool**: `scripts/bayesian_tracker.py`

---

## Tracker Commands

### System Analysis (bayesian_tracker.py)
```bash
python scripts/bayesian_tracker.py add "Hypothesis" --prior 0.6
python scripts/bayesian_tracker.py update H1 "Evidence" --preset strong_confirm
python scripts/bayesian_tracker.py report --verbose
```

### PSYCH Tier (belief_tracker.py)
```bash
python scripts/belief_tracker.py add "High Neuroticism" --prior 0.5
python scripts/belief_tracker.py update T1 "Evidence" --preset strong_indicator
python scripts/belief_tracker.py profile
```

### RAPID Tier (rapid_checker.py)
```bash
python scripts/rapid_checker.py start "Claim to validate"
python scripts/rapid_checker.py coherence data-task-match --pass
python scripts/rapid_checker.py flag methodology "No baseline"
python scripts/rapid_checker.py verdict
```

### Time-Series Review (ts_reviewer.py)
```bash
python scripts/ts_reviewer.py review data.csv --column value
```

---

## Output Artifacts

1. **Coherence Report** (Phase 0.5): Red flags, verdict
2. **Analysis Plan** (Phase 0): Tier, questions, hypotheses
3. **I/O Surface Map** (Phase 1): Channels, probe database
4. **Causal Graph** (Phase 2): Nodes, edges, loops
5. **Parametric Model** (Phase 3): Equations, parameters, uncertainty
6. **Composed Model** (Phase 4): Synthesis, emergence report
7. **Validation Report** (Phase 5): Metrics, baselines, limitations
8. **Hypothesis Registry**: All hypotheses with posteriors
9. **State Block**: End every response with current state

---

## Tool Integration

| Domain | Tools |
|--------|-------|
| Binary RE | Ghidra (free, decompilation) |
| Dynamic | Frida (hooking), Unicorn (emulation) |
| Symbolic | angr (path exploration) |
| System ID | SysIdentPy (NARMAX), SIPPY (state-space) |
| Protocol | Netzob, Wireshark, Scapy |
| Fuzzing | AFL++, libFuzzer |
| Sensitivity | SALib |
| Time Series | ts_reviewer.py (signal diagnostics) |
| Utility | strace/procmon, pefile |

**Web search triggers**: Unknown component, unexpected behavior, CVE lookup, library docs.

**Reference**: `references/tool-catalog.md`

---

## Critical Reminders

- **Tier first**: Don't over-engineer. LITE if possible.
- **Falsify, don't confirm**: Design tests to break hypotheses.
- **Quantify uncertainty**: Never report point estimates alone.
- **Emergence is real**: Component models ≠ system model.
- **Map ≠ Territory**: Your model is wrong. Is it useful?
- **Know your traps**: See `references/cognitive-traps.md`.
