# Decision Trees

Quick-reference decision trees for common protocol branch points.

## Table of Contents

- [Which Model Structure?](#which-model-structure)
- [When to Stop?](#when-to-stop)
- [Recursive Decompose?](#recursive-decompose)
- [When to Start Over?](#when-to-start-over)
- [RAPID → Next Tier?](#rapid--next-tier)
- [Tier Selection from Questionnaire](#tier-selection-from-questionnaire)

---

## Which Model Structure?

```
START
├─ Single output?
│  ├─ Linear? → ARX (ARMAX if colored noise)
│  └─ Nonlinear? → NARMAX
└─ Multiple outputs? → State-Space
   Discrete modes? → EFSM (Extended Finite State Machine — state machine
                     with variables, guards, and actions; see system-identification.md)
```

---

## When to Stop?

> "Design is based on requirements. There's no justification for designing something one bit 'better' than the requirements dictate." — Akin's Law #13

```
├─ Fidelity target met? → STOP, deliver model (don't pursue higher fidelity than required)
├─ Diminishing returns (<5% improvement per iteration)? → STOP or escalate tier
├─ Pursuing elegance over functionality? → STOP (Edison's Law: "better" is the enemy of "good")
└─ Adversarial detection? → Pause, reassess
```

---

## Recursive Decompose?

```
├─ Components > 15 or Interactions > 50? → DECOMPOSE
├─ Cognitive overload? → DECOMPOSE
└─ Fidelity plateau (3 iterations)? → DECOMPOSE or STOP
```

### Recursive Decomposition Algorithm (COMPREHENSIVE Tier)

```
RECURSIVE_DECOMPOSE(system, depth=0):
  MAX_DEPTH = 3
  COMPLEXITY_THRESHOLD = 15

  IF depth > MAX_DEPTH: RETURN shallow_model(system)
  IF count_components(system) < COMPLEXITY_THRESHOLD:
    RETURN analyze_atomic(system)  # Run STANDARD tier on subsystem

  partitions = partition_by_coupling(system)
  sub_models = [RECURSIVE_DECOMPOSE(p, depth+1) FOR p in partitions]
  composed = compose(sub_models)  # Phase 4 operators

  IF emergence_gap(composed, system) > 0.2:
    augment_emergence(composed)
  RETURN composed
```

**Definitions:**
- `analyze_atomic(s)` = Run STANDARD tier phases 0-5 on subsystem s
- `count_components(s)` = Number of distinct functional units
- `partition_by_coupling(s)` = Group components by interaction strength (see `compositional-synthesis.md` coupling assessment)
- `compose(sub_models)` = Combine sub-models using Phase 4 composition operators (serial, parallel, feedback, hierarchical — see `compositional-synthesis.md`)
- `emergence_gap(model, actual)` = |model_prediction - actual| / |actual|
- `augment_emergence(model)` = Add interaction terms, feedback loops, nonlinear coupling, or time delays to close the emergence gap below 10%

**Partitioning strategies**: Functional, Structural, Data flow, Temporal

---

## When to Start Over?

> "Sometimes, the fastest way to get to the end is to throw everything out and start over." — Akin's Law #11
> "Your best design efforts will inevitably wind up being useless in the final design." — Akin's Law #4

```
├─ Validation failure on fundamental assumption? → START OVER (re-scope from Phase 0)
├─ 3+ iterations with <5% improvement each? → START OVER (different approach)
├─ Discovered system is different archetype than assumed? → START OVER (re-classify)
├─ Model works but is wrong archetype? → CONTINUE (refine, don't restart for elegance)
└─ High time investment but no working model? → START OVER (sunk cost is irrelevant)
```

**What "start over" preserves**: Observations, falsified hypotheses, I/O measurements. These constrain the restart.
**What "start over" discards**: Model structure, composition, parameter estimates. These were the wrong path.

**Decision rule**: If you've spent >50% of budget and validation R² < 0.5, starting over is likely faster than continuing. Log the decision in `decisions.md` with trade-off rationale.

---

## RAPID → Next Tier?

```
After RAPID verdict:
├─ CREDIBLE + no follow-up needed? → DONE
├─ SKEPTICAL/DOUBTFUL + need more? → STANDARD tier
├─ REJECT + must investigate? → STANDARD tier (find root cause)
└─ Complex system revealed? → COMPREHENSIVE tier
```

---

## Tier Selection from Questionnaire

| Access | Adversary | Time | Components | → Tier |
|--------|-----------|------|------------|--------|
| Any | No | <30min | - | RAPID |
| Any | No/Unknown | <2h | <5 | LITE |
| Any | No | 2-20h | 5-15 | STANDARD |
| Any | Yes | >2h | Any | STANDARD + adversarial |
| Any | Yes/Unknown | >20h | >15 | COMPREHENSIVE |
| Human target | - | 1-4h | - | PSYCH |

---

## Cross-References

- Setup techniques: `references/setup-techniques.md`
- Simulation paradigm selection: `references/simulation-guide.md` (archetype-to-mode mapping)
- Compositional synthesis operators: `references/compositional-synthesis.md`
- Engineering laws (start-over criteria, fidelity sufficiency): `references/engineering-laws.md`
