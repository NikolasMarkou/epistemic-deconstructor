# Decision Trees

Quick-reference decision trees for common protocol branch points.

## Table of Contents

- [Which Model Structure?](#which-model-structure)
- [When to Stop?](#when-to-stop)
- [Recursive Decompose?](#recursive-decompose)
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
   Discrete modes? → EFSM
```

---

## When to Stop?

```
├─ Time budget exhausted? → STOP, document uncertainty
├─ Fidelity target met? → STOP, deliver model
├─ Diminishing returns (<5%/hr)? → STOP or escalate tier
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
- `partition_by_coupling(s)` = Group components by interaction strength
- `emergence_gap(model, actual)` = |model_prediction - actual| / |actual|

**Partitioning strategies**: Functional, Structural, Data flow, Temporal

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
