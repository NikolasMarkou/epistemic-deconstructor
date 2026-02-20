# Compositional Model Synthesis Reference

Methods for combining sub-models into whole-system models with uncertainty propagation.

## Table of Contents

- [Composition Operators](#composition-operators)
- [Coupling Strength Assessment](#coupling-strength-assessment)
- [Uncertainty Propagation Framework](#uncertainty-propagation-framework)
- [Interface Specification](#interface-specification)
- [Validation Checklist](#validation-checklist)
- [Common Composition Patterns](#common-composition-patterns)

---

## Composition Operators

### Serial Composition (M₁ → M₂)
Output of M₁ feeds input of M₂.

**Transfer function**: H_total(s) = H₁(s) · H₂(s)

**Time domain**: y = M₂(M₁(x))

**Uncertainty propagation**:
```
If independent: Var(y) = Var(M₂) + (∂M₂/∂x)² · Var(M₁)
Confidence: C_total = min(C₁, C₂) × coupling_factor
```

**When to use**: Pipeline architectures, sequential processing

### Parallel Composition (M₁ ‖ M₂)
Same input to both, outputs combined.

**Transfer function**: H_total(s) = H₁(s) + H₂(s) (if additive)

**Time domain**: y = combine(M₁(x), M₂(x))

**Uncertainty propagation**:
```
If additive: Var(y) = Var(M₁) + Var(M₂) + 2·Cov(M₁,M₂)
If independent: Var(y) = Var(M₁) + Var(M₂)
```

**When to use**: Redundant systems, voting architectures

### Feedback Composition (M₁ ↔ M₂)
Closed-loop connection.

**Transfer function**: H_closed(s) = G(s) / (1 + G(s)·H(s))

Where G = forward path, H = feedback path

**Stability requirement**: Gain margin > 0 dB, Phase margin > 45°

**Uncertainty propagation**:
```
Confidence: C_total = min(C₁, C₂) × stability_margin
Stability margin ∈ [0, 1] based on phase/gain margins
```

**When to use**: Control systems, homeostatic systems

### Hierarchical Composition (M₁ ⊃ M₂)
Subsystem treated as atomic at higher level.

**Evaluation**: Recursive model call

**Interface contract**:
- Input domain of M₂ ⊆ Output domain of M₁'s internal call
- Timing constraints propagate

**Uncertainty propagation**:
```
C_total = C_parent × Π(C_children)
```

## Coupling Strength Assessment

| Coupling Level | Characteristics | Factor |
|----------------|-----------------|--------|
| Tight | Shared state, direct memory access | 0.7-0.8 |
| Moderate | Message passing, defined interface | 0.85-0.95 |
| Loose | Event-driven, async | 0.9-0.98 |
| Decoupled | No runtime dependency | 0.98-1.0 |

### Assessment Procedure
1. Identify interface between components
2. Characterize data flow (sync/async, rate, format)
3. Test sensitivity: perturb one component, measure effect on other
4. Assign coupling factor based on sensitivity

## Uncertainty Propagation Framework

### Monte Carlo Method
```python
def propagate_uncertainty_mc(models, compositions, n_samples=1000):
    """
    Monte Carlo uncertainty propagation.
    """
    results = []
    for _ in range(n_samples):
        # Sample from each model's parameter distribution
        params = {m.id: m.sample_params() for m in models}
        
        # Compose and evaluate
        composed = compose(models, compositions, params)
        results.append(composed.evaluate())
    
    return np.mean(results), np.std(results)
```

### Analytical Method (Linear)
For linear compositions with Gaussian uncertainty:
```
Serial: μ_total = μ₁ · μ₂
        σ²_total = μ₁²σ₂² + μ₂²σ₁² + σ₁²σ₂²

Parallel (sum): μ_total = μ₁ + μ₂
                σ²_total = σ₁² + σ₂² + 2ρσ₁σ₂
```

### Weakest Link Rule
```
C_system ≤ min(C_components) × Π(coupling_factors)
```

Conservative bound: actual confidence cannot exceed this.

## Interface Specification

Document for each composition:

| Property | Description |
|----------|-------------|
| Input types | Domain, units, valid ranges |
| Output types | Domain, units, valid ranges |
| Timing | Latency, throughput, deadlines |
| Error handling | How errors propagate |
| State assumptions | Required preconditions |

## Validation Checklist

### Consistency Checks
- [ ] Interface types match (input of M₂ ∈ output of M₁)
- [ ] Units are compatible
- [ ] Timing constraints satisfiable

### Conservation Laws
- [ ] What quantities are conserved across composition?
- [ ] Mass balance, energy balance, information balance

### Boundary Testing
- [ ] Composed model valid at operating limits
- [ ] Graceful degradation at boundaries

### Emergence Testing
- [ ] Does composed model match observed whole-system behavior?
- [ ] Quantify emergence gap: |y_composed - y_observed| / |y_observed|
- [ ] If gap > 20%, add emergence terms

## Common Composition Patterns

### Pipeline
```
x → M₁ → M₂ → ... → Mₙ → y
H_total = H₁ · H₂ · ... · Hₙ
```

### Fan-out / Fan-in
```
       ┌→ M₁ ─┐
x ─────┼→ M₂ ─┼──→ combine → y
       └→ M₃ ─┘
```

### Feedback Control
```
r ──→ ⊕ ──→ G ──┬──→ y
      ↑         │
      └─── H ←──┘
```

### Hierarchical / Nested
```
M_parent:
  ┌─────────────────┐
  │  M_child_1      │
  │       ↓         │
  │  M_child_2      │
  └─────────────────┘
```

## Cross-References

- Simulation paradigms (SD, MC, ABM): `references/simulation-guide.md`
- System identification inputs: `references/system-identification.md`
- Validation requirements: `references/validation-checklist.md`
