# Modeling Epistemology

Foundational reasoning principles for building models of unknown systems. Complements operational references (system-identification.md, validation-checklist.md, cognitive-traps.md) with the unified conceptual framework that generates them.

## Table of Contents

- [The Root Cause: Epistemic Incompleteness](#the-root-cause-epistemic-incompleteness)
- [Seven Principles of Rigorous Modeling](#seven-principles-of-rigorous-modeling)
- [The Fidelity-Generalizability Tradeoff](#the-fidelity-generalizability-tradeoff)
- [Dual Languages of Modeling](#dual-languages-of-modeling)
- [Three Iteration Loops](#three-iteration-loops)
- [Practical Maxims](#practical-maxims)
- [Known Blind Spots](#known-blind-spots)
- [Explicit vs Implicit Assumptions](#explicit-vs-implicit-assumptions)
- [Cross-References](#cross-references)

---

## The Root Cause: Epistemic Incompleteness

Every technique in this protocol exists to manage the gap between what needs to be known and what can be learned. This gap takes three forms:

1. **Finite observations** — We see samples, not populations. Every estimate has sampling variability. Every pattern might be noise. (Drives: regularization, baseline requirements, conformal prediction.)
2. **Limited access** — We must probe before knowing consequences. Gathering information costs time, resources, or operational exposure. (Drives: probe design, tier selection, exploration-exploitation tradeoffs.)
3. **Computational bounds** — Even with complete data, exact inference is often intractable. We must approximate. (Drives: model selection hierarchies, decomposition, information criteria.)

These three forms are irreducible. They generate the entire protocol methodology.

---

## Seven Principles of Rigorous Modeling

### 1. Epistemic Humility
Data underdetermines truth. Every model is provisional. Every prediction carries irreducible uncertainty. The analyst's first commitment is calibrated honesty about the limits of knowledge. *Key question: What do I not know, and how much does it matter?*

See `cognitive-traps.md` for countermeasures. See `evidence-calibration.md` for LR caps and prior discipline.

### 2. Principled Formulation
The hardest step is translating an ambiguous system into a precise analytical structure. The same system can be framed as inference (what is true?), optimization (what is best?), decision-making (what should I do?), or simulation (what happens if?). The choice of framing determines what can be learned.

**Before choosing methods, define**: the target question (what), available evidence (from what), and success metric (measured how). The choice of metric embeds values — symmetric error treats over/under equally; asymmetric loss encodes asymmetric costs.

### 3. Decomposition
Every analytical method decomposes complex into simple. This is not a technique within analysis — it IS the logical structure of analysis. Three forces converge to demand it:

- **Structural reality**: Systems are compositional. Conditional independence, locality, and hierarchy are features of reality.
- **Computational necessity**: Monolithic analysis is exponentially intractable. Decomposition achieves tractability.
- **Cognitive requirement**: Humans cannot design, verify, or trust what they cannot decompose.

Granularity is the control knob: coarser = more generalizable but less faithful; finer = the reverse. The choice of decomposition granularity is the single most consequential modeling decision.

See `compositional-synthesis.md` for composition operators. See `decision-trees.md` for recursive decomposition thresholds.

### 4. The Fidelity-Generalizability Tradeoff
Every model navigates a tension between fitting known data and performing on new data. This is the central challenge. See [dedicated section below](#the-fidelity-generalizability-tradeoff).

### 5. Dual Languages of Modeling
Probability and optimization are not competing philosophies — they are dual technical languages connected by precise identities. See [dedicated section below](#dual-languages-of-modeling).

### 6. Rigorous Validation
Validation is the constitutive practice that distinguishes a model from a guess. Every model is guilty until proven innocent. Three gaps that validation bridges: (1) Formulation gap — is the model a reasonable approximation? (2) Implementation gap — does the method correctly implement the model? (3) Deployment gap — will it work on unseen data?

See `validation-checklist.md` for the operational framework.

### 7. Iterative Refinement
Getting it right the first time is impossible. The workflow is inherently cyclic: formulate → decompose → fit → validate → diagnose → reformulate. See [Three Iteration Loops below](#three-iteration-loops).

---

## The Fidelity-Generalizability Tradeoff

The same tension appears under different names across every analytical tradition:

| Tradition | Fidelity Side | Generalizability Side |
|-----------|---------------|----------------------|
| Statistics | Low bias (complex model) | Low variance (simple model) |
| Regularization | Minimize empirical error | Penalty term (L1/L2/early stopping) |
| Bayesian inference | Likelihood (data fit) | Prior / marginal likelihood (parsimony) |
| Information theory | Full description | Minimum description length |
| Sequential analysis | Exploit current best | Explore uncertain options |
| Time-series | In-sample fit (R²) | Out-of-sample accuracy (MASE) |
| System identification | Model complexity (order) | Information criterion (AIC/BIC) |

**Operational rules**:
- Start simple. The best operating point is almost always simpler than expected.
- Cross-validate honestly — training performance is not a reliable indicator of future performance.
- Use information criteria (AIC, BIC, marginal likelihood) to balance fit and complexity.
- Ensembles hedge the tradeoff — combining diverse models reduces variance without proportionally increasing bias.

---

## Dual Languages of Modeling

| Probabilistic Language | Optimization Language | Identity |
|----------------------|---------------------|----------|
| Posterior inference | Regularized optimization | MAP = penalized MLE |
| Gaussian prior | L2 regularization | −log N(0,σ²) ∝ ‖θ‖² |
| Laplace prior | L1 regularization | −log Laplace ∝ ‖θ‖₁ |
| Marginal likelihood | Model evidence | ∫ p(D|θ)p(θ)dθ |
| KL divergence | Cross-entropy loss | H(p,q) = H(p) + KL(p‖q) |
| Expected utility | Loss minimization | max E[U] = min E[L] |

**When to use which**:
- **Probability**: When propagating uncertainty, combining evidence, handling missing data, or communicating calibrated predictions.
- **Optimization**: When seeking scalability, convergence guarantees, or exploiting convex structure.
- **Both**: When the problem demands it — most rigorous analysis does.

---

## Three Iteration Loops

| Loop | Scope | Trigger | Protocol Mapping |
|------|-------|---------|-----------------|
| **Inner** | Same formulation, adjust parameters | High variance (overfit) or hyperparameter sensitivity | Phase 3 model order selection, regularization tuning |
| **Middle** | Change model family or decomposition | High bias (underfit) or residual structure | Phase 3→Phase 3 (switch ARX→NARMAX, add features) |
| **Outer** | Redefine the problem | Validation reveals wrong question | Phase 5→Phase 0 (re-scope, change fidelity target) |

**Diagnostic before prescription**: High bias → add capacity. High variance → regularize or get more data. Misspecified model → reformulate. Wrong question → re-scope.

---

## Practical Maxims

### Formulation
1. Define the target question, available evidence, and success metric before choosing methods.
2. Recognize mathematical structure: is it linear? Markov? Factorable? Convex? Exploit structure.
3. Distinguish prediction (correlation suffices) from intervention (causation required).

### Modeling
4. Start with the simplest model that could possibly work. Simple models generalize better, fail more interpretably, and are easier to debug.
5. Every model has assumptions — make yours explicit. There is no assumption-free analysis.
6. Encode known structure as constraints or priors, not as features.
7. Let the data choose complexity (information criteria, marginal likelihood, cross-validation).

### Validation
8. Never evaluate on training data. The only honest measure is out-of-sample performance.
9. Validation must respect data structure. Temporal data needs temporal splits. Grouped data needs group splits.
10. Verify implementation before validating the model. Bugs are more common than misspecification.

### Epistemological
11. Quantify uncertainty — a prediction without confidence bounds is unfinished work.
12. Know what you do not know — model uncertainty about the model itself, not just parameters.

---

## Known Blind Spots

No framework is complete. Watch for these gaps:

1. **Representation**: How to discover the right decomposition or features is undertheorized relative to its importance.
2. **Social epistemology**: Analysis is treated as solo activity. Team dynamics, institutional incentives, and peer review are absent from the formalism.
3. **Values in objectives**: The choice of what to optimize embeds human values. Treating loss functions as purely technical choices obscures ethical dimensions.
4. **Non-stationarity**: Most methods assume stable data-generating processes. Distribution shift, regime changes, and concept drift are acknowledged but not deeply integrated.
5. **Knowledge vs understanding**: Models predict but don't explain. When and why a model works matters as much as whether it works.
6. **Knowledge decay**: Models degrade over time. Monitoring is covered but a deep theory of temporal knowledge validity is missing.

---

## Explicit vs Implicit Assumptions

Every model structure encodes assumptions. Understanding the distinction helps calibrate confidence:

**Explicit (structural) assumptions** are hard-coded into the model. The model *cannot* violate them regardless of data. Examples: ARX assumes linearity; causal ordering assumes past→future; Fourier basis assumes periodicity; state-space assumes Markov property.

**Implicit (procedural) assumptions** are encouraged but not enforced — through regularization, validation strategy, or data preprocessing. The model *can* violate them if data strongly contradicts. Examples: L2 regularization nudges toward smoothness; walk-forward CV encourages temporal respect; baseline comparison pressures toward parsimony.

**Design principle**: Use explicit encoding when confidence in the assumption is high. Use implicit encoding when you want to encourage a property without ruling out alternatives. When an explicit assumption is violated, the model fails silently. When an implicit assumption is violated, the model can adapt — at the cost of higher variance.

---

## Cross-References

- Cognitive traps and debiasing: `cognitive-traps.md`
- Evidence calibration rules: `evidence-calibration.md`
- Validation operational framework: `validation-checklist.md`
- Model structure selection: `system-identification.md`
- Compositional synthesis: `compositional-synthesis.md`
- Recursive decomposition: `decision-trees.md`
- Forecasting-specific model selection: `forecasting-science.md`
