# Profile Synthesis Reference

This reference provides methods for combining individual trait assessments into a unified psychological model. Used primarily in PSYCH tier Phases 4-P (Motive Synthesis) and 5-P (Validation & Prediction).

---

## Synthesis Principles

### 1. Confidence Propagation

Composed profile confidence is limited by weakest component.

```
Confidence_profile = min(Confidence_trait1, Confidence_trait2, ...) × integration_factor

Integration factors:
  Consistent traits: 0.95      (reinforcing pattern)
  Independent traits: 0.90     (no interaction)
  Contradictory traits: 0.70   (tension detected)
  Insufficient data: 0.50      (speculation)
```

### 2. Trait Interaction

Traits don't exist in isolation. Combinations create emergent patterns.

### 3. Context Dependency

All profiles are context-dependent. A profile valid in professional settings may not predict personal behavior.

### 4. Temporal Validity

Profiles decay. Reassess after significant time or life events.

---

## Trait Composition Mathematics

### OCEAN Integration

Each trait is assessed as Low (L), Medium (M), or High (H), with confidence levels.

```
Trait_score = Σ(evidence_weight × evidence_strength) / Σ(evidence_weight)

Where:
  evidence_strength: -1 (strong low), 0 (neutral), +1 (strong high)
  evidence_weight: 0.3 (inference), 0.6 (behavioral), 1.0 (direct/stressed)
```

### Dark Triad Composite

```
DT_risk = (N_score × 0.3) + (M_score × 0.4) + (P_score × 0.3)

Where each score is 0 (low) to 1 (high)

Risk levels:
  DT_risk < 0.3: Low concern
  0.3 ≤ DT_risk < 0.6: Moderate concern
  DT_risk ≥ 0.6: High concern
```

### MICE Priority Vector

```
MICE_vector = [Money_score, Ideology_score, Coercion_score, Ego_score]

Normalize: MICE_norm = MICE_vector / sum(MICE_vector)

Primary motive = argmax(MICE_norm)
Secondary motive = second_highest(MICE_norm)
```

---

## Composition Rules

### Rule 1: High-High Amplification

When two correlated traits are both high, effects amplify.

| Trait 1 | Trait 2 | Amplified Effect |
|---------|---------|------------------|
| High N + High E | | Emotional volatility + social expression = dramatic outbursts |
| High N + Low A | | Anxiety + antagonism = hostile defensiveness |
| Low A + High M | | Antagonism + strategy = calculated aggression |
| High N + High P | | Emotional reactivity + impulsivity = dangerous volatility |
| High O + High C | | Creativity + discipline = productive innovation |

### Rule 2: Trait Tension

Contradictory evidence may indicate:

| Pattern | Interpretation |
|---------|----------------|
| **Context switching** | Different behavior in different settings (authentic) |
| **Masking** | Deliberate presentation management (deceptive or adaptive) |
| **Measurement error** | Insufficient or noisy data |
| **Evolution** | Trait change in progress |
| **Pathology** | Internal conflict indicating dysfunction |

### Rule 3: Dominant Trait Override

Under stress, dominant traits override secondary ones.

```
Stress_behavior = Primary_trait × stress_multiplier + Secondary_trait × (1 - stress_multiplier)

Where stress_multiplier increases with stress intensity (0.6 → 0.9)
```

---

## Profile Template

### Basic Profile Statement

```
[Subject Name/ID]
Assessment Date: ___________
Context: [Professional/Personal/Adversarial]

OCEAN Profile: O[_] C[_] E[_] A[_] N[_]
Dark Triad: Narcissism[_] Mach[_] Psych[_]
Primary MICE: ___________
Secondary MICE: ___________

Confidence: [Low/Medium/High]
Data Quality: [Limited/Adequate/Extensive]
```

### Extended Profile Statement

```
# Psychological Profile: [Subject]

## Executive Summary
One paragraph describing core psychological structure and key behavioral predictions.

## Trait Assessment

### Big Five (OCEAN)
| Trait | Level | Confidence | Key Evidence |
|-------|-------|------------|--------------|
| Openness | | | |
| Conscientiousness | | | |
| Extraversion | | | |
| Agreeableness | | | |
| Neuroticism | | | |

### Dark Triad
| Trait | Level | Confidence | Key Evidence |
|-------|-------|------------|--------------|
| Narcissism | | | |
| Machiavellianism | | | |
| Psychopathy | | | |

### Motivation (MICE)
| Driver | Score | Confidence | Key Evidence |
|--------|-------|------------|--------------|
| Money | | | |
| Ideology | | | |
| Coercion | | | |
| Ego | | | |

## Behavioral Predictions

### Under Normal Conditions
- Prediction 1: [behavior] because [trait combination]
- Prediction 2: ...

### Under Stress
- Prediction 1: [behavior] because [dominant trait]
- Prediction 2: ...

### Vulnerabilities
- [Leverage point 1]: [how to use ethically]
- [Leverage point 2]: ...

### Resistances
- [What won't work and why]

## Interaction Strategy

### Do
- [Recommended approach 1]
- [Recommended approach 2]

### Don't
- [Avoid this because]
- [Avoid this because]

## Confidence Assessment

Overall Confidence: [Low/Medium/High]

Limiting Factors:
- [Data gap 1]
- [Uncertainty 1]

Required Validation:
- [ ] Test prediction 1 with [scenario]
- [ ] Confirm [trait] with [probe]

## Profile Validity

Valid For: [Context description]
Not Valid For: [Context where profile may not apply]
Reassess After: [Time period or trigger event]
```

---

## Common Archetypes

Pre-composed profiles for rapid classification.

### The Executive

```
OCEAN: O[M] C[H] E[H] A[L] N[L]
DT: N[M] M[M] P[L]
MICE: Ego > Money > Ideology

Characteristics:
- Confident, decisive
- Results-oriented
- Status-conscious
- Low patience for process
- Expects deference

Strategy:
- Be direct and brief
- Respect their time
- Quantify everything
- Let them make final decision
- Frame as business opportunity
```

### The Expert

```
OCEAN: O[H] C[H] E[L] A[M] N[M]
DT: N[L] M[L] P[L]
MICE: Ego (expertise type) > Ideology > Money

Characteristics:
- Deep knowledge, narrow focus
- Prefers substance over style
- May miss social cues
- Values accuracy
- Sensitive to expertise challenges

Strategy:
- Respect their knowledge
- Ask for their expert opinion
- Tolerate detail
- Don't oversimplify
- Be accurate
```

### The Politician

```
OCEAN: O[M] C[M] E[H] A[H*] N[L]
DT: N[M] M[H] P[L]
*Performed agreeableness, not authentic
MICE: Ego (status) > Money > Ideology

Characteristics:
- Charming, persuasive
- Coalition builder
- Long-term strategist
- Says what you want to hear
- Plays both sides

Strategy:
- Verify everything independently
- Document commitments
- Assume strategic positioning
- Build relationship but don't trust blindly
- Watch for alliance shifts
```

### The True Believer

```
OCEAN: O[L] C[H] E[M] A[L to in-group, L to out-group] N[M]
DT: N[L] M[L] P[L]
MICE: Ideology >> all others

Characteristics:
- Deeply committed to cause
- Black/white thinking
- Willing to sacrifice
- Suspicious of outsiders
- Resistant to conflicting information

Strategy:
- Find shared values
- Frame in their ideological terms
- Don't challenge core beliefs directly
- Show respect for their dedication
- Be patient
```

### The Manipulator

```
OCEAN: O[M] C[M] E[H] A[L] N[L]
DT: N[H] M[H] P[M]
MICE: Ego > Money > Power

Characteristics:
- Charming initially
- Information asymmetry seeking
- Tests boundaries
- Multiple personas
- Gaslighting tendency

Strategy:
- Verify everything
- Trust actions, not words
- Document extensively
- Build in checkpoints
- Have exit strategy
- Don't be alone with them
```

### The Anxious Avoider

```
OCEAN: O[L] C[L] E[L] A[H] N[H]
DT: N[L] M[L] P[L]
MICE: Coercion (fear) > Ideology (safety)

Characteristics:
- Risk averse
- Conflict avoidant
- Seeks reassurance
- May freeze under pressure
- Over-commits to please, under-delivers

Strategy:
- Provide safety and structure
- Reduce uncertainty
- Be patient
- Don't pressure
- Provide clear expectations
- Follow up without blame
```

---

## Contradiction Handling

### When Evidence Conflicts

1. **Check for context-switching**
   - Is behavior consistent within context?
   - Different roles may show different traits

2. **Check for temporal change**
   - Recent vs. historical evidence?
   - Trigger events that might cause shift?

3. **Check for data quality**
   - Direct observation vs. hearsay?
   - Stressed vs. relaxed conditions?

4. **Weight recent + stressed + direct evidence higher**

5. **Document uncertainty explicitly**
   - "High O in professional context, Low O in personal context"
   - "Possible High M, conflicting data, requires validation"

### Contradiction Resolution Matrix

| Type | Resolution | Action |
|------|------------|--------|
| Context-dependent | Maintain separate context profiles | Note switching pattern |
| Temporal change | Update to recent, note trajectory | Track evolution |
| Data quality | Weight better data higher | Seek better evidence |
| Masking detected | Note both presented and inferred | Test with stress/probe |
| Unresolvable | Document both possibilities | Assign low confidence |

---

## Validation Protocol

### Internal Validation

```
[ ] Traits are consistent with each other (no impossible combinations)
[ ] MICE aligns with observed behavior
[ ] Dark Triad assessment explains manipulation patterns (if any)
[ ] Profile explains historical behavior (backward validation)
```

### Predictive Validation

```
[ ] Generate 3 specific, testable predictions
[ ] Define test scenarios
[ ] Track prediction accuracy
[ ] Update profile based on results
```

### External Validation

```
[ ] Third-party observations consistent?
[ ] Independent information confirms/contradicts?
[ ] Subject's self-description vs. profile (if available)
```

---

## Profile Maintenance

### Update Triggers

- Significant time passage (>6 months)
- Life event (job change, relationship change, crisis)
- Prediction failure
- New significant evidence
- Context change

### Decay Function

```
Confidence_current = Confidence_original × decay_factor^(months/12)

Where decay_factor = 0.8 (stable individual) to 0.5 (volatile individual)
```

---

## Output Format

### Minimal (for quick reference)

```
[Subject]: O[H] C[M] E[L] A[L] N[H] | DT: N[L] M[H] P[L] | MICE: Ideology | Conf: Med
Strategy: Verify commitments, frame as principled, document
```

### Standard (for working profile)

```
[Subject] Profile
================
OCEAN: O[H] C[M] E[L] A[L] N[H]
DT: N[L] M[H] P[L]
MICE: Ideology (primary), Ego (secondary)
Confidence: Medium
Context: Professional negotiation

Key Predictions:
1. Will prioritize principle over relationship
2. Uses strategic ambiguity - verify specifics
3. Under stress, becomes more anxious and rigid

Strategy:
DO: Frame in value terms, be precise, document
DON'T: Appeal to personal relationship, rush
```

---

## Cross-References

- Trait definitions: `references/archetype-mapping.md`
- Elicitation for evidence: `references/elicitation-techniques.md`
- Linguistic indicators: `references/linguistic-markers.md`
- Motive details: `references/motive-analysis.md`
- Analysis bias: `references/cognitive-traps.md`
