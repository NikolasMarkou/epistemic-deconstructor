# PSYCH Tier: Psychological Profiling Protocol

For analyzing human behavior, personas, and psychological profiles. Outputs behavioral predictions, negotiation strategies, and deception detection.

## Table of Contents

- [Ethical Constraints](#ethical-constraints)
- [State Block Format](#psych-state-block-format)
- [Phase 0-P: Context & Frame](#phase-0-p-context--frame)
  - [High-Profile Subject Protocol (HPSP)](#0-px-pre-analysis-reality-check-hpsp)
- [Phase 1-P: Baseline Calibration](#phase-1-p-baseline-calibration)
- [Phase 2-P: Stimulus-Response Mapping](#phase-2-p-stimulus-response-mapping)
- [Phase 3-P: Structural Identification](#phase-3-p-structural-identification)
- [Phase 4-P: Motive Synthesis](#phase-4-p-motive-synthesis)
  - [Financial Entanglement Analysis (HPSP)](#4-px-financial-entanglement-analysis-hpsp)
- [Phase 5-P: Validation & Prediction](#phase-5-p-validation--prediction)
- [Decision Trees](#psych-tier-decision-trees)
- [Cross-Domain Integration](#cross-domain-integration)
- [Psychological Axioms](#psychological-axioms)

---

## Ethical Constraints

**REQUIRED** - Review before every analysis:

- **No Clinical Diagnosis**: Do not diagnose disorders (e.g., "Bipolar", "NPD"). Use observable traits (e.g., "High emotional volatility", "Grandiose presentation").
- **Cultural Calibration**: Adjust assessments for cultural norms. Directness, emotional expression, and self-promotion vary by culture.
- **Consent Awareness**: Document when subject is unaware of analysis.
- **Defensive Use**: Primary use is defense/negotiation, not manipulation or exploitation.

---

## PSYCH State Block Format

```
[STATE: Phase X-P | Tier: PSYCH | Archetype: Y | Rapport: Low/Med/High | Stress: Low/Med/High]
```

Example:
```
[STATE: Phase 2-P | Tier: PSYCH | Archetype: High-N/Low-A | Rapport: Med | Stress: Low]
```

Extended format with High-Profile Subject Protocol (HPSP):
```
[STATE: Phase X-P | Tier: PSYCH | PR-Discount: Applied | Financial-Entanglement: Y/N | Mask-Confidence: L/M/H]
```

---

## Phase 0-P: Context & Frame
*Budget: 10% | Output: Analysis Plan, Initial Hypotheses*

### 0-P.1 Relationship Assessment

| Dimension | Options |
|-----------|---------|
| **Power Dynamic** | Upward (they have power) / Downward (you have power) / Peer |
| **Access Channels** | Text only / Voice / Video / In-person / Proxy/Third-party |
| **Stakes** | Low / Medium / High / Critical |
| **Adversarial Status** | Cooperative / Neutral / Competitive / Adversarial |
| **Consent Status** | Aware / Unaware / Implicit |

### 0-P.2 Initial Hypothesis Seeding

Always maintain 3+ hypotheses about subject's psychological structure:
```
H1-P: [Most likely archetype/profile]
H2-P: [Alternative archetype/profile]
H3-P: [Deceptive/Mask hypothesis - presented persona differs from actual]
```

### 0-P.3 Information Inventory

- What data is available? (text logs, video, audio, documents, third-party reports)
- What's the quality? (direct observation vs. hearsay)
- What's missing? (gaps in observability)

### 0-P.4 Analysis Objectives

| Objective | Focus |
|-----------|-------|
| **Predict behavior** | Trait identification, motive mapping |
| **Detect deception** | Baseline establishment, stress testing |
| **Prepare negotiation** | Leverage points, resistance factors |
| **Build rapport** | Values alignment, communication style |
| **Assess reliability** | Trustworthiness indicators |

### 0-P.5 Stop Condition

Phase 0-P complete when:
- [ ] Relationship dynamics documented
- [ ] ≥3 hypotheses generated (including deceptive-mask)
- [ ] Available data inventoried
- [ ] Analysis objective specified
- [ ] Ethical constraints acknowledged

---

### 0-P.X Pre-Analysis Reality Check (HPSP)

**Trigger**: Subject has PR budget >$1M/year, media training, or public persona management apparatus.

**ASSUME PRESENTED PERSONA IS ENGINEERED PRODUCT.**

Before any baseline calibration:

1. **Identify PR infrastructure**: Publicists, communications staff, ghostwriters, media coaches
2. **Map narrative control assets**: Owned media (blogs, foundations), friendly journalists, book deals
3. **Treat all public statements as curated output** — not raw behavioral data

| Data Type | Trust Level | Use For |
|-----------|-------------|---------|
| Prepared interviews | LOW | Studying *intended* narrative |
| Spontaneous reactions (rare) | MEDIUM | Deviation detection |
| Resource allocation (money, time) | HIGH | Revealed preferences |
| Legal/regulatory filings | HIGH | Structural truth |
| Adversary statements | MEDIUM | Counter-narrative data |

**Axiom Addition**:
> **Persona ≠ Person**: For resourced subjects, public behavior is marketing. Analyze the *product*, not the *person*, unless accessing unscripted data.

#### Data Trust Hierarchy Quick Reference

```
HIGHEST TRUST
    │
    ├── Money movements (audited)
    ├── Legal filings / SEC disclosures
    ├── Time allocation (calendars, travel)
    ├── Who they hire / fire
    ├── Adversary accusations (triangulate)
    ├── Off-script moments (leaks, hot mics)
    ├── Prepared interviews
    ├── Authored books / blogs
    │
LOWEST TRUST
    └── PR statements / press releases
```

---

## Phase 1-P: Baseline Calibration
*Budget: 20% | Output: Baseline Profile*

Establish "normal" for this subject before looking for deviations.

### 1-P.1 Linguistic Baseline

| Metric | Measurement | Record |
|--------|-------------|--------|
| **Sentence length** | Avg words per sentence | ___ |
| **Response latency** | Typical time to respond | ___ |
| **Vocabulary level** | Complexity indicators | Low/Med/High |
| **Punctuation/emoji** | Expressive markers | ___ |
| **Hedge frequency** | Uncertainty markers | Low/Med/High |
| **Pronoun ratios** | I / We / They distribution | ___/___ /___% |

**Reference**: See `references/linguistic-markers.md` for detailed measurement.

### 1-P.2 Emotional Baseline

| Dimension | Default State |
|-----------|---------------|
| **Affective tone** | Positive / Neutral / Negative |
| **Emotional range** | Narrow / Moderate / Wide |
| **Reactivity** | Low (stable) / Medium / High (volatile) |
| **Default energy** | Low / Medium / High |

### 1-P.3 Timing Patterns

| Pattern | Observation |
|---------|-------------|
| **Response speed** | Immediate / Moderate / Delayed |
| **Interruption tendency** | Never / Sometimes / Often |
| **Topic transition** | Abrupt / Smooth / Hesitant |
| **Engagement duration** | Brief / Moderate / Extended |

### 1-P.4 Idiosyncrasy Index

Capture unique speech patterns, tics, and phrases when subject is at ease:
- Catchphrases
- Filler words
- Topic preferences
- Avoidance patterns
- Humor style

**Key principle**: Baseline is God. Only deviation from baseline is significant.

### 1-P.5 Stop Condition

Phase 1-P complete when:
- [ ] Minimum 5-10 exchanges observed at ease (non-stressful)
- [ ] Linguistic metrics documented
- [ ] Emotional baseline established
- [ ] Timing patterns recorded
- [ ] Idiosyncrasies noted

---

## Phase 2-P: Stimulus-Response Mapping
*Budget: 25% | Output: Deviation Database, Trigger Map*

Apply controlled stimuli and observe deviations from baseline.

### 2-P.1 Elicitation Probes

| Technique | Purpose | Deviation to Watch |
|-----------|---------|-------------------|
| **The Void (Silence)** | Test anxiety tolerance | Fill rate, content of fill |
| **The Challenge** | Test ego/competitiveness | Defensive vs. acceptance |
| **The Misstatement** | Test correction impulse | Speed and intensity of correction |
| **The Flattery** | Test narcissism | Acceptance vs. deflection |
| **The Hypothetical** | Access hidden views | Engagement depth |
| **Assumed Knowledge** | Test information gaps | Correction vs. acceptance |

**Reference**: See `references/elicitation-techniques.md` for detailed protocols.

### 2-P.2 Stress Response Capture

Under increased pressure, observe:

| Marker | Baseline | Under Stress | Delta |
|--------|----------|--------------|-------|
| Sentence length | ___ | ___ | ___ |
| Response latency | ___ | ___ | ___ |
| Hedge frequency | ___ | ___ | ___ |
| Pronoun shift | ___ | ___ | ___ |
| Topic avoidance | ___ | ___ | ___ |

### 2-P.3 Deception Marker Scan

Look for clusters of:
- Distancing language (passive voice, pronoun drop)
- Specificity changes (detailed → vague or vague → over-detailed)
- Convincing statements ("Honestly...", "I swear...")
- Equivocation (technically true but misleading)
- Response pattern changes

**Warning**: No single marker is diagnostic. Require pattern + context + baseline deviation.

### 2-P.4 Trigger Documentation

| Trigger Topic | Response Type | Intensity | Interpretation |
|---------------|---------------|-----------|----------------|
| [topic] | [defensive/open/avoidant] | [low/med/high] | [hypothesis] |

### 2-P.5 Stop Condition

Phase 2-P complete when:
- [ ] ≥3 elicitation probes applied
- [ ] Stress response observed (or opportunity documented)
- [ ] Deviation patterns documented
- [ ] Trigger topics identified
- [ ] Deception markers assessed (present/absent/inconclusive)

---

## Phase 3-P: Structural Identification
*Budget: 20% | Output: Trait Profile*

Map subject onto psychological frameworks.

### 3-P.1 Big Five (OCEAN) Assessment

| Trait | Evidence | Level | Confidence |
|-------|----------|-------|------------|
| **Openness** | | L / M / H | Low/Med/High |
| **Conscientiousness** | | L / M / H | Low/Med/High |
| **Extraversion** | | L / M / H | Low/Med/High |
| **Agreeableness** | | L / M / H | Low/Med/High |
| **Neuroticism** | | L / M / H | Low/Med/High |

**Reference**: See `references/archetype-mapping.md` for trait indicators.

### 3-P.2 Dark Triad Assessment (ALWAYS ASSESS ALL THREE)

| Trait | Indicators Present | Level | Confidence |
|-------|-------------------|-------|------------|
| **Narcissism** | | L / M / H | Low/Med/High |
| **Machiavellianism** | | L / M / H | Low/Med/High |
| **Psychopathy** | | L / M / H | Low/Med/High |

**DT Risk Score**: Calculate composite (see `references/archetype-mapping.md`).

### 3-P.3 Cognitive Distortion Scan

Check for persistent patterns:

| Distortion | Present? | Example |
|------------|----------|---------|
| **Black/white thinking** | Y/N | |
| **Catastrophizing** | Y/N | |
| **Victim stance** | Y/N | |
| **Mind reading** | Y/N | |
| **Personalization** | Y/N | |
| **Emotional reasoning** | Y/N | |

### 3-P.4 Stop Condition

Phase 3-P complete when:
- [ ] OCEAN profile mapped with evidence
- [ ] All three Dark Triad traits assessed
- [ ] Cognitive distortions scanned
- [ ] Confidence levels assigned to each trait

---

## Phase 4-P: Motive Synthesis
*Budget: 15% | Output: Unified Model, Drive Matrix*

Integrate traits and identify motivational drivers.

### 4-P.1 MICE Framework Application

| Driver | Evidence | Score (0-10) | Confidence |
|--------|----------|--------------|------------|
| **Money** | | | |
| **Ideology** | | | |
| **Coercion** | | | |
| **Ego** | | | |

**Primary Driver**: _____________
**Secondary Driver**: _____________

**Reference**: See `references/motive-analysis.md` for detailed framework.

### 4-P.2 Drive Matrix Construction

```
Subject: _______________

Primary: [Driver] - [Key evidence]
Secondary: [Driver] - [Key evidence]
Conflict points: [Where motives may conflict]
Leverage opportunities: [What they want/fear]
Resistance factors: [What they won't compromise on]
```

### 4-P.3 Archetype Classification

Based on OCEAN + Dark Triad + MICE, classify:

| Archetype | Match | Confidence |
|-----------|-------|------------|
| [Archetype 1] | High/Medium/Low | |
| [Archetype 2] | High/Medium/Low | |

**Reference**: See `references/profile-synthesis.md` for archetype catalog.

---

### 4-P.X Financial Entanglement Analysis (HPSP)

**Trigger**: Subject is wealthy or has institutional backing. **REQUIRED for high-profile subjects.**

Before finalizing MICE drivers, map money flows:

1. **Direct benefit**: Does subject receive money from entities they're defending?
2. **Leverage structures**: Does subject's money *multiply* via co-funding arrangements?
3. **Influence infrastructure**: Would policy change destroy subject's operating model?
4. **Network dependency**: Who loses funding if subject's position fails?

| Question | If YES |
|----------|--------|
| Subject co-funds with government/institutions? | Map leverage ratios |
| Subject's orgs receive matching funds? | Calculate true multiplier |
| Subject sets terms that others must follow? | Identify control mechanisms |
| Policy change would break funding model? | **Re-classify defense as self-interest** |

**Key Principle**:
> **Altruism and self-interest are not mutually exclusive.** A subject can genuinely believe in their cause AND be defending financial/influence infrastructure simultaneously. Map both; don't choose one.

---

### 4-P.4 Unified Model Statement

```
[Subject] is characterized by [OCEAN summary], with [Dark Triad assessment].
Primary driver is [MICE primary] as evidenced by [key evidence].
Under stress, expect [behavioral prediction].
Key vulnerabilities: [leverage points].
Key resistances: [what won't work].
```

### 4-P.5 Stop Condition

Phase 4-P complete when:
- [ ] MICE drivers ranked with evidence
- [ ] Drive matrix constructed
- [ ] Archetype classified
- [ ] Unified model statement written

---

## Phase 5-P: Validation & Prediction
*Budget: 10% | Output: Validated Profile, Behavioral Predictions*

Test the model and generate actionable predictions.

### 5-P.1 Internal Consistency Check

| Check | Pass/Fail |
|-------|-----------|
| Traits consistent with observed behavior? | |
| MICE aligns with stated and inferred goals? | |
| Archetype explains response patterns? | |
| No major contradictions unresolved? | |

### 5-P.2 Predictive Hypothesis Generation

Generate specific, testable predictions:

| Prediction | Scenario | Expected Behavior | Confidence |
|------------|----------|-------------------|------------|
| P1 | If [situation] | Then [behavior] | Low/Med/High |
| P2 | If [situation] | Then [behavior] | Low/Med/High |
| P3 | If [situation] | Then [behavior] | Low/Med/High |

### 5-P.3 Adversarial Awareness Check

| Question | Assessment |
|----------|------------|
| Is subject aware of being analyzed? | Y/N/Unknown |
| Is subject capable of counter-profiling? | Y/N/Unknown |
| Is presented persona likely a mask? | Y/N/Unknown |
| Have we accounted for intentional deception? | Y/N |

### 5-P.4 Confidence Assessment

```
Overall Profile Confidence: [Low/Medium/High]

Confidence limited by:
- [ ] Limited observation time
- [ ] Single context only
- [ ] No stress observation
- [ ] Third-party data only
- [ ] Possible counter-profiling

Required for higher confidence:
- [ ] [Specific gap to fill]
- [ ] [Specific gap to fill]
```

### 5-P.5 Interaction Strategy Output

```
## Interaction Strategy: [Subject]

### Do
- [Recommended approach 1]
- [Recommended approach 2]
- [Recommended approach 3]

### Don't
- [Avoid this because]
- [Avoid this because]

### Watch For
- [Behavioral signal indicating X]
- [Behavioral signal indicating Y]

### Exit Conditions
- [When to disengage]
- [Red lines]
```

### 5-P.6 Stop Condition

Phase 5-P complete when:
- [ ] Internal consistency verified
- [ ] ≥3 testable predictions generated
- [ ] Adversarial awareness assessed
- [ ] Confidence level assigned with limiting factors
- [ ] Interaction strategy documented

**Additional for HPSP (High-Profile Subjects):**
- [ ] PR apparatus identified and discounted
- [ ] Financial flows mapped (not just stated motives)
- [ ] Leverage structures documented
- [ ] "Cui bono" analysis completed for stated positions
- [ ] Mask vs. Face confidence explicitly rated

**Tool**: Use `scripts/belief_tracker.py` for trait confidence tracking.

---

## PSYCH Tier Decision Trees

### "Which Profile Depth?"

```
START
├─ Single interaction needed?
│  └─ Yes → Quick archetype + MICE (30min)
├─ Ongoing relationship?
│  └─ Yes → Full PSYCH tier (2-4h)
├─ High stakes negotiation?
│  └─ Yes → Full PSYCH + Dark Triad deep dive
└─ Adversarial context?
   └─ Yes → Full PSYCH + Deception focus
```

### "Mask Detected?"

```
START
├─ Presentation inconsistent across contexts?
│  └─ Yes → Likely masking
├─ Too-smooth, too-perfect presentation?
│  └─ Yes → Possible performance
├─ Stress reveals different personality?
│  └─ Yes → Baseline = mask, stress = face
└─ Third-party reports contradict subject?
   └─ Yes → Investigate discrepancy
```

---

## Cross-Domain Integration

The PSYCH tier can be combined with system analysis tiers:

**Nested Analysis**: When analyzing a system, invoke PSYCH to profile the adversary or designer.
```
[STATE: Phase 2 | Tier: STANDARD | Sub: Phase 1-P | Subject: Adversary]
```

**Designer Profiling**: Understanding the designer's psychology can explain system design choices.

**Adversary Modeling**: Predict adversarial responses based on psychological profile.

---

## Psychological Axioms

| Axiom | Implication | Checkpoint |
|-------|-------------|------------|
| Rational Actor Fallacy | Humans are predictably irrational | Are you modeling utility or emotion? |
| Verbal Map ≠ Territory | What they say ≠ What they believe | Are you analyzing content or syntax? |
| Leakage | High cognitive load forces truth leakage | Looking for slips, micro-expressions, grammar breaks? |
| Baseline is God | Only deviation from baseline is significant | Have you established "normal" for this subject? |
| Projection Trap | You assume they think like you | Have you generated 3 unlike interpretations? |
| Context Dependency | Behavior is context-bound | Have you observed across multiple contexts? |
| Motivated Reasoning | People believe what serves them | What does this belief give them? |
| Mask vs. Face | Presented self ≠ Actual self | Have you tested under stress? |

---

## Cross-References

- Archetype definitions: `references/archetype-mapping.md`
- Elicitation techniques: `references/elicitation-techniques.md`
- Linguistic markers: `references/linguistic-markers.md`
- Motive analysis: `references/motive-analysis.md`
- Profile synthesis: `references/profile-synthesis.md`
- Cognitive traps: `references/cognitive-traps.md`
