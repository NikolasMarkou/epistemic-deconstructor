# Coherence Checks (60-Second Filter)

Before detailed validation, verify basic coherence. Failures indicate fundamental confusion, AI-generated content, or incompetence.

## Data-Task Alignment

| Check | What to Verify |
|-------|----------------|
| **Data matches claimed task** | Input data appropriate for claimed output |
| **Methodology matches domain** | Techniques suitable for this type of problem |
| **Temporal alignment** | Time-series for predictions, cross-sectional for classification |
| **Correct resolution** | Data granularity matches prediction granularity |

**Test**: Can you trace: raw data → model input → prediction target → claimed task?

If not → incoherent claim.

## Metric-Task Alignment

| Task Type | Valid Metrics | Invalid Metrics |
|-----------|---------------|-----------------|
| **Continuous prediction** | RMSE, MAE, MAPE, R² | Accuracy, Precision, Recall, F1 |
| **Classification** | Accuracy, Precision, Recall, F1, AUC | RMSE, MAE |
| **Ranking** | NDCG, MAP, MRR | RMSE, Accuracy |
| **Probability estimation** | Brier score, log loss, calibration | Accuracy alone |

**Instant reject**: Metrics that don't match task type indicate fundamental confusion.

## Internal Consistency

| Check | Requirement |
|-------|-------------|
| Results match across sections | Abstract = tables = conclusion |
| Sample sizes consistent | N in methodology = N in results |
| Date/time ranges align | Training/test periods consistent |
| Model count matches | If 5 models described, 5 evaluated |
| Terminology consistent | Same terms used throughout |

## AI-Slop Indicators

| Red Flag | Interpretation |
|----------|----------------|
| Irrelevant domain text | Copy-paste from unrelated source |
| Inconsistent terminology | Switches between unrelated concepts |
| Generic methodology + specific claims | Boilerplate + specific numbers |
| Mismatched figures/captions | Figure shows X, caption describes Y |
| Nonsensical sentences | Grammatically correct, semantically empty |
| Citation-claim mismatch | Cites source A, attributes claim B |
| Abrupt topic changes | Sections don't connect logically |

## Minimum Competence Checks

| Check | Requirement |
|-------|-------------|
| Domain understanding | Correct terminology and concepts |
| Implementable methodology | Steps could actually produce results |
| Data accessibility | Source exists with claimed features |
| Plausible results | Not physically/statistically impossible |
| Logical flow | Claim follows from evidence |

## Extraordinary Claims Filter

Claims requiring extraordinary evidence:

| Claim Type | Threshold | Required Evidence |
|------------|-----------|-------------------|
| **Perfect accuracy** | 99%+ on noisy data | Independent replication + audit |
| **Beats state-of-art by large margin** | >20% improvement | Independent replication |
| **Solves "impossible" problem** | Previously unsolved | Extraordinary scrutiny |
| **Results across many domains** | Universal applicability | Validation in each domain |

**Rule**: Extraordinary claims without extraordinary evidence = suspect.

## Methodological Coherence

| Check | Failure Example |
|-------|-----------------|
| Techniques compatible | Incompatible preprocessing + model |
| Pipeline logical | Each step connects to next |
| Jargon defined | Invented terms without definition |
| Empirical vs position | Claims results but no implementation |
| Cause matches effect | Claimed mechanism could produce claimed result |

## Publication Pattern Red Flags

| Pattern | Interpretation |
|---------|----------------|
| Wide topic range from single source | Expertise inflation |
| High volume, low depth | Quantity over quality |
| Consistent near-perfect claims | Fabrication pattern |
| Systematic omissions | Intentional obfuscation |
| Always positive results | Publication bias or worse |

## The 60-Second Protocol

1. **Scan abstract and conclusion** (15s): Do claims match?
2. **Check metrics vs task type** (10s): Appropriate?
3. **Verify data-task alignment** (15s): Makes sense?
4. **Look for AI-slop indicators** (10s): Signs of generation?
5. **Gut check plausibility** (10s): Within physical limits?

**Result of any failure**: Proceed with extreme caution or reject immediately.

## Coherence Verdict

| Result | Criteria | Action |
|--------|----------|--------|
| **COHERENT** | All checks pass | Proceed to detailed validation |
| **QUESTIONABLE** | 1-2 minor issues | Proceed with documented concerns |
| **INCOHERENT** | Any major failure | REJECT or request clarification |
