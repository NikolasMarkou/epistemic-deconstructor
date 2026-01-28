# Red Flags Catalog

Comprehensive list of indicators for invalid claims, cargo-cult methodology, and scientifically meaningless assertions across any domain.

## Instant Reject Conditions

These indicate fundamental failures that invalidate any claim:

| Condition | Interpretation | Action |
|-----------|----------------|--------|
| **Impossibility** | Results violate known physical/domain limits | REJECT |
| **Memorization** | Perfect fit on training/calibration data | REJECT |
| **Contamination** | Test/validation > training/calibration performance | REJECT |
| **Incoherence** | Internal contradictions in methodology or results | REJECT |
| **Unverifiability** | No way to validate claims independently | REJECT (or flag) |

## Methodology Red Flags

| Red Flag | Interpretation |
|----------|----------------|
| No baseline comparison | Can't assess if method adds value |
| "Novel method" + no comparison | Complexity theater |
| Complex method, minimal validation | Inverted priorities |
| Results metric doesn't match task | Fundamental confusion |
| Random/inappropriate train-test split | Invalid methodology |
| No forecastability/feasibility analysis | Didn't check if task is possible |
| Inappropriate techniques for domain | Square peg, round hole |

## Documentation Red Flags

| Red Flag | Interpretation |
|----------|----------------|
| Vague data source | "We used data" without specifics |
| Missing preprocessing steps | Can't reproduce |
| Hyperparameters not disclosed | Can't reproduce |
| "Standard" anything without definition | Undefined = unverifiable |
| "Code available upon request" | Usually never available |
| "Proprietary data" | Results unverifiable |
| Missing split methodology | Can't assess validity |

## Results Red Flags

| Red Flag | Interpretation |
|----------|----------------|
| Perfect fit on training data | Memorization, not learning |
| Test > Train performance | Statistical impossibility |
| Too-good-to-be-true metrics | Likely leakage or fabrication |
| No confidence intervals | Can't assess reliability |
| No significance test vs baseline | May not beat random |
| Suspiciously round numbers | Possible fabrication |
| Cherry-picked evaluation window | Hiding poor performance |

## Claims Red Flags

| Red Flag | Interpretation |
|----------|----------------|
| Extraordinary claims, ordinary evidence | Extraordinary claims require extraordinary evidence |
| Beats well-funded competitors easily | Implausible without explanation |
| 99%+ on noisy/difficult tasks | Physically impossible |
| Results across wildly different domains | Jack of all trades, master of none |
| "Breakthrough" without independent replication | Unvalidated marketing |
| Vague success metrics | Hiding behind ambiguity |

## Tool Worship (Cargo-Cult) Red Flags

| Red Flag | Interpretation |
|----------|----------------|
| "We used [fancy tool]" as main contribution | Tool doesn't create validity |
| 90% architecture description, 10% validation | Inverted priorities |
| Beats baseline on cherry-picked metrics | Doesn't beat on decision-relevant metrics |
| No analysis of whether task is feasible | Assumed prediction is possible |
| Complex pipeline, simple validation | Complexity hiding invalidity |

## Publication/Presentation Red Flags

| Pattern | Interpretation |
|---------|----------------|
| Cross-domain "breakthroughs" from same source | Spam pattern |
| Consistent near-perfect results across noisy domains | Fabrication pattern |
| Systematic detail omission | Intentional obfuscation |
| Position papers sold as empirical | "We propose..." with no implementation |
| Undefined jargon | Made-up terms = unverifiable |
| Irrelevant text inserted | AI-slop or copy-paste error |
| Figures don't match captions | Assembly error or fabrication |

## Statistical Red Flags

| Red Flag | Problem |
|----------|---------|
| No confidence intervals | Can't assess reliability |
| No significance test vs baseline | May not beat random |
| No multiple testing correction | Data snooped |
| p = 0.049 on primary result | Suspiciously borderline |
| "Statistically significant" with tiny sample | Likely overfit |
| In-sample significance only | Meaningless for prediction |

## The Meta-Rule

**If a claim triggers 3+ red flags from different categories, treat the entire work as suspect regardless of any individual metric looking reasonable.**

Invalid methodology often gets one thing right while failing systematically elsewhere. The accumulation of flags across different categories indicates systemic problems, not isolated issues.

## Red Flag Severity Levels

| Level | Count | Interpretation |
|-------|-------|----------------|
| **Clean** | 0-1 flags | Proceed with normal scrutiny |
| **Caution** | 2-3 flags | Increased skepticism, verify key claims |
| **Suspect** | 4-5 flags | Treat as unreliable until proven otherwise |
| **Invalid** | 6+ flags OR any instant reject | Do not rely on this work |

## Using This Catalog

1. **Scan systematically**: Check each category, don't just look for obvious failures
2. **Count across categories**: Accumulation matters more than individual flags
3. **Document findings**: Note which specific flags triggered
4. **Apply consistently**: Same standards for claims you like and dislike
5. **Update priors**: Red flags should lower confidence, not just raise questions
