# Validation Checklist

Consolidated requirements for validating models, claims, and methodologies.

## Overfitting Detection

### Indicators
| Indicator | Interpretation |
|-----------|----------------|
| Train fit = Perfect (100%) | Model memorized data; zero scientific value |
| Train >> Test performance | Classic overfitting; gap >20% is severe |
| Test > Train performance | Statistical impossibility; contamination |
| Perfect fit on noisy data | Impossible without error |
| Model complexity >> data size | Capacity to memorize |

### Required Checks
```
[ ] Train performance < 100%?
[ ] Test performance within reasonable gap of train?
[ ] Train > Test (correct direction)?
[ ] Model complexity justified by data size?
[ ] Regularization/early stopping used?
```

## Data Leakage Detection

### Leakage Types
| Type | Description | Detection |
|------|-------------|-----------|
| **Future leakage** | Features use future information | Verify temporal ordering |
| **Target leakage** | Target info encoded in features | Check feature-target independence |
| **Train-test contamination** | Overlap between sets | Verify strict separation |
| **Look-ahead bias** | Decisions based on future info | Audit decision points |

### Required Checks
```
[ ] All features use only past/concurrent information?
[ ] No target information in features?
[ ] Train/test sets strictly separated?
[ ] Preprocessing fit only on training data?
[ ] No information flow from test to train?
```

## Baseline Requirements

### Minimum Baselines
| Baseline Type | Description | When Required |
|---------------|-------------|---------------|
| **Null hypothesis** | "No effect" or "random" | Always |
| **Naive persistence** | "Tomorrow = Today" | Time series |
| **Simple average** | Mean of training set | Regression |
| **Majority class** | Most common class | Classification |
| **Domain simple** | Simplest domain method | Always |

### Required Checks
```
[ ] Null baseline included?
[ ] Naive/simple baseline included?
[ ] Domain-appropriate baseline included?
[ ] Statistical significance vs baseline?
[ ] Improvement over baseline quantified?
```

## Walk-Forward / Temporal Validation

For any sequential data:

### Requirements
| Requirement | Description |
|-------------|-------------|
| Temporal split | No random shuffling of time series |
| Walk-forward | Train on past, test on future |
| Gap period | Buffer between train and test |
| Multiple windows | Not just one split |

### Required Checks
```
[ ] Temporal ordering preserved?
[ ] Train data precedes test data?
[ ] Gap period to prevent adjacent leakage?
[ ] Multiple validation windows tested?
[ ] Results consistent across windows?
```

## Uncertainty Quantification

### Required Measures
| Measure | Purpose |
|---------|---------|
| Confidence intervals | Reliability of predictions |
| Standard errors | Reliability of metrics |
| Bootstrap analysis | Robustness of results |
| Significance tests | vs baseline and null |

### Required Checks
```
[ ] Confidence intervals on predictions?
[ ] Standard errors on metrics?
[ ] Bootstrap or cross-validation robustness?
[ ] Significance test vs baseline?
[ ] Multiple testing correction (if applicable)?
```

### P-Value Guidelines
| Situation | Correction |
|-----------|------------|
| Single comparison | α = 0.05 |
| 2-5 comparisons | Bonferroni (α/n) |
| Many comparisons | False Discovery Rate |
| Data snooping likely | Assume p inflated 10-100x |

## Reproducibility Checklist

### Data Specification
```
[ ] Exact data source specified?
[ ] Date ranges documented?
[ ] Preprocessing steps listed?
[ ] Feature list complete?
[ ] Missing data handling described?
```

### Method Specification
```
[ ] Model architecture detailed?
[ ] Hyperparameters disclosed?
[ ] Training procedure documented?
[ ] Random seeds specified?
[ ] Optimization details included?
```

### Environment Specification
```
[ ] Software versions listed?
[ ] Hardware requirements noted?
[ ] Dependencies documented?
[ ] Code available?
[ ] Data available (or obtainable)?
```

### Reproducibility Test
Could an independent researcher, using only the documentation, reproduce these results?
- If no → results are unverifiable claims, not science.

## Master Validation Checklist

### Quick Validation (5 minutes)
```
[ ] Coherence: Data matches task?
[ ] Coherence: Metrics match task type?
[ ] Coherence: Internal consistency?
[ ] Overfitting: Train ≠ 100%?
[ ] Overfitting: Train > Test?
[ ] Baseline: At least one baseline?
[ ] Domain: Within plausibility bounds?
```

### Standard Validation (30 minutes)
All Quick checks, plus:
```
[ ] Leakage: All types checked?
[ ] Baseline: Multiple baselines?
[ ] Baseline: Statistical significance?
[ ] Temporal: Walk-forward (if applicable)?
[ ] Uncertainty: Confidence intervals?
[ ] Reproducibility: Method complete?
```

### Thorough Validation (2+ hours)
All Standard checks, plus:
```
[ ] Leakage: Audit feature engineering?
[ ] Baseline: Domain-specific baselines?
[ ] Temporal: Multiple windows?
[ ] Uncertainty: Bootstrap analysis?
[ ] Uncertainty: Multiple testing correction?
[ ] Reproducibility: All specs complete?
[ ] Reproducibility: Attempt reproduction?
```

## Validation Scoring

| Score | Criteria | Interpretation |
|-------|----------|----------------|
| **6/6** | All checks pass | Credible |
| **4-5/6** | Minor issues | Skeptical |
| **2-3/6** | Major gaps | Doubtful |
| **0-1/6** | Critical failures | Reject |

For Quick Validation: 7 checks, scale appropriately.
For Thorough Validation: Full audit required, no simple score.
