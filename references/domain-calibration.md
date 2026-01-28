# Domain Calibration

Understand what's actually achievable in a domain before evaluating claims.

## Why Calibration Matters

Without domain-appropriate baselines, you cannot distinguish:
- Genuine improvement from noise
- Skill from luck
- Valid results from methodological artifacts

**Key principle**: Before accepting a claim, know what's plausible, excellent, and suspicious for that domain.

## How to Establish Plausibility Bounds

### Step 1: Identify Domain Experts and Benchmarks
- What do established practitioners achieve?
- What are published benchmark results?
- What theoretical limits exist?

### Step 2: Understand Irreducible Uncertainty
- How noisy is the underlying phenomenon?
- What's the theoretical best-case?
- What portion is inherently unpredictable?

### Step 3: Build Calibration Table
For each key metric, establish:
- **Suspicious**: Results so good they indicate error/fraud
- **Plausible**: Typical range for competent work
- **Excellent**: Top-tier but achievable results

### Step 4: Apply Consistently
- Compare claimed results to calibration table
- Flag anything in "suspicious" range
- Demand extraordinary evidence for extraordinary claims

## Calibration Table Template

| Metric | Suspicious | Plausible | Excellent | Notes |
|--------|------------|-----------|-----------|-------|
| [Metric 1] | > X | Y - Z | Z - W | [Context] |
| [Metric 2] | > A | B - C | C - D | [Context] |

## Example Domain: ML/AI Classification

| Metric | Suspicious | Plausible | Excellent |
|--------|------------|-----------|-----------|
| Accuracy (balanced) | >99% | 70-90% | 90-98% |
| Accuracy (imbalanced) | >95% | 60-85% | 85-95% |
| F1 Score | >0.99 | 0.70-0.90 | 0.90-0.98 |
| AUC-ROC | >0.999 | 0.70-0.90 | 0.90-0.98 |

**Notes**:
- Balanced accuracy near 99% on non-trivial tasks indicates overfitting or leakage
- Imbalanced data inflates accuracy; use F1 or AUC instead
- Perfect scores (1.0) are instant red flags

## Example Domain: Financial Prediction

| Metric | Suspicious | Plausible | Excellent |
|--------|------------|-----------|-----------|
| Directional accuracy (daily) | >65% | 52-58% | 58-62% |
| Sharpe ratio (after costs) | >3.0 | 0.5-1.5 | 1.5-2.5 |
| R² on returns | >0.3 | 0.01-0.05 | 0.05-0.15 |
| R² on price levels (test) | >0.95 | N/A | N/A |

**Notes**:
- Renaissance Medallion (best known fund) achieves ~66% directional before costs
- R² on price levels is meaningless due to non-stationarity
- Claims beating these bounds without extraordinary evidence = reject

## Example Domain: Engineering/Physical Systems

| Metric | Suspicious | Plausible | Excellent |
|--------|------------|-----------|-----------|
| Model fit R² (in-sample) | =1.0 | 0.85-0.95 | 0.95-0.99 |
| Model fit R² (out-sample) | >0.99 | 0.80-0.90 | 0.90-0.98 |
| Prediction error (MAPE) | <1% | 5-15% | 2-5% |
| Generalization gap | <0% | 5-15% | 2-5% |

**Notes**:
- Physical systems with known physics can achieve high R²
- Perfect in-sample fit still indicates overfitting risk
- Generalization gap (train - test) should be positive and small

## Example Domain: Medical/Biological

| Metric | Suspicious | Plausible | Excellent |
|--------|------------|-----------|-----------|
| Diagnostic sensitivity | >99% | 70-85% | 85-95% |
| Diagnostic specificity | >99% | 70-85% | 85-95% |
| AUC-ROC | >0.99 | 0.70-0.85 | 0.85-0.95 |
| Effect size (Cohen's d) | >2.0 | 0.2-0.8 | 0.8-1.5 |

**Notes**:
- Biological variability limits achievable accuracy
- Small effect sizes are common and valid
- Claims of perfect discrimination require extraordinary scrutiny

## The "Too Good" Rule

| If claimed... | Then require... |
|---------------|-----------------|
| Results in "suspicious" range | Independent replication + audit |
| Results at domain ceiling | Extraordinary evidence |
| Results exceeding theoretical limits | Instant reject |
| Results dramatically better than state-of-art | Exceptional scrutiny |

## Building Domain-Specific Calibration

When entering a new domain:

1. **Research baseline**: What do simple methods achieve?
2. **Find state-of-art**: What do best methods achieve?
3. **Identify limits**: What's theoretically possible?
4. **Document sources**: Citation for each calibration point
5. **Update regularly**: Domains evolve

## Application Protocol

1. Identify appropriate domain calibration
2. Map claimed metrics to calibration table
3. Classify each metric: Suspicious / Plausible / Excellent
4. Apply "Too Good" rule for suspicious results
5. Document calibration assessment in verdict

## When Calibration Is Missing

If no established calibration exists for a domain:

1. **Establish baseline**: What does naive/simple method achieve?
2. **Apply general principles**: Perfect = suspicious, noise exists
3. **Be more skeptical**: Unknown domain = higher burden of proof
4. **Build calibration**: Document results for future reference
