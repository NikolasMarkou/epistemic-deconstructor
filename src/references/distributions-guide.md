# Statistical Distributions Guide

Selecting the right distribution for Monte Carlo parameter uncertainty, DES inter-arrival/service times, and ABM agent state initialization.

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [Distribution Selection Decision Tree](#distribution-selection-decision-tree)
3. [Distribution Specifications](#distribution-specifications)
4. [Distribution Relationships](#distribution-relationships)
5. [Phase Integration](#phase-integration)

---

## Quick Reference

All distributions are specified as JSON dicts for `simulator.py`'s `--param_distributions`, DES config, and ABM agent state fields.

| Distribution | Type | Support | Parameters | Primary Use Case |
|---|---|---|---|---|
| `normal` | Continuous | (-inf, +inf) | `mean`, `std` | Symmetric uncertainty, measurement error |
| `uniform` | Continuous | [a, b] | `low`, `high` | Maximum ignorance, bounded range |
| `lognormal` | Continuous | (0, inf) | `mu`, `sigma` | Multiplicative processes, financial returns |
| `triangular` | Continuous | [a, b] | `left`, `mode`, `right` | Expert elicitation with min/mode/max |
| `beta` | Continuous | [0, 1] | `a`, `b` | Probabilities, proportions, rates |
| `exponential` | Continuous | [0, inf) | `scale` | Time between events, memoryless waiting |
| `gamma` | Continuous | (0, inf) | `shape`, `scale` | Waiting for k-th event, rate priors |
| `weibull` | Continuous | [0, inf) | `a`, `scale` | Reliability, time-to-failure |
| `poisson` | Discrete | {0, 1, 2, ...} | `lam` | Event counts in fixed interval |
| `binomial` | Discrete | {0, ..., n} | `n`, `p` | Successes in n independent trials |
| `chisquare` | Continuous | (0, inf) | `df` | Variance CIs, goodness-of-fit |
| `constant` | Deterministic | {v} | `value` | Fixed parameters (no uncertainty) |

---

## Distribution Selection Decision Tree

```
Is the parameter discrete or continuous?
|
+-- Discrete
|   +-- Single trial, binary outcome? -----------> binomial (n=1)
|   +-- Fixed n trials, counting successes? -----> binomial
|   +-- Counting events in an interval? ---------> poisson
|
+-- Continuous
    +-- Between two fixed bounds?
    |   +-- All values equally likely? -----------> uniform
    |   +-- Know min/mode/max from expert? -------> triangular
    |   +-- Bounded to [0,1] (probability)? ------> beta
    |
    +-- Always positive, right-skewed?
    |   +-- Time between events (memoryless)? ----> exponential
    |   +-- Time until k-th event / rate prior? --> gamma
    |   +-- Multiplicative growth process? -------> lognormal
    |   +-- Failure/reliability with aging? ------> weibull
    |   +-- Sum of squared normals? --------------> chisquare
    |
    +-- Symmetric, bell-shaped?
        +-- Known or estimated mean and std? -----> normal
```

### Common Phase 3 Scenarios

| Scenario | Recommended Distribution | Example Spec |
|---|---|---|
| Parameter estimated via OLS with CI | `normal` | `{"dist": "normal", "mean": -0.5, "std": 0.05}` |
| Rate parameter with conjugate prior | `gamma` | `{"dist": "gamma", "shape": 2.0, "scale": 0.5}` |
| Probability parameter from data | `beta` | `{"dist": "beta", "a": 10, "b": 30}` |
| Expert says "between 2 and 8, likely 5" | `triangular` | `{"dist": "triangular", "left": 2, "mode": 5, "right": 8}` |
| Component lifetime (wear-out) | `weibull` | `{"dist": "weibull", "a": 2.5, "scale": 1000}` |
| No information, bounded range | `uniform` | `{"dist": "uniform", "low": 0.1, "high": 0.9}` |

---

## Distribution Specifications

### normal

Symmetric uncertainty around a central estimate. Use when Phase 3 produces parameter +/- standard error.

```json
{"dist": "normal", "mean": 5.0, "std": 1.0}
```

| Property | Value |
|---|---|
| Mean | mu |
| Variance | sigma^2 |
| 95% of mass | mu +/- 1.96*sigma |

### uniform

Maximum ignorance within bounds. Use when you know the range but nothing else.

```json
{"dist": "uniform", "low": 0.0, "high": 10.0}
```

### lognormal

Product of many small positive factors. Use for financial returns, concentrations, sizes.

```json
{"dist": "lognormal", "mu": 0.0, "sigma": 0.5}
```

Note: `mu` and `sigma` are parameters of the underlying normal (log-scale), not the lognormal itself. Mean of lognormal = exp(mu + sigma^2/2).

### triangular

Expert elicitation with minimum, most likely, and maximum values.

```json
{"dist": "triangular", "left": 0.0, "mode": 5.0, "right": 10.0}
```

### beta

Probabilities and proportions bounded to [0, 1]. Conjugate prior for binomial likelihood. For A/B testing posteriors, set `a` = successes + 1, `b` = failures + 1.

```json
{"dist": "beta", "a": 2.0, "b": 5.0}
```

| Shape | Condition |
|---|---|
| Uniform | a = b = 1 |
| Symmetric bell | a = b > 1 |
| Left-skewed (mass near 0) | a < b |
| Right-skewed (mass near 1) | a > b |

### exponential

Time between events in a Poisson process. Memoryless: past waiting doesn't affect future. Default for DES inter-arrival times.

```json
{"dist": "exponential", "scale": 2.0}
```

Mean = scale. Variance = scale^2.

### gamma

Generalizes exponential. Use for waiting time until the k-th event, or as a conjugate prior for Poisson/exponential rate parameters.

```json
{"dist": "gamma", "shape": 2.0, "scale": 1.5}
```

| Special Case | Parameters |
|---|---|
| Exponential | shape = 1 |
| Chi-squared(v) | shape = v/2, scale = 2 |

Mean = shape * scale. Variance = shape * scale^2.

### weibull

Reliability and survival analysis. The shape parameter `a` controls failure rate behavior.

```json
{"dist": "weibull", "a": 2.0, "scale": 3.0}
```

| a | Failure Rate | Interpretation |
|---|---|---|
| a < 1 | Decreasing | Infant mortality (early failures) |
| a = 1 | Constant | Random failures (= exponential) |
| a > 1 | Increasing | Wear-out (aging) |

### poisson

Count of events in a fixed interval. Returns integer values (cast to float for array compatibility).

```json
{"dist": "poisson", "lam": 5.0}
```

Mean = Variance = lam. Use when events are independent and rate is constant.

### binomial

Number of successes in `n` independent trials with probability `p`. Returns integer values.

```json
{"dist": "binomial", "n": 10, "p": 0.3}
```

Mean = n*p. Variance = n*p*(1-p). Use for discrete outcome counts (defects per batch, conversions per cohort).

### chisquare

Sum of squares of `df` independent standard normal variables. Use for variance confidence intervals and goodness-of-fit statistics.

```json
{"dist": "chisquare", "df": 5}
```

Mean = df. Variance = 2*df.

### constant

Deterministic value (no uncertainty). Use to fix parameters while varying others in MC runs.

```json
{"dist": "constant", "value": 3.14}
```

---

## Distribution Relationships

```
Bernoulli(p) = binomial(n=1, p)
    +-- n trials -----------------> binomial(n, p)
                                        +-- n large, p small --> poisson(lam=np)
                                                                    +-- inter-arrival --> exponential(scale=1/lam)
                                                                                            +-- k events --> gamma(k, 1/lam)

normal(0, 1)
    +-- exp(X) -------------------> lognormal(mu, sigma)
    +-- Z1^2 + ... + Zv^2 -------> chisquare(v)
                                        +-- special case --> gamma(v/2, 2)

exponential(scale) = gamma(shape=1, scale) = weibull(a=1, scale)
beta(1, 1) = uniform(0, 1)
```

---

## Phase Integration

| Phase | Distribution Role | Guidance |
|---|---|---|
| **Phase 3** | Parameterize uncertainty bounds from system ID | Use normal for OLS estimates, gamma for rate priors, beta for probability parameters |
| **Phase 4 (MC)** | `--param_distributions` for Monte Carlo runs | Match distribution family to parameter semantics using the decision tree above |
| **Phase 4 (DES)** | `arrival` and `service` time distributions | Default: exponential. Use weibull for aging service, gamma for k-stage service |
| **Phase 4 (ABM)** | Agent state initialization via `state` field | Use any distribution for heterogeneous agent populations |
| **Phase 5** | Interpret MC output percentile bands | Skewed output distributions (from lognormal/weibull inputs) require percentile-based CIs, not mean +/- std |
