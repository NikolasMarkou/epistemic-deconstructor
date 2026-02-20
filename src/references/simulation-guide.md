# Simulation Guide

Take identified models from Phases 3-4 and run them forward: project trajectories, quantify risk, test counterfactuals, and stress-test parameter spaces. The simulation engine (`scripts/simulator.py`) supports five paradigms and feeds results back into Phase 5 validation.

## Table of Contents

- [Domain Fit Gate](#domain-fit-gate)
- [Simulation Paradigms](#simulation-paradigms)
  - [System Dynamics (SD)](#1-system-dynamics-sd)
  - [Monte Carlo (MC)](#2-monte-carlo-mc)
  - [Agent-Based Model (ABM)](#3-agent-based-model-abm)
  - [Discrete-Event Simulation (DES)](#4-discrete-event-simulation-des)
  - [Sensitivity Analysis](#5-sensitivity-analysis)
- [Archetype-to-Mode Mapping](#archetype-to-mode-mapping)
- [Model Conversion Recipes](#model-conversion-recipes)
- [Validation Bridge](#validation-bridge)
- [Convergence Diagnostics](#convergence-diagnostics)
- [Visualization Outputs](#visualization-outputs)
- [State Block Extension](#state-block-extension)
- [Workflow](#workflow)
- [Dependencies](#dependencies)
- [Critical Reminders](#critical-reminders)
- [Cross-References](#cross-references)

---

## Domain Fit Gate

Not every analysis benefits from simulation. Run this gate before committing.

### Simulate When

| Signal | Example |
|--------|---------|
| Parametric model exists (Phase 3+) | ARX, state-space, NARMAX coefficients |
| System evolves over time | Feedback loops, accumulation, delays |
| Uncertainty in parameters matters | Bootstrap CIs, Bayesian posteriors |
| Counterfactuals requested | "What if parameter X were 2x higher?" |
| Nonlinear interactions suspected | Emergence gap > 20% from Phase 4 |
| Risk/tail analysis needed | "How bad can it get?" |
| Multiple agents with local rules | Market participants, network nodes |

### Do NOT Simulate When

| Signal | Instead |
|--------|---------|
| Static classification (no dynamics) | Decision trees, lookup |
| Pure text/document analysis | Stay in PSYCH tier |
| RAPID tier (claim validation) | Use `rapid_checker.py` |
| No model identified yet (Phase 0-1) | Continue deconstruction first |
| Analytical solution is exact | Compute directly |

### Gate Decision Tree

```
SIMULATE?
├─ Parametric model identified? (Phase 3+)
│  ├─ NO → STOP. Complete identification first.
│  └─ YES
│     ├─ Dynamics (time evolution)? → Full simulation
│     │  └─ NO → Static sensitivity only
│     ├─ Stochastic elements? → Monte Carlo mode
│     │  └─ NO → Deterministic (still useful for counterfactuals)
│     └─ Multiple interacting agents? → Agent-based mode
│        └─ NO → System dynamics or discrete-event
```

---

## Simulation Paradigms

### 1. System Dynamics (SD)

For continuous feedback systems with stocks and flows: controllers, supply chains, epidemics, population dynamics.

**Input**: State-space (A, B, C, D) or ODE system from Phase 3.

**Features**:
- Linear state-space simulation via RK4 or Euler integration
- Nonlinear ODE support via `ode_code` field (scipy RK45 adaptive)
- Eigenvalue stability analysis
- Settling time estimation (2% band)
- Phase portrait generation (2+ states)

**Integrators**: `rk4` (default), `euler` (fast/debug), `rk45` (stiff systems via scipy).

### 2. Monte Carlo (MC)

For parameter uncertainty, risk quantification, tail analysis.

**Input**: Any model + parameter distribution specs from Phase 3 uncertainty bounds.

**Supported distributions**: `normal`, `uniform`, `lognormal`, `triangular`, `beta`, `exponential`, `constant`.

**Outputs**: Percentile bands (5/25/50/75/95), terminal stats (mean, std, skew, kurtosis, VaR, CVaR), convergence diagnostics.

**Convergence**: Use `--convergence_check` flag. Checks relative change in terminal mean across checkpoints (100, 200, 500, ..., 10000 runs). Converged when relative change < 1%.

### 3. Agent-Based Model (ABM)

For decentralized systems where macro behavior emerges from micro rules: markets, networks, social dynamics.

**Input**: Agent rules from Phase 2 causal graph + Phase 3 parameters. Config JSON file.

**Topologies**: `complete`, `grid`, `small_world` (Watts-Strogatz), `scale_free` (Barabási-Albert).

**Agent actions**: `increment`, `decrement`, `adopt_neighbor_mean`, `random_walk`.

**Macro tracking**: Mean state, standard deviation, active fraction — all per time step.

### 4. Discrete-Event Simulation (DES)

For event-driven systems with queues, arrivals, departures: protocols, service systems, pipelines.

**Input**: Arrival/service distributions, server count. Config JSON file.

**Outputs**: Throughput, queue stats (mean/max length, mean/p95 wait), per-server utilization.

**Architecture**: Single-queue / multi-server with priority heap. Events: ARRIVAL, DEPARTURE.

### 5. Sensitivity Analysis

For parameter impact ranking without dynamics. Works with any Phase 3 model expression.

**Methods**:
- `sobol` — Quantitative global sensitivity (first-order S1 and total-order ST indices). Recommended for final analysis.
- `morris` — Screening method (mu* and sigma). Cheaper, good for identifying important factors.
- OAT fallback — One-at-a-time when SALib is not installed. Basic tornado diagram.

---

## Archetype-to-Mode Mapping

Use Phase 4 archetype to select the primary simulation mode:

| Phase 4 Archetype | Primary Mode | Secondary Mode | Key Parameters |
|-------------------|-------------|----------------|----------------|
| **State Machine** | DES | SD (if continuous timing) | States, transitions, event rates |
| **Pipeline** | DES | Sensitivity | Stage times, bottleneck identification |
| **Controller** | SD | MC (param uncertainty) | Gains, setpoints, disturbance models |
| **Pub/Sub** | DES | ABM (if many publishers) | Message rates, queue capacities |
| **Network** | ABM | MC (failure cascades) | Topology, node rules, coupling strength |
| **Adaptive** | MC | SD (with drift) | Learning rates, concept drift bounds |

---

## Model Conversion Recipes

### ARX → MC or SD

```python
# From Phase 3 ARX identification:
# y(t) = -a1*y(t-1) - a2*y(t-2) + b1*u(t-1)
# With uncertainty: a1 = -0.5 ± 0.05, a2 = 0.3 ± 0.03, b1 = 1.0 ± 0.1

# Deterministic (SD mode):
model = '{"type": "arx", "a": [-0.5, 0.3], "b": [1.0], "nk": 1}'

# Stochastic (MC mode):
param_dists = '{
  "a[0]": {"dist": "normal", "mean": -0.5, "std": 0.05},
  "a[1]": {"dist": "normal", "mean": 0.3, "std": 0.03},
  "b[0]": {"dist": "normal", "mean": 1.0, "std": 0.1}
}'
```

### State-Space → SD

```python
# From Phase 3 subspace identification (N4SID):
# x' = Ax + Bu, y = Cx + Du

model = '{"A": [[-1, 0.5], [0, -2]], "B": [[1], [0]], "C": [[1, 0]], "D": [[0]]}'
```

### NARMAX → SD (nonlinear)

```python
# From Phase 3 NARMAX via SINDy or FROLS:
# dx/dt = f(x, u) discovered nonlinear dynamics

model = '{"ode_code": "def f(t, x, u):\\n    return [x[1], -0.5*x[0]**3 - 0.1*x[1] + u]"}'
```

---

## Validation Bridge

After simulation, generate Phase 5-ready validation data:

```bash
python scripts/simulator.py bridge \
  --sim_output sim_mc.json --output validation_bridge.json
```

This maps simulation outputs to Phase 5 checks:

| Simulator Output | Phase 5 Use |
|-----------------|-------------|
| SD trajectories | Interpolation R² check |
| MC percentiles | Uncertainty bounds validation |
| Sensitivity indices | Parameter importance ranking |
| ABM macro series | Emergence gap quantification |
| DES queue stats | Performance validation |

### Interpolation Validation (R² > 0.95)

1. Run SD with training-set inputs
2. Compare sim output to actual training data
3. Compute R² = 1 - SS_res/SS_tot

### Extrapolation Validation (R² > 0.80)

1. Run SD with held-out test inputs (unseen during Phase 3)
2. Compare sim output to actual test data
3. Compute R²

### Counterfactual Validation

1. Define counterfactual scenarios (parameter changes, input changes)
2. Run SD/MC for each scenario
3. Check: do predictions align with domain expert expectations?
4. Document: which counterfactuals are plausible vs. out-of-model-range

### Uncertainty Validation

1. Run MC with Phase 3 parameter uncertainty
2. Check: do actual observations fall within MC percentile bands?
3. If >5% of observations outside 90% band → model underestimates uncertainty

### Emergence Validation

1. Run ABM or composed SD model
2. Compare macro output to Phase 4 composed model prediction
3. Emergence gap = |predicted - actual| / |actual|
4. If gap > 0.20 → emergence present, model incomplete

---

## Convergence Diagnostics

### Monte Carlo Convergence

| Metric | Threshold | Action if Fail |
|--------|-----------|----------------|
| Mean relative change | < 1% between n and n/2 | Double n_runs |
| Std relative change | < 5% between n and n/2 | Double n_runs |
| 5th percentile stable | < 2% change | Increase to capture tails |
| Max n_runs reached | 100,000 | Report unconverged + current bounds |

### SD Numerical Stability

| Check | Method | Remediation |
|-------|--------|-------------|
| Eigenvalues | Re(λ) < 0 for all λ | Verify model identification |
| Step response bounded | max(\|y\|) < 100× steady-state | Reduce dt, use RK45 |
| Energy conservation | If applicable, ΔE/E < 1e-6 | Use symplectic integrator |
| Stiffness ratio | max(\|λ\|)/min(\|λ\|) > 1000 | Switch to implicit solver |

### ABM Ergodicity

| Check | Method | Remediation |
|-------|--------|-------------|
| Ensemble mean converges | 10 runs with different seeds | Increase if variance high |
| Transient length | Plot macro metric, find plateau | Discard initial transient |
| Sensitivity to topology | Try 2+ topologies | Report if qualitatively different |

---

## Visualization Outputs

All modes generate matplotlib PNGs when `--plot` is passed:

| Plot | Mode | Shows |
|------|------|-------|
| `*_trajectory.png` | SD | State variables over time |
| `*_phase.png` | SD | State-space phase portrait |
| `*_fan.png` | MC | Percentile fan chart |
| `*_histogram.png` | MC | Terminal distribution |
| `*_sobol.png` | Sensitivity | Sobol index bar chart |
| `*_morris.png` | Sensitivity | Morris μ* vs σ scatter |
| `*_tornado.png` | Sensitivity | OAT tornado diagram |
| `*_macro.png` | ABM | Macro dynamics + active fraction |
| `*_queue.png` | DES | Queue length + server utilization |

Add `--report path.md` to any mode for a markdown summary report.

---

## State Block Extension

Append simulation state to the deconstructor state block:

```
[STATE: Phase 3→SIM | Tier: STANDARD | Mode: MC | Runs: 10000 | Converged: YES | Lead: H2 (82%)]
[STATE: Phase 4→SIM | Tier: COMPREHENSIVE | Mode: ABM | Agents: 1000 | Steps: 500/500 | Emergence: 0.15]
[STATE: Phase 5→SIM | Tier: STANDARD | Mode: SD | Validation: PASS (R2=0.94) | Counterfactuals: 3/3]
```

---

## Workflow

```
1. GATE:         Domain fit check (above)
2. SELECT:       Choose paradigm from archetype/model type
3. CONFIGURE:    Map deconstructor outputs to simulator inputs
4. RUN:          Execute simulation (scripts/simulator.py)
5. DIAGNOSE:     Check convergence, stability, sanity
6. SIGNAL-CHECK: Route time-series output through ts_reviewer.py quick (stationarity, forecastability, baselines)
7. VISUALIZE:    --plot flag generates PNGs
8. BRIDGE:       Generate validation_bridge.json for Phase 5
9. ITERATE:      Adjust parameters, re-run if needed
```

**Step 6 detail**: After simulation produces trajectories (SD) or fan charts (MC), run `ts_reviewer.py quick` on the simulated output to validate signal quality. If ts_reviewer flags stationarity issues or the simulated signal has PE > 0.95 (effectively random), the simulation parameterization may be wrong. Compare simulated PE and baselines against observed data from Phase 1 — large discrepancies indicate model misspecification, not emergence.

---

## Dependencies

**Required**: `numpy`, `scipy`, `matplotlib` (all pip-installable).

**Optional**: `SALib` (Sobol/Morris global sensitivity).

**Security note**: `simulator.py` uses `exec()` for nonlinear ODE code strings and `eval()` for ABM rule triggers and sensitivity model expressions. These run with restricted builtins but in the caller's process. Only use with trusted model definitions.

---

## Critical Reminders

- **Gate first**: Don't simulate what doesn't need simulating.
- **Model before simulation**: Simulation without an identified model is theater.
- **Convergence is non-negotiable**: MC results without convergence checks are noise.
- **Validate against data**: Simulation matching your model but not reality is self-consistent hallucination.
- **Uncertainty in, uncertainty out**: Propagate parameter uncertainty or your predictions are overconfident.
- **Emergence check**: If sim output differs from composed model prediction, the model is incomplete.
- **Seed everything**: Reproducibility requires `--seed`.

---

## Cross-References

- System identification (model inputs): `references/system-identification.md`
- Compositional synthesis (MC uncertainty, emergence): `references/compositional-synthesis.md`
- Validation requirements: `references/validation-checklist.md`
- Tool catalog: `references/tool-catalog.md`
- Sensitivity analysis algorithms: `references/tools-sensitivity.md`
- Time-series diagnostics: `references/timeseries-review.md`
- Forecasting science: `references/forecasting-science.md`
