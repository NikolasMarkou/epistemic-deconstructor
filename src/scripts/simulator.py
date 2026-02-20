#!/usr/bin/env python3
"""
Epistemic Simulator — simulation engine for deconstructor models.

Supports five paradigms: System Dynamics (SD), Monte Carlo (MC),
Agent-Based Modeling (ABM), Discrete-Event Simulation (DES),
and Sensitivity Analysis (Morris/Sobol/OAT).

Requires: numpy (hard dependency), scipy (for RK45/stats), matplotlib (plots).
See references/simulation-guide.md for full protocol.
"""

import argparse
import json
import sys
import os
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Optional
import warnings

import numpy as np

from common import save_json as _common_save_json

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _seed_rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)


def _save_json(data: dict, path: str):
    """Save simulation output as JSON using common.py's locked I/O."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Pre-serialize numpy types before handing to common save_json
    serializable = json.loads(json.dumps(data, default=_json_default))
    _common_save_json(path, serializable)
    print(f"[simulator] Saved → {path}")


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _sample_distribution(spec: dict, rng: np.random.Generator, size: int = 1):
    """Sample from a distribution specification dict."""
    dist = spec["dist"]
    if dist == "normal":
        return rng.normal(spec["mean"], spec["std"], size)
    elif dist == "uniform":
        return rng.uniform(spec["low"], spec["high"], size)
    elif dist == "lognormal":
        return rng.lognormal(spec.get("mu", 0), spec.get("sigma", 1), size)
    elif dist == "triangular":
        return rng.triangular(spec["left"], spec["mode"], spec["right"], size)
    elif dist == "beta":
        return rng.beta(spec["a"], spec["b"], size)
    elif dist == "exponential":
        return rng.exponential(spec["scale"], size)
    elif dist == "constant":
        return np.full(size, spec["value"])
    else:
        raise ValueError(f"Unknown distribution: {dist}")


# ---------------------------------------------------------------------------
# Input functions for SD mode
# ---------------------------------------------------------------------------

def _make_input_func(name: str, amplitude: float = 1.0, freq: float = 1.0,
                     t_on: float = 0.0) -> Callable:
    """Factory for common input/excitation signals."""
    if name == "step":
        return lambda t: amplitude * (1.0 if t >= t_on else 0.0)
    elif name == "impulse":
        return lambda t: amplitude if abs(t - t_on) < 1e-6 else 0.0
    elif name == "sine":
        return lambda t: amplitude * np.sin(2 * np.pi * freq * t)
    elif name == "ramp":
        return lambda t: amplitude * max(0, t - t_on)
    elif name == "zero":
        return lambda t: 0.0
    else:
        raise ValueError(f"Unknown input function: {name}")


# ═══════════════════════════════════════════════════════════════════════════
# MODE 1: System Dynamics (SD)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SDResult:
    t: np.ndarray
    x: np.ndarray          # (n_steps, n_states)
    y: np.ndarray          # (n_steps, n_outputs)
    u_history: np.ndarray  # (n_steps, n_inputs)
    eigenvalues: np.ndarray
    is_stable: bool
    settling_time: Optional[float]
    metadata: dict = field(default_factory=dict)


def _sd_linear(model: dict, x0: np.ndarray, u_func: Callable,
               t_end: float, dt: float, integrator: str) -> SDResult:
    """Simulate a linear state-space model: x' = Ax + Bu, y = Cx + Du."""
    A = np.array(model["A"], dtype=float)
    B = np.array(model["B"], dtype=float)
    C = np.array(model.get("C", np.eye(A.shape[0])), dtype=float)
    D = np.array(model.get("D", np.zeros((C.shape[0], B.shape[1]))), dtype=float)

    n_states = A.shape[0]
    n_inputs = B.shape[1]
    n_outputs = C.shape[0]

    eigenvalues = np.linalg.eigvals(A)
    is_stable = np.all(np.real(eigenvalues) < 0)

    t_arr = np.arange(0, t_end + dt, dt)
    n_steps = len(t_arr)
    x = np.zeros((n_steps, n_states))
    y = np.zeros((n_steps, n_outputs))
    u_history = np.zeros((n_steps, n_inputs))

    x[0] = x0

    def _u_vec(t):
        val = u_func(t)
        if np.isscalar(val):
            return np.full(n_inputs, val)
        return np.asarray(val)

    def _dxdt(t, state, u_val):
        return A @ state + B @ u_val

    for i in range(n_steps):
        u_val = _u_vec(t_arr[i])
        u_history[i] = u_val
        y[i] = C @ x[i] + D @ u_val

        if i < n_steps - 1:
            if integrator == "euler":
                x[i + 1] = x[i] + dt * _dxdt(t_arr[i], x[i], u_val)
            elif integrator == "rk4":
                k1 = _dxdt(t_arr[i], x[i], u_val)
                k2 = _dxdt(t_arr[i] + dt / 2, x[i] + dt / 2 * k1, u_val)
                k3 = _dxdt(t_arr[i] + dt / 2, x[i] + dt / 2 * k2, u_val)
                k4 = _dxdt(t_arr[i] + dt, x[i] + dt * k3, u_val)
                x[i + 1] = x[i] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            elif integrator == "rk45":
                from scipy.integrate import solve_ivp
                sol = solve_ivp(lambda t, s: _dxdt(t, s, u_val),
                                [t_arr[i], t_arr[i + 1]], x[i],
                                method="RK45", max_step=dt)
                x[i + 1] = sol.y[:, -1]

    # Settling time estimate (2% band)
    settling_time = None
    if is_stable and n_outputs > 0:
        ss_val = y[-1, 0]
        if abs(ss_val) > 1e-10:
            within_band = np.abs(y[:, 0] - ss_val) < 0.02 * abs(ss_val)
            settled_idx = np.where(within_band)[0]
            if len(settled_idx) > 0:
                # Find last crossing out of band
                for j in range(len(settled_idx) - 1, -1, -1):
                    if settled_idx[j] < n_steps - 1 and not within_band[settled_idx[j] + 1] if settled_idx[j] + 1 < n_steps else False:
                        continue
                    settling_time = t_arr[settled_idx[j]]
                    break

    return SDResult(
        t=t_arr, x=x, y=y, u_history=u_history,
        eigenvalues=eigenvalues, is_stable=is_stable,
        settling_time=settling_time,
        metadata={"integrator": integrator, "dt": dt, "n_states": n_states}
    )


def _sd_nonlinear(model: dict, x0: np.ndarray, u_func: Callable,
                  t_end: float, dt: float) -> SDResult:
    """Simulate a nonlinear ODE system using scipy.integrate.solve_ivp.

    model["ode_code"]: string of Python code defining f(t, x, u) -> dx/dt
    """
    from scipy.integrate import solve_ivp

    # SECURITY: exec used for user-defined ODE expressions (e.g. nonlinear dynamics).
    # Runs in caller's process — only use with trusted model definitions.
    namespace = {"np": np}
    exec(model["ode_code"], namespace)  # noqa: S102
    ode_func = namespace["f"]

    t_span = (0, t_end)
    t_eval = np.arange(0, t_end + dt, dt)

    def rhs(t, state):
        u_val = u_func(t)
        return ode_func(t, state, u_val)

    sol = solve_ivp(rhs, t_span, x0, t_eval=t_eval, method="RK45",
                    max_step=dt * 10, rtol=1e-8, atol=1e-10)

    x = sol.y.T
    u_history = np.array([u_func(t) for t in sol.t])
    if u_history.ndim == 1:
        u_history = u_history.reshape(-1, 1)

    return SDResult(
        t=sol.t, x=x, y=x.copy(), u_history=u_history,
        eigenvalues=np.array([]), is_stable=sol.success,
        settling_time=None,
        metadata={"integrator": "RK45_adaptive", "dt": dt, "nonlinear": True}
    )


def run_sd(args):
    """Entry point for System Dynamics mode."""
    model = json.loads(args.model)
    x0 = np.array(json.loads(args.x0), dtype=float)
    u_func = _make_input_func(args.u_func, amplitude=args.amplitude,
                              freq=args.freq, t_on=args.t_on)

    if "ode_code" in model:
        result = _sd_nonlinear(model, x0, u_func, args.t_end, args.dt)
    else:
        result = _sd_linear(model, x0, u_func, args.t_end, args.dt, args.integrator)

    output = {
        "mode": "sd",
        "t": result.t,
        "x": result.x,
        "y": result.y,
        "u": result.u_history,
        "eigenvalues_real": np.real(result.eigenvalues),
        "eigenvalues_imag": np.imag(result.eigenvalues),
        "is_stable": result.is_stable,
        "settling_time": result.settling_time,
        "metadata": result.metadata,
    }
    _save_json(output, args.output)

    if args.plot:
        _plot_sd(result, args.output)

    # Print summary
    print(f"[SD] States: {result.x.shape[1]} | Steps: {len(result.t)} | "
          f"Stable: {result.is_stable} | Settling: {result.settling_time}")

    if args.report:
        _report_sd(result, args.report)

    return result


def _plot_sd(result: SDResult, output_base: str):
    """Generate SD plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = Path(output_base).with_suffix("")

    # Trajectory plot
    fig, axes = plt.subplots(result.x.shape[1], 1, figsize=(10, 3 * result.x.shape[1]),
                             squeeze=False, sharex=True)
    for i in range(result.x.shape[1]):
        axes[i, 0].plot(result.t, result.x[:, i], linewidth=1.5)
        axes[i, 0].set_ylabel(f"x[{i}]")
        axes[i, 0].grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel("Time")
    fig.suptitle("System Dynamics — State Trajectories", fontsize=13)
    fig.tight_layout()
    traj_path = f"{base}_trajectory.png"
    fig.savefig(traj_path, dpi=150)
    plt.close(fig)
    print(f"[plot] → {traj_path}")

    # Phase portrait (if 2+ states)
    if result.x.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot(result.x[:, 0], result.x[:, 1], linewidth=1, alpha=0.8)
        ax.plot(result.x[0, 0], result.x[0, 1], "go", markersize=8, label="Start")
        ax.plot(result.x[-1, 0], result.x[-1, 1], "rs", markersize=8, label="End")
        ax.set_xlabel("x[0]")
        ax.set_ylabel("x[1]")
        ax.set_title("Phase Portrait")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        phase_path = f"{base}_phase.png"
        fig.savefig(phase_path, dpi=150)
        plt.close(fig)
        print(f"[plot] → {phase_path}")


def _report_sd(result: SDResult, path: str):
    """Generate markdown report for SD simulation."""
    eigs = result.eigenvalues
    lines = [
        "# System Dynamics Simulation Report\n",
        f"## Configuration",
        f"- Integrator: {result.metadata.get('integrator', 'unknown')}",
        f"- dt: {result.metadata.get('dt', 'N/A')}",
        f"- States: {result.x.shape[1]}",
        f"- Time steps: {len(result.t)}",
        f"- Duration: {result.t[-1]:.2f}\n",
        f"## Stability",
        f"- Stable: **{result.is_stable}**",
        f"- Eigenvalues: {', '.join(f'{e:.4f}' for e in eigs)}" if len(eigs) > 0 else "- Eigenvalues: N/A (nonlinear)",
        f"- Settling time (2%): {result.settling_time}\n",
        f"## Final State",
    ]
    for i in range(result.x.shape[1]):
        lines.append(f"- x[{i}] = {result.x[-1, i]:.6f}")

    Path(path).write_text("\n".join(lines))
    print(f"[report] → {path}")


# ═══════════════════════════════════════════════════════════════════════════
# MODE 2: Monte Carlo (MC)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MCResult:
    n_runs: int
    t: np.ndarray
    trajectories: np.ndarray   # (n_runs, n_steps, n_states)
    percentiles: dict          # {5: arr, 25: arr, 50: arr, 75: arr, 95: arr}
    terminal_stats: dict       # {mean, std, skew, kurtosis, min, max, VaR_5, CVaR_5}
    converged: bool
    convergence_n: Optional[int]
    param_samples: dict
    metadata: dict = field(default_factory=dict)


def _mc_run_single(model: dict, params: dict, x0: np.ndarray,
                   u_func: Callable, t_end: float, dt: float) -> np.ndarray:
    """Run a single MC realization with sampled parameters."""
    # Inject sampled params into model
    mod = json.loads(json.dumps(model, default=_json_default))
    for key, val in params.items():
        parts = key.replace("]", "").replace("[", ".").split(".")
        target = mod
        for p in parts[:-1]:
            if p.isdigit():
                target = target[int(p)]
            else:
                target = target[p]
        last = parts[-1]
        if last.isdigit():
            target[int(last)] = val
        else:
            target[last] = val

    # Build state-space or ARX
    if "A" in mod:
        A = np.array(mod["A"], dtype=float)
        B = np.array(mod["B"], dtype=float)
        n = A.shape[0]
        t_arr = np.arange(0, t_end + dt, dt)
        x = np.zeros((len(t_arr), n))
        x[0] = x0[:n] if len(x0) >= n else np.zeros(n)
        for i in range(len(t_arr) - 1):
            u_val = u_func(t_arr[i])
            if np.isscalar(u_val):
                u_val = np.full(B.shape[1], u_val)
            k1 = A @ x[i] + B @ np.asarray(u_val)
            k2 = A @ (x[i] + dt / 2 * k1) + B @ np.asarray(u_val)
            k3 = A @ (x[i] + dt / 2 * k2) + B @ np.asarray(u_val)
            k4 = A @ (x[i] + dt * k3) + B @ np.asarray(u_val)
            x[i + 1] = x[i] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return x
    elif "a" in mod or "type" in mod and mod.get("type") == "arx":
        # ARX: y(t) = -a1*y(t-1) - ... + b0*u(t-nk) + ...
        a = np.array(mod.get("a", []), dtype=float)
        b = np.array(mod.get("b", [1.0]), dtype=float)
        nk = int(mod.get("nk", 1))
        t_arr = np.arange(0, t_end + dt, dt)
        n_steps = len(t_arr)
        y = np.zeros(n_steps)
        u_arr = np.array([u_func(t) for t in t_arr])
        if u_arr.ndim > 1:
            u_arr = u_arr[:, 0]
        na = len(a)
        nb = len(b)
        for i in range(max(na, nb + nk - 1), n_steps):
            for j, aj in enumerate(a):
                y[i] += -aj * y[i - j - 1]
            for j, bj in enumerate(b):
                idx = i - nk - j
                if 0 <= idx < n_steps:
                    y[i] += bj * u_arr[idx]
        return y.reshape(-1, 1)
    else:
        raise ValueError("MC mode requires model with 'A'/'B' (state-space) or 'a'/'b' (ARX)")


def run_mc(args):
    """Entry point for Monte Carlo mode."""
    model = json.loads(args.model)
    param_dists = json.loads(args.param_distributions)
    x0 = np.array(json.loads(args.x0), dtype=float) if args.x0 else np.zeros(1)
    u_func = _make_input_func(args.u_func, amplitude=args.amplitude)
    rng = _seed_rng(args.seed)

    n_runs = args.n_runs
    dt = args.dt
    t_end = args.t_end
    t_arr = np.arange(0, t_end + dt, dt)
    n_steps = len(t_arr)

    # Pre-sample all parameters
    param_samples = {}
    for key, spec in param_dists.items():
        param_samples[key] = _sample_distribution(spec, rng, size=n_runs)

    # Run simulations
    all_traj = []
    for i in range(n_runs):
        params_i = {k: float(v[i]) for k, v in param_samples.items()}
        try:
            traj = _mc_run_single(model, params_i, x0, u_func, t_end, dt)
            # Ensure consistent length
            if len(traj) >= n_steps:
                traj = traj[:n_steps]
            else:
                pad = np.full((n_steps - len(traj), traj.shape[1] if traj.ndim > 1 else 1), np.nan)
                traj = np.vstack([traj, pad]) if traj.ndim > 1 else np.concatenate([traj, pad.flatten()])
            all_traj.append(traj)
        except Exception as e:
            if args.verbose:
                print(f"[MC] Run {i} failed: {e}")
            continue

        if args.verbose and (i + 1) % max(1, n_runs // 10) == 0:
            print(f"[MC] {i + 1}/{n_runs} complete")

    trajectories = np.array(all_traj)
    if trajectories.ndim == 2:
        trajectories = trajectories[:, :, np.newaxis]

    actual_runs = trajectories.shape[0]
    print(f"[MC] Completed {actual_runs}/{n_runs} runs")

    # Percentiles
    pcts = {}
    for p in [5, 25, 50, 75, 95]:
        pcts[p] = np.nanpercentile(trajectories[:, :, 0], p, axis=0)

    # Terminal statistics
    terminal = trajectories[:, -1, 0]
    terminal = terminal[~np.isnan(terminal)]
    from scipy import stats as sp_stats
    term_stats = {
        "mean": float(np.mean(terminal)),
        "std": float(np.std(terminal)),
        "skew": float(sp_stats.skew(terminal)),
        "kurtosis": float(sp_stats.kurtosis(terminal)),
        "min": float(np.min(terminal)),
        "max": float(np.max(terminal)),
        "VaR_5": float(np.percentile(terminal, 5)),
        "CVaR_5": float(np.mean(terminal[terminal <= np.percentile(terminal, 5)])),
    }

    # Convergence check
    converged = False
    convergence_n = None
    if args.convergence_check and actual_runs >= 200:
        checkpoints = [100, 200, 500, 1000, 2000, 5000, 10000]
        checkpoints = [c for c in checkpoints if c <= actual_runs]
        prev_mean = None
        for cp in checkpoints:
            sub = trajectories[:cp, -1, 0]
            sub = sub[~np.isnan(sub)]
            curr_mean = np.mean(sub)
            if prev_mean is not None and abs(prev_mean) > 1e-10:
                rel_change = abs(curr_mean - prev_mean) / abs(curr_mean)
                if rel_change < 0.01:
                    converged = True
                    convergence_n = cp
                    break
            prev_mean = curr_mean

    result = MCResult(
        n_runs=actual_runs, t=t_arr, trajectories=trajectories,
        percentiles=pcts, terminal_stats=term_stats,
        converged=converged, convergence_n=convergence_n,
        param_samples={k: v.tolist() for k, v in param_samples.items()},
        metadata={"seed": args.seed, "dt": dt}
    )

    output = {
        "mode": "mc",
        "n_runs": actual_runs,
        "t": t_arr,
        "percentiles": {str(k): v for k, v in pcts.items()},
        "terminal_stats": term_stats,
        "converged": converged,
        "convergence_n": convergence_n,
        "metadata": result.metadata,
    }
    _save_json(output, args.output)

    if args.plot:
        _plot_mc(result, args.output)

    print(f"[MC] Converged: {converged} (at n={convergence_n}) | "
          f"Terminal mean: {term_stats['mean']:.4f} ± {term_stats['std']:.4f}")

    if args.report:
        _report_mc(result, args.report)

    return result


def _plot_mc(result: MCResult, output_base: str):
    """Generate MC plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = Path(output_base).with_suffix("")

    # Fan chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(result.t, result.percentiles[5], result.percentiles[95],
                    alpha=0.15, color="C0", label="5-95%")
    ax.fill_between(result.t, result.percentiles[25], result.percentiles[75],
                    alpha=0.3, color="C0", label="25-75%")
    ax.plot(result.t, result.percentiles[50], color="C0", linewidth=2, label="Median")
    ax.set_xlabel("Time")
    ax.set_ylabel("Output")
    ax.set_title(f"Monte Carlo Fan Chart ({result.n_runs} runs)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fan_path = f"{base}_fan.png"
    fig.savefig(fan_path, dpi=150)
    plt.close(fig)
    print(f"[plot] → {fan_path}")

    # Terminal histogram
    terminal = result.trajectories[:, -1, 0]
    terminal = terminal[~np.isnan(terminal)]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(terminal, bins=min(80, max(20, result.n_runs // 50)),
            density=True, alpha=0.7, color="C0", edgecolor="white")
    ax.axvline(result.terminal_stats["mean"], color="red", linestyle="--",
               linewidth=1.5, label=f"Mean = {result.terminal_stats['mean']:.3f}")
    ax.axvline(result.terminal_stats["VaR_5"], color="orange", linestyle=":",
               linewidth=1.5, label=f"VaR 5% = {result.terminal_stats['VaR_5']:.3f}")
    ax.set_xlabel("Terminal Value")
    ax.set_ylabel("Density")
    ax.set_title("Terminal Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    hist_path = f"{base}_histogram.png"
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)
    print(f"[plot] → {hist_path}")


def _report_mc(result: MCResult, path: str):
    lines = [
        "# Monte Carlo Simulation Report\n",
        f"## Configuration",
        f"- Runs: {result.n_runs}",
        f"- Seed: {result.metadata.get('seed', 'None')}",
        f"- Converged: {result.converged} (at n={result.convergence_n})\n",
        f"## Terminal Statistics",
        f"- Mean: {result.terminal_stats['mean']:.6f}",
        f"- Std: {result.terminal_stats['std']:.6f}",
        f"- Skew: {result.terminal_stats['skew']:.4f}",
        f"- Kurtosis: {result.terminal_stats['kurtosis']:.4f}",
        f"- Min: {result.terminal_stats['min']:.6f}",
        f"- Max: {result.terminal_stats['max']:.6f}",
        f"- VaR 5%: {result.terminal_stats['VaR_5']:.6f}",
        f"- CVaR 5%: {result.terminal_stats['CVaR_5']:.6f}",
    ]
    Path(path).write_text("\n".join(lines))
    print(f"[report] → {path}")


# ═══════════════════════════════════════════════════════════════════════════
# MODE 3: Agent-Based Model (ABM)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Agent:
    agent_id: int
    agent_type: str
    state: dict
    neighbors: list = field(default_factory=list)


@dataclass
class ABMResult:
    n_agents: int
    t_steps: int
    macro_series: dict      # {metric_name: [values per step]}
    agent_snapshots: list   # sampled snapshots for analysis
    emergence_gap: Optional[float]
    metadata: dict = field(default_factory=dict)


def _build_topology(n: int, kind: str, rng: np.random.Generator,
                    k: int = 6, p_rewire: float = 0.1) -> dict:
    """Build adjacency list for agent interactions."""
    adj = {i: [] for i in range(n)}

    if kind == "complete":
        for i in range(n):
            adj[i] = [j for j in range(n) if j != i]
    elif kind == "grid":
        side = int(np.ceil(np.sqrt(n)))
        for i in range(n):
            r, c = divmod(i, side)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                j = nr * side + nc
                if 0 <= nr < side and 0 <= nc < side and j < n:
                    adj[i].append(j)
    elif kind == "small_world":
        # Watts-Strogatz
        for i in range(n):
            for j in range(1, k // 2 + 1):
                neighbor = (i + j) % n
                adj[i].append(neighbor)
                adj[neighbor].append(i)
        # Rewire
        for i in range(n):
            for j in range(len(adj[i])):
                if rng.random() < p_rewire:
                    new_target = rng.integers(0, n)
                    while new_target == i or new_target in adj[i]:
                        new_target = rng.integers(0, n)
                    adj[i][j] = new_target
    elif kind == "scale_free":
        # Barabási-Albert (m=3)
        m = min(3, n - 1)
        for i in range(m):
            for j in range(i + 1, m):
                adj[i].append(j)
                adj[j].append(i)
        degrees = np.array([len(adj[i]) for i in range(n)])
        for i in range(m, n):
            probs = degrees[:i] / degrees[:i].sum()
            targets = rng.choice(i, size=m, replace=False, p=probs)
            for t in targets:
                adj[i].append(int(t))
                adj[int(t)].append(i)
            degrees[i] = m
            for t in targets:
                degrees[int(t)] += 1

    # Deduplicate
    for i in adj:
        adj[i] = list(set(adj[i]))
    return adj


def run_abm(args):
    """Entry point for Agent-Based Model."""
    config = json.loads(Path(args.config).read_text())
    rng = _seed_rng(args.seed)
    n = args.n_agents
    t_steps = args.t_steps

    # Build topology
    topo_params = config.get("interaction", {})
    adj = _build_topology(n, args.topology,
                          rng, k=topo_params.get("k", 6),
                          p_rewire=topo_params.get("p_rewire", 0.1))

    # Initialize agents
    agents = []
    agent_types = config.get("agent_types", [{"name": "default", "fraction": 1.0, "state": {}, "rules": []}])
    type_counts = []
    for atype in agent_types:
        count = int(atype["fraction"] * n)
        type_counts.append(count)
    # Fix rounding
    type_counts[-1] = n - sum(type_counts[:-1])

    idx = 0
    for atype, count in zip(agent_types, type_counts):
        for _ in range(count):
            state = {}
            for k, spec in atype.get("state", {}).items():
                if isinstance(spec, dict) and "dist" in spec:
                    state[k] = float(_sample_distribution(spec, rng, 1)[0])
                else:
                    state[k] = spec
            agents.append(Agent(
                agent_id=idx,
                agent_type=atype["name"],
                state=state,
                neighbors=adj.get(idx, [])
            ))
            idx += 1

    # Macro tracking
    macro = {"mean_state": [], "std_state": [], "active_fraction": []}

    # Simple rule evaluation (safe subset)
    def _eval_rule(rule: dict, agent: Agent, neighbor_states: list):
        """Evaluate a simple rule. Returns True if trigger fires."""
        trigger = rule.get("trigger", "True")
        # Simple expression evaluation with agent state
        local_ns = {**agent.state, "n_neighbors": len(agent.neighbors)}
        if neighbor_states:
            local_ns["neighbor_mean"] = np.mean([
                s.get(list(agent.state.keys())[0] if agent.state else "x", 0)
                for s in neighbor_states
            ]) if neighbor_states else 0
        try:
            # SECURITY: eval with restricted builtins for ABM rule triggers.
            return bool(eval(trigger, {"__builtins__": {}}, local_ns))  # noqa: S307
        except Exception:
            return False

    def _apply_action(rule: dict, agent: Agent, rng: np.random.Generator):
        """Apply rule action to agent state."""
        action = rule.get("action", "")
        params = rule.get("params", {})
        if action == "increment":
            key = params.get("key", list(agent.state.keys())[0] if agent.state else None)
            if key:
                agent.state[key] = agent.state.get(key, 0) + params.get("amount", 1)
        elif action == "decrement":
            key = params.get("key", list(agent.state.keys())[0] if agent.state else None)
            if key:
                agent.state[key] = agent.state.get(key, 0) - params.get("amount", 1)
        elif action == "adopt_neighbor_mean":
            key = params.get("key", list(agent.state.keys())[0] if agent.state else None)
            if key and agent.neighbors:
                n_vals = [agents[ni].state.get(key, 0) for ni in agent.neighbors if ni < len(agents)]
                if n_vals:
                    agent.state[key] = np.mean(n_vals)
        elif action == "random_walk":
            key = params.get("key", list(agent.state.keys())[0] if agent.state else None)
            scale = params.get("scale", 0.1)
            if key:
                agent.state[key] = agent.state.get(key, 0) + rng.normal(0, scale)

    # Main simulation loop
    snapshots = []
    for t in range(t_steps):
        # Collect macro stats
        if agents[0].state:
            first_key = list(agents[0].state.keys())[0]
            vals = [a.state.get(first_key, 0) for a in agents]
            macro["mean_state"].append(float(np.mean(vals)))
            macro["std_state"].append(float(np.std(vals)))
            macro["active_fraction"].append(float(np.mean([1 for v in vals if v > 0]) / n))

        # Apply rules
        for atype in agent_types:
            for agent in agents:
                if agent.agent_type != atype["name"]:
                    continue
                neighbor_states = [agents[ni].state for ni in agent.neighbors if ni < len(agents)]
                for rule in atype.get("rules", []):
                    if _eval_rule(rule, agent, neighbor_states):
                        _apply_action(rule, agent, rng)

        # Snapshot every 10%
        if t % max(1, t_steps // 10) == 0:
            snapshots.append({
                "t": t,
                "sample": [{"id": a.agent_id, "type": a.agent_type, "state": dict(a.state)}
                           for a in agents[:min(20, n)]]
            })

        if args.verbose and (t + 1) % max(1, t_steps // 10) == 0:
            print(f"[ABM] Step {t + 1}/{t_steps}")

    result = ABMResult(
        n_agents=n, t_steps=t_steps,
        macro_series=macro, agent_snapshots=snapshots,
        emergence_gap=None,
        metadata={"topology": args.topology, "seed": args.seed,
                  "agent_types": [a["name"] for a in agent_types]}
    )

    output = {
        "mode": "abm",
        "n_agents": n,
        "t_steps": t_steps,
        "macro_series": macro,
        "snapshots_count": len(snapshots),
        "metadata": result.metadata,
    }
    _save_json(output, args.output)

    if args.plot:
        _plot_abm(result, args.output)

    print(f"[ABM] {n} agents | {t_steps} steps | Topology: {args.topology}")

    if args.report:
        _report_abm(result, args.report)

    return result


def _plot_abm(result: ABMResult, output_base: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = Path(output_base).with_suffix("")

    if result.macro_series.get("mean_state"):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        steps = range(len(result.macro_series["mean_state"]))
        mean = np.array(result.macro_series["mean_state"])
        std = np.array(result.macro_series["std_state"])

        ax1.plot(steps, mean, color="C0", linewidth=1.5, label="Mean")
        ax1.fill_between(steps, mean - std, mean + std, alpha=0.2, color="C0", label="±1σ")
        ax1.set_ylabel("State Value")
        ax1.set_title(f"ABM Macro Dynamics ({result.n_agents} agents)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(steps, result.macro_series["active_fraction"], color="C1", linewidth=1.5)
        ax2.set_ylabel("Active Fraction")
        ax2.set_xlabel("Time Step")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        macro_path = f"{base}_macro.png"
        fig.savefig(macro_path, dpi=150)
        plt.close(fig)
        print(f"[plot] → {macro_path}")


def _report_abm(result: ABMResult, path: str):
    lines = [
        "# Agent-Based Model Report\n",
        f"## Configuration",
        f"- Agents: {result.n_agents}",
        f"- Steps: {result.t_steps}",
        f"- Topology: {result.metadata.get('topology', 'N/A')}",
        f"- Agent types: {', '.join(result.metadata.get('agent_types', []))}\n",
        f"## Macro Statistics (final)",
    ]
    if result.macro_series.get("mean_state"):
        lines.append(f"- Final mean: {result.macro_series['mean_state'][-1]:.6f}")
        lines.append(f"- Final std: {result.macro_series['std_state'][-1]:.6f}")

    Path(path).write_text("\n".join(lines))
    print(f"[report] → {path}")


# ═══════════════════════════════════════════════════════════════════════════
# MODE 4: Discrete-Event Simulation (DES)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DESResult:
    t_end: float
    events_processed: int
    queue_stats: dict       # {queue_name: {mean_length, max_length, mean_wait}}
    utilization: dict       # {server_name: fraction_busy}
    throughput: float
    event_log: list
    metadata: dict = field(default_factory=dict)


def run_des(args):
    """Entry point for Discrete-Event Simulation."""
    config = json.loads(Path(args.config).read_text())
    rng = _seed_rng(args.seed)
    t_end = args.t_end

    # Simple single-queue / multi-server DES
    n_servers = config.get("n_servers", 1)
    arrival_dist = config.get("arrival", {"dist": "exponential", "scale": 1.0})
    service_dist = config.get("service", {"dist": "exponential", "scale": 0.8})

    # Event types: ARRIVAL, DEPARTURE
    import heapq
    events = []  # (time, event_type, metadata)
    heapq.heappush(events, (float(_sample_distribution(arrival_dist, rng, 1)[0]), "ARRIVAL", {}))

    queue = []          # waiting customers
    servers = [None] * n_servers  # None = free, else departure_time
    wait_times = []
    queue_lengths = []
    busy_time = np.zeros(n_servers)
    event_log = []
    n_processed = 0

    while events and events[0][0] <= t_end:
        t, etype, meta = heapq.heappop(events)

        if etype == "ARRIVAL":
            # Schedule next arrival
            next_arrival = t + float(_sample_distribution(arrival_dist, rng, 1)[0])
            if next_arrival <= t_end:
                heapq.heappush(events, (next_arrival, "ARRIVAL", {}))

            # Try to assign to a free server
            assigned = False
            for s in range(n_servers):
                if servers[s] is None:
                    service_time = float(_sample_distribution(service_dist, rng, 1)[0])
                    servers[s] = t + service_time
                    busy_time[s] += service_time
                    heapq.heappush(events, (t + service_time, "DEPARTURE", {"server": s}))
                    wait_times.append(0.0)
                    assigned = True
                    break
            if not assigned:
                queue.append(t)  # arrival time for wait tracking

            queue_lengths.append(len(queue))
            event_log.append({"t": t, "type": "ARRIVAL", "queue_len": len(queue)})

        elif etype == "DEPARTURE":
            s = meta["server"]
            n_processed += 1

            if queue:
                arrival_t = queue.pop(0)
                wait_times.append(t - arrival_t)
                service_time = float(_sample_distribution(service_dist, rng, 1)[0])
                servers[s] = t + service_time
                busy_time[s] += service_time
                heapq.heappush(events, (t + service_time, "DEPARTURE", {"server": s}))
            else:
                servers[s] = None

            queue_lengths.append(len(queue))
            event_log.append({"t": t, "type": "DEPARTURE", "queue_len": len(queue)})

    q_stats = {
        "main": {
            "mean_length": float(np.mean(queue_lengths)) if queue_lengths else 0,
            "max_length": int(np.max(queue_lengths)) if queue_lengths else 0,
            "mean_wait": float(np.mean(wait_times)) if wait_times else 0,
            "max_wait": float(np.max(wait_times)) if wait_times else 0,
            "p95_wait": float(np.percentile(wait_times, 95)) if wait_times else 0,
        }
    }
    util = {}
    for s in range(n_servers):
        util[f"server_{s}"] = float(busy_time[s] / t_end) if t_end > 0 else 0

    result = DESResult(
        t_end=t_end, events_processed=n_processed,
        queue_stats=q_stats, utilization=util,
        throughput=n_processed / t_end if t_end > 0 else 0,
        event_log=event_log[-1000:],  # keep last 1000
        metadata={"n_servers": n_servers, "seed": args.seed}
    )

    output = {
        "mode": "des",
        "events_processed": n_processed,
        "throughput": result.throughput,
        "queue_stats": q_stats,
        "utilization": util,
        "metadata": result.metadata,
    }
    _save_json(output, args.output)

    if args.plot:
        _plot_des(result, args.output)

    print(f"[DES] Processed: {n_processed} | Throughput: {result.throughput:.2f}/t | "
          f"Mean wait: {q_stats['main']['mean_wait']:.3f}")

    if args.report:
        _report_des(result, args.report)

    return result


def _plot_des(result: DESResult, output_base: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = Path(output_base).with_suffix("")

    # Queue length over time
    times = [e["t"] for e in result.event_log]
    q_lens = [e["queue_len"] for e in result.event_log]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
    ax1.step(times, q_lens, where="post", linewidth=0.8, color="C0")
    ax1.set_ylabel("Queue Length")
    ax1.set_title("Discrete-Event Simulation — Queue Dynamics")
    ax1.grid(True, alpha=0.3)

    # Utilization bar
    servers = list(result.utilization.keys())
    utils = [result.utilization[s] for s in servers]
    colors = ["C2" if u < 0.8 else "C1" if u < 0.95 else "C3" for u in utils]
    ax2.bar(servers, utils, color=colors, edgecolor="white")
    ax2.set_ylabel("Utilization")
    ax2.set_title("Server Utilization")
    ax2.axhline(0.8, color="gray", linestyle="--", alpha=0.5)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    des_path = f"{base}_queue.png"
    fig.savefig(des_path, dpi=150)
    plt.close(fig)
    print(f"[plot] → {des_path}")


def _report_des(result: DESResult, path: str):
    lines = [
        "# Discrete-Event Simulation Report\n",
        f"## Configuration",
        f"- Servers: {result.metadata.get('n_servers', 'N/A')}",
        f"- Duration: {result.t_end}\n",
        f"## Results",
        f"- Events processed: {result.events_processed}",
        f"- Throughput: {result.throughput:.4f}/t",
        f"- Mean queue length: {result.queue_stats['main']['mean_length']:.2f}",
        f"- Max queue length: {result.queue_stats['main']['max_length']}",
        f"- Mean wait: {result.queue_stats['main']['mean_wait']:.4f}",
        f"- P95 wait: {result.queue_stats['main']['p95_wait']:.4f}\n",
        f"## Utilization",
    ]
    for s, u in result.utilization.items():
        lines.append(f"- {s}: {u:.1%}")

    Path(path).write_text("\n".join(lines))
    print(f"[report] → {path}")


# ═══════════════════════════════════════════════════════════════════════════
# MODE 5: Sensitivity Analysis
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SensitivityResult:
    method: str
    param_names: list
    indices: dict        # {param: {S1, ST, mu_star (Morris)}}
    rankings: list       # sorted by importance
    n_samples: int
    metadata: dict = field(default_factory=dict)


def run_sensitivity(args):
    """Entry point for Sensitivity Analysis."""
    param_ranges = json.loads(args.param_ranges)
    method = args.method
    n_samples = args.n_samples

    param_names = list(param_ranges.keys())
    bounds = [param_ranges[p] for p in param_names]

    problem = {
        "num_vars": len(param_names),
        "names": param_names,
        "bounds": bounds,
    }

    # If model_func is provided as a Python expression, build it
    # For CLI use, we expect a simple polynomial or expression
    model_expr = args.model_func

    def model_fn(X):
        """Evaluate model for parameter matrix X (n_samples x n_params)."""
        results = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            local_ns = {name: X[i, j] for j, name in enumerate(param_names)}
            local_ns["np"] = np
            try:
                # SECURITY: eval with restricted builtins for sensitivity model expressions.
                results[i] = eval(model_expr, {"__builtins__": {}, "np": np}, local_ns)  # noqa: S307
            except Exception as e:
                results[i] = np.nan
        return results

    try:
        from SALib.sample import saltelli, morris as morris_sample
        from SALib.analyze import sobol as sobol_analyze, morris as morris_analyze

        if method == "sobol":
            X = saltelli.sample(problem, n_samples, calc_second_order=False)
            Y = model_fn(X)
            Si = sobol_analyze.analyze(problem, Y, calc_second_order=False)
            indices = {}
            for i, name in enumerate(param_names):
                indices[name] = {
                    "S1": float(Si["S1"][i]),
                    "ST": float(Si["ST"][i]),
                    "S1_conf": float(Si["S1_conf"][i]),
                    "ST_conf": float(Si["ST_conf"][i]),
                }
            rankings = sorted(param_names, key=lambda p: indices[p]["ST"], reverse=True)

        elif method == "morris":
            X = morris_sample.sample(problem, n_samples, num_levels=4)
            Y = model_fn(X)
            Si = morris_analyze.analyze(problem, X, Y)
            indices = {}
            for i, name in enumerate(param_names):
                indices[name] = {
                    "mu_star": float(Si["mu_star"][i]),
                    "sigma": float(Si["sigma"][i]),
                    "mu_star_conf": float(Si["mu_star_conf"][i]),
                }
            rankings = sorted(param_names, key=lambda p: indices[p]["mu_star"], reverse=True)

        else:
            raise ValueError(f"Method {method} not supported. Use 'sobol' or 'morris'.")

    except ImportError:
        print("[sensitivity] SALib not installed. Running basic variance-based analysis.")
        # Fallback: simple one-at-a-time
        rng = _seed_rng(42)
        baseline = {name: np.mean(b) for name, b in zip(param_names, bounds)}
        # SECURITY: eval with restricted builtins for OAT sensitivity fallback.
        base_val = eval(model_expr, {"__builtins__": {}, "np": np}, {**baseline})  # noqa: S307

        indices = {}
        for i, name in enumerate(param_names):
            low_params = {**baseline, name: bounds[i][0]}
            high_params = {**baseline, name: bounds[i][1]}
            y_low = eval(model_expr, {"__builtins__": {}, "np": np}, {**low_params})  # noqa: S307
            y_high = eval(model_expr, {"__builtins__": {}, "np": np}, {**high_params})  # noqa: S307
            delta = abs(y_high - y_low)
            indices[name] = {"delta": float(delta), "low": float(y_low), "high": float(y_high)}
        rankings = sorted(param_names, key=lambda p: indices[p]["delta"], reverse=True)
        method = "one_at_a_time_fallback"

    result = SensitivityResult(
        method=method, param_names=param_names,
        indices=indices, rankings=rankings,
        n_samples=n_samples,
    )

    output = {
        "mode": "sensitivity",
        "method": method,
        "indices": indices,
        "rankings": rankings,
        "n_samples": n_samples,
    }
    _save_json(output, args.output)

    if args.plot:
        _plot_sensitivity(result, args.output)

    print(f"[Sensitivity] Method: {method} | Top param: {rankings[0] if rankings else 'N/A'}")

    if args.report:
        _report_sensitivity(result, args.report)

    return result


def _plot_sensitivity(result: SensitivityResult, output_base: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = Path(output_base).with_suffix("")

    if result.method == "sobol":
        fig, ax = plt.subplots(figsize=(8, max(4, len(result.param_names) * 0.5)))
        y_pos = np.arange(len(result.rankings))
        s1 = [result.indices[p]["S1"] for p in result.rankings]
        st = [result.indices[p]["ST"] for p in result.rankings]
        ax.barh(y_pos - 0.15, s1, 0.3, label="First-order (S1)", color="C0")
        ax.barh(y_pos + 0.15, st, 0.3, label="Total (ST)", color="C1")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(result.rankings)
        ax.set_xlabel("Sensitivity Index")
        ax.set_title("Sobol Sensitivity Indices")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="x")
        fig.tight_layout()
        sobol_path = f"{base}_sobol.png"
        fig.savefig(sobol_path, dpi=150)
        plt.close(fig)
        print(f"[plot] → {sobol_path}")

    elif result.method == "morris":
        fig, ax = plt.subplots(figsize=(7, 6))
        mu_star = [result.indices[p]["mu_star"] for p in result.param_names]
        sigma = [result.indices[p]["sigma"] for p in result.param_names]
        ax.scatter(mu_star, sigma, s=80, color="C0", edgecolors="black", zorder=5)
        for i, name in enumerate(result.param_names):
            ax.annotate(name, (mu_star[i], sigma[i]), textcoords="offset points",
                        xytext=(5, 5), fontsize=9)
        ax.set_xlabel("μ* (mean absolute elementary effect)")
        ax.set_ylabel("σ (standard deviation)")
        ax.set_title("Morris Screening")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        morris_path = f"{base}_morris.png"
        fig.savefig(morris_path, dpi=150)
        plt.close(fig)
        print(f"[plot] → {morris_path}")

    else:
        # Tornado for OAT fallback
        fig, ax = plt.subplots(figsize=(8, max(4, len(result.rankings) * 0.4)))
        y_pos = np.arange(len(result.rankings))
        deltas = [result.indices[p]["delta"] for p in result.rankings]
        ax.barh(y_pos, deltas, color="C0", edgecolor="white")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(result.rankings)
        ax.set_xlabel("Output Range (|high - low|)")
        ax.set_title("Tornado Diagram — Parameter Impact")
        ax.grid(True, alpha=0.3, axis="x")
        fig.tight_layout()
        tornado_path = f"{base}_tornado.png"
        fig.savefig(tornado_path, dpi=150)
        plt.close(fig)
        print(f"[plot] → {tornado_path}")


def _report_sensitivity(result: SensitivityResult, path: str):
    lines = [
        "# Sensitivity Analysis Report\n",
        f"## Configuration",
        f"- Method: {result.method}",
        f"- Samples: {result.n_samples}",
        f"- Parameters: {len(result.param_names)}\n",
        f"## Rankings (most → least influential)",
    ]
    for i, name in enumerate(result.rankings, 1):
        idx = result.indices[name]
        detail = " | ".join(f"{k}={v:.4f}" for k, v in idx.items())
        lines.append(f"{i}. **{name}** — {detail}")

    Path(path).write_text("\n".join(lines))
    print(f"[report] → {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Validation Bridge (feeds back to Phase 5)
# ═══════════════════════════════════════════════════════════════════════════

def generate_validation_bridge(sim_output_path: str, output_path: str):
    """Convert simulation output into Phase 5-ready validation data."""
    data = json.loads(Path(sim_output_path).read_text())
    mode = data.get("mode", "unknown")

    bridge = {
        "source": sim_output_path,
        "mode": mode,
        "phase5_inputs": {}
    }

    if mode == "sd":
        bridge["phase5_inputs"]["stability"] = {
            "is_stable": data.get("is_stable"),
            "eigenvalues_real": data.get("eigenvalues_real"),
            "settling_time": data.get("settling_time"),
        }
        bridge["phase5_inputs"]["trajectory_available"] = True

    elif mode == "mc":
        bridge["phase5_inputs"]["uncertainty_bounds"] = data.get("percentiles", {})
        bridge["phase5_inputs"]["terminal_stats"] = data.get("terminal_stats", {})
        bridge["phase5_inputs"]["convergence"] = {
            "converged": data.get("converged"),
            "at_n": data.get("convergence_n"),
        }

    elif mode == "sensitivity":
        bridge["phase5_inputs"]["parameter_rankings"] = data.get("rankings", [])
        bridge["phase5_inputs"]["sensitivity_indices"] = data.get("indices", {})

    elif mode == "des":
        bridge["phase5_inputs"]["throughput"] = data.get("throughput")
        bridge["phase5_inputs"]["queue_stats"] = data.get("queue_stats", {})
        bridge["phase5_inputs"]["utilization"] = data.get("utilization", {})

    elif mode == "abm":
        bridge["phase5_inputs"]["macro_series"] = data.get("macro_series", {})

    _save_json(bridge, output_path)
    return bridge


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def build_parser():
    parser = argparse.ArgumentParser(
        description="Epistemic Simulator — simulation engine for deconstructor models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    sub = parser.add_subparsers(dest="mode", help="Simulation mode")

    # Shared args
    def _add_common(p):
        p.add_argument("--output", default="sim_output.json", help="Output JSON path")
        p.add_argument("--plot", action="store_true", help="Generate plots")
        p.add_argument("--verbose", action="store_true", help="Verbose output")
        p.add_argument("--report", default=None, help="Generate markdown report at path")

    # SD
    sd = sub.add_parser("sd", help="System Dynamics")
    sd.add_argument("--model", required=True, help="Model JSON (A,B,C,D or ode_code)")
    sd.add_argument("--x0", required=True, help="Initial state JSON array")
    sd.add_argument("--t_end", type=float, required=True, help="Simulation end time")
    sd.add_argument("--dt", type=float, default=0.01, help="Time step")
    sd.add_argument("--u_func", default="step", choices=["step", "impulse", "sine", "ramp", "zero"])
    sd.add_argument("--amplitude", type=float, default=1.0)
    sd.add_argument("--freq", type=float, default=1.0)
    sd.add_argument("--t_on", type=float, default=0.0)
    sd.add_argument("--integrator", default="rk4", choices=["euler", "rk4", "rk45"])
    _add_common(sd)

    # MC
    mc = sub.add_parser("mc", help="Monte Carlo")
    mc.add_argument("--model", required=True, help="Model JSON")
    mc.add_argument("--param_distributions", required=True, help="Parameter distributions JSON")
    mc.add_argument("--x0", default="[0]", help="Initial state JSON")
    mc.add_argument("--n_runs", type=int, default=1000, help="Number of MC runs")
    mc.add_argument("--t_end", type=float, required=True, help="Simulation end time")
    mc.add_argument("--dt", type=float, default=0.1, help="Time step")
    mc.add_argument("--u_func", default="step", choices=["step", "impulse", "sine", "ramp", "zero"])
    mc.add_argument("--amplitude", type=float, default=1.0)
    mc.add_argument("--seed", type=int, default=None, help="Random seed")
    mc.add_argument("--convergence_check", action="store_true", help="Check MC convergence")
    _add_common(mc)

    # ABM
    abm = sub.add_parser("abm", help="Agent-Based Model")
    abm.add_argument("--config", required=True, help="ABM config JSON file")
    abm.add_argument("--n_agents", type=int, default=100, help="Number of agents")
    abm.add_argument("--t_steps", type=int, default=100, help="Number of time steps")
    abm.add_argument("--topology", default="small_world",
                     choices=["complete", "grid", "small_world", "scale_free"])
    abm.add_argument("--seed", type=int, default=None)
    _add_common(abm)

    # DES
    des = sub.add_parser("des", help="Discrete-Event Simulation")
    des.add_argument("--config", required=True, help="DES config JSON file")
    des.add_argument("--t_end", type=float, required=True, help="Simulation end time")
    des.add_argument("--seed", type=int, default=None)
    _add_common(des)

    # Sensitivity
    sens = sub.add_parser("sensitivity", help="Sensitivity Analysis")
    sens.add_argument("--model_func", required=True, help="Model expression (Python)")
    sens.add_argument("--param_ranges", required=True, help="Parameter ranges JSON")
    sens.add_argument("--method", default="sobol", choices=["sobol", "morris"])
    sens.add_argument("--n_samples", type=int, default=1024, help="Number of samples")
    _add_common(sens)

    # Validation bridge
    bridge = sub.add_parser("bridge", help="Generate Phase 5 validation bridge")
    bridge.add_argument("--sim_output", required=True, help="Simulation output JSON")
    bridge.add_argument("--output", default="validation_bridge.json")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.mode:
        parser.print_help()
        sys.exit(1)

    try:
        if args.mode == "sd":
            run_sd(args)
        elif args.mode == "mc":
            run_mc(args)
        elif args.mode == "abm":
            run_abm(args)
        elif args.mode == "des":
            run_des(args)
        elif args.mode == "sensitivity":
            run_sensitivity(args)
        elif args.mode == "bridge":
            generate_validation_bridge(args.sim_output, args.output)
        else:
            parser.print_help()
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"Error: File not found — {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
