#!/usr/bin/env python3
"""
parametric_identifier.py — Parametric System Identification Framework
=====================================================================
Fits ARX / ARMAX / NARMAX models to input-output data and recovers structure,
parameter estimates, uncertainty bounds, and residual diagnostics.

Complements forecast_modeler.py (which focuses on forecasting accuracy) by
targeting *system understanding* — the output is a model ready for forward
simulation via simulator.py and validation via Phase 5.

Usage:
    from parametric_identifier import fit_arx, fit_arx_grid, compare_structures

    result = fit_arx(y, u, na=2, nb=1, nk=1)
    print(result.format_report())

    # Grid search
    ranked = fit_arx_grid(y, u, na_range=range(1, 5), nb_range=range(1, 5))
    best = ranked[0]

    # Multi-family comparison
    comp = compare_structures(y, u, families=['arx', 'armax', 'narmax'])

Dependencies (graceful degradation):
    Required : numpy
    Optional : scipy        (chi2 p-values, coherence)
               statsmodels  (ARMAX via SARIMAX, Ljung-Box)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import warnings
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

# ---------------------------------------------------------------------------
# Optional imports — framework degrades gracefully
# ---------------------------------------------------------------------------
try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    _HAS_NUMPY = False

try:
    from scipy import stats as sp_stats
    from scipy import signal as sp_signal

    _HAS_SCIPY = True
except ImportError:
    sp_stats = None  # type: ignore
    sp_signal = None  # type: ignore
    _HAS_SCIPY = False

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.stats.diagnostic import acorr_ljungbox

    _HAS_STATSMODELS = True
except ImportError:
    SARIMAX = None  # type: ignore
    acorr_ljungbox = None  # type: ignore
    _HAS_STATSMODELS = False


# ===========================================================================
# 1. DATA STRUCTURES
# ===========================================================================


@dataclass
class FitResult:
    """Result of fitting a single parametric model."""

    model_type: str                       # 'ARX' | 'ARMAX' | 'NARMAX'
    structure: Dict[str, Any]             # e.g. {'na': 2, 'nb': 1, 'nk': 1}
    param_names: List[str]
    param_values: List[float]
    param_ci_lo: List[float]
    param_ci_hi: List[float]
    param_ci_method: str                  # 'bootstrap' | 'analytic' | 'none'
    residuals: List[float]
    rss: float
    n_samples: int
    n_params: int
    criteria: Dict[str, float]
    whiteness: Dict[str, Any]
    cv: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def to_simulator_format(self) -> Dict[str, Any]:
        """Return dict compatible with simulator.py ARX mode."""
        if self.model_type == "ARX":
            na = int(self.structure.get("na", 0))
            nb = int(self.structure.get("nb", 0))
            nk = int(self.structure.get("nk", 1))
            # In the regression y[t] = phi^T * theta, the y-lag coefficients
            # are -a_j where A(q) = 1 + a1 q^-1 + ... The simulator applies
            # y[i] += -a[j]*y[i-j-1], so a[j] is the polynomial coefficient.
            y_coeffs = self.param_values[:na]
            u_coeffs = self.param_values[na:na + nb]
            return {
                "type": "arx",
                "a": [-c for c in y_coeffs],
                "b": list(u_coeffs) if nb > 0 else [0.0],
                "nk": nk,
            }
        if self.model_type == "ARMAX":
            na = int(self.structure.get("p", 0))
            nb = int(self.structure.get("nb", 0))
            # ARMAX in simulator has no native form; approximate with ARX part.
            ar_coeffs = [
                v for n, v in zip(self.param_names, self.param_values)
                if n.startswith("ar")
            ]
            exog_coeffs = [
                v for n, v in zip(self.param_names, self.param_values)
                if n.startswith("exog")
            ]
            return {
                "type": "arx",
                "a": [-c for c in ar_coeffs],
                "b": list(exog_coeffs) if exog_coeffs else [0.0],
                "nk": int(self.structure.get("nk", 1)),
            }
        if self.model_type == "NARMAX":
            return {
                "type": "narmax",
                "terms": self.structure.get("terms", []),
                "coeffs": list(self.param_values),
                "na": int(self.structure.get("na", 0)),
                "nb": int(self.structure.get("nb", 0)),
                "degree": int(self.structure.get("degree", 1)),
                "nk": int(self.structure.get("nk", 1)),
            }
        raise ValueError(f"Unknown model_type: {self.model_type}")

    # ------------------------------------------------------------------
    def to_json(self, truncate_residuals: int = 200) -> Dict[str, Any]:
        """Serializable dict for --output FILE."""
        r = list(self.residuals)
        if len(r) > truncate_residuals:
            r = r[:truncate_residuals]
            r_note = f"(truncated to first {truncate_residuals} of {len(self.residuals)})"
        else:
            r_note = None
        out = {
            "model_type": self.model_type,
            "structure": self.structure,
            "param_names": self.param_names,
            "param_values": self.param_values,
            "param_ci_lo": self.param_ci_lo,
            "param_ci_hi": self.param_ci_hi,
            "param_ci_method": self.param_ci_method,
            "rss": self.rss,
            "n_samples": self.n_samples,
            "n_params": self.n_params,
            "criteria": self.criteria,
            "whiteness": self.whiteness,
            "cv": self.cv,
            "residuals": r,
            "metadata": self.metadata,
            "simulator_format": self.to_simulator_format(),
        }
        if r_note:
            out["residuals_note"] = r_note
        return out

    # ------------------------------------------------------------------
    def format_report(self) -> str:
        """Markdown report summarising fit."""
        lines: List[str] = []
        lines.append(f"# {self.model_type} Fit Report")
        lines.append("")
        lines.append(f"**Structure**: {self.structure}")
        lines.append(f"**Samples**: {self.n_samples}  **Params**: {self.n_params}")
        lines.append(f"**RSS**: {self.rss:.6g}")
        lines.append("")
        lines.append("## Parameters")
        lines.append("")
        lines.append("| Name | Estimate | CI Lower | CI Upper |")
        lines.append("|------|---------:|---------:|---------:|")
        for name, val, lo, hi in zip(
            self.param_names, self.param_values,
            self.param_ci_lo, self.param_ci_hi,
        ):
            lines.append(f"| {name} | {val:.6g} | {lo:.6g} | {hi:.6g} |")
        lines.append("")
        lines.append(f"*CI method*: {self.param_ci_method}")
        lines.append("")
        lines.append("## Information Criteria")
        lines.append("")
        for k, v in self.criteria.items():
            lines.append(f"- **{k}**: {v:.6g}")
        lines.append("")
        lines.append("## Whiteness (Ljung-Box)")
        lines.append("")
        w = self.whiteness
        lines.append(f"- Q = {w.get('Q', float('nan')):.4f}")
        lines.append(f"- p-value = {w.get('p_value', float('nan')):.4f}")
        lines.append(f"- lags = {w.get('lags', '?')}")
        lines.append(f"- passed (alpha=0.05) = {w.get('passed', '?')}")
        if self.cv:
            lines.append("")
            lines.append("## Walk-Forward Cross-Validation")
            lines.append("")
            lines.append(
                f"- R² mean = {self.cv.get('r2_mean', float('nan')):.4f}"
                f" (std {self.cv.get('r2_std', float('nan')):.4f})"
            )
            lines.append(f"- folds = {self.cv.get('k_folds', '?')}")
            lines.append(f"- R² per fold: {self.cv.get('r2_per_fold', [])}")
        if self.metadata:
            lines.append("")
            lines.append("## Metadata")
            lines.append("")
            for k, v in self.metadata.items():
                lines.append(f"- {k}: {v}")
        return "\n".join(lines)


@dataclass
class ComparisonEntry:
    """One row in a structure comparison table."""

    family: str
    structure_desc: str
    aic: float
    bic: float
    fpe: float
    r2_cv: Optional[float]
    whiteness_pass: bool
    n_params: int
    result: Optional[FitResult] = None


# ===========================================================================
# 2. CORE UTILITIES
# ===========================================================================


def _require_numpy() -> None:
    if not _HAS_NUMPY:
        raise RuntimeError(
            "parametric_identifier requires numpy. Install with: pip install numpy"
        )


def _as_1d_array(x: Sequence) -> "np.ndarray":
    """Coerce sequence to clean 1D float numpy array, replacing NaN/Inf with 0."""
    _require_numpy()
    arr = np.asarray(x, dtype=float).reshape(-1)
    mask = np.isnan(arr) | np.isinf(arr)
    if np.any(mask):
        warnings.warn(
            f"Found {int(mask.sum())} NaN/Inf values in input; replacing with 0.0",
            RuntimeWarning,
        )
        arr = arr.copy()
        arr[mask] = 0.0
    return arr


def compute_criteria(n: int, k: int, rss: float) -> Dict[str, float]:
    """Return AIC/BIC/AICc/FPE given sample size, param count, and RSS."""
    if n <= 0 or rss <= 0:
        return {"AIC": float("nan"), "BIC": float("nan"),
                "AICc": float("nan"), "FPE": float("nan")}
    aic = n * math.log(rss / n) + 2 * k
    bic = n * math.log(rss / n) + k * math.log(n)
    if n - k - 1 > 0:
        aicc = aic + 2 * k * (k + 1) / (n - k - 1)
    else:
        aicc = float("inf")
    if n - k > 0:
        fpe = (rss / n) * (n + k) / (n - k)
    else:
        fpe = float("inf")
    return {"AIC": aic, "BIC": bic, "AICc": aicc, "FPE": fpe}


def _chi2_sf(q: float, df: int) -> float:
    """Survival function for chi-square; uses scipy if available else W-H."""
    if q <= 0 or df <= 0:
        return 1.0
    if _HAS_SCIPY:
        return float(sp_stats.chi2.sf(q, df))
    # Wilson-Hilferty approximation: ((q/df)^(1/3) - (1 - 2/(9 df))) /
    # sqrt(2/(9 df)) ~ N(0, 1)
    h = 2.0 / (9.0 * df)
    z = ((q / df) ** (1.0 / 3.0) - (1.0 - h)) / math.sqrt(h)
    # Normal survival function via erfc
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def ljung_box(
    residuals: Sequence[float],
    lags: int = 20,
    alpha: float = 0.05,
) -> Tuple[float, float, bool]:
    """
    Ljung-Box Q-test for residual whiteness.

    Returns (Q_stat, p_value, passed_at_05). Passed means residuals are
    consistent with white noise at the specified alpha.
    """
    _require_numpy()
    r = _as_1d_array(residuals)
    n = r.size
    if n < lags + 2:
        return (float("nan"), float("nan"), False)

    # Prefer statsmodels implementation for exactness.
    if _HAS_STATSMODELS:
        try:
            lb = acorr_ljungbox(r, lags=[lags], return_df=True)
            Q = float(lb["lb_stat"].iloc[0])
            p = float(lb["lb_pvalue"].iloc[0])
            return (Q, p, p >= alpha)
        except Exception:
            pass

    # Fallback: compute autocorrelations directly.
    rc = r - r.mean()
    denom = float(np.dot(rc, rc))
    if denom <= 0:
        return (0.0, 1.0, True)
    acf = np.zeros(lags + 1)
    acf[0] = 1.0
    for k in range(1, lags + 1):
        num = float(np.dot(rc[k:], rc[:-k]))
        acf[k] = num / denom
    Q = 0.0
    for k in range(1, lags + 1):
        if n - k > 0:
            Q += (acf[k] ** 2) / (n - k)
    Q = n * (n + 2) * Q
    p = _chi2_sf(Q, lags)
    return (float(Q), float(p), bool(p >= alpha))


# ===========================================================================
# 3. ARX: REGRESSOR CONSTRUCTION AND OLS
# ===========================================================================


def build_arx_regressors(
    y: Sequence[float],
    u: Optional[Sequence[float]],
    na: int,
    nb: int,
    nk: int = 1,
) -> Tuple["np.ndarray", "np.ndarray"]:
    """
    Build ARX regressor matrix.

    Model: y[t] = -a1*y[t-1] - ... - a_na*y[t-na]
                 + b0*u[t-nk] + ... + b_{nb-1}*u[t-nk-nb+1] + e[t]

    Regressor row for sample t (t >= n_start):
        [y[t-1], y[t-2], ..., y[t-na], u[t-nk], u[t-nk-1], ..., u[t-nk-nb+1]]

    Fitted coefficients correspond to [-a1, ..., -a_na, b0, ..., b_{nb-1}].
    Caller converts back to a-polynomial form if needed.

    If u is None, nb must be 0; the model is pure AR(na).
    """
    _require_numpy()
    if na < 0 or nb < 0 or nk < 1:
        raise ValueError("na>=0, nb>=0, nk>=1 required")
    if u is None and nb > 0:
        raise ValueError("nb>0 requires input signal u")
    y = _as_1d_array(y)
    if u is not None:
        u = _as_1d_array(u)
        if u.size != y.size:
            raise ValueError(f"y length {y.size} != u length {u.size}")
    N = y.size
    # Earliest index t such that all regressors are defined:
    #   need t-na >= 0 and t-nk-nb+1 >= 0
    n_start = max(na, nk + nb - 1) if nb > 0 else na
    n_start = max(n_start, 1)
    if n_start >= N:
        raise ValueError(
            f"Data too short: need > {n_start} samples for na={na}, nb={nb}, nk={nk}"
        )
    n_rows = N - n_start
    n_cols = na + nb
    if n_cols == 0:
        raise ValueError("na + nb must be > 0")
    Phi = np.zeros((n_rows, n_cols), dtype=float)
    y_out = y[n_start:].copy()
    for i, t in enumerate(range(n_start, N)):
        for j in range(na):
            Phi[i, j] = y[t - 1 - j]
        for j in range(nb):
            Phi[i, na + j] = u[t - nk - j]
    return Phi, y_out


def fit_arx_ols(
    Phi: "np.ndarray",
    y: "np.ndarray",
) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    """OLS fit via QR decomposition. Returns (theta, residuals, covariance)."""
    _require_numpy()
    Phi = np.asarray(Phi, dtype=float)
    y = np.asarray(y, dtype=float)
    if Phi.shape[0] < Phi.shape[1]:
        raise ValueError(
            f"Underdetermined system: {Phi.shape[0]} samples, {Phi.shape[1]} params"
        )
    try:
        Q, R = np.linalg.qr(Phi)
        # Check conditioning.
        diag = np.abs(np.diag(R))
        if diag.min() < 1e-12 * max(diag.max(), 1.0):
            raise np.linalg.LinAlgError("Regressor matrix effectively singular")
        theta = np.linalg.solve(R, Q.T @ y)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            f"Singular regressor matrix — try reducing na/nb. ({exc})"
        ) from exc
    residuals = y - Phi @ theta
    n, k = Phi.shape
    dof = max(n - k, 1)
    sigma2 = float(np.dot(residuals, residuals)) / dof
    try:
        inv_xtx = np.linalg.inv(Phi.T @ Phi)
        cov = sigma2 * inv_xtx
    except np.linalg.LinAlgError:
        cov = np.full((k, k), float("nan"))
    return theta, residuals, cov


def _arx_param_names(na: int, nb: int) -> List[str]:
    names = [f"a{j+1}" for j in range(na)]
    names += [f"b{j}" for j in range(nb)]
    return names


def analytic_ci_from_cov(
    theta: Sequence[float],
    cov: "np.ndarray",
    alpha: float = 0.05,
) -> Tuple[List[float], List[float]]:
    """Symmetric normal-approximation CI from covariance matrix."""
    _require_numpy()
    theta = np.asarray(theta, dtype=float)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    # Use 1.96 as default if scipy missing; scipy gives exact quantile.
    if _HAS_SCIPY:
        z = float(sp_stats.norm.ppf(1.0 - alpha / 2.0))
    else:
        z = 1.959963984540054 if abs(alpha - 0.05) < 1e-9 else 1.96
    lo = (theta - z * se).tolist()
    hi = (theta + z * se).tolist()
    return lo, hi


def fit_arx(
    y: Sequence[float],
    u: Optional[Sequence[float]],
    na: int,
    nb: int,
    nk: int = 1,
    *,
    bootstrap: int = 0,
    cv_folds: int = 0,
    alpha: float = 0.05,
    rng: Optional["np.random.Generator"] = None,
) -> FitResult:
    """Fit ARX(na, nb, nk) by OLS and return a full FitResult."""
    _require_numpy()
    t0 = time.time()
    Phi, y_trim = build_arx_regressors(y, u, na, nb, nk)
    theta, residuals, cov = fit_arx_ols(Phi, y_trim)

    names = _arx_param_names(na, nb)
    rss = float(np.dot(residuals, residuals))
    n = int(Phi.shape[0])
    k = int(Phi.shape[1])
    criteria = compute_criteria(n, k, rss)

    # Whiteness test on residuals.
    lags = min(20, max(1, n // 5))
    Q_stat, p_val, passed = ljung_box(residuals, lags=lags)

    # Uncertainty quantification.
    ci_method = "analytic"
    if bootstrap and bootstrap > 0:
        try:
            lo, hi, _samples = residual_bootstrap(
                lambda yy, uu: fit_arx(
                    yy, uu, na, nb, nk, bootstrap=0, cv_folds=0
                ),
                y=_as_1d_array(y),
                u=_as_1d_array(u) if u is not None else None,
                theta_hat=theta,
                residuals=residuals,
                structure={"na": na, "nb": nb, "nk": nk},
                n_boot=bootstrap,
                alpha=alpha,
                rng=rng,
                model_kind="ARX",
            )
            ci_method = "bootstrap"
        except Exception as exc:
            warnings.warn(f"Bootstrap failed ({exc}); falling back to analytic CI")
            lo, hi = analytic_ci_from_cov(theta, cov, alpha=alpha)
            ci_method = "analytic"
    else:
        lo, hi = analytic_ci_from_cov(theta, cov, alpha=alpha)

    # Cross-validation.
    cv_out: Optional[Dict[str, Any]] = None
    if cv_folds and cv_folds >= 2:
        try:
            cv_out = walk_forward_cv(
                lambda yy, uu: fit_arx(
                    yy, uu, na, nb, nk, bootstrap=0, cv_folds=0
                ),
                y=_as_1d_array(y),
                u=_as_1d_array(u) if u is not None else None,
                k_folds=cv_folds,
                model_kind="ARX",
                structure={"na": na, "nb": nb, "nk": nk},
            )
        except Exception as exc:
            warnings.warn(f"CV failed: {exc}")

    fit_time = time.time() - t0
    result = FitResult(
        model_type="ARX",
        structure={"na": na, "nb": nb, "nk": nk},
        param_names=names,
        param_values=theta.tolist(),
        param_ci_lo=lo,
        param_ci_hi=hi,
        param_ci_method=ci_method,
        residuals=residuals.tolist(),
        rss=rss,
        n_samples=n,
        n_params=k,
        criteria=criteria,
        whiteness={
            "Q": Q_stat, "p_value": p_val, "passed": passed, "lags": lags,
        },
        cv=cv_out,
        metadata={
            "fit_time_sec": round(fit_time, 4),
            "library": "numpy-qr",
            "has_input": u is not None,
        },
    )
    return result


def fit_arx_grid(
    y: Sequence[float],
    u: Optional[Sequence[float]],
    na_range: Sequence[int],
    nb_range: Sequence[int],
    nk_range: Sequence[int] = (1,),
    *,
    criterion: str = "bic",
    cv_folds: int = 0,
) -> List[FitResult]:
    """Grid search over ARX structures; returns results sorted by criterion."""
    _require_numpy()
    criterion = criterion.upper()
    if criterion not in ("AIC", "BIC", "AICC", "FPE"):
        raise ValueError(f"Unknown criterion: {criterion}")
    key = "AICc" if criterion == "AICC" else criterion

    results: List[FitResult] = []
    for na in na_range:
        for nb in nb_range:
            for nk in nk_range:
                if na == 0 and nb == 0:
                    continue
                if nb > 0 and u is None:
                    continue
                try:
                    r = fit_arx(y, u, na, nb, nk, cv_folds=cv_folds)
                    results.append(r)
                except Exception as exc:
                    warnings.warn(f"Skipping ARX({na},{nb},{nk}): {exc}")
    results.sort(key=lambda r: r.criteria.get(key, float("inf")))
    return results


# ===========================================================================
# 4. ARMAX via SARIMAX
# ===========================================================================


def fit_armax(
    y: Sequence[float],
    u: Optional[Sequence[float]],
    p: int,
    q: int,
    *,
    cv_folds: int = 0,
    alpha: float = 0.05,
    bootstrap: int = 0,
    rng: Optional["np.random.Generator"] = None,
) -> FitResult:
    """Fit ARMAX(p, q) using statsmodels SARIMAX backend."""
    _require_numpy()
    if not _HAS_STATSMODELS:
        raise RuntimeError(
            "fit_armax requires statsmodels. Install with: pip install statsmodels"
        )
    t0 = time.time()
    y_arr = _as_1d_array(y)
    u_arr = _as_1d_array(u) if u is not None else None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            y_arr,
            exog=u_arr,
            order=(p, 0, q),
            trend="n",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = model.fit(disp=False)

    params = np.asarray(res.params, dtype=float)
    param_names = list(res.param_names)
    try:
        cov = np.asarray(res.cov_params(), dtype=float)
    except Exception:
        cov = np.full((len(params), len(params)), float("nan"))
    residuals = np.asarray(res.resid, dtype=float)
    rss = float(np.dot(residuals, residuals))
    n = int(residuals.size)
    k = int(len(params))
    criteria = compute_criteria(n, k, rss)

    lags = min(20, max(1, n // 5))
    Q_stat, p_val, passed_w = ljung_box(residuals, lags=lags)

    # Normalize param names to stable prefixes.
    normalized: List[str] = []
    for name in param_names:
        ln = name.lower()
        if ln.startswith("ar.l") or ln.startswith("ar."):
            normalized.append(f"ar{ln.split('l')[-1] if 'l' in ln else ln.split('.')[-1]}")
        elif ln.startswith("ma.l") or ln.startswith("ma."):
            normalized.append(f"ma{ln.split('l')[-1] if 'l' in ln else ln.split('.')[-1]}")
        elif ln.startswith("exog") or ln == "x1" or ln.startswith("x"):
            normalized.append(f"exog_{ln}")
        elif ln.startswith("sigma"):
            normalized.append("sigma2")
        else:
            normalized.append(ln)

    ci_method = "analytic"
    if bootstrap and bootstrap > 0:
        try:
            lo, hi, _s = residual_bootstrap(
                lambda yy, uu: fit_armax(yy, uu, p, q, cv_folds=0, bootstrap=0),
                y=y_arr, u=u_arr,
                theta_hat=params, residuals=residuals,
                structure={"p": p, "q": q}, n_boot=bootstrap,
                alpha=alpha, rng=rng, model_kind="ARMAX",
            )
            ci_method = "bootstrap"
        except Exception as exc:
            warnings.warn(f"Bootstrap failed ({exc}); falling back to analytic")
            lo, hi = analytic_ci_from_cov(params, cov, alpha=alpha)
    else:
        lo, hi = analytic_ci_from_cov(params, cov, alpha=alpha)

    cv_out: Optional[Dict[str, Any]] = None
    if cv_folds and cv_folds >= 2:
        try:
            cv_out = walk_forward_cv(
                lambda yy, uu: fit_armax(yy, uu, p, q, cv_folds=0, bootstrap=0),
                y=y_arr, u=u_arr, k_folds=cv_folds,
                model_kind="ARMAX", structure={"p": p, "q": q},
            )
        except Exception as exc:
            warnings.warn(f"CV failed: {exc}")

    fit_time = time.time() - t0
    return FitResult(
        model_type="ARMAX",
        structure={"p": p, "q": q, "nk": 0},
        param_names=normalized,
        param_values=params.tolist(),
        param_ci_lo=lo,
        param_ci_hi=hi,
        param_ci_method=ci_method,
        residuals=residuals.tolist(),
        rss=rss,
        n_samples=n,
        n_params=k,
        criteria=criteria,
        whiteness={
            "Q": Q_stat, "p_value": p_val, "passed": passed_w, "lags": lags,
        },
        cv=cv_out,
        metadata={
            "fit_time_sec": round(fit_time, 4),
            "library": "statsmodels-SARIMAX",
            "converged": bool(getattr(res, "mle_retvals", {}).get("converged", True)),
            "has_input": u is not None,
        },
    )


def fit_armax_grid(
    y: Sequence[float],
    u: Optional[Sequence[float]],
    p_range: Sequence[int],
    q_range: Sequence[int],
    *,
    criterion: str = "bic",
    cv_folds: int = 0,
) -> List[FitResult]:
    """Grid search over ARMAX(p, q) structures."""
    if not _HAS_STATSMODELS:
        raise RuntimeError("fit_armax_grid requires statsmodels")
    key = "AICc" if criterion.upper() == "AICC" else criterion.upper()
    results: List[FitResult] = []
    for p in p_range:
        for q in q_range:
            if p == 0 and q == 0:
                continue
            try:
                r = fit_armax(y, u, p, q, cv_folds=cv_folds)
                results.append(r)
            except Exception as exc:
                warnings.warn(f"Skipping ARMAX({p},{q}): {exc}")
    results.sort(key=lambda r: r.criteria.get(key, float("inf")))
    return results


# ===========================================================================
# 5. NARMAX: polynomial basis + FROLS
# ===========================================================================


def polynomial_basis(
    y: Sequence[float],
    u: Optional[Sequence[float]],
    na: int,
    nb: int,
    degree: int,
    nk: int = 1,
) -> Tuple["np.ndarray", List[str], "np.ndarray"]:
    """
    Build polynomial NARMAX regressor matrix with all monomials up to *degree*.

    Base regressors: y[t-1], ..., y[t-na], u[t-nk], ..., u[t-nk-nb+1].
    Monomials: all unordered combinations with repetition of size 1..degree.
    Includes a constant term.

    Returns (Phi, labels, y_trim).
    """
    _require_numpy()
    if degree < 1:
        raise ValueError("degree must be >= 1")
    y = _as_1d_array(y)
    u_arr = _as_1d_array(u) if u is not None else None
    if u_arr is not None and u_arr.size != y.size:
        raise ValueError("y and u length mismatch")
    if u is None and nb > 0:
        raise ValueError("nb>0 requires input signal u")
    n_start = max(na, nk + nb - 1) if nb > 0 else max(na, 1)
    n_start = max(n_start, 1)
    N = y.size
    if n_start >= N:
        raise ValueError("Data too short for chosen na/nb/nk")
    n_rows = N - n_start

    # Base regressor labels.
    base_labels = [f"y[t-{j+1}]" for j in range(na)]
    base_labels += [f"u[t-{nk+j}]" for j in range(nb)]
    n_base = len(base_labels)
    if n_base == 0:
        raise ValueError("na+nb must be > 0")

    # Enumerate monomials as multisets of base indices.
    from itertools import combinations_with_replacement
    monomials: List[Tuple[int, ...]] = []
    labels: List[str] = ["1"]
    monomials.append(())
    for d in range(1, degree + 1):
        for combo in combinations_with_replacement(range(n_base), d):
            monomials.append(combo)
            labels.append("*".join(base_labels[i] for i in combo))

    n_cols = len(monomials)
    Phi = np.zeros((n_rows, n_cols), dtype=float)
    for i, t in enumerate(range(n_start, N)):
        base_vals = np.zeros(n_base)
        for j in range(na):
            base_vals[j] = y[t - 1 - j]
        for j in range(nb):
            base_vals[na + j] = u_arr[t - nk - j]
        for c_idx, combo in enumerate(monomials):
            if not combo:
                Phi[i, c_idx] = 1.0
            else:
                v = 1.0
                for idx in combo:
                    v *= base_vals[idx]
                Phi[i, c_idx] = v
    y_trim = y[n_start:].copy()
    return Phi, labels, y_trim


def frols(
    Phi: "np.ndarray",
    y: "np.ndarray",
    err_threshold: float = 0.99,
    max_terms: Optional[int] = None,
) -> Tuple[List[int], "np.ndarray", List[float]]:
    """
    Forward Regression Orthogonal Least Squares.

    Selects minimal terms from Phi explaining variance of y via the Error
    Reduction Ratio criterion. Returns (selected_indices, coefficients, errs).
    *coefficients* are in the ORIGINAL Phi basis (recovered via final OLS
    refit on the selected columns) — suitable for prediction.
    """
    _require_numpy()
    Phi = np.asarray(Phi, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    n, m = Phi.shape
    if n == 0 or m == 0:
        return [], np.zeros(0), []
    if max_terms is None:
        max_terms = m

    yy = float(np.dot(y, y))
    if yy <= 0:
        return [], np.zeros(0), []

    selected: List[int] = []
    errs: List[float] = []
    # Orthogonalized (w) columns for already-selected terms.
    W: List["np.ndarray"] = []
    # Track which columns are still candidates.
    remaining = set(range(m))
    total_err = 0.0

    while total_err < err_threshold and len(selected) < max_terms and remaining:
        best_err = -1.0
        best_idx: Optional[int] = None
        best_w: Optional["np.ndarray"] = None
        for j in remaining:
            col = Phi[:, j].copy()
            w = col.copy()
            for wk in W:
                ww = float(np.dot(wk, wk))
                if ww > 0:
                    w = w - (float(np.dot(wk, col)) / ww) * wk
            ww = float(np.dot(w, w))
            if ww < 1e-14:
                continue
            g = float(np.dot(w, y)) / ww
            err = (g * g * ww) / yy
            if err > best_err:
                best_err = err
                best_idx = j
                best_w = w
        if best_idx is None or best_w is None or best_err <= 0:
            break
        selected.append(best_idx)
        W.append(best_w)
        errs.append(best_err)
        total_err += best_err
        remaining.discard(best_idx)

    if not selected:
        return [], np.zeros(0), []

    # Refit selected columns on original basis by OLS for interpretability.
    Phi_s = Phi[:, selected]
    try:
        Q, R = np.linalg.qr(Phi_s)
        coeffs = np.linalg.solve(R, Q.T @ y)
    except np.linalg.LinAlgError:
        coeffs = np.linalg.lstsq(Phi_s, y, rcond=None)[0]
    return selected, coeffs, errs


def fit_narmax(
    y: Sequence[float],
    u: Optional[Sequence[float]],
    na: int,
    nb: int,
    degree: int = 2,
    nk: int = 1,
    *,
    err_threshold: float = 0.99,
    max_terms: Optional[int] = None,
    cv_folds: int = 0,
    alpha: float = 0.05,
    bootstrap: int = 0,
    rng: Optional["np.random.Generator"] = None,
) -> FitResult:
    """Fit polynomial NARMAX with FROLS term selection."""
    _require_numpy()
    t0 = time.time()
    Phi, labels, y_trim = polynomial_basis(y, u, na, nb, degree, nk)
    selected, coeffs, errs = frols(Phi, y_trim, err_threshold, max_terms)
    if not selected:
        raise ValueError("FROLS selected no terms — check data or thresholds")

    Phi_s = Phi[:, selected]
    residuals = y_trim - Phi_s @ coeffs
    rss = float(np.dot(residuals, residuals))
    n = int(Phi_s.shape[0])
    k = int(Phi_s.shape[1])
    criteria = compute_criteria(n, k, rss)

    # Analytic CI from OLS covariance on the *selected* basis.
    dof = max(n - k, 1)
    sigma2 = rss / dof
    try:
        cov = sigma2 * np.linalg.inv(Phi_s.T @ Phi_s)
    except np.linalg.LinAlgError:
        cov = np.full((k, k), float("nan"))
    lags = min(20, max(1, n // 5))
    Q_stat, p_val, passed_w = ljung_box(residuals, lags=lags)

    ci_method = "analytic"
    if bootstrap and bootstrap > 0:
        try:
            lo, hi, _ = residual_bootstrap(
                lambda yy, uu: fit_narmax(
                    yy, uu, na, nb, degree, nk,
                    err_threshold=1.0,
                    max_terms=k,  # force same size
                    cv_folds=0, bootstrap=0,
                ),
                y=_as_1d_array(y),
                u=_as_1d_array(u) if u is not None else None,
                theta_hat=coeffs, residuals=residuals,
                structure={"na": na, "nb": nb, "degree": degree,
                           "nk": nk, "terms": selected},
                n_boot=bootstrap, alpha=alpha, rng=rng,
                model_kind="NARMAX",
            )
            ci_method = "bootstrap"
        except Exception as exc:
            warnings.warn(f"Bootstrap failed ({exc}); falling back to analytic")
            lo, hi = analytic_ci_from_cov(coeffs, cov, alpha=alpha)
    else:
        lo, hi = analytic_ci_from_cov(coeffs, cov, alpha=alpha)

    cv_out: Optional[Dict[str, Any]] = None
    if cv_folds and cv_folds >= 2:
        try:
            cv_out = walk_forward_cv(
                lambda yy, uu: fit_narmax(
                    yy, uu, na, nb, degree, nk,
                    err_threshold=err_threshold, max_terms=max_terms,
                    cv_folds=0, bootstrap=0,
                ),
                y=_as_1d_array(y),
                u=_as_1d_array(u) if u is not None else None,
                k_folds=cv_folds,
                model_kind="NARMAX",
                structure={"na": na, "nb": nb, "degree": degree, "nk": nk},
            )
        except Exception as exc:
            warnings.warn(f"CV failed: {exc}")

    fit_time = time.time() - t0
    selected_labels = [labels[i] for i in selected]
    return FitResult(
        model_type="NARMAX",
        structure={
            "na": na, "nb": nb, "degree": degree, "nk": nk,
            "terms": selected_labels,
            "term_indices": list(selected),
            "err_contributions": [float(e) for e in errs],
        },
        param_names=selected_labels,
        param_values=[float(c) for c in coeffs],
        param_ci_lo=lo,
        param_ci_hi=hi,
        param_ci_method=ci_method,
        residuals=residuals.tolist(),
        rss=rss,
        n_samples=n,
        n_params=k,
        criteria=criteria,
        whiteness={
            "Q": Q_stat, "p_value": p_val, "passed": passed_w, "lags": lags,
        },
        cv=cv_out,
        metadata={
            "fit_time_sec": round(fit_time, 4),
            "library": "numpy-frols",
            "err_total": float(sum(errs)),
            "n_candidates": int(Phi.shape[1]),
            "has_input": u is not None,
        },
    )


# ===========================================================================
# 6. FORWARD SIMULATION (for bootstrap regeneration)
# ===========================================================================


def _simulate_arx(
    u: Optional["np.ndarray"],
    theta: "np.ndarray",
    na: int,
    nb: int,
    nk: int,
    n_samples: int,
    y_init: "np.ndarray",
    noise: "np.ndarray",
) -> "np.ndarray":
    """One-step-ahead simulate ARX model with injected noise sequence."""
    _require_numpy()
    y = np.zeros(n_samples)
    # Seed initial condition with y_init (first n_start samples).
    n_start = max(na, nk + nb - 1) if nb > 0 else max(na, 1)
    n_start = max(n_start, 1)
    for t in range(min(n_start, n_samples)):
        y[t] = y_init[t] if t < len(y_init) else 0.0
    for t in range(n_start, n_samples):
        val = 0.0
        for j in range(na):
            val += theta[j] * y[t - 1 - j]
        for j in range(nb):
            idx = t - nk - j
            if 0 <= idx < n_samples and u is not None:
                val += theta[na + j] * u[idx]
        # noise index is shifted by n_start (since residuals have n_samples - n_start rows)
        ni = t - n_start
        if 0 <= ni < noise.size:
            val += noise[ni]
        y[t] = val
    return y


def _simulate_narmax(
    u: Optional["np.ndarray"],
    coeffs: "np.ndarray",
    structure: Dict[str, Any],
    n_samples: int,
    y_init: "np.ndarray",
    noise: "np.ndarray",
    full_basis_labels: List[str],
    selected_indices: List[int],
) -> "np.ndarray":
    """Forward-simulate a NARMAX model. Kept simple; for bootstrap regen."""
    # Fall back to ARX-like simulation: re-evaluate polynomial basis at each t.
    _require_numpy()
    na = int(structure.get("na", 0))
    nb = int(structure.get("nb", 0))
    degree = int(structure.get("degree", 1))
    nk = int(structure.get("nk", 1))
    n_start = max(na, nk + nb - 1) if nb > 0 else max(na, 1)
    n_start = max(n_start, 1)
    y = np.zeros(n_samples)
    for t in range(min(n_start, n_samples)):
        y[t] = y_init[t] if t < len(y_init) else 0.0

    # Enumerate monomial structure identical to polynomial_basis.
    from itertools import combinations_with_replacement
    n_base = na + nb
    monomials: List[Tuple[int, ...]] = [()]
    for d in range(1, degree + 1):
        for combo in combinations_with_replacement(range(n_base), d):
            monomials.append(combo)
    for t in range(n_start, n_samples):
        base_vals = np.zeros(n_base)
        for j in range(na):
            base_vals[j] = y[t - 1 - j]
        for j in range(nb):
            if u is not None:
                base_vals[na + j] = u[t - nk - j]
        val = 0.0
        for c_idx, combo in enumerate(monomials):
            if c_idx not in selected_indices:
                continue
            v = 1.0
            for idx in combo:
                v *= base_vals[idx]
            # Find which position in selected_indices this is.
            pos = selected_indices.index(c_idx)
            val += coeffs[pos] * v
        ni = t - n_start
        if 0 <= ni < noise.size:
            val += noise[ni]
        y[t] = val
    return y


# ===========================================================================
# 7. BOOTSTRAP AND CROSS-VALIDATION
# ===========================================================================


def residual_bootstrap(
    fit_fn: Callable,
    y: "np.ndarray",
    u: Optional["np.ndarray"],
    theta_hat: "np.ndarray",
    residuals: "np.ndarray",
    structure: Dict[str, Any],
    *,
    n_boot: int = 500,
    alpha: float = 0.05,
    rng: Optional["np.random.Generator"] = None,
    model_kind: str = "ARX",
) -> Tuple[List[float], List[float], "np.ndarray"]:
    """
    Residual bootstrap for parameter confidence intervals (temporal data).

    Strategy: re-sample centered residuals with replacement, regenerate y by
    forward-simulating the fitted model through the original u plus the new
    noise, refit, collect parameter samples; return (lo, hi, samples).
    """
    _require_numpy()
    rng = rng or np.random.default_rng()
    r = np.asarray(residuals, dtype=float)
    r_centered = r - r.mean()
    y_arr = np.asarray(y, dtype=float)
    theta_hat = np.asarray(theta_hat, dtype=float)
    n_samples = y_arr.size
    samples: List["np.ndarray"] = []

    for _ in range(n_boot):
        # Resample residuals with replacement over the *residual* support.
        eps = rng.choice(r_centered, size=r_centered.size, replace=True)
        if model_kind == "ARX":
            na = int(structure["na"])
            nb = int(structure["nb"])
            nk = int(structure["nk"])
            y_boot = _simulate_arx(
                u=u, theta=theta_hat, na=na, nb=nb, nk=nk,
                n_samples=n_samples, y_init=y_arr, noise=eps,
            )
        elif model_kind == "ARMAX":
            # For ARMAX, regenerate via SARIMAX's simulate + exog.
            # Fall back: just add resampled noise to y_arr as approximation
            # — refitting on identical y would repeat original estimate.
            # Use SARIMAX simulate if available.
            if _HAS_STATSMODELS:
                try:
                    p = int(structure["p"])
                    q = int(structure["q"])
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        mod = SARIMAX(
                            y_arr, exog=u, order=(p, 0, q), trend="n",
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                        )
                        # Set params to theta_hat then simulate.
                        y_boot = mod.simulate(
                            params=theta_hat, nsimulations=n_samples,
                            measurement_shocks=eps[:n_samples] if eps.size >= n_samples
                            else np.concatenate([eps, np.zeros(n_samples - eps.size)]),
                            exog=u,
                        )
                        y_boot = np.asarray(y_boot, dtype=float)
                except Exception:
                    y_boot = y_arr + np.concatenate(
                        [eps, np.zeros(max(0, n_samples - eps.size))]
                    )[:n_samples]
            else:
                y_boot = y_arr.copy()
        elif model_kind == "NARMAX":
            # Simple: use ARX-like structural regeneration on polynomial basis.
            y_boot = y_arr.copy()
            y_boot[-eps.size:] = y_arr[-eps.size:] - r + eps
        else:
            raise ValueError(f"Unknown model_kind: {model_kind}")

        try:
            fit = fit_fn(y_boot, u)
            samples.append(np.asarray(fit.param_values, dtype=float))
        except Exception:
            continue

    if not samples:
        # Bootstrap failed entirely; return wide intervals.
        k = theta_hat.size
        return ([float("nan")] * k, [float("nan")] * k,
                np.zeros((0, k)))
    # Handle heterogeneous-length samples (NARMAX with variable terms).
    ref_len = theta_hat.size
    clean = [s for s in samples if s.size == ref_len]
    if not clean:
        k = theta_hat.size
        return ([float("nan")] * k, [float("nan")] * k,
                np.zeros((0, k)))
    S = np.vstack(clean)
    lo = np.percentile(S, 100.0 * (alpha / 2.0), axis=0).tolist()
    hi = np.percentile(S, 100.0 * (1 - alpha / 2.0), axis=0).tolist()
    return lo, hi, S


def walk_forward_cv(
    fit_fn: Callable,
    y: "np.ndarray",
    u: Optional["np.ndarray"],
    k_folds: int = 5,
    *,
    model_kind: str = "ARX",
    structure: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Expanding-window forward chaining cross-validation."""
    _require_numpy()
    y_arr = np.asarray(y, dtype=float)
    u_arr = np.asarray(u, dtype=float) if u is not None else None
    N = y_arr.size
    fold_size = N // (k_folds + 1)
    if fold_size < 5:
        raise ValueError(
            f"Data too short for {k_folds}-fold CV (need fold_size >= 5)"
        )

    r2_per_fold: List[float] = []
    rmse_per_fold: List[float] = []
    for i in range(k_folds):
        train_end = (i + 2) * fold_size
        test_start = train_end
        test_end = min(test_start + fold_size, N)
        if test_end - test_start < 3:
            break
        y_train = y_arr[:train_end]
        y_test = y_arr[test_start:test_end]
        u_train = u_arr[:train_end] if u_arr is not None else None
        u_test = u_arr[test_start:test_end] if u_arr is not None else None
        try:
            fit = fit_fn(y_train, u_train)
        except Exception as exc:
            warnings.warn(f"CV fold {i} skipped: {exc}")
            continue
        # Predict one-step-ahead on test segment using *final* training state.
        y_pred = _predict_one_step(
            fit, y_arr=y_arr, u_arr=u_arr,
            start=test_start, end=test_end, model_kind=model_kind,
        )
        if y_pred is None or y_pred.size != y_test.size:
            continue
        ss_res = float(np.sum((y_test - y_pred) ** 2))
        y_mean = float(np.mean(y_test))
        ss_tot = float(np.sum((y_test - y_mean) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        rmse = math.sqrt(ss_res / y_test.size)
        r2_per_fold.append(r2)
        rmse_per_fold.append(rmse)

    if not r2_per_fold:
        return {"r2_mean": float("nan"), "r2_std": float("nan"),
                "r2_per_fold": [], "rmse_per_fold": [], "k_folds": 0}
    return {
        "r2_mean": float(np.mean(r2_per_fold)),
        "r2_std": float(np.std(r2_per_fold)),
        "r2_per_fold": r2_per_fold,
        "rmse_per_fold": rmse_per_fold,
        "k_folds": len(r2_per_fold),
    }


def _predict_one_step(
    fit: FitResult,
    y_arr: "np.ndarray",
    u_arr: Optional["np.ndarray"],
    start: int,
    end: int,
    model_kind: str,
) -> Optional["np.ndarray"]:
    """One-step-ahead prediction for CV, using true history (teacher-forced)."""
    _require_numpy()
    n_out = end - start
    preds = np.zeros(n_out)
    if model_kind == "ARX":
        na = int(fit.structure["na"])
        nb = int(fit.structure["nb"])
        nk = int(fit.structure["nk"])
        theta = np.asarray(fit.param_values, dtype=float)
        for i, t in enumerate(range(start, end)):
            val = 0.0
            if t - 1 - (na - 1) < 0:
                return None
            for j in range(na):
                val += theta[j] * y_arr[t - 1 - j]
            for j in range(nb):
                idx = t - nk - j
                if idx < 0 or u_arr is None:
                    return None
                val += theta[na + j] * u_arr[idx]
            preds[i] = val
        return preds
    if model_kind == "ARMAX":
        # Use simple AR-part prediction (no noise-driven MA term, teacher forced).
        p = int(fit.structure["p"])
        q = int(fit.structure.get("q", 0))
        names = fit.param_names
        values = np.asarray(fit.param_values, dtype=float)
        ar_idx = [i for i, n in enumerate(names) if n.startswith("ar")]
        exog_idx = [i for i, n in enumerate(names) if n.startswith("exog")]
        ar_coefs = values[ar_idx] if ar_idx else np.zeros(0)
        exog_coefs = values[exog_idx] if exog_idx else np.zeros(0)
        for i, t in enumerate(range(start, end)):
            val = 0.0
            for j in range(min(p, ar_coefs.size)):
                if t - 1 - j < 0:
                    return None
                val += ar_coefs[j] * y_arr[t - 1 - j]
            if u_arr is not None and exog_coefs.size > 0:
                val += exog_coefs[0] * u_arr[t]
            preds[i] = val
        return preds
    if model_kind == "NARMAX":
        # For CV: evaluate selected polynomial terms on teacher-forced history.
        na = int(fit.structure["na"])
        nb = int(fit.structure["nb"])
        degree = int(fit.structure["degree"])
        nk = int(fit.structure["nk"])
        term_indices = fit.structure.get("term_indices", [])
        coeffs = np.asarray(fit.param_values, dtype=float)
        from itertools import combinations_with_replacement
        n_base = na + nb
        monomials: List[Tuple[int, ...]] = [()]
        for d in range(1, degree + 1):
            for combo in combinations_with_replacement(range(n_base), d):
                monomials.append(combo)
        for i, t in enumerate(range(start, end)):
            base = np.zeros(n_base)
            for j in range(na):
                if t - 1 - j < 0:
                    return None
                base[j] = y_arr[t - 1 - j]
            for j in range(nb):
                idx = t - nk - j
                if idx < 0 or u_arr is None:
                    return None
                base[na + j] = u_arr[idx]
            val = 0.0
            for pos, c_idx in enumerate(term_indices):
                combo = monomials[c_idx]
                v = 1.0
                for bi in combo:
                    v *= base[bi]
                val += coeffs[pos] * v
            preds[i] = val
        return preds
    return None


# ===========================================================================
# 8. COMPARE STRUCTURES
# ===========================================================================


def compare_structures(
    y: Sequence[float],
    u: Optional[Sequence[float]] = None,
    *,
    families: Sequence[str] = ("arx", "armax", "narmax"),
    criterion: str = "bic",
    na_max: int = 4,
    nb_max: int = 4,
    nk_range: Sequence[int] = (1,),
    narmax_degree: int = 2,
    cv_folds: int = 0,
) -> List[ComparisonEntry]:
    """
    Fit default grids across requested families and rank them.

    Winner = lowest criterion among models passing whiteness test. If none
    pass, winner is still lowest-criterion model but with whiteness_flag=False.
    """
    _require_numpy()
    key = criterion.upper()
    if key == "AICC":
        key = "AICc"
    entries: List[ComparisonEntry] = []

    for fam in families:
        fam = fam.lower()
        try:
            if fam == "arx":
                ranked = fit_arx_grid(
                    y, u,
                    na_range=range(1, na_max + 1),
                    nb_range=range(0 if u is None else 1, nb_max + 1),
                    nk_range=nk_range,
                    criterion=criterion,
                    cv_folds=cv_folds,
                )
                for r in ranked:
                    desc = f"ARX({r.structure['na']},{r.structure['nb']},{r.structure['nk']})"
                    entries.append(ComparisonEntry(
                        family="ARX",
                        structure_desc=desc,
                        aic=r.criteria.get("AIC", float("nan")),
                        bic=r.criteria.get("BIC", float("nan")),
                        fpe=r.criteria.get("FPE", float("nan")),
                        r2_cv=r.cv.get("r2_mean") if r.cv else None,
                        whiteness_pass=bool(r.whiteness.get("passed", False)),
                        n_params=r.n_params,
                        result=r,
                    ))
            elif fam == "armax":
                if not _HAS_STATSMODELS:
                    warnings.warn("ARMAX skipped: statsmodels not available")
                    continue
                ranked = fit_armax_grid(
                    y, u,
                    p_range=range(1, na_max + 1),
                    q_range=range(0, nb_max + 1),
                    criterion=criterion,
                    cv_folds=cv_folds,
                )
                for r in ranked:
                    desc = f"ARMAX({r.structure['p']},{r.structure['q']})"
                    entries.append(ComparisonEntry(
                        family="ARMAX",
                        structure_desc=desc,
                        aic=r.criteria.get("AIC", float("nan")),
                        bic=r.criteria.get("BIC", float("nan")),
                        fpe=r.criteria.get("FPE", float("nan")),
                        r2_cv=r.cv.get("r2_mean") if r.cv else None,
                        whiteness_pass=bool(r.whiteness.get("passed", False)),
                        n_params=r.n_params,
                        result=r,
                    ))
            elif fam == "narmax":
                # Single structure per call; vary (na, nb).
                for na in range(1, min(3, na_max) + 1):
                    for nb in range(0 if u is None else 1, min(3, nb_max) + 1):
                        try:
                            r = fit_narmax(
                                y, u, na, nb, degree=narmax_degree,
                                cv_folds=cv_folds,
                            )
                            desc = f"NARMAX({na},{nb},deg={narmax_degree})"
                            entries.append(ComparisonEntry(
                                family="NARMAX",
                                structure_desc=desc,
                                aic=r.criteria.get("AIC", float("nan")),
                                bic=r.criteria.get("BIC", float("nan")),
                                fpe=r.criteria.get("FPE", float("nan")),
                                r2_cv=r.cv.get("r2_mean") if r.cv else None,
                                whiteness_pass=bool(r.whiteness.get("passed", False)),
                                n_params=r.n_params,
                                result=r,
                            ))
                        except Exception as exc:
                            warnings.warn(f"NARMAX({na},{nb}) skipped: {exc}")
            else:
                warnings.warn(f"Unknown family: {fam}")
        except Exception as exc:
            warnings.warn(f"Family {fam} failed: {exc}")

    crit_field = {"AIC": "aic", "BIC": "bic", "FPE": "fpe", "AICc": "aic"}.get(
        key, "bic"
    )
    entries.sort(key=lambda e: getattr(e, crit_field, float("inf")) or float("inf"))
    return entries


# ===========================================================================
# 9. IDENTIFIABILITY ASSESSMENT
# ===========================================================================


def assess_identifiability(
    y: Sequence[float],
    u: Optional[Sequence[float]] = None,
    fs: float = 1.0,
) -> Dict[str, Any]:
    """
    Phase 3 pre-fit gate: is the data suitable for parametric identification?

    Returns dict with 'verdict' in {GO, MARGINAL, NO-GO}, plus diagnostic
    fields (n, snr_estimate, coherence_mean, rationale).
    """
    _require_numpy()
    y_arr = _as_1d_array(y)
    n = y_arr.size
    rationale: List[str] = []
    verdict = "GO"

    # Data length check.
    if n < 50:
        verdict = "NO-GO"
        rationale.append(f"n={n} < 50: insufficient samples for robust ID")
    elif n < 200:
        verdict = "MARGINAL"
        rationale.append(f"n={n} < 200: marginal (consider lower-order models)")
    else:
        rationale.append(f"n={n} >= 200: adequate sample size")

    # SNR estimate: ratio of variance vs residual of low-pass smoothing.
    snr_est: Optional[float] = None
    if n >= 10:
        window = max(3, min(n // 10, 21))
        if window % 2 == 0:
            window += 1
        kernel = np.ones(window) / window
        smooth = np.convolve(y_arr, kernel, mode="same")
        signal_var = float(np.var(smooth))
        noise_var = float(np.var(y_arr - smooth))
        if noise_var > 0:
            snr_est = signal_var / noise_var
            if snr_est < 1.0:
                if verdict == "GO":
                    verdict = "MARGINAL"
                rationale.append(
                    f"SNR~{snr_est:.2f} < 1: signal dominated by noise"
                )
            else:
                rationale.append(f"SNR estimate ~{snr_est:.2f}")

    # Coherence γ²(f) if u provided and scipy available.
    coh_mean: Optional[float] = None
    if u is not None:
        u_arr = _as_1d_array(u)
        if u_arr.size == y_arr.size:
            if _HAS_SCIPY:
                try:
                    nperseg = min(256, max(32, n // 4))
                    _, gamma2 = sp_signal.coherence(
                        u_arr, y_arr, fs=fs, nperseg=nperseg,
                    )
                    coh_mean = float(np.mean(gamma2))
                    if coh_mean < 0.5:
                        if verdict == "GO":
                            verdict = "MARGINAL"
                        rationale.append(
                            f"mean coherence γ²={coh_mean:.2f} < 0.5: "
                            "weak linear relationship"
                        )
                    else:
                        rationale.append(f"mean coherence γ²={coh_mean:.2f}")
                except Exception as exc:
                    rationale.append(f"coherence failed: {exc}")
            else:
                rationale.append("install scipy for coherence analysis")

    return {
        "verdict": verdict,
        "n": n,
        "snr_estimate": snr_est,
        "coherence_mean": coh_mean,
        "has_input": u is not None,
        "rationale": rationale,
    }


# ===========================================================================
# 10. CLI
# ===========================================================================


def _load_csv_columns(
    path: str, y_col: str, u_col: Optional[str] = None,
) -> Tuple[List[float], Optional[List[float]]]:
    """Load y and optional u columns from a CSV file."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        if y_col not in fields:
            raise ValueError(
                f"Column '{y_col}' not found. Available: {', '.join(fields)}"
            )
        if u_col is not None and u_col not in fields:
            raise ValueError(
                f"Input column '{u_col}' not found. Available: {', '.join(fields)}"
            )
        y_raw: List[float] = []
        u_raw: List[float] = []
        for row in reader:
            try:
                y_raw.append(float(row[y_col]))
            except (TypeError, ValueError):
                y_raw.append(float("nan"))
            if u_col is not None:
                try:
                    u_raw.append(float(row[u_col]))
                except (TypeError, ValueError):
                    u_raw.append(float("nan"))
    return y_raw, (u_raw if u_col is not None else None)


def _cli_fit(args: argparse.Namespace) -> int:
    y, u = _load_csv_columns(args.file, args.column, args.input_column)
    family = args.family.lower()
    criterion = args.criterion

    if args.grid:
        na_rng = range(1, args.na_max + 1)
        nb_rng = range(0 if u is None else 1, args.nb_max + 1)
        nk_rng = range(1, args.nk_max + 1)
        if family == "arx":
            ranked = fit_arx_grid(
                y, u, na_rng, nb_rng, nk_rng,
                criterion=criterion, cv_folds=args.cv_folds,
            )
        elif family == "armax":
            ranked = fit_armax_grid(
                y, u, na_rng, nb_rng,
                criterion=criterion, cv_folds=args.cv_folds,
            )
        else:
            print(f"Grid search not supported for family={family}", file=sys.stderr)
            return 2
        if not ranked:
            print("No successful fits in grid", file=sys.stderr)
            return 1
        result = ranked[0]
        print(f"# Grid search winner ({criterion.upper()})\n")
    else:
        if family == "arx":
            result = fit_arx(
                y, u, args.na, args.nb, args.nk,
                bootstrap=args.bootstrap,
                cv_folds=args.cv_folds,
            )
        elif family == "armax":
            result = fit_armax(
                y, u, args.na, args.nb,
                bootstrap=args.bootstrap,
                cv_folds=args.cv_folds,
            )
        elif family == "narmax":
            result = fit_narmax(
                y, u, args.na, args.nb,
                degree=args.degree, nk=args.nk,
                bootstrap=args.bootstrap,
                cv_folds=args.cv_folds,
            )
        else:
            print(f"Unknown family: {family}", file=sys.stderr)
            return 2

    print(result.format_report())
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result.to_json(), f, indent=2, default=str)
        print(f"\nJSON saved to {args.output}")
    if args.report:
        with open(args.report, "w") as f:
            f.write(result.format_report())
        print(f"Markdown report saved to {args.report}")
    return 0


def _cli_assess(args: argparse.Namespace) -> int:
    y, u = _load_csv_columns(args.file, args.column, args.input_column)
    verdict = assess_identifiability(y, u, fs=args.fs)
    print("# Identifiability Assessment")
    print(f"\n**Verdict**: {verdict['verdict']}")
    print(f"**n**: {verdict['n']}")
    if verdict.get("snr_estimate") is not None:
        print(f"**SNR estimate**: {verdict['snr_estimate']:.3f}")
    if verdict.get("coherence_mean") is not None:
        print(f"**Mean coherence**: {verdict['coherence_mean']:.3f}")
    print("\n## Rationale\n")
    for r in verdict["rationale"]:
        print(f"- {r}")
    return 0


def _cli_compare(args: argparse.Namespace) -> int:
    y, u = _load_csv_columns(args.file, args.column, args.input_column)
    families = [s.strip() for s in args.families.split(",") if s.strip()]
    entries = compare_structures(
        y, u, families=families, criterion=args.criterion,
        cv_folds=args.cv_folds,
    )
    print(f"# Structure Comparison (criterion: {args.criterion.upper()})\n")
    print(f"{'Family':<8} {'Structure':<24} {'AIC':>10} {'BIC':>10} "
          f"{'FPE':>10} {'White':>6} {'k':>4}")
    print("-" * 74)
    for e in entries[:20]:
        w = "yes" if e.whiteness_pass else "no"
        print(f"{e.family:<8} {e.structure_desc:<24} "
              f"{e.aic:>10.2f} {e.bic:>10.2f} {e.fpe:>10.4g} "
              f"{w:>6} {e.n_params:>4}")
    # Determine winner: lowest criterion among whiteness-pass models.
    passed = [e for e in entries if e.whiteness_pass]
    winner = passed[0] if passed else (entries[0] if entries else None)
    if winner:
        flag = "" if winner.whiteness_pass else " (WARNING: whiteness failed)"
        print(f"\nWinner: {winner.family} {winner.structure_desc}{flag}")
    if args.output and winner:
        out = {
            "entries": [
                {
                    "family": e.family, "structure": e.structure_desc,
                    "aic": e.aic, "bic": e.bic, "fpe": e.fpe,
                    "r2_cv": e.r2_cv, "whiteness_pass": e.whiteness_pass,
                    "n_params": e.n_params,
                }
                for e in entries
            ],
            "winner": {
                "family": winner.family,
                "structure": winner.structure_desc,
                "whiteness_pass": winner.whiteness_pass,
                "fit": winner.result.to_json() if winner.result else None,
            },
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\nJSON saved to {args.output}")
    return 0


def _demo() -> int:
    """Synthetic ARX(2,1,1) demo, full pipeline."""
    _require_numpy()
    print("=" * 72)
    print("  parametric_identifier.py — DEMO")
    print("=" * 72)
    print(f"  Libraries: numpy={_HAS_NUMPY}, scipy={_HAS_SCIPY}, "
          f"statsmodels={_HAS_STATSMODELS}")
    print()

    rng = np.random.default_rng(42)
    n = 500
    # True system: A(q) = 1 - 0.7 q^-1 + 0.2 q^-2, B(q) = 0.5 q^-1
    # y[t] = 0.7 y[t-1] - 0.2 y[t-2] + 0.5 u[t-1] + e[t]
    u = rng.normal(0, 1, n)
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = 0.7 * y[t - 1] - 0.2 * y[t - 2] + 0.5 * u[t - 1] + rng.normal(0, 0.1)

    print("Synthetic system: y[t] = 0.7 y[t-1] - 0.2 y[t-2] + 0.5 u[t-1] + noise")
    print(f"Samples: {n}, noise std: 0.1\n")

    # Assess
    print("--- Identifiability ---")
    verdict = assess_identifiability(y, u)
    print(f"Verdict: {verdict['verdict']}, n={verdict['n']}, "
          f"SNR~{verdict['snr_estimate']:.2f}")
    if verdict.get("coherence_mean") is not None:
        print(f"Mean coherence: {verdict['coherence_mean']:.3f}")
    print()

    # Fit single-structure ARX(2,1,1)
    print("--- ARX(2,1,1) Fit ---")
    res = fit_arx(y, u, na=2, nb=1, nk=1, cv_folds=5)
    # Recovered coefficients: [-a1, -a2, b0] = [0.7, -0.2, 0.5]
    print(f"Param estimates: {[round(v, 4) for v in res.param_values]}")
    print(f"RSS: {res.rss:.4f}")
    print(f"AIC: {res.criteria['AIC']:.2f}, BIC: {res.criteria['BIC']:.2f}")
    print(f"Whiteness: Q={res.whiteness['Q']:.2f}, "
          f"p={res.whiteness['p_value']:.3f}, passed={res.whiteness['passed']}")
    if res.cv:
        print(f"CV R² mean: {res.cv['r2_mean']:.4f} "
              f"(std {res.cv['r2_std']:.4f})")
    sim = res.to_simulator_format()
    print(f"Simulator format: a={sim['a']}, b={sim['b']}, nk={sim['nk']}")
    print()

    # Grid search
    print("--- Grid Search ARX ---")
    ranked = fit_arx_grid(
        y, u, na_range=range(1, 5), nb_range=range(1, 3),
        nk_range=[1], criterion="bic",
    )
    print(f"Top 3 by BIC:")
    for r in ranked[:3]:
        s = r.structure
        print(f"  ARX({s['na']},{s['nb']},{s['nk']}): "
              f"BIC={r.criteria['BIC']:.2f}, k={r.n_params}")
    print()

    # Compare structures
    print("--- Structure Comparison ---")
    entries = compare_structures(
        y, u, families=["arx", "narmax"], criterion="bic",
        na_max=3, nb_max=2,
    )
    print(f"{'Family':<8} {'Structure':<24} {'BIC':>10} {'k':>4}")
    for e in entries[:6]:
        print(f"{e.family:<8} {e.structure_desc:<24} {e.bic:>10.2f} {e.n_params:>4}")
    print()

    print("Demo complete.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parametric Identifier — fit ARX/ARMAX/NARMAX models",
    )
    sub = parser.add_subparsers(dest="cmd", help="Commands")

    # assess
    ap = sub.add_parser("assess", help="Identifiability assessment (Phase 3 gate)")
    ap.add_argument("file", help="CSV file path")
    ap.add_argument("--column", required=True, help="Output column (y)")
    ap.add_argument("--input-column", default=None, help="Input column (u)")
    ap.add_argument("--fs", type=float, default=1.0, help="Sampling frequency")

    # fit
    fp = sub.add_parser("fit", help="Fit a single model or grid")
    fp.add_argument("file", help="CSV file path")
    fp.add_argument("--column", required=True, help="Output column (y)")
    fp.add_argument("--input-column", default=None, help="Input column (u)")
    fp.add_argument("--family", default="arx",
                    choices=["arx", "armax", "narmax"])
    fp.add_argument("--na", type=int, default=2)
    fp.add_argument("--nb", type=int, default=1)
    fp.add_argument("--nk", type=int, default=1)
    fp.add_argument("--degree", type=int, default=2,
                    help="NARMAX polynomial degree")
    fp.add_argument("--grid", action="store_true", help="Grid search over ranges")
    fp.add_argument("--na-max", type=int, default=5)
    fp.add_argument("--nb-max", type=int, default=5)
    fp.add_argument("--nk-max", type=int, default=1)
    fp.add_argument("--criterion", default="bic",
                    choices=["aic", "bic", "aicc", "fpe"])
    fp.add_argument("--bootstrap", type=int, default=0,
                    help="Bootstrap samples for CI (0 = analytic)")
    fp.add_argument("--cv-folds", type=int, default=0,
                    help="Walk-forward CV folds (0 = skip)")
    fp.add_argument("--output", default=None, help="JSON output path")
    fp.add_argument("--report", default=None, help="Markdown report path")

    # compare
    cp = sub.add_parser("compare", help="Compare families")
    cp.add_argument("file", help="CSV file path")
    cp.add_argument("--column", required=True, help="Output column (y)")
    cp.add_argument("--input-column", default=None, help="Input column (u)")
    cp.add_argument("--families", default="arx,armax,narmax")
    cp.add_argument("--criterion", default="bic")
    cp.add_argument("--cv-folds", type=int, default=0)
    cp.add_argument("--output", default=None, help="JSON output path")

    # demo
    sub.add_parser("demo", help="Run built-in synthetic demo")

    args = parser.parse_args()

    if not _HAS_NUMPY:
        print("Error: numpy is required. Install with: pip install numpy",
              file=sys.stderr)
        return 2

    try:
        if args.cmd == "assess":
            return _cli_assess(args)
        if args.cmd == "fit":
            return _cli_fit(args)
        if args.cmd == "compare":
            return _cli_compare(args)
        if args.cmd == "demo":
            return _demo()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
