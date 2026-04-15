#!/usr/bin/env python3
"""Tests for src/scripts/parametric_identifier.py"""

import math
import os
import subprocess
import sys
import unittest
import warnings

# Allow importing from src/scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "scripts"))

import numpy as np

from parametric_identifier import (
    FitResult,
    ComparisonEntry,
    build_arx_regressors,
    fit_arx_ols,
    fit_arx,
    fit_arx_grid,
    fit_armax,
    fit_armax_grid,
    fit_narmax,
    polynomial_basis,
    frols,
    compute_criteria,
    ljung_box,
    analytic_ci_from_cov,
    residual_bootstrap,
    walk_forward_cv,
    assess_identifiability,
    compare_structures,
    _HAS_NUMPY,
    _HAS_STATSMODELS,
    _HAS_SCIPY,
)


# ---------------------------------------------------------------------------
# Helpers: synthetic data generators with fixed seeds
# ---------------------------------------------------------------------------


def _gen_arx(
    n: int = 500,
    noise_std: float = 0.1,
    seed: int = 0,
    a1: float = 0.7,
    a2: float = -0.2,
    b0: float = 0.5,
):
    """Generate y[t] = a1 y[t-1] + a2 y[t-2] + b0 u[t-1] + e[t]."""
    rng = np.random.default_rng(seed)
    u = rng.normal(0, 1, n)
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = a1 * y[t - 1] + a2 * y[t - 2] + b0 * u[t - 1] + rng.normal(
            0, noise_std
        )
    return y, u


def _gen_ar(n: int = 400, phi: float = 0.6, noise_std: float = 0.2, seed: int = 1):
    """Pure AR(1)."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + rng.normal(0, noise_std)
    return y


# ===========================================================================
# 1. BUILD_ARX_REGRESSORS
# ===========================================================================


class TestBuildArxRegressors(unittest.TestCase):

    def test_shape_and_content(self):
        y = np.arange(10.0)
        u = np.arange(10.0) * 0.1
        Phi, y_out = build_arx_regressors(y, u, na=2, nb=1, nk=1)
        # n_start = max(2, 1+1-1)=2, so 10-2 = 8 rows, 3 cols
        self.assertEqual(Phi.shape, (8, 3))
        self.assertEqual(y_out.size, 8)
        # First row: y[2] = [y[1], y[0], u[1]]
        np.testing.assert_allclose(Phi[0], [1.0, 0.0, 0.1])
        self.assertEqual(y_out[0], 2.0)

    def test_pure_ar(self):
        y = np.arange(6.0)
        Phi, y_out = build_arx_regressors(y, None, na=2, nb=0, nk=1)
        # n_start = 2; 4 rows, 2 cols
        self.assertEqual(Phi.shape, (4, 2))
        np.testing.assert_allclose(Phi[0], [1.0, 0.0])

    def test_input_delay(self):
        y = np.arange(10.0)
        u = np.arange(10.0)
        Phi, y_out = build_arx_regressors(y, u, na=1, nb=1, nk=2)
        # n_start = max(1, 2+1-1)=2; 8 rows, 2 cols
        self.assertEqual(Phi.shape, (8, 2))
        # first row at t=2: regressor = [y[1]=1, u[0]=0]
        np.testing.assert_allclose(Phi[0], [1.0, 0.0])

    def test_too_short_raises(self):
        y = np.arange(3.0)
        with self.assertRaises(ValueError):
            build_arx_regressors(y, None, na=5, nb=0)

    def test_nb_without_u_raises(self):
        with self.assertRaises(ValueError):
            build_arx_regressors(np.arange(10.0), None, na=1, nb=1)


# ===========================================================================
# 2. OLS FITTING
# ===========================================================================


class TestFitArxOls(unittest.TestCase):

    def test_exact_recovery_noise_free(self):
        rng = np.random.default_rng(0)
        n = 300
        u = rng.normal(0, 1, n)
        y = np.zeros(n)
        for t in range(2, n):
            y[t] = 0.8 * y[t - 1] - 0.15 * y[t - 2] + 0.6 * u[t - 1]
        Phi, y_trim = build_arx_regressors(y, u, na=2, nb=1, nk=1)
        theta, res, cov = fit_arx_ols(Phi, y_trim)
        np.testing.assert_allclose(theta, [0.8, -0.15, 0.6], atol=1e-8)
        self.assertLess(np.abs(res).max(), 1e-8)

    def test_singular_raises(self):
        Phi = np.ones((10, 2))  # rank 1
        y = np.arange(10.0)
        with self.assertRaises(ValueError):
            fit_arx_ols(Phi, y)


# ===========================================================================
# 3. HIGH-LEVEL FIT_ARX (Known-answer recovery)
# ===========================================================================


class TestFitArxKnownAnswer(unittest.TestCase):

    def test_arx_211_recovery(self):
        y, u = _gen_arx(n=500, noise_std=0.05, seed=42)
        r = fit_arx(y, u, na=2, nb=1, nk=1)
        # [theta] = [-a1, -a2, b0] with model y = a1 y[-1] + a2 y[-2] + b0 u[-1]
        self.assertAlmostEqual(r.param_values[0], 0.7, delta=0.05)
        self.assertAlmostEqual(r.param_values[1], -0.2, delta=0.05)
        self.assertAlmostEqual(r.param_values[2], 0.5, delta=0.05)
        self.assertTrue(r.whiteness["passed"])

    def test_ar_only_fit(self):
        y = _gen_ar(n=400, phi=0.6, noise_std=0.2, seed=3)
        r = fit_arx(y, None, na=1, nb=0, nk=1)
        self.assertAlmostEqual(r.param_values[0], 0.6, delta=0.1)
        self.assertEqual(r.n_params, 1)

    def test_simulator_format_keys(self):
        y, u = _gen_arx(n=300, seed=2)
        r = fit_arx(y, u, na=2, nb=1, nk=1)
        sim = r.to_simulator_format()
        self.assertIn("a", sim)
        self.assertIn("b", sim)
        self.assertIn("nk", sim)
        self.assertEqual(sim["type"], "arx")
        # a-polynomial signs flipped from theta
        self.assertAlmostEqual(sim["a"][0], -r.param_values[0], delta=1e-9)
        self.assertAlmostEqual(sim["a"][1], -r.param_values[1], delta=1e-9)
        self.assertAlmostEqual(sim["b"][0], r.param_values[2], delta=1e-9)


# ===========================================================================
# 4. GRID SEARCH
# ===========================================================================


class TestArxGrid(unittest.TestCase):

    def test_grid_picks_true_or_close(self):
        y, u = _gen_arx(n=500, noise_std=0.1, seed=7)
        ranked = fit_arx_grid(
            y, u,
            na_range=range(1, 5),
            nb_range=range(1, 5),
            nk_range=[1],
            criterion="bic",
        )
        winner = ranked[0]
        self.assertLessEqual(winner.structure["na"], 3)
        self.assertLessEqual(winner.structure["nb"], 3)
        self.assertTrue(winner.whiteness["passed"])

    def test_grid_monotone_ordering(self):
        y, u = _gen_arx(n=300, seed=11)
        ranked = fit_arx_grid(y, u, range(1, 4), range(1, 4), [1], criterion="bic")
        bics = [r.criteria["BIC"] for r in ranked]
        self.assertEqual(bics, sorted(bics))


# ===========================================================================
# 5. BOOTSTRAP CI COVERAGE
# ===========================================================================


class TestBootstrapCoverage(unittest.TestCase):

    def test_bootstrap_ci_covers_truth(self):
        """Run multiple datasets, check true coefs fall in CI most of the time."""
        true_a1, true_a2, true_b0 = 0.7, -0.2, 0.5
        covered = [0, 0, 0]
        n_trials = 20
        for seed in range(n_trials):
            y, u = _gen_arx(n=400, noise_std=0.1, seed=seed)
            r = fit_arx(y, u, na=2, nb=1, nk=1, bootstrap=80)
            for i, true_val in enumerate([true_a1, true_a2, true_b0]):
                if r.param_ci_lo[i] <= true_val <= r.param_ci_hi[i]:
                    covered[i] += 1
        # Expect each coefficient covered in at least 70% of trials
        # (some slack for bootstrap noise with n_boot=80).
        for c in covered:
            self.assertGreaterEqual(
                c, int(0.7 * n_trials),
                f"Coverage too low: {c}/{n_trials}",
            )


# ===========================================================================
# 6. INFORMATION CRITERIA
# ===========================================================================


class TestCriteria(unittest.TestCase):

    def test_compute_criteria_reference(self):
        n, k, rss = 100, 3, 10.0
        crit = compute_criteria(n, k, rss)
        # AIC = n*log(rss/n) + 2k = 100*log(0.1) + 6
        expected_aic = 100 * math.log(0.1) + 6
        expected_bic = 100 * math.log(0.1) + 3 * math.log(100)
        self.assertAlmostEqual(crit["AIC"], expected_aic, places=6)
        self.assertAlmostEqual(crit["BIC"], expected_bic, places=6)
        # FPE = (rss/n)(n+k)/(n-k) = 0.1 * 103/97
        self.assertAlmostEqual(crit["FPE"], 0.1 * 103 / 97, places=6)
        # AICc = AIC + 2k(k+1)/(n-k-1)
        self.assertAlmostEqual(
            crit["AICc"],
            expected_aic + 2 * 3 * 4 / (100 - 3 - 1),
            places=6,
        )

    def test_degenerate_inputs(self):
        c = compute_criteria(0, 1, 1.0)
        self.assertTrue(math.isnan(c["AIC"]))
        c2 = compute_criteria(10, 5, -1.0)
        self.assertTrue(math.isnan(c2["AIC"]))


# ===========================================================================
# 7. LJUNG-BOX WHITENESS
# ===========================================================================


class TestLjungBox(unittest.TestCase):

    def test_white_noise_passes(self):
        rng = np.random.default_rng(0)
        residuals = rng.normal(0, 1, 500)
        Q, p, passed = ljung_box(residuals, lags=10)
        self.assertTrue(passed)
        self.assertGreater(p, 0.05)

    def test_correlated_fails(self):
        # AR(1)-colored residuals: strongly autocorrelated
        rng = np.random.default_rng(1)
        r = np.zeros(500)
        for t in range(1, 500):
            r[t] = 0.9 * r[t - 1] + rng.normal(0, 1)
        Q, p, passed = ljung_box(r, lags=10)
        self.assertFalse(passed)
        self.assertLess(p, 0.05)

    def test_under_order_fit_fails_whiteness(self):
        # AR(2) data, fit AR(1) — residuals should be structured.
        rng = np.random.default_rng(2)
        n = 500
        y = np.zeros(n)
        for t in range(2, n):
            y[t] = 0.6 * y[t - 1] - 0.3 * y[t - 2] + rng.normal(0, 0.2)
        r = fit_arx(y, None, na=1, nb=0)
        self.assertFalse(r.whiteness["passed"])

    def test_correct_order_passes(self):
        rng = np.random.default_rng(3)
        n = 500
        y = np.zeros(n)
        for t in range(2, n):
            y[t] = 0.6 * y[t - 1] - 0.3 * y[t - 2] + rng.normal(0, 0.2)
        r = fit_arx(y, None, na=2, nb=0)
        self.assertTrue(r.whiteness["passed"])


# ===========================================================================
# 8. WALK-FORWARD CV
# ===========================================================================


class TestWalkForwardCV(unittest.TestCase):

    def test_ar_r2_high(self):
        # Use strong AR process (phi=0.9) with small noise — asymptotic R² ≈ phi²
        y = _gen_ar(n=700, phi=0.9, noise_std=0.1, seed=10)
        r = fit_arx(y, None, na=1, nb=0, cv_folds=5)
        self.assertIsNotNone(r.cv)
        # Per-fold R² can vary with stationarity. Require majority of folds
        # to clear a moderate bar rather than mean (mean sensitive to outliers).
        good = sum(1 for v in r.cv["r2_per_fold"] if v > 0.5)
        self.assertGreaterEqual(good, max(1, r.cv["k_folds"] // 2))
        self.assertGreaterEqual(r.cv["k_folds"], 4)

    def test_cv_short_data_raises(self):
        with self.assertRaises(ValueError):
            walk_forward_cv(
                lambda y, u: fit_arx(y, u, na=1, nb=0),
                y=np.arange(10.0),
                u=None,
                k_folds=5,
            )


# ===========================================================================
# 9. POLYNOMIAL BASIS
# ===========================================================================


class TestPolynomialBasis(unittest.TestCase):

    def test_column_count(self):
        # na=2, nb=2 → 4 base regressors; degree=2 →
        # constant(1) + 4 linear + C(4+1, 2)=10 quadratic = 15 total
        y = np.arange(20.0)
        u = np.arange(20.0) * 0.1
        Phi, labels, y_trim = polynomial_basis(y, u, na=2, nb=2, degree=2, nk=1)
        self.assertEqual(Phi.shape[1], 15)
        self.assertEqual(labels[0], "1")

    def test_degree_one_matches_arx(self):
        y = np.arange(10.0)
        u = np.arange(10.0) * 0.1
        Phi_p, labels, y_p = polynomial_basis(y, u, na=1, nb=1, degree=1, nk=1)
        Phi_a, y_a = build_arx_regressors(y, u, na=1, nb=1, nk=1)
        # Polynomial basis has an extra constant term in column 0; strip it.
        np.testing.assert_allclose(Phi_p[:, 1:], Phi_a, atol=1e-12)
        np.testing.assert_allclose(y_p, y_a)


# ===========================================================================
# 10. FROLS TERM SELECTION
# ===========================================================================


class TestFrols(unittest.TestCase):

    def test_frols_picks_correct_terms(self):
        # y generated from a sparse polynomial structure: y = 2*x1 - 0.5*x2^2 + noise
        rng = np.random.default_rng(12)
        n = 300
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        y = 2.0 * x1 - 0.5 * x2 ** 2 + rng.normal(0, 0.05, n)
        # Build candidate basis: x1, x2, x1*x1, x1*x2, x2*x2, const
        Phi = np.column_stack([x1, x2, x1 * x1, x1 * x2, x2 * x2, np.ones(n)])
        selected, coeffs, errs = frols(Phi, y, err_threshold=0.999, max_terms=3)
        # Top-2 terms should include x1 (col 0) and x2*x2 (col 4)
        top2 = set(selected[:2])
        self.assertIn(0, top2)
        self.assertIn(4, top2)


class TestFitNarmax(unittest.TestCase):

    def test_narmax_runs_on_arx_data(self):
        y, u = _gen_arx(n=400, seed=20, noise_std=0.1)
        r = fit_narmax(y, u, na=2, nb=1, degree=2)
        self.assertEqual(r.model_type, "NARMAX")
        self.assertGreater(r.n_params, 0)
        # Should select few terms since data is linear
        self.assertLessEqual(r.n_params, 12)


# ===========================================================================
# 11. COMPARE STRUCTURES
# ===========================================================================


class TestCompareStructures(unittest.TestCase):

    def test_arx_beats_narmax_on_linear_data(self):
        y, u = _gen_arx(n=500, seed=30, noise_std=0.05)
        entries = compare_structures(
            y, u, families=["arx", "narmax"], criterion="bic",
            na_max=3, nb_max=2,
        )
        self.assertGreater(len(entries), 0)
        # Winner should be ARX
        self.assertEqual(entries[0].family, "ARX")


# ===========================================================================
# 12. ASSESS IDENTIFIABILITY
# ===========================================================================


class TestAssessIdentifiability(unittest.TestCase):

    def test_short_data_no_go(self):
        y = list(range(20))
        v = assess_identifiability(y)
        self.assertIn(v["verdict"], ("NO-GO", "MARGINAL"))

    def test_long_data_go_or_marginal(self):
        y, u = _gen_arx(n=600, seed=5)
        v = assess_identifiability(y, u)
        self.assertIn(v["verdict"], ("GO", "MARGINAL"))
        self.assertGreaterEqual(v["n"], 600)

    def test_with_input_coherence(self):
        y, u = _gen_arx(n=500, seed=6)
        v = assess_identifiability(y, u)
        self.assertTrue(v["has_input"])


# ===========================================================================
# 13. ARMAX (statsmodels-dependent)
# ===========================================================================


@unittest.skipUnless(_HAS_STATSMODELS, "statsmodels not available")
class TestArmax(unittest.TestCase):

    def test_armax_fit_runs(self):
        y, u = _gen_arx(n=400, seed=50, noise_std=0.1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = fit_armax(y, u, p=2, q=1)
        self.assertEqual(r.model_type, "ARMAX")
        self.assertEqual(r.n_samples, 400)
        self.assertGreater(r.n_params, 0)


# ===========================================================================
# 14. CLI SMOKE TEST
# ===========================================================================


class TestCliSmoke(unittest.TestCase):

    def test_demo_cli(self):
        script = os.path.join(
            os.path.dirname(__file__), "..", "src", "scripts",
            "parametric_identifier.py",
        )
        result = subprocess.run(
            [sys.executable, script, "demo"],
            capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(result.returncode, 0,
                         msg=f"stderr: {result.stderr}")
        self.assertIn("DEMO", result.stdout)
        self.assertIn("ARX(2,1,1)", result.stdout)
        self.assertIn("Demo complete", result.stdout)


if __name__ == "__main__":
    unittest.main()
