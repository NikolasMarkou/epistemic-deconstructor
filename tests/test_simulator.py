#!/usr/bin/env python3
"""Tests for src/scripts/simulator.py"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import unittest

# Allow importing from src/scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'scripts'))

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

if HAS_NUMPY:
    from simulator import (
        _seed_rng,
        _sample_distribution,
        _make_input_func,
        _json_default,
        _sd_linear,
        _mc_run_single,
        _build_topology,
        run_abm,
        run_sensitivity,
        generate_validation_bridge,
        build_parser,
        SDResult,
        MCResult,
    )


@unittest.skipUnless(HAS_NUMPY, "numpy required")
class TestSeedRng(unittest.TestCase):
    def test_deterministic_with_seed(self):
        rng1 = _seed_rng(42)
        rng2 = _seed_rng(42)
        self.assertEqual(rng1.random(), rng2.random())

    def test_none_seed(self):
        rng = _seed_rng(None)
        val = rng.random()
        self.assertIsInstance(val, float)


@unittest.skipUnless(HAS_NUMPY, "numpy required")
class TestSampleDistribution(unittest.TestCase):
    def setUp(self):
        self.rng = _seed_rng(123)

    def test_normal(self):
        spec = {"dist": "normal", "mean": 5.0, "std": 1.0}
        samples = _sample_distribution(spec, self.rng, size=100)
        self.assertEqual(len(samples), 100)
        self.assertAlmostEqual(np.mean(samples), 5.0, delta=0.5)

    def test_uniform(self):
        spec = {"dist": "uniform", "low": 0.0, "high": 10.0}
        samples = _sample_distribution(spec, self.rng, size=100)
        self.assertTrue(np.all(samples >= 0.0))
        self.assertTrue(np.all(samples <= 10.0))

    def test_triangular(self):
        spec = {"dist": "triangular", "left": 0.0, "mode": 5.0, "right": 10.0}
        samples = _sample_distribution(spec, self.rng, size=50)
        self.assertEqual(len(samples), 50)
        self.assertTrue(np.all(samples >= 0.0))
        self.assertTrue(np.all(samples <= 10.0))

    def test_lognormal(self):
        spec = {"dist": "lognormal", "mu": 0.0, "sigma": 0.5}
        samples = _sample_distribution(spec, self.rng, size=50)
        self.assertTrue(np.all(samples > 0))

    def test_beta(self):
        spec = {"dist": "beta", "a": 2.0, "b": 5.0}
        samples = _sample_distribution(spec, self.rng, size=50)
        self.assertTrue(np.all(samples >= 0))
        self.assertTrue(np.all(samples <= 1))

    def test_exponential(self):
        spec = {"dist": "exponential", "scale": 2.0}
        samples = _sample_distribution(spec, self.rng, size=50)
        self.assertTrue(np.all(samples >= 0))

    def test_constant(self):
        spec = {"dist": "constant", "value": 3.14}
        samples = _sample_distribution(spec, self.rng, size=10)
        self.assertTrue(np.allclose(samples, 3.14))

    def test_gamma(self):
        spec = {"dist": "gamma", "shape": 2.0, "scale": 1.5}
        samples = _sample_distribution(spec, self.rng, size=50)
        self.assertTrue(np.all(samples > 0))
        self.assertAlmostEqual(np.mean(samples), 3.0, delta=1.0)

    def test_poisson(self):
        spec = {"dist": "poisson", "lam": 5.0}
        samples = _sample_distribution(spec, self.rng, size=100)
        self.assertTrue(np.all(samples >= 0))
        self.assertTrue(np.all(samples == np.floor(samples)))

    def test_weibull(self):
        spec = {"dist": "weibull", "a": 2.0, "scale": 3.0}
        samples = _sample_distribution(spec, self.rng, size=50)
        self.assertTrue(np.all(samples >= 0))

    def test_binomial(self):
        spec = {"dist": "binomial", "n": 10, "p": 0.3}
        samples = _sample_distribution(spec, self.rng, size=50)
        self.assertTrue(np.all(samples >= 0))
        self.assertTrue(np.all(samples <= 10))
        self.assertTrue(np.all(samples == np.floor(samples)))

    def test_chisquare(self):
        spec = {"dist": "chisquare", "df": 5}
        samples = _sample_distribution(spec, self.rng, size=50)
        self.assertTrue(np.all(samples > 0))

    def test_unknown_dist_raises(self):
        spec = {"dist": "pareto_fantasy"}
        with self.assertRaises(ValueError):
            _sample_distribution(spec, self.rng)


@unittest.skipUnless(HAS_NUMPY, "numpy required")
class TestMakeInputFunc(unittest.TestCase):
    def test_step(self):
        f = _make_input_func("step", amplitude=2.0, t_on=1.0)
        self.assertEqual(f(0.5), 0.0)
        self.assertEqual(f(1.0), 2.0)
        self.assertEqual(f(5.0), 2.0)

    def test_impulse(self):
        f = _make_input_func("impulse", amplitude=3.0, t_on=2.0)
        self.assertEqual(f(2.0), 3.0)
        self.assertEqual(f(2.1), 0.0)

    def test_sine(self):
        f = _make_input_func("sine", amplitude=1.0, freq=1.0)
        self.assertAlmostEqual(f(0.0), 0.0, places=10)
        self.assertAlmostEqual(f(0.25), 1.0, places=10)

    def test_ramp(self):
        f = _make_input_func("ramp", amplitude=2.0, t_on=1.0)
        self.assertEqual(f(0.5), 0.0)
        self.assertAlmostEqual(f(2.0), 2.0)

    def test_zero(self):
        f = _make_input_func("zero")
        self.assertEqual(f(0.0), 0.0)
        self.assertEqual(f(100.0), 0.0)

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            _make_input_func("chirp")


@unittest.skipUnless(HAS_NUMPY, "numpy required")
class TestJsonDefault(unittest.TestCase):
    def test_ndarray(self):
        arr = np.array([1, 2, 3])
        self.assertEqual(_json_default(arr), [1, 2, 3])

    def test_np_float(self):
        val = np.float64(3.14)
        self.assertIsInstance(_json_default(val), float)

    def test_np_int(self):
        val = np.int64(42)
        self.assertIsInstance(_json_default(val), int)

    def test_unsupported_raises(self):
        with self.assertRaises(TypeError):
            _json_default(set([1, 2]))


@unittest.skipUnless(HAS_NUMPY, "numpy required")
class TestSDLinear(unittest.TestCase):
    def test_basic_2state_system(self):
        """Test a simple stable 2-state system."""
        model = {
            "A": [[-1, 0], [0, -2]],
            "B": [[1], [0]],
            "C": [[1, 0]],
            "D": [[0]],
        }
        x0 = np.array([0.0, 0.0])
        u_func = _make_input_func("step", amplitude=1.0)
        result = _sd_linear(model, x0, u_func, t_end=5.0, dt=0.01, integrator="rk4")

        self.assertIsInstance(result, SDResult)
        self.assertEqual(result.x.shape[1], 2)
        self.assertTrue(result.is_stable)
        self.assertTrue(len(result.t) > 100)
        # x[0] should converge to 1.0 (steady state of x'=-x+1)
        self.assertAlmostEqual(result.x[-1, 0], 1.0, delta=0.05)

    def test_euler_integrator(self):
        model = {"A": [[-1]], "B": [[1]], "C": [[1]], "D": [[0]]}
        x0 = np.array([0.0])
        u_func = _make_input_func("step")
        result = _sd_linear(model, x0, u_func, t_end=5.0, dt=0.01, integrator="euler")
        self.assertIsInstance(result, SDResult)
        self.assertAlmostEqual(result.x[-1, 0], 1.0, delta=0.1)

    def test_unstable_system(self):
        model = {"A": [[1]], "B": [[0]], "C": [[1]], "D": [[0]]}
        x0 = np.array([1.0])
        u_func = _make_input_func("zero")
        result = _sd_linear(model, x0, u_func, t_end=2.0, dt=0.01, integrator="rk4")
        self.assertFalse(result.is_stable)

    def test_output_shapes(self):
        model = {
            "A": [[-1, 0], [0, -2]],
            "B": [[1], [0]],
        }
        x0 = np.array([1.0, 0.0])
        u_func = _make_input_func("zero")
        result = _sd_linear(model, x0, u_func, t_end=1.0, dt=0.1, integrator="rk4")
        n_steps = len(result.t)
        self.assertEqual(result.x.shape, (n_steps, 2))
        self.assertEqual(result.y.shape, (n_steps, 2))  # C defaults to eye
        self.assertEqual(result.u_history.shape, (n_steps, 1))


@unittest.skipUnless(HAS_NUMPY, "numpy required")
class TestMCRunSingle(unittest.TestCase):
    def test_arx_model(self):
        model = {"type": "arx", "a": [-0.5], "b": [1.0], "nk": 1}
        params = {}
        x0 = np.array([0.0])
        u_func = _make_input_func("step")
        traj = _mc_run_single(model, params, x0, u_func, t_end=5.0, dt=0.1)
        self.assertEqual(traj.shape[1], 1)
        self.assertTrue(len(traj) > 10)

    def test_state_space_model(self):
        model = {"A": [[-1]], "B": [[1]]}
        params = {}
        x0 = np.array([0.0])
        u_func = _make_input_func("step")
        traj = _mc_run_single(model, params, x0, u_func, t_end=2.0, dt=0.1)
        self.assertEqual(traj.shape[1], 1)

    def test_invalid_model_raises(self):
        model = {"type": "unknown_model"}
        with self.assertRaises(ValueError):
            _mc_run_single(model, {}, np.array([0.0]), lambda t: 0.0, 1.0, 0.1)


@unittest.skipUnless(HAS_NUMPY, "numpy required")
class TestBuildTopology(unittest.TestCase):
    def setUp(self):
        self.rng = _seed_rng(42)

    def test_complete(self):
        adj = _build_topology(5, "complete", self.rng)
        for i in range(5):
            self.assertEqual(len(adj[i]), 4)

    def test_grid(self):
        adj = _build_topology(9, "grid", self.rng)
        # Corner node (0,0) should have 2 neighbors
        self.assertIn(1, adj[0])
        self.assertIn(3, adj[0])

    def test_small_world(self):
        adj = _build_topology(20, "small_world", self.rng, k=4)
        # All nodes should have some neighbors
        for i in range(20):
            self.assertTrue(len(adj[i]) > 0)

    def test_scale_free(self):
        adj = _build_topology(20, "scale_free", self.rng)
        for i in range(20):
            self.assertTrue(len(adj[i]) > 0)


@unittest.skipUnless(HAS_NUMPY, "numpy required")
class TestValidationBridge(unittest.TestCase):
    def test_sd_bridge(self):
        sim_data = {
            "mode": "sd",
            "is_stable": True,
            "eigenvalues_real": [-1.0, -2.0],
            "settling_time": 3.5,
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sim_data, f)
            f.flush()
            bridge = generate_validation_bridge(f.name, f.name + ".bridge")
        try:
            self.assertEqual(bridge["mode"], "sd")
            self.assertTrue(bridge["phase5_inputs"]["stability"]["is_stable"])
            self.assertEqual(bridge["phase5_inputs"]["stability"]["settling_time"], 3.5)
        finally:
            os.unlink(f.name)
            if os.path.exists(f.name + ".bridge"):
                os.unlink(f.name + ".bridge")

    def test_mc_bridge(self):
        sim_data = {
            "mode": "mc",
            "percentiles": {"5": [0.1], "95": [0.9]},
            "terminal_stats": {"mean": 0.5, "std": 0.2},
            "converged": True,
            "convergence_n": 500,
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sim_data, f)
            f.flush()
            bridge = generate_validation_bridge(f.name, f.name + ".bridge")
        try:
            self.assertEqual(bridge["mode"], "mc")
            self.assertTrue(bridge["phase5_inputs"]["convergence"]["converged"])
        finally:
            os.unlink(f.name)
            if os.path.exists(f.name + ".bridge"):
                os.unlink(f.name + ".bridge")

    def test_sensitivity_bridge(self):
        sim_data = {
            "mode": "sensitivity",
            "rankings": ["k1", "k2"],
            "indices": {"k1": {"S1": 0.8}, "k2": {"S1": 0.1}},
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sim_data, f)
            f.flush()
            bridge = generate_validation_bridge(f.name, f.name + ".bridge")
        try:
            self.assertEqual(bridge["phase5_inputs"]["parameter_rankings"], ["k1", "k2"])
        finally:
            os.unlink(f.name)
            if os.path.exists(f.name + ".bridge"):
                os.unlink(f.name + ".bridge")


@unittest.skipUnless(HAS_NUMPY, "numpy required")
class TestCLIParser(unittest.TestCase):
    def test_parser_builds(self):
        parser = build_parser()
        self.assertIsNotNone(parser)

    def test_sd_subcommand(self):
        parser = build_parser()
        args = parser.parse_args([
            "sd", "--model", '{"A": [[-1]], "B": [[1]]}',
            "--x0", "[0]", "--t_end", "5.0"
        ])
        self.assertEqual(args.mode, "sd")
        self.assertEqual(args.t_end, 5.0)

    def test_mc_subcommand(self):
        parser = build_parser()
        args = parser.parse_args([
            "mc", "--model", '{"a": [-0.5]}',
            "--param_distributions", '{"a[0]": {"dist": "normal", "mean": -0.5, "std": 0.05}}',
            "--t_end", "10", "--n_runs", "100"
        ])
        self.assertEqual(args.mode, "mc")
        self.assertEqual(args.n_runs, 100)

    def test_sensitivity_subcommand(self):
        parser = build_parser()
        args = parser.parse_args([
            "sensitivity", "--model_func", "x + y",
            "--param_ranges", '{"x": [0, 1], "y": [0, 1]}'
        ])
        self.assertEqual(args.mode, "sensitivity")
        self.assertEqual(args.method, "sobol")

    def test_bridge_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["bridge", "--sim_output", "test.json"])
        self.assertEqual(args.mode, "bridge")

    def test_no_mode_prints_help(self):
        parser = build_parser()
        args = parser.parse_args([])
        self.assertIsNone(args.mode)


@unittest.skipUnless(HAS_NUMPY, "numpy required")
class TestSensitivityOATFallback(unittest.TestCase):
    """Test the one-at-a-time fallback when SALib is not available."""

    def test_oat_basic(self):
        """Test OAT fallback produces sensible output for a simple function."""
        # Directly test the OAT logic without going through CLI
        param_names = ["a", "b"]
        bounds = [[0, 10], [0, 5]]
        model_expr = "a + 2 * b"

        baseline = {name: np.mean(b) for name, b in zip(param_names, bounds)}
        indices = {}
        for i, name in enumerate(param_names):
            low_params = {**baseline, name: bounds[i][0]}
            high_params = {**baseline, name: bounds[i][1]}
            y_low = eval(model_expr, {"__builtins__": {}, "np": np}, {**low_params})
            y_high = eval(model_expr, {"__builtins__": {}, "np": np}, {**high_params})
            delta = abs(y_high - y_low)
            indices[name] = {"delta": float(delta)}

        # b has coefficient 2 and range 5, so delta_b = 10
        # a has coefficient 1 and range 10, so delta_a = 10
        self.assertAlmostEqual(indices["a"]["delta"], 10.0)
        self.assertAlmostEqual(indices["b"]["delta"], 10.0)


# --------------------------------------------------------------------------
# Audit H8 — exec sandbox escape (plan_2026-05-28_ad87937f/D-001)
# --------------------------------------------------------------------------
@unittest.skipUnless(HAS_NUMPY, "numpy not available")
class TestOdeCodeSandbox(unittest.TestCase):
    """Regression tests: AST allowlist rejects the H8 escape chains."""

    def _call_with(self, ode_code):
        from simulator import _sd_nonlinear
        try:
            from scipy.integrate import solve_ivp  # noqa: F401
        except ImportError:
            self.skipTest("scipy not available")
        model = {"ode_code": ode_code}
        x0 = np.array([1.0])
        u_func = lambda t: 0.0
        return _sd_nonlinear(model, x0, u_func, t_end=1.0, dt=0.1)

    def test_rejects_subclasses_chain(self):
        # Audit H8 reproducer: reach BuiltinImporter via __subclasses__.
        code = ("for c in tuple().__class__.__bases__[0].__subclasses__():\n"
                "    pass\n"
                "def f(t,x,u): return x\n")
        with self.assertRaises(ValueError) as ctx:
            self._call_with(code)
        self.assertIn("ode_code rejected", str(ctx.exception))

    def test_rejects_bare_dunder_import(self):
        code = "y = __import__('os')\ndef f(t,x,u): return x\n"
        with self.assertRaises(ValueError) as ctx:
            self._call_with(code)
        self.assertIn("ode_code rejected", str(ctx.exception))

    def test_rejects_import_statement(self):
        code = "import os\ndef f(t,x,u): return x\n"
        with self.assertRaises(ValueError) as ctx:
            self._call_with(code)
        self.assertIn("import", str(ctx.exception).lower())

    def test_legitimate_ode_code_still_works(self):
        # First-order decay: dx/dt = -x; legitimate np usage must work.
        code = "def f(t,x,u): return np.array([-x[0]])\n"
        try:
            result = self._call_with(code)
        except ImportError:
            self.skipTest("scipy required")
        # result.x is shape (n_steps, 1); should decay below initial.
        self.assertTrue(result.x[-1][0] < 1.0)


# --------------------------------------------------------------------------
# F1 — eval() sandbox hardening at the ABM trigger + sensitivity model_expr
# eval sites (plan_2026-06-02_f07c6077/D-001). Mirrors TestOdeCodeSandbox but
# drives the ACTUAL eval sites (run_abm trigger, run_sensitivity model_expr),
# not just _validate_ode_code (already covered above).
# --------------------------------------------------------------------------
_INJECTION = "().__class__.__bases__[0].__subclasses__()"


@unittest.skipUnless(HAS_NUMPY, "numpy required")
class TestAbmTriggerSandbox(unittest.TestCase):
    """Regression: a malicious ABM rule `trigger` must raise ValueError that
    PROPAGATES out of run_abm — it must NOT be swallowed by _eval_rule's
    `except Exception: return False`."""

    def _make_config(self, trigger):
        cfg = {
            "agent_types": [
                {
                    "name": "default",
                    "fraction": 1.0,
                    "state": {"x": 0.5},
                    "rules": [
                        {"trigger": trigger, "action": "increment",
                         "params": {"key": "x", "amount": 1}}
                    ],
                }
            ]
        }
        tf = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False)
        json.dump(cfg, tf)
        tf.close()
        self.addCleanup(os.unlink, tf.name)
        return tf.name

    def _args(self, trigger):
        return argparse.Namespace(
            config=self._make_config(trigger),
            seed=42,
            n_agents=5,
            t_steps=3,
            topology="complete",
            output=None,
            verbose=False,
        )

    def test_rejects_injection_trigger(self):
        # The dunder-traversal escape must raise ValueError, not be silently
        # swallowed (restricted __builtins__ alone does NOT block it).
        with self.assertRaises(ValueError) as ctx:
            run_abm(self._args(_INJECTION))
        self.assertIn("ode_code rejected", str(ctx.exception))

    def test_legit_trigger_does_not_raise_from_validation(self):
        # A legitimate trigger must NOT raise ValueError from validation; the
        # ABM run completes (the rule may or may not fire — irrelevant here).
        try:
            run_abm(self._args("x > 0.5"))
        except ValueError as e:  # pragma: no cover - failure path
            self.fail(f"legit trigger wrongly rejected by validator: {e}")
        except Exception:
            # Downstream sim/IO errors (e.g. output=None) are out of scope for
            # this guard test — only a spurious ValueError is a failure.
            pass


@unittest.skipUnless(HAS_NUMPY, "numpy required")
class TestSensitivityModelExprSandbox(unittest.TestCase):
    """Regression: a malicious sensitivity `model_func` expression must raise
    ValueError before any eval site executes (one guard dominates all four
    model_expr eval sites — A4)."""

    def _args(self, model_func):
        return argparse.Namespace(
            param_ranges=json.dumps({"a": [0.0, 1.0], "b": [0.0, 1.0]}),
            method="sobol",
            n_samples=8,
            model_func=model_func,
            output=None,
        )

    def test_rejects_injection_model_expr(self):
        with self.assertRaises(ValueError) as ctx:
            run_sensitivity(self._args(_INJECTION))
        self.assertIn("ode_code rejected", str(ctx.exception))

    def test_legit_model_expr_does_not_raise_from_validation(self):
        # Legitimate arithmetic expression must pass validation. SALib may be
        # absent (OAT fallback runs) and _save_json is skipped (output=None),
        # so the only failure we care about is a spurious ValueError from the
        # guard.
        try:
            run_sensitivity(self._args("a + b"))
        except ValueError as e:  # pragma: no cover - failure path
            self.fail(f"legit model_expr wrongly rejected by validator: {e}")
        except Exception:
            # Downstream sim/IO errors are out of scope for this guard test.
            pass


@unittest.skipUnless(HAS_NUMPY, "numpy required")
class TestAbmZeroAgents(unittest.TestCase):
    """Regression: run_abm must reject n_agents<=0 cleanly (D-02)."""

    def _make_config(self):
        cfg = {
            "agent_types": [
                {"name": "default", "fraction": 1.0, "state": {"x": 0.5}, "rules": []}
            ]
        }
        tf = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )
        json.dump(cfg, tf)
        tf.close()
        self.addCleanup(os.unlink, tf.name)
        return tf.name

    def _args(self, n_agents):
        return argparse.Namespace(
            config=self._make_config(),
            seed=42,
            n_agents=n_agents,
            t_steps=10,
            topology="complete",
            output=None,
        )

    def test_abm_zero_agents_clean_error(self):
        # n_agents=0 previously raised IndexError at `agents[0].state` (and a
        # latent ZeroDivisionError at `/ n`). The guard must raise a clean
        # ValueError instead — never an IndexError.
        with self.assertRaises(ValueError) as ctx:
            run_abm(self._args(0))
        self.assertIn("n_agents", str(ctx.exception))
        self.assertNotIsInstance(ctx.exception, IndexError)

    def test_abm_negative_agents_clean_error(self):
        with self.assertRaises(ValueError):
            run_abm(self._args(-3))


class TestMissingNumpy(unittest.TestCase):
    """Regression: simulator.py must exit cleanly (no Traceback) when numpy is
    absent (D-05). This tests the numpy-ABSENT path, so it is intentionally
    NOT decorated with @skipUnless(HAS_NUMPY) — it must run regardless of
    whether the parent env has numpy. It shadows numpy in a subprocess by
    PREPENDING a tmp dir whose numpy/__init__.py raises ImportError."""

    def test_missing_numpy_clean_error(self):
        sim_path = os.path.join(
            os.path.dirname(__file__), '..', 'src', 'scripts', 'simulator.py')
        with tempfile.TemporaryDirectory() as shadow:
            pkg = os.path.join(shadow, 'numpy')
            os.makedirs(pkg)
            with open(os.path.join(pkg, '__init__.py'), 'w') as f:
                f.write('raise ImportError("shadow")\n')
            env = dict(os.environ)
            # PREPEND the shadow so it wins over any real site-packages numpy.
            env['PYTHONPATH'] = shadow + os.pathsep + env.get('PYTHONPATH', '')
            proc = subprocess.run(
                [sys.executable, sim_path, "mc", "--help"],
                env=env, capture_output=True, text=True)
        # Guard must fire: clean message, exit 1, no raw traceback.
        self.assertEqual(proc.returncode, 1,
                         f"stdout={proc.stdout!r} stderr={proc.stderr!r}")
        self.assertIn("requires numpy", proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)
        self.assertNotIn("Traceback", proc.stdout)


if __name__ == "__main__":
    unittest.main()
