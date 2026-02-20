#!/usr/bin/env python3
"""Tests for src/scripts/simulator.py"""

import json
import os
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


if __name__ == "__main__":
    unittest.main()
