#!/usr/bin/env python3
"""Tests for src/scripts/common.py"""

import json
import os
import sys
import tempfile
import unittest

# Allow importing from src/scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'scripts'))

from common import (
    POSTERIOR_EPSILON,
    clamp_probability,
    bayesian_update,
    load_json,
    save_json,
)


class TestClampProbability(unittest.TestCase):

    def test_clamp_zero(self):
        self.assertAlmostEqual(clamp_probability(0.0), POSTERIOR_EPSILON)

    def test_clamp_one(self):
        self.assertAlmostEqual(clamp_probability(1.0), 1.0 - POSTERIOR_EPSILON)

    def test_normal_value_unchanged(self):
        self.assertAlmostEqual(clamp_probability(0.5), 0.5)

    def test_near_zero(self):
        result = clamp_probability(1e-20)
        self.assertGreater(result, 0.0)

    def test_near_one(self):
        result = clamp_probability(1.0 - 1e-20)
        self.assertLess(result, 1.0)

    def test_custom_epsilon(self):
        self.assertAlmostEqual(clamp_probability(0.001, eps=0.01), 0.01)


class TestBayesianUpdate(unittest.TestCase):

    def test_neutral_lr(self):
        """LR=1.0 should not change the posterior."""
        self.assertAlmostEqual(bayesian_update(0.5, 1.0), 0.5, places=6)
        self.assertAlmostEqual(bayesian_update(0.3, 1.0), 0.3, places=6)

    def test_strong_confirm(self):
        """LR=10 should increase posterior."""
        result = bayesian_update(0.5, 10.0)
        self.assertGreater(result, 0.5)
        self.assertAlmostEqual(result, 10.0 / 11.0, places=6)

    def test_strong_disconfirm(self):
        """LR=0.1 should decrease posterior."""
        result = bayesian_update(0.5, 0.1)
        self.assertLess(result, 0.5)

    def test_falsify(self):
        """LR=0 should return 0.0."""
        self.assertEqual(bayesian_update(0.5, 0.0), 0.0)
        self.assertEqual(bayesian_update(0.99, 0.0), 0.0)

    def test_repeated_strong_confirm_no_crash(self):
        """Repeated strong confirms should not crash (division-by-zero guard)."""
        posterior = 0.5
        for _ in range(100):
            posterior = bayesian_update(posterior, 10.0)
        # Should be very close to 1 but not exactly 1
        self.assertGreater(posterior, 0.99)
        self.assertLessEqual(posterior, 1.0)

    def test_repeated_strong_disconfirm_no_crash(self):
        """Repeated strong disconfirms should not crash."""
        posterior = 0.5
        for _ in range(100):
            posterior = bayesian_update(posterior, 0.1)
        self.assertLess(posterior, 0.01)
        self.assertGreaterEqual(posterior, 0.0)

    def test_prior_at_zero(self):
        """Prior at 0 should not crash (clamped)."""
        result = bayesian_update(0.0, 5.0)
        self.assertGreater(result, 0.0)

    def test_prior_at_one(self):
        """Prior at 1 should not crash (clamped)."""
        result = bayesian_update(1.0, 0.5)
        self.assertLess(result, 1.0)


class TestJsonIO(unittest.TestCase):

    def test_roundtrip(self):
        """save_json then load_json should return same data."""
        data = {"key": "value", "number": 42, "list": [1, 2, 3]}
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            save_json(path, data)
            loaded = load_json(path)
            self.assertEqual(loaded, data)
        finally:
            os.unlink(path)

    def test_load_nonexistent(self):
        """load_json on nonexistent file should return None."""
        self.assertIsNone(load_json('/tmp/nonexistent_test_file_12345.json'))

    def test_overwrite(self):
        """save_json should overwrite existing data."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            save_json(path, {"old": True})
            save_json(path, {"new": True})
            loaded = load_json(path)
            self.assertEqual(loaded, {"new": True})
        finally:
            os.unlink(path)


if __name__ == '__main__':
    unittest.main()
