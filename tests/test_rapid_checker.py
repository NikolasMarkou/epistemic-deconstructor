#!/usr/bin/env python3
"""Tests for src/scripts/rapid_checker.py"""

import json
import os
import sys
import tempfile
import unittest

# Allow importing from src/scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'scripts'))

from rapid_checker import RapidChecker


DOMAINS_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'src', 'config', 'domains.json'
)


class TestDomainsConfigSchema(unittest.TestCase):
    """Structural invariants for src/config/domains.json (Opportunity #3 audit)."""

    @classmethod
    def setUpClass(cls):
        with open(DOMAINS_PATH) as f:
            cls.data = json.load(f)

    def test_has_schema_field(self):
        self.assertIn('_schema', self.data,
                      "domains.json must declare its format in a _schema field (parity with archetypes.json, trace_catalog.json)")

    def test_has_comment_field(self):
        self.assertIn('_comment', self.data)

    def test_every_metric_is_four_tuple_of_numbers(self):
        meta_keys = {'_comment', '_schema'}
        for domain_id, metrics in self.data.items():
            if domain_id in meta_keys:
                continue
            self.assertIsInstance(metrics, dict, f"domain {domain_id} must be a dict")
            self.assertGreater(len(metrics), 0, f"domain {domain_id} must have ≥1 metric")
            for metric_name, bounds in metrics.items():
                self.assertIsInstance(bounds, list,
                                     f"{domain_id}.{metric_name} must be a list")
                self.assertEqual(len(bounds), 4,
                                f"{domain_id}.{metric_name} must be a 4-tuple, got {len(bounds)}")
                for v in bounds:
                    self.assertIsInstance(v, (int, float),
                                         f"{domain_id}.{metric_name} contains non-number: {v!r}")


class TestRapidChecker(unittest.TestCase):

    def setUp(self):
        self.tmpfile = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
        self.tmpfile.close()
        os.unlink(self.tmpfile.name)  # start fresh
        self.checker = RapidChecker(self.tmpfile.name)

    def tearDown(self):
        if os.path.exists(self.tmpfile.name):
            os.unlink(self.tmpfile.name)

    def test_start_session(self):
        aid = self.checker.start("Test Claim")
        self.assertTrue(aid.startswith("A"))
        self.assertEqual(self.checker.assessment.title, "Test Claim")

    def test_require_session_raises(self):
        """Operations without a session should raise RuntimeError."""
        with self.assertRaises(RuntimeError):
            self.checker.add_coherence("data-task-match", True)

    def test_coherence_tracking(self):
        self.checker.start("Test")
        self.checker.add_coherence("data-task-match", True)
        self.checker.add_coherence("metric-task-match", False, notes="Wrong metrics")
        summary = self.checker.coherence_summary()
        self.assertEqual(summary['passed'], 1)
        self.assertEqual(summary['failed'], 1)
        self.assertFalse(summary['all_passed'])

    def test_flag_tracking(self):
        self.checker.start("Test")
        fid = self.checker.add_flag("methodology", "No baseline")
        self.assertEqual(fid, "F1")
        counts = self.checker.flag_count()
        self.assertEqual(counts['total'], 1)
        self.assertEqual(counts['by_category']['methodology'], 1)

    def test_flag_invalid_category(self):
        self.checker.start("Test")
        with self.assertRaises(ValueError):
            self.checker.add_flag("invalid_category", "desc")

    def test_verdict_credible(self):
        """Clean assessment -> CREDIBLE."""
        self.checker.start("Test")
        self.checker.add_coherence("data-task-match", True)
        v = self.checker.compute_verdict()
        self.assertEqual(v['verdict'], 'CREDIBLE')

    def test_verdict_reject_critical_flag(self):
        """Critical flag -> REJECT."""
        self.checker.start("Test")
        self.checker.add_flag("methodology", "Fatal", severity="critical")
        v = self.checker.compute_verdict()
        self.assertEqual(v['verdict'], 'REJECT')

    def test_verdict_reject_coherence_failure(self):
        """Coherence failure -> REJECT."""
        self.checker.start("Test")
        self.checker.add_coherence("data-task-match", False)
        v = self.checker.compute_verdict()
        self.assertEqual(v['verdict'], 'REJECT')

    def test_calibration(self):
        self.checker.start("Test")
        result = self.checker.calibrate("accuracy", 0.99, "ml_classification")
        self.assertEqual(result['assessment'], 'suspicious')

    def test_calibration_plausible(self):
        self.checker.start("Test")
        result = self.checker.calibrate("accuracy", 0.85, "ml_classification")
        self.assertEqual(result['assessment'], 'plausible')

    def test_save_load_roundtrip(self):
        self.checker.start("Test Roundtrip")
        self.checker.add_coherence("data-task-match", True)
        self.checker.add_flag("results", "Suspicious")

        checker2 = RapidChecker(self.tmpfile.name)
        self.assertEqual(checker2.assessment.title, "Test Roundtrip")
        self.assertEqual(len(checker2.assessment.red_flags), 1)
        self.assertIn("data-task-match", checker2.assessment.coherence_checks)


if __name__ == '__main__':
    unittest.main()
