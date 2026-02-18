#!/usr/bin/env python3
"""Tests for src/scripts/rapid_checker.py"""

import os
import sys
import tempfile
import unittest

# Allow importing from src/scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'scripts'))

from rapid_checker import RapidChecker


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
