#!/usr/bin/env python3
"""Tests for src/scripts/bayesian_tracker.py"""

import os
import sys
import tempfile
import unittest

# Allow importing from src/scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'scripts'))

from bayesian_tracker import BayesianTracker, Status


class TestBayesianTracker(unittest.TestCase):

    def setUp(self):
        self.tmpfile = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
        self.tmpfile.close()
        os.unlink(self.tmpfile.name)  # start fresh
        self.tracker = BayesianTracker(self.tmpfile.name)

    def tearDown(self):
        if os.path.exists(self.tmpfile.name):
            os.unlink(self.tmpfile.name)

    def test_add_hypothesis(self):
        hid = self.tracker.add("Test hypothesis", phase="P0", prior=0.6)
        self.assertEqual(hid, "H1")
        h = self.tracker.hypotheses[hid]
        self.assertEqual(h.statement, "Test hypothesis")
        self.assertAlmostEqual(h.prior, 0.6)
        self.assertAlmostEqual(h.posterior, 0.6)
        self.assertEqual(h.status, Status.ACTIVE.value)

    def test_add_invalid_prior(self):
        with self.assertRaises(ValueError):
            self.tracker.add("Bad prior", prior=0.0)
        with self.assertRaises(ValueError):
            self.tracker.add("Bad prior", prior=1.0)

    def test_update_hypothesis(self):
        hid = self.tracker.add("H", prior=0.5)
        new_p = self.tracker.update(hid, "Evidence", preset="strong_confirm")
        self.assertGreater(new_p, 0.5)

    def test_update_invalid_id(self):
        with self.assertRaises(KeyError):
            self.tracker.update("H999", "Evidence", preset="neutral")

    def test_remove_hypothesis(self):
        hid = self.tracker.add("To remove", prior=0.5)
        self.assertTrue(self.tracker.remove(hid))
        self.assertNotIn(hid, self.tracker.hypotheses)

    def test_remove_nonexistent(self):
        self.assertFalse(self.tracker.remove("H999"))

    def test_save_load_roundtrip(self):
        self.tracker.add("H1", prior=0.5)
        self.tracker.add("H2", prior=0.7)
        self.tracker.update("H1", "ev1", preset="strong_confirm")
        self.tracker.add_flag("methodology", "No baseline")
        self.tracker.add_coherence("data-task-match", "PASS")

        # Load from same file
        tracker2 = BayesianTracker(self.tmpfile.name)
        self.assertEqual(len(tracker2.hypotheses), 2)
        self.assertEqual(len(tracker2.red_flags), 1)
        self.assertEqual(len(tracker2.coherence_checks), 1)
        self.assertAlmostEqual(
            tracker2.hypotheses["H1"].posterior,
            self.tracker.hypotheses["H1"].posterior
        )

    def test_monotonic_ids(self):
        h1 = self.tracker.add("First", prior=0.5)
        h2 = self.tracker.add("Second", prior=0.5)
        self.tracker.remove(h1)
        h3 = self.tracker.add("Third", prior=0.5)
        # H3 should be H3, not H1 (IDs are monotonic, not reused)
        self.assertEqual(h1, "H1")
        self.assertEqual(h2, "H2")
        self.assertEqual(h3, "H3")

    def test_verdict_credible(self):
        """No flags -> CREDIBLE."""
        v = self.tracker.get_verdict()
        self.assertEqual(v['verdict'], 'CREDIBLE')

    def test_verdict_reject_critical_flag(self):
        """Critical flag -> REJECT."""
        self.tracker.add_flag("methodology", "Fatal flaw", severity="critical")
        v = self.tracker.get_verdict()
        self.assertEqual(v['verdict'], 'REJECT')

    def test_repeated_strong_confirm_no_crash(self):
        """50x strong_confirm should not crash (div-by-zero guard)."""
        hid = self.tracker.add("Test", prior=0.5)
        for _ in range(50):
            self.tracker.update(hid, "evidence", preset="strong_confirm")
        h = self.tracker.hypotheses[hid]
        self.assertGreater(h.posterior, 0.99)
        self.assertEqual(h.status, Status.CONFIRMED.value)


if __name__ == '__main__':
    unittest.main()
