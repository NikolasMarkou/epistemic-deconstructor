#!/usr/bin/env python3
"""Tests for src/scripts/belief_tracker.py"""

import os
import sys
import tempfile
import unittest

# Allow importing from src/scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'scripts'))

from belief_tracker import BeliefTracker


class TestBeliefTracker(unittest.TestCase):

    def setUp(self):
        self.tmpfile = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
        self.tmpfile.close()
        os.unlink(self.tmpfile.name)  # start fresh
        self.tracker = BeliefTracker(self.tmpfile.name)

    def tearDown(self):
        if os.path.exists(self.tmpfile.name):
            os.unlink(self.tmpfile.name)

    def test_add_trait(self):
        tid = self.tracker.add_trait("High Neuroticism", category="neuroticism",
                                      polarity="high", prior=0.5)
        self.assertEqual(tid, "T1")
        t = self.tracker.traits[tid]
        self.assertEqual(t.trait, "High Neuroticism")
        self.assertAlmostEqual(t.prior, 0.5)

    def test_add_invalid_prior(self):
        with self.assertRaises(ValueError):
            self.tracker.add_trait("Bad", prior=0.0)

    def test_update_trait(self):
        tid = self.tracker.add_trait("Test", prior=0.5)
        new_p = self.tracker.update_trait(tid, "Evidence", preset="strong_indicator")
        self.assertGreater(new_p, 0.5)

    def test_update_invalid_id(self):
        with self.assertRaises(KeyError):
            self.tracker.update_trait("T999", "Evidence", preset="neutral")

    def test_save_load_roundtrip(self):
        self.tracker.set_subject("Test Subject", "Unit test")
        self.tracker.add_trait("T1", prior=0.5)
        self.tracker.add_baseline("linguistic", "Short sentences")
        self.tracker.add_deviation("Longer sentences", significance="moderate")

        tracker2 = BeliefTracker(self.tmpfile.name)
        self.assertEqual(tracker2.subject_name, "Test Subject")
        self.assertEqual(len(tracker2.traits), 1)
        self.assertEqual(len(tracker2.baselines), 1)
        self.assertEqual(len(tracker2.deviations), 1)

    def test_repeated_strong_indicator_no_crash(self):
        """50x strong_indicator should not crash (div-by-zero guard)."""
        tid = self.tracker.add_trait("Test", prior=0.5)
        for _ in range(50):
            self.tracker.update_trait(tid, "evidence", preset="strong_indicator")
        t = self.tracker.traits[tid]
        self.assertGreater(t.posterior, 0.99)
        self.assertEqual(t.status, "CONFIRMED")

    def test_baseline_roundtrip(self):
        bid = self.tracker.add_baseline("linguistic", "Uses 'we' often", value="frequent")
        baselines = self.tracker.get_baselines("linguistic")
        self.assertEqual(len(baselines), 1)
        self.assertEqual(baselines[0].id, bid)
        self.assertEqual(baselines[0].value, "frequent")

    def test_deviation_invalid_significance(self):
        with self.assertRaises(ValueError):
            self.tracker.add_deviation("test", significance="extreme")


if __name__ == '__main__':
    unittest.main()
