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


    def test_update_refuted_trait_raises(self):
        """Updating a REFUTED trait should raise ValueError."""
        tid = self.tracker.add_trait("Test", prior=0.5)
        # Use disconfirm preset (LR=0.1) repeatedly to push below 0.10
        for _ in range(20):
            try:
                self.tracker.update_trait(tid, "Counter evidence", preset="disconfirm")
            except ValueError:
                break
        self.assertEqual(self.tracker.traits[tid].status, "REFUTED")
        with self.assertRaises(ValueError) as ctx:
            self.tracker.update_trait(tid, "New evidence", preset="strong_indicator")
        self.assertIn("REFUTED", str(ctx.exception))

    def test_ocean_profile_empty(self):
        """Empty tracker -> all OCEAN entries unknown/UNASSESSED."""
        profile = self.tracker.get_ocean_profile()
        for abbrev in ('O', 'C', 'E', 'A', 'N'):
            self.assertEqual(profile[abbrev]['level'], 'unknown')
            self.assertEqual(profile[abbrev]['status'], 'UNASSESSED')

    def test_ocean_profile_with_traits(self):
        """Added OCEAN traits appear correctly in profile."""
        self.tracker.add_trait("High Openness", category="openness",
                                polarity="high", prior=0.7)
        self.tracker.add_trait("Low Conscientiousness", category="conscientiousness",
                                polarity="low", prior=0.6)
        profile = self.tracker.get_ocean_profile()
        self.assertEqual(profile['O']['level'], 'high')
        self.assertAlmostEqual(profile['O']['confidence'], 0.7)
        self.assertEqual(profile['C']['level'], 'low')
        self.assertEqual(profile['E']['level'], 'unknown')

    def test_dark_triad_profile_empty(self):
        """Empty tracker -> all DT entries unknown."""
        profile = self.tracker.get_dark_triad_profile()
        for abbrev in ('N', 'M', 'P'):
            self.assertEqual(profile[abbrev]['level'], 'unknown')
            self.assertEqual(profile[abbrev]['status'], 'UNASSESSED')

    def test_dark_triad_profile_with_traits(self):
        """Added DT traits map correctly."""
        self.tracker.add_trait("High Narcissism", category="narcissism",
                                polarity="high", prior=0.8)
        profile = self.tracker.get_dark_triad_profile()
        self.assertEqual(profile['N']['level'], 'high')
        self.assertAlmostEqual(profile['N']['confidence'], 0.8)
        self.assertEqual(profile['M']['level'], 'unknown')

    def test_mice_profile_empty(self):
        """Empty tracker -> all MICE scores 0, UNASSESSED."""
        profile = self.tracker.get_mice_profile()
        for name in ('Money', 'Ideology', 'Coercion', 'Ego'):
            self.assertEqual(profile[name]['score'], 0)
            self.assertEqual(profile[name]['status'], 'UNASSESSED')

    def test_mice_profile_with_traits(self):
        """MICE scores map from trait posteriors."""
        self.tracker.add_trait("Money motivated", category="money",
                                polarity="high", prior=0.7)
        self.tracker.add_trait("Ego driven", category="ego",
                                polarity="high", prior=0.6)
        profile = self.tracker.get_mice_profile()
        self.assertAlmostEqual(profile['Money']['score'], 0.7)
        self.assertAlmostEqual(profile['Ego']['score'], 0.6)
        self.assertEqual(profile['Ideology']['score'], 0)

    def test_dt_risk_no_traits(self):
        """No DT traits -> risk = 0.0."""
        risk = self.tracker.calculate_dt_risk()
        self.assertAlmostEqual(risk, 0.0)

    def test_dt_risk_high_all(self):
        """High on all three DT traits -> risk near 0.9."""
        self.tracker.add_trait("High N", category="narcissism",
                                polarity="high", prior=0.9)
        self.tracker.add_trait("High M", category="machiavellianism",
                                polarity="high", prior=0.9)
        self.tracker.add_trait("High P", category="psychopathy",
                                polarity="high", prior=0.9)
        risk = self.tracker.calculate_dt_risk()
        # 0.9*0.3 + 0.9*0.4 + 0.9*0.3 = 0.9
        self.assertAlmostEqual(risk, 0.9)

    def test_dt_risk_low_polarity(self):
        """Low polarity inverts confidence in risk calculation."""
        self.tracker.add_trait("Low Narcissism", category="narcissism",
                                polarity="low", prior=0.9)
        risk = self.tracker.calculate_dt_risk()
        # low polarity -> 1-0.9 = 0.1 contribution for N, others 0
        # 0.1*0.3 = 0.03
        self.assertAlmostEqual(risk, 0.03)


if __name__ == '__main__':
    unittest.main()
