#!/usr/bin/env python3
"""Tests for src/scripts/rapid_checker.py"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

# Allow importing from src/scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'scripts'))

from rapid_checker import RapidChecker


SCRIPT_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'src', 'scripts', 'rapid_checker.py'
)


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


class TestRapidCheckerSubcommandSurface(unittest.TestCase):
    """Tests for CLI subcommands previously without coverage:
    `flag-remove`, `domains`, `status` (plan_2026-05-25_cdd1f345 audit fix).
    """

    def setUp(self):
        self.tmpfile = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
        self.tmpfile.close()
        os.unlink(self.tmpfile.name)
        self.checker = RapidChecker(self.tmpfile.name)

    def tearDown(self):
        if os.path.exists(self.tmpfile.name):
            os.unlink(self.tmpfile.name)

    def test_flag_remove_existing_flag(self):
        """`flag-remove` (RapidChecker.remove_flag) deletes by flag_id."""
        self.checker.start("Test")
        fid = self.checker.add_flag("results", "Suspicious accuracy")
        self.assertEqual(len(self.checker.assessment.red_flags), 1)
        removed = self.checker.remove_flag(fid)
        self.assertTrue(removed)
        self.assertEqual(len(self.checker.assessment.red_flags), 0)

    def test_flag_remove_nonexistent_returns_false(self):
        """Removing a non-existent flag_id returns False without error."""
        self.checker.start("Test")
        self.checker.add_flag("results", "Initial")
        self.assertFalse(self.checker.remove_flag("F-DOES-NOT-EXIST"))
        self.assertEqual(len(self.checker.assessment.red_flags), 1)

    def test_flag_remove_requires_session(self):
        """`remove_flag` must enforce the session-required invariant."""
        with self.assertRaises(RuntimeError):
            self.checker.remove_flag("F-1")

    def test_status_returns_formatted_summary(self):
        """`status` (RapidChecker.status) returns a multi-line summary string."""
        self.checker.start("Status Test")
        self.checker.add_coherence("data-task-match", True)
        self.checker.add_flag("results", "Sample flag")
        out = self.checker.status()
        self.assertIn("Status Test", out)
        self.assertIn("Coherence:", out)
        self.assertIn("Red Flags:", out)

    def test_status_requires_session(self):
        with self.assertRaises(RuntimeError):
            self.checker.status()

    def test_domains_subcommand_invokes_calibration_map(self):
        """`domains` subcommand reads DOMAIN_CALIBRATION; verify the map is
        present and well-formed (4-tuple bounds per metric)."""
        from rapid_checker import DOMAIN_CALIBRATION
        self.assertIsInstance(DOMAIN_CALIBRATION, dict)
        self.assertGreater(len(DOMAIN_CALIBRATION), 0)
        for domain, metrics in DOMAIN_CALIBRATION.items():
            self.assertIsInstance(metrics, dict, f"{domain} metrics not a dict")
            self.assertGreater(len(metrics), 0, f"{domain} has no metrics")
            for metric, bounds in metrics.items():
                self.assertEqual(len(bounds), 4,
                                 f"{domain}.{metric} bounds not a 4-tuple")


class TestDomainCalibrationValidator(unittest.TestCase):
    """Audit H13 (plan_2026-05-28_ad87937f/D-004): _validate_domain_calibration
    rejects non-numeric, non-finite, or wrong-shape bounds."""

    def test_rejects_non_numeric_bound(self):
        from rapid_checker import _validate_domain_calibration
        bad = {"ml": {"accuracy": [0.99, "bad", 0.90, 0.98]}}
        with self.assertRaises(ValueError) as ctx:
            _validate_domain_calibration(bad, "<test>")
        self.assertIn("not numeric", str(ctx.exception))

    def test_rejects_wrong_length(self):
        from rapid_checker import _validate_domain_calibration
        bad = {"ml": {"accuracy": [0.99, 0.70, 0.90]}}  # 3 elements
        with self.assertRaises(ValueError) as ctx:
            _validate_domain_calibration(bad, "<test>")
        self.assertIn("4-element", str(ctx.exception))

    def test_rejects_non_finite_bound(self):
        from rapid_checker import _validate_domain_calibration
        import math as _math
        bad = {"ml": {"accuracy": [_math.inf, 0.70, 0.90, 0.98]}}
        with self.assertRaises(ValueError) as ctx:
            _validate_domain_calibration(bad, "<test>")
        self.assertIn("finite", str(ctx.exception))

    def test_accepts_lower_is_better_non_monotonic(self):
        # mape: lower is better → [0.01, 0.15, 0.05, 0.02] intentionally non-monotonic.
        from rapid_checker import _validate_domain_calibration
        ok = {"ml_regression": {"mape": [0.01, 0.15, 0.05, 0.02]}}
        _validate_domain_calibration(ok, "<test>")  # must not raise


class TestLoadGuard(unittest.TestCase):
    """RC2: a {} / malformed-JSON state file must exit 1 cleanly (no traceback)."""

    def _run(self, content):
        d = tempfile.mkdtemp()
        try:
            path = os.path.join(d, 'rapid_assessment.json')
            with open(path, 'w') as f:
                f.write(content)
            return subprocess.run(
                [sys.executable, SCRIPT_PATH, '--file', path, 'status'],
                capture_output=True, text=True,
            )
        finally:
            import shutil
            shutil.rmtree(d, ignore_errors=True)

    def test_empty_dict_exits_clean(self):
        r = self._run('{}')
        self.assertEqual(r.returncode, 1)
        self.assertNotIn('Traceback', r.stderr)
        self.assertIn('missing required fields', r.stderr)

    def test_malformed_json_exits_clean(self):
        r = self._run('{not json')
        self.assertEqual(r.returncode, 1)
        self.assertNotIn('Traceback', r.stderr)


if __name__ == '__main__':
    unittest.main()
