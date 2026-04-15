#!/usr/bin/env python3
"""Tests for src/scripts/scope_auditor.py"""

import json
import math
import os
import subprocess
import sys
import tempfile
import unittest

# Allow importing from src/scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'scripts'))

from scope_auditor import (
    ScopeAuditor,
    load_archetype_library,
    pearson_correlation,
    pearson_pvalue,
    read_csv_column,
    load_indices_dir,
)


SCRIPT_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'src', 'scripts', 'scope_auditor.py'
)


class TestArchetypeLibrary(unittest.TestCase):

    def test_load_default_library(self):
        """Default library loads the shipped archetypes.json."""
        lib = load_archetype_library()
        # Should have at least 8 archetypes per acceptance criterion
        self.assertGreaterEqual(len(lib), 8)
        # Cyprus example archetype must be present
        self.assertIn('speculative_asset_market', lib)

    def test_library_schema_per_archetype(self):
        """Every archetype must have name, description, accomplices."""
        lib = load_archetype_library()
        for aid, entry in lib.items():
            self.assertIn('name', entry, f"{aid} missing name")
            self.assertIn('accomplices', entry, f"{aid} missing accomplices")
            self.assertIsInstance(entry['accomplices'], list)
            for acc in entry['accomplices']:
                self.assertIn('domain', acc)
                self.assertIn('mechanism', acc)
                self.assertIn('prior', acc)
                self.assertTrue(0.0 <= acc['prior'] <= 1.0)

    def test_cyprus_archetype_has_all_missed_domains(self):
        """
        Acceptance test: speculative_asset_market archetype must name
        illicit finance, investment legislation, immigration, and
        tourism/nomad domains as accomplices — the four missed by the
        original Cyprus analysis.
        """
        lib = load_archetype_library()
        sam = lib['speculative_asset_market']
        domains_joined = ' | '.join(a['domain'].lower() for a in sam['accomplices'])
        self.assertIn('illicit', domains_joined)
        self.assertIn('legislation', domains_joined)
        self.assertIn('immigration', domains_joined)
        # Nomad appears in the same accomplice label as immigration
        self.assertIn('nomad', domains_joined)

    def test_load_missing_config_fallback(self):
        """Missing config file yields a minimal fallback library."""
        lib = load_archetype_library('/nonexistent/path/archetypes.json')
        # Fallback library has at least one archetype
        self.assertGreaterEqual(len(lib), 1)

    def test_load_invalid_json_raises(self):
        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
            f.write("{not valid json")
            bad = f.name
        try:
            with self.assertRaises(RuntimeError):
                load_archetype_library(bad)
        finally:
            os.unlink(bad)


class TestScopeAuditorState(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
        self.tmp.close()
        os.unlink(self.tmp.name)
        self.auditor = ScopeAuditor(self.tmp.name)

    def tearDown(self):
        if os.path.exists(self.tmp.name):
            os.unlink(self.tmp.name)

    def test_start_creates_session(self):
        aid = self.auditor.start("Cyprus real estate")
        self.assertTrue(aid.startswith("SA"))
        self.assertEqual(self.auditor.audit.target, "Cyprus real estate")

    def test_start_refuses_overwrite(self):
        self.auditor.start("First")
        with self.assertRaises(RuntimeError):
            self.auditor.start("Second")

    def test_start_force_overwrite(self):
        self.auditor.start("First")
        self.auditor.start("Second", force=True)
        self.assertEqual(self.auditor.audit.target, "Second")

    def test_require_session_raises_without_start(self):
        with self.assertRaises(RuntimeError):
            self.auditor.add_candidate("x", "y", 0.2, "manual")

    def test_enumerate_adds_candidates(self):
        self.auditor.start("test")
        accomplices = self.auditor.enumerate_archetype('speculative_asset_market')
        self.assertGreaterEqual(len(accomplices), 4)
        self.assertGreaterEqual(len(self.auditor.audit.candidates), len(accomplices))
        self.assertIn('speculative_asset_market',
                      self.auditor.audit.archetypes_queried)

    def test_enumerate_unknown_archetype_raises(self):
        self.auditor.start("test")
        with self.assertRaises(KeyError):
            self.auditor.enumerate_archetype('nonexistent_archetype_xyz')

    def test_trace_records_both_directions(self):
        self.auditor.start("test")
        entries = self.auditor.trace_flows(
            inputs=['capital', 'labor'],
            outputs=['product', 'emissions'],
        )
        self.assertEqual(len(entries), 4)
        directions = [e['direction'] for e in entries]
        self.assertEqual(directions.count('input'), 2)
        self.assertEqual(directions.count('output'), 2)

    def test_steelman_also_logs_candidate(self):
        self.auditor.start("test")
        self.auditor.add_steelman('journalist', 'illicit finance',
                                   'sanctions arbitrage')
        self.assertEqual(len(self.auditor.audit.steelman), 1)
        self.assertEqual(len(self.auditor.audit.candidates), 1)
        self.assertIn('M4:steelman', self.auditor.audit.candidates[0]['source'])

    def test_dedupe_candidates(self):
        self.auditor.start("test")
        self.auditor.add_candidate('same domain', 'mech1', 0.2, 'M2:a')
        self.auditor.add_candidate('Same Domain', 'mech2', 0.3, 'M2:b')
        self.auditor.add_candidate('different', 'mech3', 0.1, 'M2:c')
        removed = self.auditor.dedupe_candidates()
        self.assertEqual(removed, 1)
        self.assertEqual(len(self.auditor.audit.candidates), 2)

    def test_persistence_round_trip(self):
        """State persists across ScopeAuditor instances."""
        self.auditor.start("persistent")
        self.auditor.enumerate_archetype('speculative_asset_market')
        n_candidates = len(self.auditor.audit.candidates)

        # Reload
        reloaded = ScopeAuditor(self.tmp.name)
        self.assertIsNotNone(reloaded.audit)
        self.assertEqual(reloaded.audit.target, "persistent")
        self.assertEqual(len(reloaded.audit.candidates), n_candidates)

    def test_gate_fails_with_no_archetype(self):
        self.auditor.start("test")
        gate = self.auditor.gate_status()
        self.assertFalse(gate['pass'])

    def test_gate_passes_with_cyprus_example(self):
        """Running the Cyprus worked example must pass the gate."""
        self.auditor.start("Cyprus real estate")
        self.auditor.enumerate_archetype('speculative_asset_market')
        gate = self.auditor.gate_status()
        self.assertTrue(gate['pass'], f"Gate should pass; got {gate}")
        self.assertGreaterEqual(gate['candidates_unique'], 3)

    def test_report_mentions_candidates(self):
        self.auditor.start("test")
        self.auditor.enumerate_archetype('speculative_asset_market')
        report = self.auditor.report()
        self.assertIn('Scope Audit', report)
        self.assertIn('illicit', report.lower())


class TestPearsonCorrelation(unittest.TestCase):

    def test_perfect_positive(self):
        r = pearson_correlation([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
        self.assertAlmostEqual(r, 1.0, places=10)

    def test_perfect_negative(self):
        r = pearson_correlation([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])
        self.assertAlmostEqual(r, -1.0, places=10)

    def test_zero_correlation(self):
        # Symmetric data — zero correlation
        r = pearson_correlation([1, 2, 3, 4, 5], [3, 1, 4, 1, 5])
        self.assertLess(abs(r), 0.5)

    def test_degenerate_input(self):
        # Constant y → zero variance → return 0 not NaN
        r = pearson_correlation([1, 2, 3], [7, 7, 7])
        self.assertEqual(r, 0.0)
        # Mismatched lengths → 0
        r = pearson_correlation([1, 2], [1, 2, 3])
        self.assertEqual(r, 0.0)

    def test_pvalue_strong_correlation_small_sample(self):
        """
        Strong correlation (r=1) in 10 points has p << 0.05.
        Weak correlation (r=0.1) in 10 points has p > 0.05.
        """
        p_strong = pearson_pvalue(0.95, 10)
        p_weak = pearson_pvalue(0.1, 10)
        self.assertLess(p_strong, 0.01)
        self.assertGreater(p_weak, 0.2)

    def test_pvalue_small_n(self):
        self.assertEqual(pearson_pvalue(0.5, 2), 1.0)


class TestCSVHelpers(unittest.TestCase):

    def test_read_numeric_column_no_header(self):
        with tempfile.NamedTemporaryFile('w', suffix='.csv', delete=False) as f:
            f.write("1.0\n2.0\n3.0\n4.0\n")
            path = f.name
        try:
            values = read_csv_column(path)
            self.assertEqual(values, [1.0, 2.0, 3.0, 4.0])
        finally:
            os.unlink(path)

    def test_read_with_header(self):
        with tempfile.NamedTemporaryFile('w', suffix='.csv', delete=False) as f:
            f.write("date,value\n2020-01,1.0\n2020-02,2.5\n2020-03,3.3\n")
            path = f.name
        try:
            values = read_csv_column(path, column='value')
            self.assertEqual(values, [1.0, 2.5, 3.3])
        finally:
            os.unlink(path)

    def test_load_indices_dir(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, 'index_a.csv'), 'w') as f:
                f.write("1\n2\n3\n4\n5\n")
            with open(os.path.join(d, 'index_b.csv'), 'w') as f:
                f.write("5\n4\n3\n2\n1\n")
            with open(os.path.join(d, 'readme.txt'), 'w') as f:
                f.write("not csv")
            indices = load_indices_dir(d)
            self.assertEqual(set(indices.keys()), {'index_a', 'index_b'})
            self.assertEqual(indices['index_a'], [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_load_missing_dir(self):
        with self.assertRaises(FileNotFoundError):
            load_indices_dir('/nonexistent/dir/xyz')


class TestResidualMatching(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
        self.tmp.close()
        os.unlink(self.tmp.name)
        self.auditor = ScopeAuditor(self.tmp.name)
        self.auditor.start("residual test")

    def tearDown(self):
        if os.path.exists(self.tmp.name):
            os.unlink(self.tmp.name)

    def test_residual_match_flags_strong_correlation(self):
        # Known correlated pair
        residuals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        indices = {
            'strong_corr': [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
            'no_corr': [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        }
        matches = self.auditor.residual_match(residuals, indices)
        strong = next(m for m in matches if m['index'] == 'strong_corr')
        no_corr = next(m for m in matches if m['index'] == 'no_corr')
        self.assertTrue(strong['flagged'])
        self.assertAlmostEqual(strong['r'], 1.0, places=6)
        self.assertFalse(no_corr['flagged'])

    def test_residual_match_adds_candidates_for_flagged(self):
        residuals = list(range(1, 21))
        indices = {
            'driver_a': [x * 2.0 for x in residuals],
        }
        matches = self.auditor.residual_match(residuals, indices)
        self.assertTrue(matches[0]['flagged'])
        # Candidate logged
        self.assertTrue(any('driver_a' in c['domain']
                            for c in self.auditor.audit.candidates))

    def test_residual_match_short_series(self):
        residuals = [1.0, 2.0]
        indices = {'short': [1.0, 2.0]}
        matches = self.auditor.residual_match(residuals, indices)
        self.assertFalse(matches[0]['flagged'])


class TestCLISmoke(unittest.TestCase):
    """Subprocess smoke tests — verify --help and basic commands run."""

    def test_help_exits_zero(self):
        result = subprocess.run(
            [sys.executable, SCRIPT_PATH, '--help'],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn('scope_auditor.py', result.stdout)

    def test_list_archetypes_cli(self):
        result = subprocess.run(
            [sys.executable, SCRIPT_PATH, 'list-archetypes'],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn('speculative_asset_market', result.stdout)

    def test_cyprus_end_to_end_cli(self):
        """Full Cyprus workflow via CLI must pass the gate."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            os.unlink(path)
            env_kwargs = dict(capture_output=True, text=True)
            subprocess.run(
                [sys.executable, SCRIPT_PATH, '--file', path,
                 'start', 'Cyprus real estate'],
                check=True, **env_kwargs,
            )
            subprocess.run(
                [sys.executable, SCRIPT_PATH, '--file', path,
                 'enumerate', '--archetype', 'speculative_asset_market'],
                check=True, **env_kwargs,
            )
            result = subprocess.run(
                [sys.executable, SCRIPT_PATH, '--file', path, 'gate'],
                **env_kwargs,
            )
            self.assertEqual(result.returncode, 0,
                             f"Gate should pass. Output: {result.stdout}")
            self.assertIn('pass: True', result.stdout)
        finally:
            if os.path.exists(path):
                os.unlink(path)


if __name__ == '__main__':
    unittest.main()
