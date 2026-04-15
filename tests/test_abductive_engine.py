#!/usr/bin/env python3
"""Tests for src/scripts/abductive_engine.py

No mocks — all tests use real file I/O and real JSON round-trips.
Target: >=90% line coverage on abductive_engine.py.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

# Allow importing from src/scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'scripts'))

from abductive_engine import (
    AbductiveEngine,
    AbductiveState,
    load_trace_catalog,
    load_archetype_library_for_analogy,
    LLM_PARAMETRIC_MAX_PRIOR,
    LLM_PARAMETRIC_MAX_LR,
    DEFAULT_COVERAGE_THRESHOLD,
    VALID_SOURCES,
    _odds_update,
    _tokenize,
    _token_overlap,
)


SCRIPT_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'src', 'scripts', 'abductive_engine.py'
)
TRACKER_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'src', 'scripts', 'bayesian_tracker.py'
)


# ---------------------------------------------------------------------------
# Catalog loader
# ---------------------------------------------------------------------------

class TestTraceCatalog(unittest.TestCase):

    def test_default_catalog_loads(self):
        """Shipped trace_catalog.json loads and has expected categories."""
        cat = load_trace_catalog()
        self.assertIsInstance(cat, dict)
        # Shipped seed includes timing, resource, output_anomaly, failure,
        # behavioral_deviation, generic
        self.assertIn('timing', cat)
        self.assertIn('resource', cat)
        self.assertIn('generic', cat)

    def test_catalog_schema(self):
        """Each category has a candidates list with required fields."""
        cat = load_trace_catalog()
        for category_id, entry in cat.items():
            self.assertIn('candidates', entry, f"{category_id} missing candidates")
            for cand in entry['candidates']:
                self.assertIn('cause', cand)
                self.assertIn('mechanism', cand)
                self.assertIn('prior', cand)
                self.assertTrue(0 < cand['prior'] < 1)

    def test_missing_config_fallback(self):
        """Missing config yields minimal fallback."""
        cat = load_trace_catalog('/nonexistent/path/trace_catalog.json')
        self.assertIn('generic', cat)
        self.assertGreaterEqual(len(cat['generic']['candidates']), 1)

    def test_invalid_json_raises(self):
        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
            f.write("{not valid")
            bad = f.name
        try:
            with self.assertRaises(RuntimeError):
                load_trace_catalog(bad)
        finally:
            os.unlink(bad)

    def test_archetype_library_for_analogy(self):
        """Loading archetype library for AR filters out entries without trace_signatures."""
        lib = load_archetype_library_for_analogy()
        # At least the seeded archetypes carry trace_signatures
        self.assertGreaterEqual(len(lib), 1)
        for aid, entry in lib.items():
            self.assertIn('trace_signatures', entry)
            self.assertIsInstance(entry['trace_signatures'], list)

    def test_archetype_library_missing_file_returns_empty(self):
        """Missing archetypes.json returns empty dict (graceful)."""
        lib = load_archetype_library_for_analogy('/nonexistent/xyz.json')
        self.assertEqual(lib, {})


# ---------------------------------------------------------------------------
# State + session
# ---------------------------------------------------------------------------

class TestAbductiveState(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
        self.tmp.close()
        os.unlink(self.tmp.name)
        self.engine = AbductiveEngine(self.tmp.name)

    def tearDown(self):
        if os.path.exists(self.tmp.name):
            os.unlink(self.tmp.name)
        lock = self.tmp.name + '.lock'
        if os.path.exists(lock):
            os.unlink(lock)

    def test_start_creates_session(self):
        sid = self.engine.start()
        self.assertTrue(sid.startswith("AE"))
        self.assertIsNotNone(self.engine.state)

    def test_start_refuses_overwrite(self):
        self.engine.start()
        with self.assertRaises(RuntimeError):
            self.engine.start()

    def test_start_force_overwrite(self):
        s1 = self.engine.start()
        s2 = self.engine.start(force=True)
        # force=True creates a fresh session
        self.assertIsNotNone(s2)

    def test_require_session_raises_without_start(self):
        with self.assertRaises(RuntimeError):
            self.engine.add_observation("O1", "test")

    def test_add_observation_idempotent(self):
        self.engine.start()
        self.engine.add_observation("O1", "first")
        # Second add should return existing entry, not duplicate
        entry = self.engine.add_observation("O1", "different text")
        self.assertEqual(entry['text'], 'first')
        self.assertEqual(len(self.engine.state.observations), 1)

    def test_persistence_round_trip(self):
        """State persists across AbductiveEngine instances."""
        self.engine.start()
        self.engine.add_observation("O1", "persisted", "timing")
        self.engine.add_candidate(
            cause="test cause",
            mechanism="test mech",
            prior=0.2,
            source='analyst',
            observation_ids=['O1'],
        )
        reloaded = AbductiveEngine(self.tmp.name)
        self.assertIsNotNone(reloaded.state)
        self.assertEqual(len(reloaded.state.observations), 1)
        self.assertEqual(len(reloaded.state.candidates), 1)
        self.assertEqual(reloaded.state.candidates[0]['cause'], 'test cause')


# ---------------------------------------------------------------------------
# Candidate management + provenance caps
# ---------------------------------------------------------------------------

class TestCandidateProvenance(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
        self.tmp.close()
        os.unlink(self.tmp.name)
        self.engine = AbductiveEngine(self.tmp.name)
        self.engine.start()

    def tearDown(self):
        if os.path.exists(self.tmp.name):
            os.unlink(self.tmp.name)
        lock = self.tmp.name + '.lock'
        if os.path.exists(lock):
            os.unlink(lock)

    def test_valid_sources(self):
        for source in VALID_SOURCES:
            cid = self.engine.add_candidate(
                cause=f"c-{source}", mechanism="m", prior=0.2, source=source,
            )
            self.assertTrue(cid.startswith("CAND"))

    def test_invalid_source_rejected(self):
        with self.assertRaises(ValueError):
            self.engine.add_candidate(
                cause="c", mechanism="m", prior=0.2, source='wikipedia',
            )

    def test_llm_parametric_prior_cap_enforced(self):
        """LLM-parametric prior cap is enforced in add_candidate."""
        # At the cap: allowed
        self.engine.add_candidate(
            cause="ok", mechanism="m", prior=LLM_PARAMETRIC_MAX_PRIOR,
            source='llm_parametric',
        )
        # Over the cap: rejected
        with self.assertRaises(ValueError) as cm:
            self.engine.add_candidate(
                cause="bad", mechanism="m", prior=0.5,
                source='llm_parametric',
            )
        self.assertIn("llm-parametric", str(cm.exception).lower())

    def test_non_llm_source_can_have_high_prior(self):
        """Library/analyst sources are NOT capped."""
        self.engine.add_candidate(
            cause="library cause", mechanism="m", prior=0.8,
            source='library',
        )
        self.engine.add_candidate(
            cause="analyst cause", mechanism="m", prior=0.85,
            source='analyst',
        )

    def test_invalid_prior_bounds(self):
        with self.assertRaises(ValueError):
            self.engine.add_candidate(cause="c", mechanism="m", prior=0.0, source='analyst')
        with self.assertRaises(ValueError):
            self.engine.add_candidate(cause="c", mechanism="m", prior=1.0, source='analyst')

    def test_invalid_complexity(self):
        with self.assertRaises(ValueError):
            self.engine.add_candidate(
                cause="c", mechanism="m", prior=0.2, source='analyst',
                complexity=0.0,
            )

    def test_get_candidate_not_found(self):
        with self.assertRaises(KeyError):
            self.engine.get_candidate("CAND999")

    def test_observation_explained_by_updates(self):
        self.engine.add_observation("O1", "test")
        cid = self.engine.add_candidate(
            cause="c", mechanism="m", prior=0.2, source='library',
            observation_ids=['O1'],
        )
        obs = self.engine.state.observations[0]
        self.assertIn(cid, obs['explained_by'])


# ---------------------------------------------------------------------------
# TI — Trace Inversion
# ---------------------------------------------------------------------------

class TestTraceInversion(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
        self.tmp.close()
        os.unlink(self.tmp.name)
        self.engine = AbductiveEngine(self.tmp.name)
        self.engine.start()

    def tearDown(self):
        if os.path.exists(self.tmp.name):
            os.unlink(self.tmp.name)
        lock = self.tmp.name + '.lock'
        if os.path.exists(lock):
            os.unlink(lock)

    def test_invert_produces_library_candidates(self):
        produced = self.engine.invert(
            obs_id="O1", text="p99 latency spike at noon", category="timing",
        )
        self.assertGreaterEqual(len(produced), 1)
        # All library-sourced from catalog
        self.assertTrue(any(c['source'] == 'library' for c in produced))
        # Observation was recorded
        self.assertEqual(len(self.engine.state.observations), 1)

    def test_invert_with_llm_parametric_candidates(self):
        produced = self.engine.invert(
            obs_id="O2", text="unknown symptom", category="generic",
            llm_parametric_candidates=[
                {'cause': 'hypothetical', 'mechanism': 'h', 'prior': 0.25},
            ],
        )
        sources = [c['source'] for c in produced]
        self.assertIn('llm_parametric', sources)

    def test_invert_llm_parametric_prior_capped(self):
        """LLM-parametric prior is silently capped at 0.30 during invert."""
        produced = self.engine.invert(
            obs_id="O3", text="x", category="generic",
            llm_parametric_candidates=[
                {'cause': 'capped', 'mechanism': 'm', 'prior': 0.9},
            ],
        )
        llm_candidates = [c for c in produced if c['source'] == 'llm_parametric']
        for c in llm_candidates:
            self.assertLessEqual(c['prior'], LLM_PARAMETRIC_MAX_PRIOR)

    def test_invert_unknown_category_falls_back(self):
        """Unknown category falls back to generic."""
        produced = self.engine.invert(
            obs_id="O4", text="x", category="nonexistent_category",
        )
        # Should still produce generic candidates
        self.assertGreaterEqual(len(produced), 1)


# ---------------------------------------------------------------------------
# AA — Absence Audit
# ---------------------------------------------------------------------------

class TestAbsenceAudit(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
        self.tmp.close()
        os.unlink(self.tmp.name)
        self.engine = AbductiveEngine(self.tmp.name)
        self.engine.start()

    def tearDown(self):
        if os.path.exists(self.tmp.name):
            os.unlink(self.tmp.name)
        lock = self.tmp.name + '.lock'
        if os.path.exists(lock):
            os.unlink(lock)

    def test_absence_audit_queues_predictions(self):
        entries = self.engine.absence_audit(
            "H1", ["log entry X", "metric Y > 10", "signal Z"]
        )
        self.assertEqual(len(entries), 3)
        for e in entries:
            self.assertEqual(e['status'], 'pending')
            self.assertEqual(e['hypothesis_id'], 'H1')

    def test_absence_audit_empty_predictions_rejected(self):
        with self.assertRaises(ValueError):
            self.engine.absence_audit("H1", [])

    def test_absence_audit_filters_blank_predictions(self):
        entries = self.engine.absence_audit("H1", ["", "valid", "   "])
        self.assertEqual(len(entries), 1)

    def test_close_prediction_observed(self):
        entries = self.engine.absence_audit("H1", ["log X"])
        pid = entries[0]['id']
        closed = self.engine.close_prediction(pid, 'observed', note="found it")
        self.assertEqual(closed['status'], 'observed')
        self.assertEqual(closed['note'], 'found it')

    def test_close_prediction_absent(self):
        entries = self.engine.absence_audit("H1", ["log Y"])
        pid = entries[0]['id']
        closed = self.engine.close_prediction(pid, 'absent')
        self.assertEqual(closed['status'], 'absent')

    def test_close_prediction_invalid_outcome(self):
        entries = self.engine.absence_audit("H1", ["log"])
        with self.assertRaises(ValueError):
            self.engine.close_prediction(entries[0]['id'], 'maybe')

    def test_close_prediction_not_found(self):
        with self.assertRaises(KeyError):
            self.engine.close_prediction("PP999", 'observed')


# ---------------------------------------------------------------------------
# SA — Surplus Audit
# ---------------------------------------------------------------------------

class TestSurplusAudit(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
        self.tmp.close()
        os.unlink(self.tmp.name)
        self.engine = AbductiveEngine(self.tmp.name)
        self.engine.start()

    def tearDown(self):
        if os.path.exists(self.tmp.name):
            os.unlink(self.tmp.name)
        lock = self.tmp.name + '.lock'
        if os.path.exists(lock):
            os.unlink(lock)

    def test_surplus_audit_flags_unexplained(self):
        self.engine.add_observation("O1", "orphaned", "generic")
        self.engine.add_observation("O2", "explained", "generic")
        self.engine.add_candidate(
            cause="c", mechanism="m", prior=0.2, source='library',
            observation_ids=['O2'],
        )
        unexplained = self.engine.surplus_audit()
        self.assertEqual(len(unexplained), 1)
        self.assertEqual(unexplained[0]['obs_id'], 'O1')

    def test_surplus_audit_empty_when_all_explained(self):
        self.engine.add_observation("O1", "a", "generic")
        self.engine.add_candidate(
            cause="c", mechanism="m", prior=0.2, source='library',
            observation_ids=['O1'],
        )
        unexplained = self.engine.surplus_audit()
        self.assertEqual(len(unexplained), 0)

    def test_surplus_audit_persists_to_surplus_log(self):
        self.engine.add_observation("O1", "x", "generic")
        self.engine.surplus_audit()
        self.assertEqual(len(self.engine.state.surplus_log), 1)
        # Second call is idempotent (dedupe by obs_id)
        self.engine.surplus_audit()
        self.assertEqual(len(self.engine.state.surplus_log), 1)


# ---------------------------------------------------------------------------
# AR — Analogical Retrieval
# ---------------------------------------------------------------------------

class TestAnalogicalRetrieval(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
        self.tmp.close()
        os.unlink(self.tmp.name)
        self.engine = AbductiveEngine(self.tmp.name)
        self.engine.start()

    def tearDown(self):
        if os.path.exists(self.tmp.name):
            os.unlink(self.tmp.name)
        lock = self.tmp.name + '.lock'
        if os.path.exists(lock):
            os.unlink(lock)

    def test_analogize_matches_seeded_archetype(self):
        """Phrase that matches a trace_signature should return it."""
        matches = self.engine.analogize("price rising despite flat local demand")
        # Shipped archetype has this exact seed signature
        self.assertGreaterEqual(len(matches), 1)
        top = matches[0]
        self.assertIn('archetype_id', top)
        self.assertGreater(top['similarity'], 0.0)

    def test_analogize_sorted_by_similarity(self):
        matches = self.engine.analogize(
            "cross-border capital inflow surge and foreign buyer dominance"
        )
        if len(matches) >= 2:
            for i in range(len(matches) - 1):
                self.assertGreaterEqual(
                    matches[i]['similarity'], matches[i+1]['similarity']
                )

    def test_analogize_no_match_returns_empty(self):
        matches = self.engine.analogize("qwertyuiopasdf xxyyzz")
        # Nothing in the library will match random tokens
        self.assertEqual(matches, [])

    def test_analogize_persists(self):
        self.engine.analogize("price rising despite flat demand")
        self.assertEqual(len(self.engine.state.analogies), 1)


# ---------------------------------------------------------------------------
# IC — Inference Chains
# ---------------------------------------------------------------------------

class TestInferenceChains(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
        self.tmp.close()
        os.unlink(self.tmp.name)
        self.engine = AbductiveEngine(self.tmp.name)
        self.engine.start()

    def tearDown(self):
        if os.path.exists(self.tmp.name):
            os.unlink(self.tmp.name)
        lock = self.tmp.name + '.lock'
        if os.path.exists(lock):
            os.unlink(lock)

    def test_chain_start_step_close(self):
        cid = self.engine.chain_start("H1", "observation O1 fires")
        self.engine.chain_step(cid, "implies X", lr=1.5, source='analyst')
        self.engine.chain_step(cid, "X implies Y", lr=1.8, source='library')
        chain = self.engine.chain_close(cid, seed_prior=0.3)
        self.assertEqual(chain['status'], 'closed')
        self.assertIsNotNone(chain['final_posterior'])
        # Posterior should have increased from 0.3 with LR>1 updates
        self.assertGreater(chain['final_posterior'], 0.3)

    def test_chain_step_invalid_source(self):
        cid = self.engine.chain_start("H1", "p")
        with self.assertRaises(ValueError):
            self.engine.chain_step(cid, "x", lr=1.0, source='badsource')

    def test_chain_step_negative_lr_rejected(self):
        cid = self.engine.chain_start("H1", "p")
        with self.assertRaises(ValueError):
            self.engine.chain_step(cid, "x", lr=-0.1, source='analyst')

    def test_chain_step_llm_parametric_lr_cap(self):
        """LLM-parametric chain steps are hard-capped at LR 2.0."""
        cid = self.engine.chain_start("H1", "p")
        # At cap: allowed
        self.engine.chain_step(cid, "x", lr=LLM_PARAMETRIC_MAX_LR,
                               source='llm_parametric')
        # Over cap: rejected
        with self.assertRaises(ValueError) as cm:
            self.engine.chain_step(cid, "y", lr=5.0, source='llm_parametric')
        self.assertIn("llm-parametric", str(cm.exception).lower())

    def test_chain_non_llm_can_exceed_cap(self):
        """Library/analyst steps can have any LR >= 0."""
        cid = self.engine.chain_start("H1", "p")
        self.engine.chain_step(cid, "x", lr=10.0, source='library')

    def test_chain_close_requires_step(self):
        cid = self.engine.chain_start("H1", "p")
        with self.assertRaises(RuntimeError):
            self.engine.chain_close(cid)

    def test_chain_close_twice_rejected(self):
        cid = self.engine.chain_start("H1", "p")
        self.engine.chain_step(cid, "x", lr=1.5, source='analyst')
        self.engine.chain_close(cid)
        with self.assertRaises(RuntimeError):
            self.engine.chain_close(cid)

    def test_chain_step_after_close_rejected(self):
        cid = self.engine.chain_start("H1", "p")
        self.engine.chain_step(cid, "x", lr=1.5, source='analyst')
        self.engine.chain_close(cid)
        with self.assertRaises(RuntimeError):
            self.engine.chain_step(cid, "y", lr=1.5, source='analyst')

    def test_chain_not_found(self):
        with self.assertRaises(KeyError):
            self.engine.chain_step("IC999", "x", lr=1.0, source='analyst')
        with self.assertRaises(KeyError):
            self.engine.chain_close("IC999")
        with self.assertRaises(KeyError):
            self.engine.chain_audit("IC999")

    def test_chain_audit_detects_short_closed_chain(self):
        cid = self.engine.chain_start("H1", "p")
        self.engine.chain_step(cid, "x", lr=1.5, source='analyst')
        self.engine.chain_close(cid)
        audit = self.engine.chain_audit(cid)
        # 1 step + closed is a gap (min 2 required)
        self.assertFalse(audit['ok'])
        self.assertTrue(any('minimum 2' in g for g in audit['gaps']))

    def test_chain_audit_clean_chain(self):
        cid = self.engine.chain_start("H1", "p")
        self.engine.chain_step(cid, "a", lr=1.5, source='analyst')
        self.engine.chain_step(cid, "b", lr=1.5, source='analyst')
        audit = self.engine.chain_audit(cid)
        # Open chain with 2 steps and no gaps
        self.assertTrue(audit['ok'], f"unexpected gaps: {audit['gaps']}")

    def test_chain_close_invalid_seed(self):
        cid = self.engine.chain_start("H1", "p")
        self.engine.chain_step(cid, "x", lr=1.5, source='analyst')
        with self.assertRaises(ValueError):
            self.engine.chain_close(cid, seed_prior=0.0)


# ---------------------------------------------------------------------------
# Coverage gate + promotion
# ---------------------------------------------------------------------------

class TestCoverageGate(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
        self.tmp.close()
        os.unlink(self.tmp.name)
        self.engine = AbductiveEngine(self.tmp.name)
        self.engine.start()

    def tearDown(self):
        if os.path.exists(self.tmp.name):
            os.unlink(self.tmp.name)
        lock = self.tmp.name + '.lock'
        if os.path.exists(lock):
            os.unlink(lock)

    def _make_observations(self, n: int):
        for i in range(n):
            self.engine.add_observation(f"O{i+1}", f"obs {i+1}", "generic")

    def test_coverage_score_zero_for_no_observations_explained(self):
        self._make_observations(5)
        cid = self.engine.add_candidate(
            cause="c", mechanism="m", prior=0.2, source='library',
        )
        self.assertEqual(self.engine.coverage_score(cid), 0.0)

    def test_coverage_score_unit_complexity(self):
        self._make_observations(4)
        cid = self.engine.add_candidate(
            cause="c", mechanism="m", prior=0.2, source='library',
            observation_ids=['O1', 'O2'],
        )
        # 2/4 = 0.5, complexity 1.0 → 0.5
        self.assertAlmostEqual(self.engine.coverage_score(cid), 0.5)

    def test_coverage_score_penalizes_complexity(self):
        self._make_observations(4)
        simple = self.engine.add_candidate(
            cause="simple", mechanism="m", prior=0.2, source='library',
            observation_ids=['O1', 'O2'], complexity=1.0,
        )
        complex_ = self.engine.add_candidate(
            cause="complex", mechanism="m", prior=0.2, source='library',
            observation_ids=['O1', 'O2'], complexity=2.0,
        )
        self.assertGreater(
            self.engine.coverage_score(simple),
            self.engine.coverage_score(complex_),
        )

    def test_low_coverage_candidate_rejected_at_promotion(self):
        """PRIMARY failure-mode mitigation: coverage gate blocks promotion."""
        self._make_observations(10)
        cid = self.engine.add_candidate(
            cause="narrow", mechanism="m", prior=0.2, source='library',
            observation_ids=['O1'],  # 1/10 = 0.1 coverage
        )
        # Threshold 0.3 → should reject
        with self.assertRaises(RuntimeError) as cm:
            self.engine.promote(cid, threshold=0.3)
        self.assertIn("coverage", str(cm.exception).lower())

    def test_high_coverage_candidate_allowed(self):
        self._make_observations(4)
        cid = self.engine.add_candidate(
            cause="broad", mechanism="m", prior=0.2, source='library',
            observation_ids=['O1', 'O2', 'O3'],  # 3/4 = 0.75
        )
        result = self.engine.promote(cid, threshold=0.3, tracker_path=None)
        self.assertTrue(result['promoted'])
        # Re-promotion is idempotent blocker
        with self.assertRaises(RuntimeError):
            self.engine.promote(cid, threshold=0.3)

    def test_promotion_writes_to_tracker(self):
        """End-to-end: candidate promoted into bayesian_tracker.py state."""
        self._make_observations(3)
        cid = self.engine.add_candidate(
            cause="test cause", mechanism="test mech", prior=0.2, source='library',
            observation_ids=['O1', 'O2', 'O3'],  # full coverage
        )
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tf:
            tracker_path = tf.name
        try:
            os.unlink(tracker_path)
            result = self.engine.promote(
                cid, threshold=0.3, tracker_path=tracker_path,
            )
            self.assertTrue(result['promoted'])
            self.assertIsNotNone(result['hypothesis_id'])
            # Verify the tracker file now contains the hypothesis
            with open(tracker_path) as f:
                data = json.load(f)
            self.assertGreaterEqual(len(data.get('hypotheses', [])), 1)
            statement = data['hypotheses'][0]['statement']
            self.assertIn("H_ABDUCT_", statement)
        finally:
            if os.path.exists(tracker_path):
                os.unlink(tracker_path)
            lockpath = tracker_path + '.lock'
            if os.path.exists(lockpath):
                os.unlink(lockpath)

    def test_list_candidates_sorted_by_score(self):
        self._make_observations(4)
        low = self.engine.add_candidate(
            cause="low", mechanism="m", prior=0.2, source='library',
            observation_ids=['O1'],
        )
        high = self.engine.add_candidate(
            cause="high", mechanism="m", prior=0.2, source='library',
            observation_ids=['O1', 'O2', 'O3', 'O4'],
        )
        items = self.engine.list_candidates()
        self.assertEqual(items[0]['id'], high)
        self.assertEqual(items[1]['id'], low)

    def test_list_candidates_excludes_promoted_by_default(self):
        self._make_observations(3)
        cid = self.engine.add_candidate(
            cause="c", mechanism="m", prior=0.2, source='library',
            observation_ids=['O1', 'O2', 'O3'],
        )
        self.engine.promote(cid, threshold=0.3)
        self.assertEqual(len(self.engine.list_candidates()), 0)
        self.assertEqual(len(self.engine.list_candidates(include_promoted=True)), 1)


# ---------------------------------------------------------------------------
# Catalog bootstrap / review
# ---------------------------------------------------------------------------

class TestCatalogBootstrap(unittest.TestCase):

    def test_bootstrap_prompt_emits_schema(self):
        prompt = AbductiveEngine.bootstrap_prompt("timing")
        self.assertIn("timing", prompt)
        self.assertIn("llm_parametric", prompt)
        self.assertIn("pending_review", prompt)
        self.assertIn("0.30", prompt)

    def test_catalog_review_stages_candidates(self):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as ef:
            engine_path = ef.name
        os.unlink(engine_path)
        engine = AbductiveEngine(engine_path)
        engine.start()

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False,
                                         mode='w') as bf:
            json.dump({
                'category': 'custom',
                'candidates': [
                    {'cause': 'c1', 'mechanism': 'm1', 'prior': 0.2},
                    {'cause': 'c2', 'mechanism': 'm2', 'prior': 0.9,  # capped
                     'complexity': 1.5},
                ],
            }, bf)
            boot_path = bf.name

        try:
            summary = engine.catalog_review(boot_path)
            self.assertEqual(summary['staged'], 2)
            self.assertEqual(summary['category'], 'custom')
            # Both should be llm_parametric
            for c in engine.state.candidates:
                self.assertEqual(c['source'], 'llm_parametric')
                self.assertLessEqual(c['prior'], LLM_PARAMETRIC_MAX_PRIOR)
        finally:
            for p in (engine_path, boot_path, engine_path + '.lock'):
                if os.path.exists(p):
                    os.unlink(p)

    def test_catalog_review_missing_path(self):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as ef:
            p = ef.name
        os.unlink(p)
        engine = AbductiveEngine(p)
        engine.start()
        try:
            with self.assertRaises(FileNotFoundError):
                engine.catalog_review('/nonexistent/bootstrap.json')
        finally:
            for q in (p, p + '.lock'):
                if os.path.exists(q):
                    os.unlink(q)


# ---------------------------------------------------------------------------
# Gate status + report
# ---------------------------------------------------------------------------

class TestGateAndReport(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
        self.tmp.close()
        os.unlink(self.tmp.name)
        self.engine = AbductiveEngine(self.tmp.name)
        self.engine.start()

    def tearDown(self):
        if os.path.exists(self.tmp.name):
            os.unlink(self.tmp.name)
        lock = self.tmp.name + '.lock'
        if os.path.exists(lock):
            os.unlink(lock)

    def test_gate_fails_with_no_observations(self):
        gate = self.engine.gate_status()
        self.assertFalse(gate['pass'])

    def test_gate_passes_after_three_inversions(self):
        self.engine.invert("O1", "obs 1", "generic")
        self.engine.invert("O2", "obs 2", "generic")
        self.engine.invert("O3", "obs 3", "generic")
        self.engine.surplus_audit()
        gate = self.engine.gate_status()
        self.assertTrue(gate['pass'], f"gate did not pass: {gate}")
        self.assertGreaterEqual(gate['observations_inverted'], 3)

    def test_report_generates_markdown(self):
        self.engine.invert("O1", "test", "generic")
        out = self.engine.report()
        self.assertIn("Abductive Expansion Report", out)
        self.assertIn("Candidates", out)

    def test_report_verbose_includes_chains(self):
        cid = self.engine.chain_start("H1", "p")
        self.engine.chain_step(cid, "a", lr=1.5, source='analyst')
        self.engine.chain_step(cid, "b", lr=1.5, source='analyst')
        self.engine.chain_close(cid)
        out = self.engine.report(verbose=True)
        self.assertIn("Inference Chains", out)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

class TestHelpers(unittest.TestCase):

    def test_odds_update_positive(self):
        p = _odds_update(0.3, 2.0)
        self.assertGreater(p, 0.3)

    def test_odds_update_negative(self):
        p = _odds_update(0.7, 0.5)
        self.assertLess(p, 0.7)

    def test_odds_update_zero_lr_falsifies(self):
        p = _odds_update(0.5, 0.0)
        self.assertAlmostEqual(p, 0.001, places=3)

    def test_tokenize_basic(self):
        toks = _tokenize("price rising, FLAT demand")
        self.assertIn('price', toks)
        self.assertIn('rising', toks)
        self.assertIn('flat', toks)  # lowercased

    def test_tokenize_empty(self):
        self.assertEqual(_tokenize(""), set())

    def test_token_overlap_identical(self):
        self.assertEqual(_token_overlap({'a', 'b'}, {'a', 'b'}), 1.0)

    def test_token_overlap_disjoint(self):
        self.assertEqual(_token_overlap({'a'}, {'b'}), 0.0)

    def test_token_overlap_empty(self):
        self.assertEqual(_token_overlap(set(), {'a'}), 0.0)


# ---------------------------------------------------------------------------
# CLI smoke tests
# ---------------------------------------------------------------------------

class TestCLISmoke(unittest.TestCase):

    def _tmp_state(self):
        tf = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
        tf.close()
        os.unlink(tf.name)
        return tf.name

    def _cleanup(self, path):
        for p in (path, path + '.lock'):
            if os.path.exists(p):
                os.unlink(p)

    def test_help_exits_zero(self):
        result = subprocess.run(
            [sys.executable, SCRIPT_PATH, '--help'],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn('abductive_engine.py', result.stdout)

    def test_bootstrap_cli_no_session_required(self):
        result = subprocess.run(
            [sys.executable, SCRIPT_PATH, 'catalog', 'bootstrap',
             '--category', 'timing'],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn('timing', result.stdout)
        self.assertIn('llm_parametric', result.stdout)

    def test_end_to_end_all_five_operators(self):
        path = self._tmp_state()
        try:
            # start
            r = subprocess.run(
                [sys.executable, SCRIPT_PATH, '--file', path, 'start'],
                capture_output=True, text=True, check=True,
            )
            self.assertIn('AE', r.stdout)

            # TI
            r = subprocess.run(
                [sys.executable, SCRIPT_PATH, '--file', path, 'invert',
                 '--obs-id', 'O1', '--text', 'latency spike', '--category', 'timing'],
                capture_output=True, text=True, check=True,
            )
            self.assertIn('TI produced', r.stdout)

            r = subprocess.run(
                [sys.executable, SCRIPT_PATH, '--file', path, 'invert',
                 '--obs-id', 'O2', '--text', 'memory climb', '--category', 'resource'],
                capture_output=True, text=True, check=True,
            )

            r = subprocess.run(
                [sys.executable, SCRIPT_PATH, '--file', path, 'invert',
                 '--obs-id', 'O3', '--text', 'crash', '--category', 'failure'],
                capture_output=True, text=True, check=True,
            )

            # AA
            r = subprocess.run(
                [sys.executable, SCRIPT_PATH, '--file', path, 'absence-audit',
                 '--hypothesis', 'H1',
                 '--predictions', 'log entry;metric spike;disk write'],
                capture_output=True, text=True, check=True,
            )
            self.assertIn('AA queued', r.stdout)

            # SA
            r = subprocess.run(
                [sys.executable, SCRIPT_PATH, '--file', path, 'surplus-audit'],
                capture_output=True, text=True, check=True,
            )
            self.assertIn('SA found', r.stdout)

            # AR
            r = subprocess.run(
                [sys.executable, SCRIPT_PATH, '--file', path, 'analogize',
                 '--signature', 'price rising flat demand'],
                capture_output=True, text=True, check=True,
            )
            self.assertIn('AR found', r.stdout)

            # IC
            r = subprocess.run(
                [sys.executable, SCRIPT_PATH, '--file', path, 'chain', 'start',
                 '--target', 'H1', '--premise', 'observation O1 fires'],
                capture_output=True, text=True, check=True,
            )
            self.assertIn('IC', r.stdout)
            chain_id = [t for t in r.stdout.split() if t.startswith('IC')][0]

            subprocess.run(
                [sys.executable, SCRIPT_PATH, '--file', path, 'chain', 'step',
                 '--id', chain_id, '--claim', 'therefore X', '--lr', '1.5',
                 '--source', 'analyst'],
                capture_output=True, text=True, check=True,
            )
            subprocess.run(
                [sys.executable, SCRIPT_PATH, '--file', path, 'chain', 'step',
                 '--id', chain_id, '--claim', 'X implies Y', '--lr', '1.5',
                 '--source', 'library'],
                capture_output=True, text=True, check=True,
            )
            r = subprocess.run(
                [sys.executable, SCRIPT_PATH, '--file', path, 'chain', 'close',
                 '--id', chain_id, '--seed-prior', '0.4'],
                capture_output=True, text=True, check=True,
            )
            self.assertIn('final_posterior', r.stdout)

            r = subprocess.run(
                [sys.executable, SCRIPT_PATH, '--file', path, 'chain', 'audit',
                 '--id', chain_id],
                capture_output=True, text=True,
            )
            self.assertEqual(r.returncode, 0)
            self.assertIn('OK', r.stdout)

            # candidates list
            r = subprocess.run(
                [sys.executable, SCRIPT_PATH, '--file', path,
                 'candidates', 'list'],
                capture_output=True, text=True, check=True,
            )
            self.assertIn('CAND', r.stdout)

            # gate status
            r = subprocess.run(
                [sys.executable, SCRIPT_PATH, '--file', path, 'gate'],
                capture_output=True, text=True,
            )
            # Should pass (3 inverted + surplus run)
            self.assertEqual(r.returncode, 0)

            # report
            r = subprocess.run(
                [sys.executable, SCRIPT_PATH, '--file', path, 'report',
                 '--verbose'],
                capture_output=True, text=True, check=True,
            )
            self.assertIn('Abductive Expansion Report', r.stdout)

            # Verify state file is valid JSON with expected content
            with open(path) as f:
                data = json.load(f)
            self.assertIn('observations', data)
            self.assertGreaterEqual(len(data['observations']), 3)
            self.assertIn('inference_chains', data)
            self.assertGreaterEqual(len(data['inference_chains']), 1)
        finally:
            self._cleanup(path)

    def test_cli_error_propagates_nonzero(self):
        """CLI errors (e.g. unknown candidate ID) return nonzero exit."""
        path = self._tmp_state()
        try:
            subprocess.run(
                [sys.executable, SCRIPT_PATH, '--file', path, 'start'],
                capture_output=True, text=True, check=True,
            )
            r = subprocess.run(
                [sys.executable, SCRIPT_PATH, '--file', path,
                 'candidates', 'promote', '--id', 'CAND999'],
                capture_output=True, text=True,
            )
            self.assertNotEqual(r.returncode, 0)
        finally:
            self._cleanup(path)


if __name__ == '__main__':
    unittest.main()
