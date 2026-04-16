#!/usr/bin/env python3
"""Tests for src/scripts/domain_orienter.py (Phase 0.3 Domain Orientation).

No mocks — all tests use real file I/O and real JSON round-trips.
Mirrors the structure of tests/test_abductive_engine.py.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

# Allow importing from src/scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'scripts'))

from domain_orienter import (
    DomainOrienter,
    DomainOrientationState,
    CandidateTerm,
    GroundedTerm,
    Metric,
    Alias,
    Source,
    VALID_SOURCES,
    VALID_TIERS,
    VALID_SOURCE_CATEGORIES,
    LLM_PARAMETRIC_CONFIDENCE_CAP,
    ANALYST_CONFIDENCE_CAP,
    CHAIN_DERIVED_CONFIDENCE_CAP,
    GATE_MIN_TERMS_STANDARD,
    GATE_MIN_TERMS_LITE,
    GATE_MIN_METRICS,
    GATE_MIN_VERIFIED_SOURCES,
    GATE_MIN_LIBRARY_FRACTION,
    _parse_bool,
    _parse_plausibility,
    _parse_aliases,
    _recover_counter,
)


SCRIPT_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'src', 'scripts', 'domain_orienter.py'
)


def _fresh_tmp():
    """Return a tmp .json path that does NOT exist on disk."""
    tf = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
    tf.close()
    os.unlink(tf.name)
    return tf.name


def _cleanup(path):
    for p in (path, path + '.lock'):
        if os.path.exists(p):
            os.unlink(p)


# ---------------------------------------------------------------------------
# TestStart — session lifecycle
# ---------------------------------------------------------------------------

class TestStart(unittest.TestCase):

    def setUp(self):
        self.state_path = _fresh_tmp()
        self.orienter = DomainOrienter(self.state_path)

    def tearDown(self):
        _cleanup(self.state_path)

    def test_start_creates_session(self):
        rid = self.orienter.start()
        self.assertTrue(rid.startswith("domain-"))
        self.assertIsNotNone(self.orienter.state)
        self.assertTrue(os.path.exists(self.state_path))

    def test_start_is_idempotent_without_force(self):
        """A second start() without --force raises FileExistsError."""
        self.orienter.start()
        with self.assertRaises(FileExistsError):
            self.orienter.start()

    def test_start_force_overwrites_existing_session(self):
        rid1 = self.orienter.start()
        rid2 = self.orienter.start(force=True)
        self.assertNotEqual(rid1, rid2)
        self.assertEqual(self.orienter.state.run_id, rid2)

    def test_start_records_tier_field(self):
        self.orienter.start(tier="LITE", domain="cred_deriv")
        self.assertEqual(self.orienter.state.tier, "LITE")
        self.assertEqual(self.orienter.state.domain_declared, "cred_deriv")

    def test_start_rejects_invalid_tier(self):
        with self.assertRaises(ValueError):
            self.orienter.start(tier="BOGUS_TIER")


# ---------------------------------------------------------------------------
# TestExtract — TE operator
# ---------------------------------------------------------------------------

class TestExtract(unittest.TestCase):

    def setUp(self):
        self.state_path = _fresh_tmp()
        self.orienter = DomainOrienter(self.state_path)
        self.orienter.start()

    def tearDown(self):
        _cleanup(self.state_path)

    def test_extract_acronyms(self):
        """Uppercase acronyms are extracted."""
        produced = self.orienter.extract("CDS OIS ISDA and MiFID are keys.")
        texts = {c.text for c in produced}
        # CDS, OIS, ISDA are acronyms ([A-Z]{2,}); MiFID is capword
        self.assertIn("CDS", texts)
        self.assertIn("OIS", texts)
        self.assertIn("ISDA", texts)

    def test_extract_case_insensitive_dedup(self):
        """Alnum tokens like OD600/od600 dedupe by lowercase key."""
        produced = self.orienter.extract("OD600 measured OD600 repeated od600 again")
        # Expect one entry with freq=3 (all three variants collapsed)
        ods = [c for c in produced if c.text.lower() == "od600"]
        self.assertEqual(len(ods), 1)
        self.assertEqual(ods[0].frequency, 3)
        # First casing preserved
        self.assertEqual(ods[0].text, "OD600")

    def test_extract_frequency_ranking(self):
        """Candidates are returned sorted by descending frequency."""
        produced = self.orienter.extract("AAA BBB AAA CCC AAA BBB")
        self.assertEqual(len(produced), 3)
        # AAA(3), BBB(2), CCC(1)
        freqs = [c.frequency for c in produced]
        self.assertEqual(freqs, sorted(freqs, reverse=True))
        self.assertEqual(produced[0].text, "AAA")
        self.assertEqual(produced[0].frequency, 3)

    def test_extract_empty_input_returns_empty_list(self):
        produced = self.orienter.extract("")
        self.assertEqual(produced, [])
        self.assertEqual(len(self.orienter.state.candidate_terms), 0)

    def test_extract_records_source_ref_inline(self):
        """Raw-text input carries source_ref == '<inline>'."""
        produced = self.orienter.extract("CDS and OIS are tools.")
        self.assertGreater(len(produced), 0)
        for c in produced:
            self.assertEqual(c.source_ref, "<inline>")

    def test_extract_records_source_ref_file(self):
        """File input carries source_ref == basename(path)."""
        # Write a temp text file
        tf = tempfile.NamedTemporaryFile(suffix='.md', delete=False, mode='w')
        tf.write("ISDA and CDS appear here.")
        tf.close()
        try:
            produced = self.orienter.extract(tf.name)
            for c in produced:
                self.assertEqual(c.source_ref, os.path.basename(tf.name))
        finally:
            os.unlink(tf.name)

    def test_extract_skips_already_registered_terms(self):
        """Subsequent extracts do not re-register existing candidate texts."""
        self.orienter.extract("CDS OIS")
        # Second extract should not create duplicates for the same tokens
        produced = self.orienter.extract("CDS OIS ISDA")
        texts = [c.text for c in produced]
        # Only ISDA is new
        self.assertEqual(texts, ["ISDA"])


# ---------------------------------------------------------------------------
# TestGroundCaps — provenance caps on grounding
# ---------------------------------------------------------------------------

class TestGroundCaps(unittest.TestCase):

    def setUp(self):
        self.state_path = _fresh_tmp()
        self.orienter = DomainOrienter(self.state_path)
        self.orienter.start()

    def tearDown(self):
        _cleanup(self.state_path)

    def test_library_term_default_confidence_one(self):
        g = self.orienter.ground(
            term="LIB", definition="d", source="library", allow_novel=True,
        )
        self.assertAlmostEqual(g.confidence, 1.0)

    def test_library_term_explicit_nondefault_confidence_raises(self):
        """Library confidence is fixed at 1.0; passing 0.5 raises."""
        with self.assertRaises(ValueError):
            self.orienter.ground(
                term="LIB2", definition="d", source="library",
                confidence=0.5, allow_novel=True,
            )

    def test_analyst_cap_enforced(self):
        """analyst confidence ≤ 0.80 is accepted."""
        g = self.orienter.ground(
            term="AN1", definition="d", source="analyst",
            confidence=ANALYST_CONFIDENCE_CAP, allow_novel=True,
        )
        self.assertAlmostEqual(g.confidence, ANALYST_CONFIDENCE_CAP)

    def test_analyst_cap_violation_raises(self):
        """analyst confidence > 0.80 raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            self.orienter.ground(
                term="AN2", definition="d", source="analyst",
                confidence=0.81, allow_novel=True,
            )
        self.assertIn("analyst", str(cm.exception).lower())

    def test_llm_parametric_cap_enforced(self):
        """llm_parametric confidence ≤ 0.60 is accepted."""
        g = self.orienter.ground(
            term="LP1", definition="d", source="llm_parametric",
            confidence=LLM_PARAMETRIC_CONFIDENCE_CAP, allow_novel=True,
        )
        self.assertAlmostEqual(g.confidence, LLM_PARAMETRIC_CONFIDENCE_CAP)

    def test_llm_parametric_cap_violation_raises(self):
        """llm_parametric confidence > 0.60 raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            self.orienter.ground(
                term="LP2", definition="d", source="llm_parametric",
                confidence=0.61, allow_novel=True,
            )
        self.assertIn("llm_parametric", str(cm.exception).lower())

    def test_chain_derived_cap_enforced(self):
        """chain_derived confidence ≤ 0.90 is accepted."""
        g = self.orienter.ground(
            term="CD1", definition="d", source="chain_derived",
            confidence=CHAIN_DERIVED_CONFIDENCE_CAP, allow_novel=True,
        )
        self.assertAlmostEqual(g.confidence, CHAIN_DERIVED_CONFIDENCE_CAP)

    def test_chain_derived_cap_violation_raises(self):
        """chain_derived confidence > 0.90 raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            self.orienter.ground(
                term="CD2", definition="d", source="chain_derived",
                confidence=0.91, allow_novel=True,
            )
        self.assertIn("chain_derived", str(cm.exception).lower())

    def test_unknown_source_raises(self):
        with self.assertRaises(ValueError):
            self.orienter.ground(
                term="X", definition="d", source="wikipedia",
                allow_novel=True,
            )

    def test_ground_nonextracted_term_without_allow_novel_raises(self):
        """Grounding a term not in candidate_terms requires allow_novel=True."""
        with self.assertRaises(ValueError) as cm:
            self.orienter.ground(
                term="FOO", definition="d", source="library",
            )
        self.assertIn("candidate_terms", str(cm.exception).lower())

    def test_ground_allows_novel_when_flag_set(self):
        """With allow_novel=True, a brand-new term grounds successfully."""
        g = self.orienter.ground(
            term="NOVEL", definition="d", source="analyst",
            allow_novel=True,
        )
        self.assertTrue(g.id.startswith("TERM-"))

    def test_ground_extracted_candidate_promotes_it(self):
        """Grounding a previously extracted candidate moves it from pending list."""
        self.orienter.extract("ZZZ observed zzz again")
        # Pre: ZZZ in candidate_terms
        self.assertTrue(any(
            c["text"].lower() == "zzz" for c in self.orienter.state.candidate_terms
        ))
        g = self.orienter.ground(
            term="ZZZ", definition="dummy defn", source="library",
        )
        # Post: ZZZ no longer in candidate_terms; now in grounded_terms
        self.assertFalse(any(
            c["text"].lower() == "zzz" for c in self.orienter.state.candidate_terms
        ))
        self.assertEqual(g.text, "ZZZ")

    def test_duplicate_grounding_raises(self):
        self.orienter.ground(
            term="DUP", definition="d", source="library", allow_novel=True,
        )
        with self.assertRaises(ValueError) as cm:
            self.orienter.ground(
                term="DUP", definition="d2", source="analyst",
                allow_novel=True,
            )
        self.assertIn("already grounded", str(cm.exception).lower())

    def test_valid_sources_constant(self):
        """VALID_SOURCES matches the documented set."""
        self.assertEqual(
            VALID_SOURCES,
            {"library", "analyst", "llm_parametric", "chain_derived"},
        )


# ---------------------------------------------------------------------------
# TestMetricsMapping — MM operator
# ---------------------------------------------------------------------------

class TestMetricsMapping(unittest.TestCase):

    def setUp(self):
        self.state_path = _fresh_tmp()
        self.orienter = DomainOrienter(self.state_path)
        self.orienter.start(domain="credit_derivatives")

    def tearDown(self):
        _cleanup(self.state_path)

    def test_add_metric_schema(self):
        """Registering a metric yields all expected fields and a MET- id."""
        m = self.orienter.add_metric(
            name="cs01",
            units="USD/bp",
            higher_is_better=False,
            plausibility=(None, 100.0, 10000.0, None),
            source="library",
            url="https://www.isda.org/",
        )
        self.assertTrue(m.id.startswith("MET-"))
        self.assertEqual(m.name, "cs01")
        self.assertEqual(m.units, "USD/bp")
        self.assertEqual(list(m.plausibility), [None, 100.0, 10000.0, None])
        self.assertFalse(m.promoted)

    def test_parse_plausibility_nulls(self):
        """'null,0.5,1.5,null' parses to [None, 0.5, 1.5, None]."""
        parsed = _parse_plausibility("null,0.5,1.5,null")
        self.assertEqual(list(parsed), [None, 0.5, 1.5, None])

    def test_parse_plausibility_bad_length_raises(self):
        with self.assertRaises(ValueError):
            _parse_plausibility("1,2,3")

    def test_parse_plausibility_bad_value_raises(self):
        with self.assertRaises(ValueError):
            _parse_plausibility("null,abc,2,null")

    def test_llm_parametric_metric_cannot_be_promoted(self):
        """promote_candidate on an llm_parametric metric raises RuntimeError."""
        m = self.orienter.add_metric(
            name="shady",
            units="x",
            higher_is_better=True,
            plausibility=(None, 1.0, 2.0, None),
            source="llm_parametric",
            url="https://example.com",
        )
        with self.assertRaises(RuntimeError) as cm:
            self.orienter.promote_candidate(m.id)
        self.assertIn("llm_parametric", str(cm.exception).lower())

    def test_llm_parametric_metric_promoted_at_add_time_raises(self):
        """Cannot pass promoted=True for an llm_parametric metric."""
        with self.assertRaises(ValueError):
            self.orienter.add_metric(
                name="shady2",
                units="x",
                higher_is_better=True,
                plausibility=(None, 1.0, 2.0, None),
                source="llm_parametric",
                url="https://example.com",
                promoted=True,
            )

    def test_higher_is_better_flag_stored(self):
        m = self.orienter.add_metric(
            name="score",
            units="unitless",
            higher_is_better=True,
            plausibility=(None, 0.0, 1.0, None),
            source="analyst",
        )
        self.assertTrue(m.higher_is_better)
        # Verify persistence round-trip too
        reloaded = DomainOrienter(self.state_path)
        persisted = [x for x in reloaded.state.metrics if x["id"] == m.id][0]
        self.assertTrue(persisted["higher_is_better"])

    def test_missing_required_field_raises(self):
        """Empty name raises ValueError."""
        with self.assertRaises(ValueError):
            self.orienter.add_metric(
                name="",
                units="x",
                higher_is_better=True,
                plausibility=(None, 1.0, 2.0, None),
                source="analyst",
            )

    def test_missing_units_raises(self):
        with self.assertRaises(ValueError):
            self.orienter.add_metric(
                name="m1",
                units="",
                higher_is_better=True,
                plausibility=(None, 1.0, 2.0, None),
                source="analyst",
            )

    def test_invalid_source_raises(self):
        with self.assertRaises(ValueError):
            self.orienter.add_metric(
                name="m1",
                units="x",
                higher_is_better=True,
                plausibility=(None, 1.0, 2.0, None),
                source="wikipedia",
            )

    def test_non_analyst_metric_requires_url(self):
        """library/llm_parametric/chain_derived metrics must carry a URL."""
        with self.assertRaises(ValueError) as cm:
            self.orienter.add_metric(
                name="no_url",
                units="x",
                higher_is_better=True,
                plausibility=(None, 1.0, 2.0, None),
                source="library",
            )
        self.assertIn("url", str(cm.exception).lower())

    def test_domain_keying_used_in_render(self):
        """render_metrics groups under the metric's declared domain."""
        m = self.orienter.add_metric(
            name="cs01",
            units="USD/bp",
            higher_is_better=False,
            plausibility=(None, 100.0, 10000.0, None),
            source="library",
            url="https://x",
            domain="credit_derivatives",
        )
        self.orienter.promote_candidate(m.id)
        out = self.orienter.render_metrics()
        self.assertIn("credit_derivatives", out)
        self.assertIn("cs01", out["credit_derivatives"])
        self.assertEqual(
            out["credit_derivatives"]["cs01"],
            [None, 100.0, 10000.0, None],
        )

    def test_duplicate_metric_name_same_domain_raises(self):
        self.orienter.add_metric(
            name="dup", units="x", higher_is_better=True,
            plausibility=(None, 1.0, 2.0, None),
            source="analyst", domain="d1",
        )
        with self.assertRaises(ValueError):
            self.orienter.add_metric(
                name="dup", units="x", higher_is_better=True,
                plausibility=(None, 1.0, 2.0, None),
                source="analyst", domain="d1",
            )


# ---------------------------------------------------------------------------
# TestAliases — AM operator
# ---------------------------------------------------------------------------

class TestAliases(unittest.TestCase):

    def setUp(self):
        self.state_path = _fresh_tmp()
        self.orienter = DomainOrienter(self.state_path)
        self.orienter.start()

    def tearDown(self):
        _cleanup(self.state_path)

    def test_alias_canonical_linkage(self):
        g = self.orienter.ground(
            term="Credit Default Swap", definition="d", source="library",
            allow_novel=True,
        )
        a = self.orienter.alias(
            canonical="Credit Default Swap",
            aliases_list=["CDS", "default swap"],
        )
        self.assertEqual(a.canonical_id, g.id)
        self.assertEqual(a.canonical_text, "Credit Default Swap")
        self.assertEqual(a.aliases, ["CDS", "default swap"])

    def test_alias_region_stored(self):
        self.orienter.ground(
            term="ISDA", definition="d", source="library", allow_novel=True,
        )
        a = self.orienter.alias(
            canonical="ISDA",
            aliases_list=["Master Agreement"],
            region="global",
            source="library",
        )
        self.assertEqual(a.region, "global")
        self.assertEqual(a.source, "library")

    def test_alias_missing_canonical_raises(self):
        """Aliasing a term that is not grounded raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            self.orienter.alias(
                canonical="NonGrounded",
                aliases_list=["alt"],
            )
        self.assertIn("grounded_terms", str(cm.exception).lower())

    def test_alias_empty_list_raises(self):
        self.orienter.ground(
            term="FOO", definition="d", source="library", allow_novel=True,
        )
        with self.assertRaises(ValueError):
            self.orienter.alias(canonical="FOO", aliases_list=[])

    def test_alias_invalid_source_raises(self):
        self.orienter.ground(
            term="FOO2", definition="d", source="library", allow_novel=True,
        )
        with self.assertRaises(ValueError):
            self.orienter.alias(
                canonical="FOO2",
                aliases_list=["f2"],
                source="wikipedia",
            )


# ---------------------------------------------------------------------------
# TestSources — CS operator (add + verify)
# ---------------------------------------------------------------------------

class TestSources(unittest.TestCase):

    def setUp(self):
        self.state_path = _fresh_tmp()
        self.orienter = DomainOrienter(self.state_path)
        self.orienter.start()

    def tearDown(self):
        _cleanup(self.state_path)

    def test_source_registered_unverified_by_default(self):
        s = self.orienter.add_source(
            title="ISDA 2014 Definitions",
            category="standard",
            url="https://www.isda.org/",
        )
        self.assertFalse(s.verified)
        self.assertTrue(s.id.startswith("SID-"))
        self.assertEqual(s.category, "standard")

    def test_verify_with_http_status_200_flips_verified(self):
        s = self.orienter.add_source(
            title="Basel III",
            category="regulator",
            url="https://www.bis.org/",
        )
        self.assertFalse(s.verified)
        s2 = self.orienter.verify_source(s.id, http_status=200)
        self.assertTrue(s2.verified)
        self.assertEqual(s2.http_status, 200)

    def test_verify_with_non_200_raises(self):
        """HTTP != 200 (without 'citation' fallback) is rejected."""
        s = self.orienter.add_source(
            title="Offline Source",
            category="textbook",
        )
        with self.assertRaises(ValueError) as cm:
            self.orienter.verify_source(s.id, http_status=404)
        self.assertIn("not verified", str(cm.exception).lower())

    def test_verify_citation_path(self):
        """verified_by='citation' accepts offline confirmation (DOI/ISBN)."""
        s = self.orienter.add_source(
            title="Hull Options Futures",
            category="textbook",
        )
        s2 = self.orienter.verify_source(s.id, verified_by="citation")
        self.assertTrue(s2.verified)
        self.assertEqual(s2.verified_by, "citation")

    def test_unverified_source_blocks_promotion(self):
        """promote_candidate refuses unverified sources with RuntimeError."""
        s = self.orienter.add_source(
            title="Unverified Ref",
            category="textbook",
        )
        with self.assertRaises(RuntimeError) as cm:
            self.orienter.promote_candidate(s.id)
        self.assertIn("verify", str(cm.exception).lower())

    def test_verified_source_promotes(self):
        """A verified source can be promoted to canonical status."""
        s = self.orienter.add_source(
            title="Verified Ref",
            category="textbook",
        )
        self.orienter.verify_source(s.id, http_status=200)
        result = self.orienter.promote_candidate(s.id)
        self.assertTrue(result["promoted"])

    def test_invalid_category_raises(self):
        with self.assertRaises(ValueError) as cm:
            self.orienter.add_source(
                title="X", category="bogus_kind",
            )
        self.assertIn("category", str(cm.exception).lower())

    def test_duplicate_source_title_raises(self):
        self.orienter.add_source(title="Same", category="textbook")
        with self.assertRaises(ValueError):
            self.orienter.add_source(title="Same", category="regulator")

    def test_verify_missing_source_id_raises(self):
        with self.assertRaises(KeyError):
            self.orienter.verify_source("SID-999", http_status=200)


# ---------------------------------------------------------------------------
# TestGate — Phase 0.3 exit gate
# ---------------------------------------------------------------------------

class TestGate(unittest.TestCase):

    def setUp(self):
        self.state_path = _fresh_tmp()
        self.orienter = DomainOrienter(self.state_path)
        # default STANDARD unless a test overrides
        self.orienter.start(tier="STANDARD", domain="test")

    def tearDown(self):
        _cleanup(self.state_path)

    def _fill_standard_floor(self, orienter):
        """Helper: meet all STANDARD gate criteria."""
        # 10 grounded library terms → lib_fraction = 1.0 (>0.30)
        for i in range(GATE_MIN_TERMS_STANDARD):
            orienter.ground(
                term=f"T{i}", definition=f"d{i}",
                source="library", allow_novel=True,
            )
        # 3 metrics promoted
        for i in range(GATE_MIN_METRICS):
            m = orienter.add_metric(
                name=f"met{i}", units="x", higher_is_better=True,
                plausibility=(None, 1.0, 2.0, None),
                source="library", url="https://x",
            )
            orienter.promote_candidate(m.id)
        # 2 verified sources
        for i in range(GATE_MIN_VERIFIED_SOURCES):
            s = orienter.add_source(
                title=f"book{i}", category="textbook",
            )
            orienter.verify_source(s.id, http_status=200)
        # 1 alias
        orienter.alias(canonical="T0", aliases_list=["t0-alias"])

    def test_gate_passes_when_all_criteria_met(self):
        self._fill_standard_floor(self.orienter)
        passed, details = self.orienter.gate()
        self.assertTrue(passed, f"Expected PASS, got failures: {details['failures']}")
        self.assertEqual(details["failures"], [])

    def test_gate_fails_with_empty_state(self):
        passed, details = self.orienter.gate()
        self.assertFalse(passed)
        self.assertGreater(len(details["failures"]), 0)

    def test_gate_fails_on_insufficient_grounded_terms(self):
        """Only 5 grounded terms → STANDARD floor (10) fails."""
        for i in range(5):
            self.orienter.ground(
                term=f"T{i}", definition="d",
                source="library", allow_novel=True,
            )
        for i in range(GATE_MIN_METRICS):
            m = self.orienter.add_metric(
                name=f"m{i}", units="x", higher_is_better=True,
                plausibility=(None, 1.0, 2.0, None),
                source="library", url="https://x",
            )
            self.orienter.promote_candidate(m.id)
        for i in range(GATE_MIN_VERIFIED_SOURCES):
            s = self.orienter.add_source(
                title=f"b{i}", category="textbook",
            )
            self.orienter.verify_source(s.id, http_status=200)
        self.orienter.alias(canonical="T0", aliases_list=["a"])
        passed, details = self.orienter.gate()
        self.assertFalse(passed)
        self.assertTrue(any("grounded_terms" in f for f in details["failures"]))

    def test_gate_fails_on_insufficient_metrics(self):
        """10 terms but only 1 metric promoted → fails metrics floor."""
        for i in range(GATE_MIN_TERMS_STANDARD):
            self.orienter.ground(
                term=f"T{i}", definition="d",
                source="library", allow_novel=True,
            )
        # Only 1 promoted metric (below floor of 3)
        m = self.orienter.add_metric(
            name="m0", units="x", higher_is_better=True,
            plausibility=(None, 1.0, 2.0, None),
            source="library", url="https://x",
        )
        self.orienter.promote_candidate(m.id)
        for i in range(GATE_MIN_VERIFIED_SOURCES):
            s = self.orienter.add_source(
                title=f"b{i}", category="textbook",
            )
            self.orienter.verify_source(s.id, http_status=200)
        self.orienter.alias(canonical="T0", aliases_list=["a"])
        passed, details = self.orienter.gate()
        self.assertFalse(passed)
        self.assertTrue(any("metrics_promoted" in f for f in details["failures"]))

    def test_gate_fails_on_insufficient_verified_sources(self):
        for i in range(GATE_MIN_TERMS_STANDARD):
            self.orienter.ground(
                term=f"T{i}", definition="d",
                source="library", allow_novel=True,
            )
        for i in range(GATE_MIN_METRICS):
            m = self.orienter.add_metric(
                name=f"m{i}", units="x", higher_is_better=True,
                plausibility=(None, 1.0, 2.0, None),
                source="library", url="https://x",
            )
            self.orienter.promote_candidate(m.id)
        # Only 1 verified source
        s = self.orienter.add_source(
            title="book0", category="textbook",
        )
        self.orienter.verify_source(s.id, http_status=200)
        self.orienter.alias(canonical="T0", aliases_list=["a"])
        passed, details = self.orienter.gate()
        self.assertFalse(passed)
        self.assertTrue(any("verified_sources" in f for f in details["failures"]))

    def test_gate_fails_on_low_library_fraction(self):
        """lib_fraction < 0.30 fails even when term count is sufficient."""
        # 2 library + 8 analyst → fraction = 0.20 < 0.30
        for i in range(2):
            self.orienter.ground(
                term=f"L{i}", definition="d",
                source="library", allow_novel=True,
            )
        for i in range(8):
            self.orienter.ground(
                term=f"A{i}", definition="d",
                source="analyst", allow_novel=True,
            )
        for i in range(GATE_MIN_METRICS):
            m = self.orienter.add_metric(
                name=f"m{i}", units="x", higher_is_better=True,
                plausibility=(None, 1.0, 2.0, None),
                source="library", url="https://x",
            )
            self.orienter.promote_candidate(m.id)
        for i in range(GATE_MIN_VERIFIED_SOURCES):
            s = self.orienter.add_source(
                title=f"b{i}", category="textbook",
            )
            self.orienter.verify_source(s.id, http_status=200)
        self.orienter.alias(canonical="L0", aliases_list=["a"])
        passed, details = self.orienter.gate()
        self.assertFalse(passed)
        self.assertTrue(any(
            "library_sourced_fraction" in f for f in details["failures"]
        ))

    def test_gate_lite_relaxes_terms_floor_to_5(self):
        """LITE tier uses GATE_MIN_TERMS_LITE=5 instead of 10."""
        # Fresh LITE session
        path = _fresh_tmp()
        try:
            lite = DomainOrienter(path)
            lite.start(tier="LITE")
            # 5 library terms → meets LITE floor
            for i in range(GATE_MIN_TERMS_LITE):
                lite.ground(
                    term=f"T{i}", definition="d",
                    source="library", allow_novel=True,
                )
            # 2 verified sources
            for i in range(GATE_MIN_VERIFIED_SOURCES):
                s = lite.add_source(
                    title=f"b{i}", category="textbook",
                )
                lite.verify_source(s.id, http_status=200)
            # LITE does NOT require metrics or aliases
            passed, details = lite.gate()
            self.assertTrue(
                passed,
                f"LITE should pass at {GATE_MIN_TERMS_LITE} terms; "
                f"failures: {details['failures']}",
            )
        finally:
            _cleanup(path)

    def test_gate_lite_fails_below_lite_floor(self):
        path = _fresh_tmp()
        try:
            lite = DomainOrienter(path)
            lite.start(tier="LITE")
            # Only 4 terms — below LITE floor of 5
            for i in range(GATE_MIN_TERMS_LITE - 1):
                lite.ground(
                    term=f"T{i}", definition="d",
                    source="library", allow_novel=True,
                )
            for i in range(GATE_MIN_VERIFIED_SOURCES):
                s = lite.add_source(
                    title=f"b{i}", category="textbook",
                )
                lite.verify_source(s.id, http_status=200)
            passed, details = lite.gate()
            self.assertFalse(passed)
            self.assertTrue(any(
                "grounded_terms" in f and "LITE" in f
                for f in details["failures"]
            ))
        finally:
            _cleanup(path)

    def test_gate_persists_last_checked(self):
        self._fill_standard_floor(self.orienter)
        self.orienter.gate()
        self.assertIsNotNone(self.orienter.state.gate_last_checked)
        self.assertEqual(self.orienter.state.gate_status, "PASS")


# ---------------------------------------------------------------------------
# TestRendering — glossary / metrics / sources output
# ---------------------------------------------------------------------------

class TestRendering(unittest.TestCase):

    def setUp(self):
        self.state_path = _fresh_tmp()
        self.orienter = DomainOrienter(self.state_path)
        self.orienter.start(domain="credit_derivatives")

    def tearDown(self):
        _cleanup(self.state_path)

    def test_render_glossary_contains_term_heading(self):
        self.orienter.ground(
            term="CDS", definition="credit default swap",
            source="library", allow_novel=True,
        )
        md = self.orienter.render_glossary()
        # Each grounded term is rendered as a ### heading
        self.assertIn("### CDS", md)
        self.assertIn("credit default swap", md)

    def test_render_metrics_matches_domains_schema(self):
        """render_metrics returns {domain: {metric: plausibility_list}}."""
        m = self.orienter.add_metric(
            name="cs01",
            units="USD/bp",
            higher_is_better=False,
            plausibility=(None, 100.0, 10000.0, None),
            source="library",
            url="https://x",
            domain="credit_derivatives",
        )
        self.orienter.promote_candidate(m.id)
        out = self.orienter.render_metrics()
        self.assertIsInstance(out, dict)
        self.assertIn("credit_derivatives", out)
        self.assertIsInstance(out["credit_derivatives"], dict)
        self.assertIn("cs01", out["credit_derivatives"])
        self.assertIsInstance(out["credit_derivatives"]["cs01"], list)
        self.assertEqual(len(out["credit_derivatives"]["cs01"]), 4)

    def test_render_metrics_excludes_unpromoted(self):
        """Staged (promoted=False) metrics are NOT rendered."""
        self.orienter.add_metric(
            name="staged",
            units="x",
            higher_is_better=True,
            plausibility=(None, 1.0, 2.0, None),
            source="analyst",
        )
        out = self.orienter.render_metrics()
        # credit_derivatives key may not appear at all
        self.assertNotIn("staged", str(out))

    def test_render_sources_groups_by_category(self):
        self.orienter.add_source(
            title="Hull Textbook", category="textbook",
        )
        self.orienter.add_source(
            title="ISDA Standard", category="standard",
        )
        md = self.orienter.render_sources()
        self.assertIn("Textbooks", md)
        self.assertIn("Standards", md)
        self.assertIn("Hull Textbook", md)
        self.assertIn("ISDA Standard", md)

    def test_render_glossary_without_grounded_terms_returns_minimal_string(self):
        """Renders a non-crashing string with a placeholder."""
        md = self.orienter.render_glossary()
        self.assertIsInstance(md, str)
        self.assertIn("# Domain Glossary", md)
        self.assertIn("No grounded terms", md)

    def test_render_sources_without_sources_returns_minimal_string(self):
        md = self.orienter.render_sources()
        self.assertIsInstance(md, str)
        self.assertIn("# Domain Sources", md)
        self.assertIn("No sources registered", md)


# ---------------------------------------------------------------------------
# TestSkip — skip_block helper
# ---------------------------------------------------------------------------

class TestSkip(unittest.TestCase):

    def setUp(self):
        self.state_path = _fresh_tmp()
        self.orienter = DomainOrienter(self.state_path)
        self.orienter.start(tier="STANDARD", domain="d")

    def tearDown(self):
        _cleanup(self.state_path)

    def test_skip_block_contains_reason(self):
        block = self.orienter.skip_block("domain_familiarity=high, 10yr exp")
        self.assertIn("Phase 0.3 skipped", block)
        self.assertIn("domain_familiarity=high, 10yr exp", block)
        self.assertIn("Attestation", block)

    def test_skip_does_not_mutate_state(self):
        """Skip is a session-manager concern — the tool only emits text."""
        before_terms = len(self.orienter.state.grounded_terms)
        before_metrics = len(self.orienter.state.metrics)
        before_sources = len(self.orienter.state.sources)
        self.orienter.skip_block("reason text")
        # State should be unchanged after skip_block
        self.assertEqual(
            len(self.orienter.state.grounded_terms), before_terms,
        )
        self.assertEqual(len(self.orienter.state.metrics), before_metrics)
        self.assertEqual(len(self.orienter.state.sources), before_sources)

    def test_skip_empty_reason_raises(self):
        with self.assertRaises(ValueError):
            self.orienter.skip_block("")


# ---------------------------------------------------------------------------
# TestReport — markdown report
# ---------------------------------------------------------------------------

class TestReport(unittest.TestCase):

    def setUp(self):
        self.state_path = _fresh_tmp()
        self.orienter = DomainOrienter(self.state_path)
        self.orienter.start(domain="d")
        self.orienter.ground(
            term="X", definition="d", source="library", allow_novel=True,
        )
        self.orienter.add_metric(
            name="m1", units="x", higher_is_better=True,
            plausibility=(None, 1.0, 2.0, None),
            source="analyst",
        )
        self.orienter.add_source(title="Book", category="textbook")

    def tearDown(self):
        _cleanup(self.state_path)

    def test_report_compact_contains_heading(self):
        out = self.orienter.report(verbose=False)
        self.assertIn("Domain Orientation Report", out)
        self.assertIn("Tier:", out)

    def test_report_verbose_contains_all_sections(self):
        out = self.orienter.report(verbose=True)
        self.assertIn("Candidate Terms", out)
        self.assertIn("Grounded Terms", out)
        self.assertIn("Metrics", out)
        self.assertIn("Sources", out)

    def test_report_verbose_is_longer_than_compact(self):
        compact = self.orienter.report(verbose=False)
        verbose = self.orienter.report(verbose=True)
        self.assertGreater(len(verbose), len(compact))


# ---------------------------------------------------------------------------
# TestPersistence — round-trip + helper functions
# ---------------------------------------------------------------------------

class TestPersistence(unittest.TestCase):

    def setUp(self):
        self.state_path = _fresh_tmp()

    def tearDown(self):
        _cleanup(self.state_path)

    def test_round_trip_preserves_all_entity_lists(self):
        o1 = DomainOrienter(self.state_path)
        o1.start(tier="STANDARD", domain="d")
        o1.ground("T", "def", "library", allow_novel=True)
        o1.add_metric(
            "m", "x", True, (None, 1.0, 2.0, None),
            source="analyst",
        )
        o1.add_source("book", "textbook")
        o1.alias("T", ["t-alias"])
        # New engine re-reads state file
        o2 = DomainOrienter(self.state_path)
        self.assertIsNotNone(o2.state)
        self.assertEqual(len(o2.state.grounded_terms), 1)
        self.assertEqual(len(o2.state.metrics), 1)
        self.assertEqual(len(o2.state.sources), 1)
        self.assertEqual(len(o2.state.aliases), 1)

    def test_require_session_raises_without_start(self):
        o = DomainOrienter(self.state_path)
        with self.assertRaises(RuntimeError):
            o.require_session()


# ---------------------------------------------------------------------------
# TestCandidates — candidate management
# ---------------------------------------------------------------------------

class TestCandidates(unittest.TestCase):

    def setUp(self):
        self.state_path = _fresh_tmp()
        self.orienter = DomainOrienter(self.state_path)
        self.orienter.start()

    def tearDown(self):
        _cleanup(self.state_path)

    def test_list_candidates_terms(self):
        self.orienter.extract("AAA BBB")
        items = self.orienter.list_candidates("terms")
        self.assertEqual(len(items), 2)

    def test_list_candidates_invalid_kind_raises(self):
        with self.assertRaises(ValueError):
            self.orienter.list_candidates("bogus")

    def test_promote_missing_metric_raises_key_error(self):
        with self.assertRaises(KeyError):
            self.orienter.promote_candidate("MET-999")

    def test_promote_bad_prefix_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.orienter.promote_candidate("TERM-001")


# ---------------------------------------------------------------------------
# TestHelpers — pure helper functions
# ---------------------------------------------------------------------------

class TestHelpers(unittest.TestCase):

    def test_parse_bool_true_variants(self):
        for val in ("true", "True", "1", "yes", "y", "t"):
            self.assertTrue(_parse_bool(val))

    def test_parse_bool_false_variants(self):
        for val in ("false", "False", "0", "no", "n", "f"):
            self.assertFalse(_parse_bool(val))

    def test_parse_bool_invalid_raises(self):
        with self.assertRaises(ValueError):
            _parse_bool("maybe")

    def test_parse_aliases_strips_and_filters_empty(self):
        out = _parse_aliases(" a , b ,, c ,")
        self.assertEqual(out, ["a", "b", "c"])

    def test_recover_counter_finds_max(self):
        entries = [{"id": "TERM-001"}, {"id": "TERM-005"}, {"id": "TERM-003"}]
        self.assertEqual(_recover_counter(entries, "TERM-"), 6)

    def test_recover_counter_empty_returns_one(self):
        self.assertEqual(_recover_counter([], "TERM-"), 1)


# ---------------------------------------------------------------------------
# TestSmokeCLI — subprocess round-trips
# ---------------------------------------------------------------------------

class TestSmokeCLI(unittest.TestCase):

    def _tmp_state(self):
        return _fresh_tmp()

    def _cleanup(self, path):
        _cleanup(path)

    def test_help_exits_zero(self):
        result = subprocess.run(
            [sys.executable, SCRIPT_PATH, "--help"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("domain_orienter.py", result.stdout)

    def test_start_exits_zero(self):
        path = self._tmp_state()
        try:
            result = subprocess.run(
                [sys.executable, SCRIPT_PATH, "--file", path, "start",
                 "--tier", "STANDARD"],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0)
            self.assertIn("Started domain-orientation session", result.stdout)
            self.assertIn("domain-", result.stdout)
            self.assertTrue(os.path.exists(path))
        finally:
            self._cleanup(path)

    def test_gate_on_empty_state_exits_one(self):
        """Gate FAIL (fresh session, no grounding) → exit 1."""
        path = self._tmp_state()
        try:
            subprocess.run(
                [sys.executable, SCRIPT_PATH, "--file", path, "start"],
                capture_output=True, text=True, check=True,
            )
            result = subprocess.run(
                [sys.executable, SCRIPT_PATH, "--file", path, "gate"],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("pass: False", result.stdout)
        finally:
            self._cleanup(path)

    def test_ground_bad_confidence_exits_two(self):
        """ValueError path (analyst confidence > cap) yields exit 2."""
        path = self._tmp_state()
        try:
            subprocess.run(
                [sys.executable, SCRIPT_PATH, "--file", path, "start"],
                capture_output=True, text=True, check=True,
            )
            result = subprocess.run(
                [sys.executable, SCRIPT_PATH, "--file", path, "ground",
                 "--term", "FOO", "--definition", "d",
                 "--source", "analyst", "--confidence", "0.95",
                 "--allow-novel"],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("ERROR", result.stderr)
            self.assertIn("analyst confidence cap", result.stderr)
        finally:
            self._cleanup(path)

    def test_ground_invalid_source_rejected_by_argparse(self):
        """argparse choices= rejects unknown --source before our code runs."""
        path = self._tmp_state()
        try:
            subprocess.run(
                [sys.executable, SCRIPT_PATH, "--file", path, "start"],
                capture_output=True, text=True, check=True,
            )
            result = subprocess.run(
                [sys.executable, SCRIPT_PATH, "--file", path, "ground",
                 "--term", "X", "--definition", "d",
                 "--source", "wikipedia", "--allow-novel"],
                capture_output=True, text=True,
            )
            # argparse exits with 2 for choice-validation errors
            self.assertEqual(result.returncode, 2)
            self.assertIn("invalid choice", result.stderr)
        finally:
            self._cleanup(path)

    def test_end_to_end_extract_ground_gate(self):
        """Full mini-flow: start → extract → ground → gate reports results."""
        path = self._tmp_state()
        try:
            subprocess.run(
                [sys.executable, SCRIPT_PATH, "--file", path, "start",
                 "--tier", "STANDARD", "--domain", "test"],
                capture_output=True, text=True, check=True,
            )
            # extract inline text
            r = subprocess.run(
                [sys.executable, SCRIPT_PATH, "--file", path, "extract",
                 "--input", "CDS OIS ISDA and Basel"],
                capture_output=True, text=True,
            )
            self.assertEqual(r.returncode, 0)
            self.assertIn("TE produced", r.stdout)

            # ground a candidate
            r = subprocess.run(
                [sys.executable, SCRIPT_PATH, "--file", path, "ground",
                 "--term", "CDS",
                 "--definition", "credit default swap",
                 "--source", "library"],
                capture_output=True, text=True,
            )
            self.assertEqual(r.returncode, 0)
            self.assertIn("Grounded", r.stdout)

            # report should now mention the grounded term
            r = subprocess.run(
                [sys.executable, SCRIPT_PATH, "--file", path, "report",
                 "--verbose"],
                capture_output=True, text=True,
            )
            self.assertEqual(r.returncode, 0)
            self.assertIn("CDS", r.stdout)

            # Verify state file contains expected JSON fields
            with open(path) as f:
                data = json.load(f)
            self.assertIn("grounded_terms", data)
            self.assertEqual(len(data["grounded_terms"]), 1)
            self.assertIn("tier", data)
            self.assertEqual(data["tier"], "STANDARD")
        finally:
            self._cleanup(path)

    def test_skip_cli_prints_block_and_exits_zero(self):
        path = self._tmp_state()
        try:
            subprocess.run(
                [sys.executable, SCRIPT_PATH, "--file", path, "start"],
                capture_output=True, text=True, check=True,
            )
            r = subprocess.run(
                [sys.executable, SCRIPT_PATH, "--file", path, "skip",
                 "--reason", "analyst declared high familiarity"],
                capture_output=True, text=True,
            )
            self.assertEqual(r.returncode, 0)
            self.assertIn("Phase 0.3 skipped", r.stdout)
            self.assertIn("analyst declared high familiarity", r.stdout)
        finally:
            self._cleanup(path)


if __name__ == "__main__":
    unittest.main()
