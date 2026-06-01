#!/usr/bin/env python3
"""Tests for src/scripts/mermaid_render.py

No mocks. Pure-function determinism, escaping, and real-data round-trips
(PHASE_SEQUENCE from session_manager; a realistic inference chain). CLI smoke
via subprocess.
"""

import os
import re
import subprocess
import sys
import unittest

# Allow importing from src/scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'scripts'))

from mermaid_render import (
    fence,
    _sanitize_id,
    _escape_label,
    phase_sequence_to_mermaid,
    inference_chain_to_mermaid,
    coverage_graph_to_mermaid,
    causal_graph_to_mermaid,
    adjacency_to_mermaid,
)
from session_manager import PHASE_SEQUENCE

SCRIPT = os.path.join(
    os.path.dirname(__file__), '..', 'src', 'scripts', 'mermaid_render.py'
)

ID_RE = re.compile(r'^[a-z_][a-z0-9_]*$')


def first_nonempty_line(body):
    for line in body.splitlines():
        if line.strip():
            return line.strip()
    return ""


# Small realistic fixtures ---------------------------------------------------

SINGLE_TIER = {"0": "0.3", "0.3": "1", "1": None}

INFERENCE_CHAIN = {
    "id": "IC1",
    "target": "H1",
    "premise": "Observed latency spike correlates with deploy",
    "steps": [
        {"idx": 1, "claim": "Deploy changed the query plan", "lr": 1.5},
        {"idx": 2, "claim": "New plan does a full scan", "lr": 3.0},
    ],
    "status": "closed",
    "final_posterior": 0.82,
}

OBSERVATIONS = [
    {"id": "O1", "text": "Latency spike", "explained_by": ["CAND1"]},
    {"id": "O2", "text": "CPU saturation", "explained_by": []},
]
CANDIDATES = [
    {"id": "CAND1", "cause": "Bad query plan", "observations_explained": ["O1", "O2"]},
]

CAUSAL_NODES = [
    {"id": "load", "name": "System load"},
    {"id": "latency", "name": "Latency"},
]
CAUSAL_EDGES = [
    {"source": "load", "target": "latency", "sign": "+"},
    {"source": "latency", "target": "load", "sign": "-"},
]

ADJ = {"a": ["b", "c"], "b": ["a"], "c": ["a"]}


class TestEscaping(unittest.TestCase):
    def test_sanitize_id_phase_label(self):
        out = _sanitize_id("Phase 0.3 (low)")
        self.assertTrue(ID_RE.match(out), out)
        self.assertNotIn('"', out)
        self.assertNotIn('\n', out)

    def test_sanitize_id_digit_prefix(self):
        out = _sanitize_id("0.3")
        self.assertTrue(ID_RE.match(out), out)
        self.assertTrue(out.startswith("n"))

    def test_sanitize_id_empty_and_symbols(self):
        self.assertTrue(ID_RE.match(_sanitize_id("")))
        self.assertTrue(ID_RE.match(_sanitize_id("→⊕()")))

    def test_sanitize_id_deterministic(self):
        self.assertEqual(_sanitize_id("1-P.5"), _sanitize_id("1-P.5"))

    def test_escape_label_quotes_and_newlines(self):
        out = _escape_label('he said "hi"\nthen left')
        self.assertNotIn('"', out)
        self.assertNotIn('\n', out)

    def test_escape_label_transliterates_arrows(self):
        out = _escape_label("A → B")
        self.assertNotIn("→", out)
        self.assertIn("->", out)

    def test_escape_label_brackets(self):
        out = _escape_label("step[1]{x}")
        self.assertNotIn("[", out)
        self.assertNotIn("]", out)
        self.assertNotIn("{", out)


class TestHeaders(unittest.TestCase):
    def test_phase_sequence_header(self):
        body = phase_sequence_to_mermaid(SINGLE_TIER)
        self.assertEqual(first_nonempty_line(body), "stateDiagram-v2")

    def test_inference_chain_header(self):
        body = inference_chain_to_mermaid(INFERENCE_CHAIN)
        self.assertTrue(first_nonempty_line(body).startswith("flowchart"))

    def test_coverage_header(self):
        body = coverage_graph_to_mermaid(OBSERVATIONS, CANDIDATES)
        self.assertTrue(first_nonempty_line(body).startswith("graph"))

    def test_causal_header(self):
        body = causal_graph_to_mermaid(CAUSAL_NODES, CAUSAL_EDGES)
        self.assertTrue(first_nonempty_line(body).startswith("flowchart"))

    def test_adjacency_header(self):
        body = adjacency_to_mermaid(ADJ)
        self.assertTrue(first_nonempty_line(body).startswith("graph"))


class TestDeterminism(unittest.TestCase):
    def test_phase_sequence_deterministic(self):
        self.assertEqual(
            phase_sequence_to_mermaid(PHASE_SEQUENCE, tier="STANDARD", current="1"),
            phase_sequence_to_mermaid(PHASE_SEQUENCE, tier="STANDARD", current="1"),
        )

    def test_inference_chain_deterministic(self):
        self.assertEqual(
            inference_chain_to_mermaid(INFERENCE_CHAIN),
            inference_chain_to_mermaid(INFERENCE_CHAIN),
        )

    def test_coverage_deterministic(self):
        self.assertEqual(
            coverage_graph_to_mermaid(OBSERVATIONS, CANDIDATES),
            coverage_graph_to_mermaid(OBSERVATIONS, CANDIDATES),
        )

    def test_causal_deterministic(self):
        self.assertEqual(
            causal_graph_to_mermaid(CAUSAL_NODES, CAUSAL_EDGES),
            causal_graph_to_mermaid(CAUSAL_NODES, CAUSAL_EDGES),
        )

    def test_adjacency_deterministic(self):
        self.assertEqual(adjacency_to_mermaid(ADJ), adjacency_to_mermaid(ADJ))


class TestPhaseSequenceRoundTrip(unittest.TestCase):
    def _assert_state_diagram(self, body):
        self.assertEqual(first_nonempty_line(body), "stateDiagram-v2")
        self.assertIn("[*] -->", body)
        self.assertIn("--> [*]", body)
        # No unterminated node: every non-header, non-classDef line is an edge
        # or a state declaration.
        for line in body.splitlines()[1:]:
            s = line.strip()
            if not s or s.startswith("classDef") or s.startswith("class ") \
                    or s.startswith("state "):
                continue
            self.assertIn("-->", s, "unexpected line: %r" % s)

    def test_standard_tier(self):
        body = phase_sequence_to_mermaid(PHASE_SEQUENCE, tier="STANDARD")
        self._assert_state_diagram(body)

    def test_psych_tier(self):
        body = phase_sequence_to_mermaid(PHASE_SEQUENCE, tier="PSYCH")
        self._assert_state_diagram(body)

    def test_all_tiers_roundtrip(self):
        for tier in PHASE_SEQUENCE:
            body = phase_sequence_to_mermaid(PHASE_SEQUENCE, tier=tier)
            self._assert_state_diagram(body)

    def test_current_highlight(self):
        body = phase_sequence_to_mermaid(PHASE_SEQUENCE, tier="STANDARD", current="1")
        self.assertIn("classDef current", body)
        self.assertIn("class ", body)

    def test_labels_form(self):
        body = phase_sequence_to_mermaid(SINGLE_TIER, labels={"0": "Setup"})
        self.assertIn('state "Setup" as', body)

    def test_single_tier_map_no_tier_arg(self):
        body = phase_sequence_to_mermaid(SINGLE_TIER)
        self.assertEqual(first_nonempty_line(body), "stateDiagram-v2")
        self.assertIn("[*] -->", body)
        self.assertIn("--> [*]", body)


class TestInferenceChainRoundTrip(unittest.TestCase):
    def test_lr_labels_present(self):
        body = inference_chain_to_mermaid(INFERENCE_CHAIN)
        self.assertTrue(first_nonempty_line(body).startswith("flowchart"))
        self.assertIn("LR 1.5", body)
        self.assertIn("LR 3", body)

    def test_closed_shows_posterior(self):
        body = inference_chain_to_mermaid(INFERENCE_CHAIN)
        self.assertIn("posterior", body)

    def test_open_chain_no_close(self):
        chain = dict(INFERENCE_CHAIN)
        chain["status"] = "open"
        body = inference_chain_to_mermaid(chain)
        self.assertNotIn("posterior", body)


class TestCoverageGraph(unittest.TestCase):
    def test_edge_from_candidate_to_observation(self):
        body = coverage_graph_to_mermaid(OBSERVATIONS, CANDIDATES)
        self.assertIn("cand_cand1 --> obs_o1", body)
        self.assertIn("cand_cand1 --> obs_o2", body)

    def test_classdefs_present(self):
        body = coverage_graph_to_mermaid(OBSERVATIONS, CANDIDATES)
        self.assertIn("classDef observation", body)
        self.assertIn("classDef candidate", body)


class TestCausalGraph(unittest.TestCase):
    def test_signed_edges(self):
        body = causal_graph_to_mermaid(CAUSAL_NODES, CAUSAL_EDGES)
        self.assertIn("|+|", body)
        self.assertIn("|-|", body)


class TestFence(unittest.TestCase):
    def test_fence_balanced(self):
        out = fence("flowchart LR\n    a --> b")
        self.assertEqual(out.count("```"), 2)
        self.assertTrue(out.startswith("```mermaid"))
        self.assertTrue(out.endswith("```"))


class TestCLI(unittest.TestCase):
    def test_help_exit_zero(self):
        r = subprocess.run(
            [sys.executable, SCRIPT, "--help"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        self.assertEqual(r.returncode, 0)

    def test_demo_exit_zero_and_diagram(self):
        r = subprocess.run(
            [sys.executable, SCRIPT, "demo"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        self.assertEqual(r.returncode, 0)
        out = r.stdout.decode()
        self.assertIn("stateDiagram-v2", out)
        self.assertIn("```mermaid", out)


SESSION_MGR = os.path.join(
    os.path.dirname(__file__), '..', 'src', 'scripts', 'session_manager.py'
)
ABDUCTIVE = os.path.join(
    os.path.dirname(__file__), '..', 'src', 'scripts', 'abductive_engine.py'
)


class TestDiagramCLI(unittest.TestCase):
    """Subprocess tests for the S3 read-only diagram subcommands."""

    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, argv):
        return subprocess.run(
            argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

    # --- session_manager.py diagram --------------------------------------

    def test_session_diagram_standard(self):
        r = self._run([sys.executable, SESSION_MGR, "--base-dir", self.tmp,
                       "diagram", "--tier", "STANDARD"])
        self.assertEqual(r.returncode, 0, r.stderr.decode())
        out = r.stdout.decode()
        self.assertIn("stateDiagram-v2", out)
        # Balanced ```mermaid fence pair.
        self.assertIn("```mermaid", out)
        self.assertEqual(out.count("```"), 2, out)

    def test_session_diagram_psych(self):
        r = self._run([sys.executable, SESSION_MGR, "--base-dir", self.tmp,
                       "diagram", "--tier", "PSYCH"])
        self.assertEqual(r.returncode, 0, r.stderr.decode())
        self.assertIn("stateDiagram-v2", r.stdout.decode())

    def test_session_diagram_is_readonly(self):
        before = sorted(os.listdir(self.tmp))
        r = self._run([sys.executable, SESSION_MGR, "--base-dir", self.tmp,
                       "diagram", "--tier", "STANDARD"])
        self.assertEqual(r.returncode, 0, r.stderr.decode())
        after = sorted(os.listdir(self.tmp))
        self.assertEqual(before, after, "diagram wrote files: %r -> %r" % (before, after))

    # --- abductive_engine.py chain-diagram / coverage-diagram ------------

    def _fixture(self):
        """Build a minimal abductive_state.json in self.tmp via the CLI itself.

        Returns the fixture path. Starts a session, inverts one observation
        (populates observations + candidates), and builds one closed inference
        chain so chain-diagram has real content.
        """
        path = os.path.join(self.tmp, "abductive_state.json")
        # start
        r = self._run([sys.executable, ABDUCTIVE, "--file", path, "start", "--force"])
        self.assertEqual(r.returncode, 0, r.stderr.decode())
        # invert -> observation O1 + candidates
        r = self._run([sys.executable, ABDUCTIVE, "--file", path, "invert",
                       "--obs-id", "O1", "--text", "Latency spike after deploy",
                       "--category", "generic"])
        self.assertEqual(r.returncode, 0, r.stderr.decode())
        # one closed chain
        r = self._run([sys.executable, ABDUCTIVE, "--file", path, "chain", "start",
                       "--target", "H1", "--premise", "Deploy changed query plan"])
        self.assertEqual(r.returncode, 0, r.stderr.decode())
        r = self._run([sys.executable, ABDUCTIVE, "--file", path, "chain", "step",
                       "--id", "IC1", "--claim", "Plan now does a full scan",
                       "--lr", "3.0", "--source", "analyst"])
        self.assertEqual(r.returncode, 0, r.stderr.decode())
        r = self._run([sys.executable, ABDUCTIVE, "--file", path, "chain", "close",
                       "--id", "IC1", "--seed-prior", "0.5"])
        self.assertEqual(r.returncode, 0, r.stderr.decode())
        return path

    def test_abductive_coverage_diagram(self):
        path = self._fixture()
        r = self._run([sys.executable, ABDUCTIVE, "--file", path, "coverage-diagram"])
        self.assertEqual(r.returncode, 0, r.stderr.decode())
        out = r.stdout.decode()
        self.assertIn("graph", out)
        self.assertIn("```mermaid", out)

    def test_abductive_chain_diagram(self):
        path = self._fixture()
        r = self._run([sys.executable, ABDUCTIVE, "--file", path, "chain-diagram"])
        self.assertEqual(r.returncode, 0, r.stderr.decode())
        out = r.stdout.decode()
        self.assertIn("flowchart", out)
        self.assertIn("```mermaid", out)

    def test_abductive_chain_diagram_empty_graceful(self):
        # A started-but-empty state: chain-diagram must still exit 0 (graceful).
        path = os.path.join(self.tmp, "empty_state.json")
        r = self._run([sys.executable, ABDUCTIVE, "--file", path, "start", "--force"])
        self.assertEqual(r.returncode, 0, r.stderr.decode())
        r = self._run([sys.executable, ABDUCTIVE, "--file", path, "chain-diagram"])
        self.assertEqual(r.returncode, 0, r.stderr.decode())
        self.assertIn("flowchart", r.stdout.decode())


if __name__ == "__main__":
    unittest.main()
