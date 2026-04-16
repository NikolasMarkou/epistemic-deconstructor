#!/usr/bin/env python3
"""End-to-end integration tests for Phase 0.3 Domain Orientation.

Exercises the full STANDARD-tier flow:
  session_manager new
  → analyst declares domain_familiarity=low
  → domain_orienter start/extract/ground/add-metric/source/verify/render/gate
  → bayesian_tracker rename (post-Phase-0.3 hypothesis re-framing)
  → scope_auditor enumerate --glossary <session_glossary>

Plus the SKIP path:
  session_manager new
  → analyst declares domain_familiarity=high
  → session_manager skip 0.3 "<reason>"

Tests run via subprocess to mirror real analyst usage. Each test isolates
its session in a fresh tempdir so they can run in parallel.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "src" / "scripts"
SM = SCRIPTS / "session_manager.py"
DO = SCRIPTS / "domain_orienter.py"
BT = SCRIPTS / "bayesian_tracker.py"
SA = SCRIPTS / "scope_auditor.py"


def run(argv, **kwargs):
    """Run subprocess with text capture and return CompletedProcess."""
    return subprocess.run(
        [sys.executable] + [str(a) for a in argv],
        capture_output=True, text=True, **kwargs,
    )


class TestPhase03IntegrationStandardLowFamiliarity(unittest.TestCase):
    """STANDARD tier with domain_familiarity=low: full Phase 0.3 walkthrough."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="phase03_int_")
        # session_manager creates analyses/ relative to the cwd it's invoked in
        self.session_args = ["--base-dir", self.tmpdir]

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _path(self, filename):
        """Resolve a session file path via session_manager."""
        result = run([SM] + self.session_args + ["path", filename])
        self.assertEqual(result.returncode, 0,
                         f"path lookup failed: {result.stderr}")
        return result.stdout.strip()

    def test_full_flow_credit_derivatives(self):
        # 1. Bootstrap session
        result = run([SM] + self.session_args + ["new", "Credit-derivatives pricing engine"])
        self.assertEqual(result.returncode, 0, result.stderr)

        # 2. domain_orienter start (lazy state file creation)
        do_path = self._path("domain_orientation.json")
        result = run([DO, "--file", do_path, "start",
                     "--tier", "STANDARD", "--domain", "credit_derivatives"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(os.path.exists(do_path))

        # 3. TE — extract from inline text
        text = ("CDS spread is the annual premium paid on a credit default swap. "
                "DV01 measures dollar duration. CS01 captures spread sensitivity. "
                "ISDA 2014 governs the standard. Recovery rate is critical. "
                "ARCH model captures volatility clustering. NARMAX is nonlinear. "
                "Lookback is the historical window. Cox process underlies the model. "
                "Hazard rate determines default intensity. Tenor is the contract length.")
        result = run([DO, "--file", do_path, "extract", "--input", text])
        self.assertEqual(result.returncode, 0, result.stderr)

        # 4. TG — ground at least 10 terms (5 library + 5 analyst is fine)
        # Library terms (no confidence cap; default 1.0)
        for i, term in enumerate([
            "CDS spread", "DV01", "CS01", "ISDA", "Recovery rate",
        ]):
            result = run([DO, "--file", do_path, "ground",
                         "--term", term,
                         "--definition", f"Authoritative definition of {term} per ISDA.",
                         "--source", "library",
                         "--url", "https://www.isda.org/",
                         "--allow-novel"])
            self.assertEqual(result.returncode, 0,
                             f"ground {term} failed: {result.stderr}")
        # Analyst terms (cap 0.80)
        for term in ["ARCH model", "NARMAX", "Lookback", "Cox process", "Hazard rate"]:
            result = run([DO, "--file", do_path, "ground",
                         "--term", term,
                         "--definition", f"Analyst-grounded definition of {term}.",
                         "--source", "analyst",
                         "--confidence", "0.7",
                         "--allow-novel"])
            self.assertEqual(result.returncode, 0,
                             f"ground {term} failed: {result.stderr}")

        # 5. MM — add 3 metrics with plausibility bounds, then promote
        for name, units, plaus in [
            ("cs01", "USD/bp", "null,100,10000,null"),
            ("recovery_rate", "fraction", "0.99,0.2,0.6,0.8"),
            ("default_intensity", "1/yr", "10.0,0.001,0.05,0.5"),
        ]:
            result = run([DO, "--file", do_path, "add-metric",
                         "--name", name, "--units", units,
                         "--higher-is-better", "false",
                         "--plausibility", plaus,
                         "--source", "library", "--url", "https://www.isda.org/",
                         "--domain", "credit_derivatives"])
            self.assertEqual(result.returncode, 0,
                             f"add-metric {name} failed: {result.stderr}")
        # Promote each so they count toward metrics_promoted threshold
        for mid in ["MET-001", "MET-002", "MET-003"]:
            result = run([DO, "--file", do_path, "candidates", "promote",
                         "--id", mid])
            self.assertEqual(result.returncode, 0,
                             f"promote {mid} failed: {result.stderr}")

        # 5b. AM — register at least one alias cluster
        result = run([DO, "--file", do_path, "alias",
                     "--canonical", "CDS spread",
                     "--aliases", "credit default swap spread,CDS premium",
                     "--source", "library"])
        self.assertEqual(result.returncode, 0,
                         f"alias failed: {result.stderr}")

        # 6. CS — register and verify 2 sources
        for title in ["ISDA 2014 Credit Derivatives Definitions",
                      "Credit Risk Modeling (Lando)"]:
            result = run([DO, "--file", do_path, "source",
                         "--title", title, "--category", "standard",
                         "--url", "https://www.isda.org/"])
            self.assertEqual(result.returncode, 0)

        # Verify both — get IDs from candidates list
        for sid in ["SID-001", "SID-002"]:
            result = run([DO, "--file", do_path, "verify",
                         "--source-id", sid, "--http-status", "200"])
            self.assertEqual(result.returncode, 0,
                             f"verify {sid} failed: {result.stderr}")

        # 7. Render artifacts
        glossary_path = self._path("domain_glossary.md")
        metrics_path = self._path("domain_metrics.json")
        sources_path = self._path("domain_sources.md")
        for cmd, output in [
            ("glossary", glossary_path),
            ("metrics", metrics_path),
            ("sources", sources_path),
        ]:
            result = run([DO, "--file", do_path, cmd, "render", "--output", output])
            self.assertEqual(result.returncode, 0,
                             f"{cmd} render failed: {result.stderr}")
            self.assertTrue(os.path.exists(output),
                            f"{cmd} render produced no file at {output}")

        # 8. Gate check — must PASS (exit 0)
        result = run([DO, "--file", do_path, "gate"])
        self.assertEqual(result.returncode, 0,
                         f"Gate did not PASS. stdout={result.stdout} stderr={result.stderr}")

        # 9. Verify rendered metrics JSON has the expected schema
        with open(metrics_path) as f:
            metrics = json.load(f)
        self.assertIn("credit_derivatives", metrics)
        self.assertIn("cs01", metrics["credit_derivatives"])
        # Plausibility tuple length
        self.assertEqual(len(metrics["credit_derivatives"]["cs01"]), 4)

        # 10. Glossary should mention CDS spread
        with open(glossary_path) as f:
            glossary_text = f.read()
        self.assertIn("CDS spread", glossary_text)

        # 11. Bayesian rename — post-Phase-0.3 hypothesis re-framing
        hyp_path = self._path("hypotheses.json")
        result = run([BT, "--file", hyp_path, "add",
                     "Generic linear pricing function", "--prior", "0.5"])
        self.assertEqual(result.returncode, 0, result.stderr)
        result = run([BT, "--file", hyp_path, "rename", "H1",
                     "Reduced-form intensity model with stochastic recovery"])
        self.assertEqual(result.returncode, 0, result.stderr)
        # Verify rename took effect (hypotheses are a list keyed by id)
        with open(hyp_path) as f:
            hyp_data = json.load(f)
        h1 = next(h for h in hyp_data["hypotheses"] if h["id"] == "H1")
        self.assertEqual(
            h1["statement"],
            "Reduced-form intensity model with stochastic recovery",
        )

        # 12. scope_auditor enumerate --glossary should produce alignment advisory
        scope_path = self._path("scope_audit.json")
        result = run([SA, "--file", scope_path, "start", "Credit-derivatives pricing engine"])
        self.assertEqual(result.returncode, 0, result.stderr)
        result = run([SA, "--file", scope_path, "enumerate",
                     "--archetype", "speculative_asset_market",
                     "--glossary", do_path])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Glossary alignment", result.stdout)


class TestPhase03IntegrationSkipPath(unittest.TestCase):
    """High-familiarity skip path: declare and bypass via session_manager skip."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="phase03_skip_")
        self.session_args = ["--base-dir", self.tmpdir]

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_skip_logs_decisions_and_state(self):
        # Bootstrap
        result = run([SM] + self.session_args + ["new", "Familiar domain analysis"])
        self.assertEqual(result.returncode, 0, result.stderr)

        # Skip Phase 0.3
        result = run([SM] + self.session_args + ["skip", "0.3",
                     "domain_familiarity=high; analyst is SME with 10y experience"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Skipped Phase 0.3", result.stdout)

        # decisions.md should reference the skip
        decisions_path_result = run([SM] + self.session_args + ["path", "decisions.md"])
        decisions_path = decisions_path_result.stdout.strip()
        with open(decisions_path) as f:
            decisions = f.read()
        self.assertIn("SKIP Phase 0.3", decisions)
        self.assertIn("domain_familiarity=high", decisions)

        # state.md should record the transition
        state_path_result = run([SM] + self.session_args + ["path", "state.md"])
        state_path = state_path_result.stdout.strip()
        with open(state_path) as f:
            state = f.read()
        self.assertIn("SKIP Phase 0.3", state)


class TestPhase03IntegrationGateFailure(unittest.TestCase):
    """A session that does not meet thresholds must FAIL the gate (exit 1)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="phase03_gatefail_")
        self.session_args = ["--base-dir", self.tmpdir]

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _path(self, filename):
        result = run([SM] + self.session_args + ["path", filename])
        return result.stdout.strip()

    def test_empty_state_gate_fails_exit_1(self):
        result = run([SM] + self.session_args + ["new", "Empty session"])
        self.assertEqual(result.returncode, 0, result.stderr)
        do_path = self._path("domain_orientation.json")
        run([DO, "--file", do_path, "start", "--tier", "STANDARD"])
        result = run([DO, "--file", do_path, "gate"])
        # Empty state cannot meet 10-term floor
        self.assertEqual(result.returncode, 1,
                         f"Expected gate FAIL exit 1, got {result.returncode}")


if __name__ == "__main__":
    unittest.main()
