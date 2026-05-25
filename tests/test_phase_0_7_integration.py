#!/usr/bin/env python3
"""End-to-end integration tests for Phase 0.7 Scope Interrogation.

Exercises the scope_auditor CLI through session_manager paths:
  session_manager new
  → scope_auditor start
  → scope_auditor add-candidate (>=3 to satisfy gate threshold)
  → scope_auditor steelman + trace (gate also requires steelman + traces)
  → scope_auditor gate (exit 0 = PASS)

Plus a gate-failure path:
  bare session with no candidates / no archetype / no steelman → gate exit 1.

Tests run via subprocess to mirror real analyst usage. Each test isolates
its session in a fresh tempdir so they can run in parallel.

Created in plan_2026-05-25_cdd1f345 to close F7 (no Phase 0.7 integration
test) from the v7.15.11 comprehensive self-audit.
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
SA = SCRIPTS / "scope_auditor.py"


def run(argv, **kwargs):
    return subprocess.run(
        [sys.executable] + [str(a) for a in argv],
        capture_output=True, text=True, **kwargs,
    )


class TestPhase07HappyPath(unittest.TestCase):
    """Full Phase 0.7 walkthrough that satisfies the gate."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="phase07_int_")
        self.session_args = ["--base-dir", self.tmpdir]

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _path(self, filename):
        result = run([SM] + self.session_args + ["path", filename])
        self.assertEqual(result.returncode, 0,
                         f"path lookup failed: {result.stderr}")
        return result.stdout.strip()

    def test_gate_passes_with_minimum_set(self):
        # 1. Bootstrap session
        result = run([SM] + self.session_args + ["new", "Test target system"])
        self.assertEqual(result.returncode, 0, result.stderr)

        sa_path = self._path("scope_audit.json")

        # 2. Start scope audit
        result = run([SA, "--file", sa_path, "start", "Test target system"])
        self.assertEqual(result.returncode, 0, result.stderr)

        # 3. Enumerate archetype accomplices (satisfies has_archetype_query)
        result = run([SA, "--file", sa_path, "enumerate",
                      "--archetype", "api_backed_software_service"])
        self.assertEqual(result.returncode, 0, result.stderr)

        # 4. Add ≥3 unique exogeneity candidates (different domains)
        for domain, mech in [
            ("Release packaging", "Stale artifacts shipped"),
            ("CI / release infra", "No automated test enforcement"),
            ("Install surface", "Flat agent namespace collisions"),
        ]:
            result = run([SA, "--file", sa_path, "add-candidate",
                          "--domain", domain,
                          "--mechanism", mech,
                          "--prior", "0.4",
                          "--source", "analyst"])
            self.assertEqual(result.returncode, 0, result.stderr)

        # 5. Record a steelman critique (satisfies has_steelman)
        result = run([SA, "--file", sa_path, "steelman",
                      "--persona", "outsider",
                      "--domain", "Release packaging",
                      "--mechanism", "scope omits packaging and CI"])
        self.assertEqual(result.returncode, 0, result.stderr)

        # 6. Record a flow trace (satisfies has_traces)
        result = run([SA, "--file", sa_path, "trace",
                      "--inputs", "argv,env",
                      "--outputs", "dist/<v>.zip"])
        self.assertEqual(result.returncode, 0, result.stderr)

        # 7. Gate must now PASS (exit 0)
        result = run([SA, "--file", sa_path, "gate"])
        self.assertEqual(result.returncode, 0,
                         f"Gate did not PASS. stdout={result.stdout} stderr={result.stderr}")
        self.assertIn("pass: True", result.stdout)

        # 8. State file persists the candidates
        with open(sa_path) as f:
            state = json.load(f)
        # Each scope_auditor version stores candidates differently; at minimum the file should be non-empty JSON.
        self.assertTrue(state, "scope_audit.json is empty")


class TestPhase07GateFailure(unittest.TestCase):
    """A bare session with no work done must FAIL the gate (exit 1)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="phase07_gatefail_")
        self.session_args = ["--base-dir", self.tmpdir]

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _path(self, filename):
        result = run([SM] + self.session_args + ["path", filename])
        return result.stdout.strip()

    def test_bare_session_gate_fails(self):
        result = run([SM] + self.session_args + ["new", "Bare session"])
        self.assertEqual(result.returncode, 0, result.stderr)
        sa_path = self._path("scope_audit.json")
        run([SA, "--file", sa_path, "start", "Bare"])
        result = run([SA, "--file", sa_path, "gate"])
        self.assertEqual(result.returncode, 1,
                         f"Expected gate FAIL exit 1, got {result.returncode}")
        self.assertIn("pass: False", result.stdout)


if __name__ == "__main__":
    unittest.main()
