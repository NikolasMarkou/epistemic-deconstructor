"""Tests for src/scripts/phase_gate.py.

plan_2026-05-28_b0332cf7/D-001: structural exit-gate dispatcher covering the 13
non-terminal FSM phases that previously advanced on file-presence only. Audit
H1/H10 remediation.
"""

import os
import subprocess
import sys
import tempfile
import unittest


SCRIPT = os.path.join(
    os.path.dirname(__file__), '..', 'src', 'scripts', 'phase_gate.py')


def run_gate(path):
    """Invoke phase_gate.py for *path* and return CompletedProcess."""
    return subprocess.run(
        [sys.executable, SCRIPT, '--file', path, 'gate'],
        capture_output=True, text=True)


class _PhaseGateBase(unittest.TestCase):

    def setUp(self):
        self.session = tempfile.mkdtemp()
        self.phase_out = os.path.join(self.session, 'phase_outputs')
        os.makedirs(self.phase_out, exist_ok=True)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.session, ignore_errors=True)

    def _write(self, name, content):
        path = os.path.join(self.phase_out, name)
        with open(path, 'w') as f:
            f.write(content)
        return path

    def _seed_hypotheses(self, n=3):
        """Seed <session>/hypotheses.json with n minimal entries so the
        Phase-0 REQUIRED hypotheses_json_seeded gate (D-003) passes."""
        import json
        path = os.path.join(self.session, 'hypotheses.json')
        data = {"hypotheses": [
            {"id": f"H{i}", "statement": f"hypothesis {i}", "prior": 0.5}
            for i in range(1, n + 1)
        ]}
        with open(path, 'w') as f:
            json.dump(data, f)
        return path


class TestPhase0Gate(_PhaseGateBase):

    GOOD = (
        "# Phase 0 — Setup & Framing\n\n"
        "Hypothesis seeds: H1, H2, H3 with [H_S] / [H_S_prime].\n"
        "Framing per SKILL.md; H1 is candidate cause, H2 adversarial.\n"
        "Tracked in hypotheses.json. " + "x" * 200
    )

    def test_pass_well_formed(self):
        path = self._write('phase_0.md', self.GOOD)
        self._seed_hypotheses(3)
        r = run_gate(path)
        self.assertEqual(r.returncode, 0, msg=r.stderr)
        self.assertIn("pass: PASS", r.stdout)

    def test_fail_no_hypotheses_json(self):
        # D-003: hypotheses.json absent -> REQUIRED gate fails (exit 1).
        path = self._write('phase_0.md', self.GOOD)
        r = run_gate(path)
        self.assertEqual(r.returncode, 1, msg=r.stdout)
        self.assertIn("hypotheses_json_seeded: MISS", r.stderr)
        self.assertIn("pass: FAIL", r.stderr)

    def test_fail_too_few_hypotheses_json_entries(self):
        # D-003: hypotheses.json with <3 entries -> REQUIRED gate fails.
        path = self._write('phase_0.md', self.GOOD)
        self._seed_hypotheses(2)
        r = run_gate(path)
        self.assertEqual(r.returncode, 1, msg=r.stdout)
        self.assertIn("hypotheses_json_seeded: MISS", r.stderr)

    def test_fail_empty(self):
        path = self._write('phase_0.md', "")
        r = run_gate(path)
        self.assertEqual(r.returncode, 1)
        self.assertIn("[REQUIRED] min_size_200B: MISS", r.stderr)
        self.assertIn("pass: FAIL", r.stderr)

    def test_fail_too_few_h_refs(self):
        path = self._write('phase_0.md',
                            "# Phase 0\nOnly one hypothesis: H1. "
                            + "framing " * 50)
        r = run_gate(path)
        self.assertEqual(r.returncode, 1)
        self.assertIn("at_least_3_H_refs: MISS", r.stderr)

    def test_fail_no_framing_keyword(self):
        # Has 3 H refs and enough bytes but no hypothesis / framing keyword.
        bulk = "X" * 250
        path = self._write('phase_0.md',
                            f"# Phase 0\nThings: H1 H2 H3.\n{bulk}\n"
                            "Nothing semantic here.")
        r = run_gate(path)
        self.assertEqual(r.returncode, 1)
        self.assertIn("framing_keyword: MISS", r.stderr)


class TestPhase05Gate(_PhaseGateBase):

    def test_pass_with_credible_verdict(self):
        path = self._write('phase_0_5.md',
                            "# Phase 0.5 RAPID screening\n"
                            "Coherence checks all pass. Verdict: CREDIBLE.\n"
                            + "x" * 100)
        r = run_gate(path)
        self.assertEqual(r.returncode, 0, msg=r.stderr)

    def test_fail_no_verdict(self):
        path = self._write('phase_0_5.md',
                            "# Phase 0.5\n" + "x" * 200)
        r = run_gate(path)
        self.assertEqual(r.returncode, 1)
        self.assertIn("rapid_verdict_present: MISS", r.stderr)


class TestPhase1Gate(_PhaseGateBase):

    def test_pass_with_observation_reference(self):
        path = self._write('phase_1.md',
                            "# Phase 1 Boundary Mapping\n"
                            "See obs_001_cli_surface.md and observations.md.\n"
                            + "x" * 200)
        r = run_gate(path)
        self.assertEqual(r.returncode, 0, msg=r.stderr)

    def test_fail_no_observation_keyword(self):
        path = self._write('phase_1.md',
                            "# Phase 1\nJust some text " + "y" * 250)
        r = run_gate(path)
        self.assertEqual(r.returncode, 1)
        self.assertIn("observation_reference: MISS", r.stderr)


class TestPhase2Gate(_PhaseGateBase):

    def test_pass_with_causal_keyword(self):
        path = self._write('phase_2.md',
                            "# Phase 2 Causal Analysis\n"
                            "Causal DAG built; H1 refuted via falsification.\n"
                            + "x" * 350)
        r = run_gate(path)
        self.assertEqual(r.returncode, 0, msg=r.stderr)

    def test_fail_no_causal_keyword(self):
        path = self._write('phase_2.md',
                            "# Phase 2\nGeneric text " + "z" * 400)
        r = run_gate(path)
        self.assertEqual(r.returncode, 1)
        self.assertIn("causal_or_falsification_keyword: MISS", r.stderr)


class TestPhase3Gate(_PhaseGateBase):

    def test_pass_with_model_keyword(self):
        path = self._write('phase_3.md',
                            "# Phase 3\nARX model fit; residual whiteness OK; "
                            "R² = 0.92; 95% CI computed.\n" + "x" * 200)
        r = run_gate(path)
        self.assertEqual(r.returncode, 0, msg=r.stderr)

    def test_recommended_miss_still_passes(self):
        # Has REQUIRED 'model' keyword but no uncertainty keyword.
        path = self._write('phase_3.md',
                            "# Phase 3\nA model was fit. " + "x" * 250)
        r = run_gate(path)
        self.assertEqual(r.returncode, 0, msg=r.stderr)
        self.assertIn("uncertainty_keyword: MISS", r.stderr)


class TestPhase4Gate(_PhaseGateBase):

    def test_pass_with_synthesis_keyword(self):
        path = self._write('phase_4.md',
                            "# Phase 4 Synthesis\nArchetype identified; "
                            "sub-models composed; emergence tested.\n"
                            + "x" * 200)
        r = run_gate(path)
        self.assertEqual(r.returncode, 0, msg=r.stderr)


class TestPhase5Gate(_PhaseGateBase):

    def test_pass_with_validation_and_summary(self):
        path = self._write('phase_5.md',
                            "# Phase 5\nValidation hierarchy passed; "
                            "summary.md written; verdict: CLOSE.\n"
                            + "x" * 200)
        r = run_gate(path)
        self.assertEqual(r.returncode, 0, msg=r.stderr)

    def test_fail_missing_summary_or_verdict(self):
        path = self._write('phase_5.md',
                            "# Phase 5\nValidation hierarchy executed. "
                            + "x" * 250)
        r = run_gate(path)
        self.assertEqual(r.returncode, 1)
        self.assertIn("summary_or_verdict_keyword: MISS", r.stderr)


class TestPsychVariants(_PhaseGateBase):

    def test_phase_2_P_distinguished_from_phase_2(self):
        """Phase 2-P uses behavioral terms (not just causal) — distinct
        from phase_2.md rule set."""
        # Content has 'behavioral' but not 'causal' — should still pass for 2-P.
        path = self._write('phase_2_P.md',
                            "# Phase 2-P\nStimulus-response patterns; "
                            "behavioral traits observed.\n" + "x" * 350)
        r = run_gate(path)
        self.assertEqual(r.returncode, 0, msg=r.stderr)

    def test_phase_4_P_motive_keyword(self):
        path = self._write('phase_4_P.md',
                            "# Phase 4-P Motive Analysis\nMICE framework; "
                            "ideology and coercion mapped; intent inferred.\n"
                            + "x" * 200)
        r = run_gate(path)
        self.assertEqual(r.returncode, 0, msg=r.stderr)

    def test_phase_4_P_fail_no_motive_keyword(self):
        path = self._write('phase_4_P.md',
                            "# Phase 4-P\nGeneric content " + "y" * 250)
        r = run_gate(path)
        self.assertEqual(r.returncode, 1)
        self.assertIn("motive_keyword: MISS", r.stderr)


class TestDispatcherUsage(_PhaseGateBase):

    def test_unknown_basename_exit_2(self):
        path = self._write('not_a_phase.md', "x" * 200)
        r = run_gate(path)
        self.assertEqual(r.returncode, 2)
        self.assertIn("does not handle phase", r.stderr)

    def test_phase_0_3_routes_elsewhere(self):
        """phase_0_3.md is served by domain_orienter.py, not phase_gate.py.
        Calling phase_gate.py on it must exit 2 with an informative message
        pointing the user at the correct script."""
        path = self._write('phase_0_3.md', "x" * 200)
        r = run_gate(path)
        self.assertEqual(r.returncode, 2)
        self.assertIn("domain_orienter.py", r.stderr)

    def test_missing_file(self):
        """File path that does not exist → all REQUIRED rules MISS → exit 1."""
        path = os.path.join(self.phase_out, 'phase_0.md')  # never created
        r = run_gate(path)
        self.assertEqual(r.returncode, 1)
        self.assertIn("min_size_200B: MISS", r.stderr)


class TestRunPhaseGateIntegration(unittest.TestCase):
    """Verify session_manager._run_phase_gate dispatches through to
    phase_gate.py correctly for a newly-added phase."""

    def setUp(self):
        self.workdir = tempfile.mkdtemp()
        self.sm = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), '..', 'src', 'scripts',
                         'session_manager.py'),
            '--base-dir', self.workdir,
        ]

    def tearDown(self):
        import shutil
        shutil.rmtree(self.workdir, ignore_errors=True)

    def _sm(self, *args, input_text=None):
        return subprocess.run(self.sm + list(args),
                              capture_output=True, text=True,
                              input=input_text)

    def test_advance_refused_on_stub_phase_0(self):
        self._sm('new', '--tier', 'STANDARD', '--domain-familiarity', 'high',
                 'integration smoke')
        self._sm('write', 'phase_outputs/phase_0.md', input_text="# stub\n")
        r = self._sm('advance', 'try-stub')
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("gate FAIL", r.stderr)

    def test_advance_passes_with_well_formed_phase_0(self):
        self._sm('new', '--tier', 'STANDARD', '--domain-familiarity', 'high',
                 'integration smoke')
        good = ("# Phase 0 — Framing\n\n"
                "Hypothesis seeds H1, H2, H3. [H_S] and [H_S_prime] tracked.\n"
                "Hypotheses captured in hypotheses.json.\n" + "x" * 200)
        self._sm('write', 'phase_outputs/phase_0.md', input_text=good)
        # D-003: Phase-0 gate now REQUIRES a >=3-entry hypotheses.json.
        self._sm('write', 'hypotheses.json', input_text=(
            '{"hypotheses": [{"id": "H1"}, {"id": "H2"}, {"id": "H3"}]}'))
        r = self._sm('advance', 'try-good')
        self.assertEqual(r.returncode, 0, msg=r.stderr)
        self.assertIn("Phase 0 → Phase 0.3", r.stdout)


class TestFenceStrippingHardening(_PhaseGateBase):
    """Keyword-class gate predicates must ignore keywords that appear ONLY
    inside fenced code/diagram blocks (e.g. a ```mermaid node label), so a stub
    deliverable cannot pass a gate by hiding the keyword in a diagram. The
    byte-size predicate still counts full content. (plan_2026-06-01_a27c8aac/S6)
    """

    def test_wellformed_deliverable_with_diagram_still_passes(self):
        # Regression: a legitimate Phase 0 deliverable whose prose satisfies the
        # gate (framing keyword + 3 H-refs + bytes) AND also carries a mermaid
        # diagram must still PASS — fence-stripping must not break it.
        content = (
            "# Phase 0 — Setup & Framing\n\n"
            "Hypothesis seeds: H1, H2, H3 with [H_S] / [H_S_prime].\n"
            "Framing per SKILL.md; H1 is candidate cause, H2 adversarial.\n"
            "Tracked in hypotheses.json.\n\n"
            "```mermaid\n"
            "stateDiagram-v2\n"
            "  [*] --> Setup\n"
            "  Setup --> Boundary\n"
            "```\n"
            + "x" * 200
        )
        path = self._write('phase_0.md', content)
        self._seed_hypotheses(3)
        r = run_gate(path)
        self.assertEqual(r.returncode, 0, msg=r.stderr)
        self.assertIn("pass: PASS", r.stdout)

    def test_keyword_only_inside_mermaid_label_fails(self):
        # The hole this closes: the Phase 2 required keyword
        # (causal/falsif/mechanism/refut/weaken) appears ONLY inside a mermaid
        # node label and NOWHERE in prose. Prose has everything else (>300 bytes
        # so the size check passes) so the failure isolates to the keyword gate.
        prose_filler = (
            "Phase 2 analysis. We examine how inputs drive outputs and how the "
            "graph of relationships explains the observed behavior. The directed "
            "edges encode strength and sign. We tested each hypothesis H1 against "
            "the data and reviewed the supporting evidence at length here. "
        ) * 3
        content = (
            "# Phase 2 — Analysis\n\n"
            + prose_filler + "\n\n"
            "```mermaid\n"
            "flowchart LR\n"
            "  A[causal mechanism] --> B[falsified node]\n"
            "```\n"
        )
        # Sanity: deliverable is well over the 300B size floor.
        self.assertGreater(len(content.encode('utf-8')), 300)
        path = self._write('phase_2.md', content)
        r = run_gate(path)
        self.assertEqual(r.returncode, 1, msg=r.stdout)
        self.assertIn("causal_or_falsification_keyword: MISS", r.stderr)


if __name__ == '__main__':
    unittest.main()
