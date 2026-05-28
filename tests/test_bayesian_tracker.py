#!/usr/bin/env python3
"""Tests for src/scripts/bayesian_tracker.py"""

import os
import sys
import tempfile
import unittest

# Allow importing from src/scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'scripts'))

from bayesian_tracker import BayesianTracker, Status


class TestBayesianTracker(unittest.TestCase):

    def setUp(self):
        self.tmpfile = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
        self.tmpfile.close()
        os.unlink(self.tmpfile.name)  # start fresh
        self.tracker = BayesianTracker(self.tmpfile.name)

    def tearDown(self):
        if os.path.exists(self.tmpfile.name):
            os.unlink(self.tmpfile.name)

    def test_add_hypothesis(self):
        hid = self.tracker.add("Test hypothesis", phase="P0", prior=0.6)
        self.assertEqual(hid, "H1")
        h = self.tracker.hypotheses[hid]
        self.assertEqual(h.statement, "Test hypothesis")
        self.assertAlmostEqual(h.prior, 0.6)
        self.assertAlmostEqual(h.posterior, 0.6)
        self.assertEqual(h.status, Status.ACTIVE.value)

    def test_add_invalid_prior(self):
        with self.assertRaises(ValueError):
            self.tracker.add("Bad prior", prior=0.0)
        with self.assertRaises(ValueError):
            self.tracker.add("Bad prior", prior=1.0)

    def test_update_hypothesis(self):
        hid = self.tracker.add("H", prior=0.5)
        new_p = self.tracker.update(hid, "Evidence", preset="strong_confirm")
        self.assertGreater(new_p, 0.5)

    def test_update_invalid_id(self):
        with self.assertRaises(KeyError):
            self.tracker.update("H999", "Evidence", preset="neutral")

    def test_remove_hypothesis(self):
        hid = self.tracker.add("To remove", prior=0.5)
        self.assertTrue(self.tracker.remove(hid))
        self.assertNotIn(hid, self.tracker.hypotheses)

    def test_remove_nonexistent(self):
        self.assertFalse(self.tracker.remove("H999"))

    def test_rename_preserves_evidence_and_posterior(self):
        """rename rewrites the statement without disturbing prior, posterior,
        or the evidence trail — intended for glossary-informed re-framing."""
        hid = self.tracker.add("Generic framing", prior=0.5)
        self.tracker.update(hid, "observation 1", preset="strong_confirm")
        self.tracker.update(hid, "observation 2", preset="moderate_confirm")
        posterior_before = self.tracker.hypotheses[hid].posterior
        evidence_count_before = len(self.tracker.hypotheses[hid].evidence)

        self.assertTrue(self.tracker.rename(hid, "Native-idiom framing"))

        h = self.tracker.hypotheses[hid]
        self.assertEqual(h.statement, "Native-idiom framing")
        self.assertEqual(h.prior, 0.5)
        self.assertAlmostEqual(h.posterior, posterior_before)
        self.assertEqual(len(h.evidence), evidence_count_before)

    def test_rename_nonexistent(self):
        """rename returns False when the ID is not found, mirrors remove()."""
        self.assertFalse(self.tracker.rename("H999", "anything"))

    def test_rename_empty_raises(self):
        """Empty or whitespace-only statements are rejected by the cap check."""
        hid = self.tracker.add("H", prior=0.5)
        with self.assertRaises(ValueError):
            self.tracker.rename(hid, "")
        with self.assertRaises(ValueError):
            self.tracker.rename(hid, "   ")

    def test_rename_persists_across_reload(self):
        """rename changes survive a save/load roundtrip."""
        hid = self.tracker.add("Old statement", prior=0.5)
        self.tracker.rename(hid, "New statement")
        tracker2 = BayesianTracker(self.tmpfile.name)
        self.assertEqual(tracker2.hypotheses[hid].statement, "New statement")

    def test_save_load_roundtrip(self):
        self.tracker.add("H1", prior=0.5)
        self.tracker.add("H2", prior=0.7)
        self.tracker.update("H1", "ev1", preset="strong_confirm")
        self.tracker.add_flag("methodology", "No baseline")
        self.tracker.add_coherence("data-task-match", "PASS")

        # Load from same file
        tracker2 = BayesianTracker(self.tmpfile.name)
        self.assertEqual(len(tracker2.hypotheses), 2)
        self.assertEqual(len(tracker2.red_flags), 1)
        self.assertEqual(len(tracker2.coherence_checks), 1)
        self.assertAlmostEqual(
            tracker2.hypotheses["H1"].posterior,
            self.tracker.hypotheses["H1"].posterior
        )

    def test_monotonic_ids(self):
        h1 = self.tracker.add("First", prior=0.5)
        h2 = self.tracker.add("Second", prior=0.5)
        self.tracker.remove(h1)
        h3 = self.tracker.add("Third", prior=0.5)
        # H3 should be H3, not H1 (IDs are monotonic, not reused)
        self.assertEqual(h1, "H1")
        self.assertEqual(h2, "H2")
        self.assertEqual(h3, "H3")

    def test_verdict_credible(self):
        """No flags -> CREDIBLE."""
        v = self.tracker.get_verdict()
        self.assertEqual(v['verdict'], 'CREDIBLE')

    def test_verdict_reject_critical_flag(self):
        """Critical flag -> REJECT."""
        self.tracker.add_flag("methodology", "Fatal flaw", severity="critical")
        v = self.tracker.get_verdict()
        self.assertEqual(v['verdict'], 'REJECT')

    def test_repeated_strong_confirm_no_crash(self):
        """50x strong_confirm should not crash (div-by-zero guard)."""
        hid = self.tracker.add("Test", prior=0.5)
        for _ in range(50):
            self.tracker.update(hid, "evidence", preset="strong_confirm")
        h = self.tracker.hypotheses[hid]
        self.assertGreater(h.posterior, 0.99)
        self.assertEqual(h.status, Status.CONFIRMED.value)


    def test_update_killed_hypothesis_raises(self):
        """Updating a REFUTED hypothesis should raise ValueError."""
        hid = self.tracker.add("Test", prior=0.5)
        self.tracker.update(hid, "Falsifying evidence", preset="falsify")
        self.assertEqual(self.tracker.hypotheses[hid].status, Status.REFUTED.value)
        with self.assertRaises(ValueError) as ctx:
            self.tracker.update(hid, "New evidence", preset="strong_confirm")
        self.assertIn("REFUTED", str(ctx.exception))

    def test_verdict_skeptical_two_flags(self):
        """2 flags -> SKEPTICAL."""
        self.tracker.add_flag("methodology", "No baseline")
        self.tracker.add_flag("results", "Suspicious pattern")
        v = self.tracker.get_verdict()
        self.assertEqual(v['verdict'], 'SKEPTICAL')

    def test_verdict_skeptical_three_flags(self):
        """3 flags in 2 categories -> SKEPTICAL."""
        self.tracker.add_flag("methodology", "Flag 1")
        self.tracker.add_flag("methodology", "Flag 2")
        self.tracker.add_flag("results", "Flag 3")
        v = self.tracker.get_verdict()
        self.assertEqual(v['verdict'], 'SKEPTICAL')

    def test_verdict_doubtful_four_flags(self):
        """4 flags -> DOUBTFUL."""
        self.tracker.add_flag("methodology", "Flag 1")
        self.tracker.add_flag("methodology", "Flag 2")
        self.tracker.add_flag("results", "Flag 3")
        self.tracker.add_flag("results", "Flag 4")
        v = self.tracker.get_verdict()
        self.assertEqual(v['verdict'], 'DOUBTFUL')

    def test_verdict_doubtful_three_categories(self):
        """3 categories with flags -> DOUBTFUL (Meta-Rule)."""
        self.tracker.add_flag("methodology", "Flag 1")
        self.tracker.add_flag("results", "Flag 2")
        self.tracker.add_flag("claims", "Flag 3")
        v = self.tracker.get_verdict()
        self.assertEqual(v['verdict'], 'DOUBTFUL')

    def test_verdict_reject_many_flags(self):
        """6 flags -> REJECT."""
        for i in range(6):
            self.tracker.add_flag("methodology", f"Flag {i+1}")
        v = self.tracker.get_verdict()
        self.assertEqual(v['verdict'], 'REJECT')

    def test_verdict_reject_four_categories(self):
        """4 categories -> REJECT."""
        self.tracker.add_flag("methodology", "Flag 1")
        self.tracker.add_flag("results", "Flag 2")
        self.tracker.add_flag("claims", "Flag 3")
        self.tracker.add_flag("documentation", "Flag 4")
        v = self.tracker.get_verdict()
        self.assertEqual(v['verdict'], 'REJECT')

    def test_verdict_credible_one_flag(self):
        """1 flag -> still CREDIBLE."""
        self.tracker.add_flag("methodology", "Minor issue")
        v = self.tracker.get_verdict()
        self.assertEqual(v['verdict'], 'CREDIBLE')


class TestLRCapEnforcement(unittest.TestCase):
    """plan_2026-05-19_8608e41f/D-003: SKILL.md Evidence Rule 1 cap enforcement
    via CLI. The cap is enforced in main() — exercising via subprocess.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.hyp_file = os.path.join(self.tmpdir, "hypotheses.json")
        self.state_file = os.path.join(self.tmpdir, "state.md")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_state(self, phase):
        with open(self.state_file, "w") as f:
            f.write(f"# Current State\n## Phase: {phase}\n## Tier: STANDARD\n")

    def _run_cli(self, *args):
        import subprocess
        script = os.path.join(
            os.path.dirname(__file__), '..', 'src', 'scripts', 'bayesian_tracker.py')
        result = subprocess.run(
            ['python3', script, '--file', self.hyp_file] + list(args),
            capture_output=True, text=True)
        return result

    def test_cap_rejects_lr_above_phase_0_cap(self):
        """LR=4.0 rejected when session phase is 0 (cap 3.0)."""
        self._write_state("0")
        self._run_cli('add', 'Test', '--prior', '0.5')
        result = self._run_cli('update', 'H1', 'evidence', '--lr', '4.0')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("exceeds", result.stderr)

    def test_override_cap_accepts_with_logged_reason(self):
        """--override-cap allows above-cap LR; writes LR-OVERRIDE to decisions.md.

        Note (plan_2026-05-28_000d7a7a/D-001): LR=4.0 from prior 0.5 produces
        posterior 0.80 which exactly crosses the Rule 5 disconfirm gate. This
        test isolates --override-cap; we also pass --override-disconfirm to
        keep the cap test independent of the Rule 5 gate.
        """
        self._write_state("0")
        self._run_cli('add', 'Test', '--prior', '0.5')
        result = self._run_cli(
            'update', 'H1', 'evidence', '--lr', '4.0',
            '--override-cap', 'experimental direct falsification',
            '--override-disconfirm', 'isolating cap-override test from Rule 5')
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        decisions = os.path.join(self.tmpdir, "decisions.md")
        self.assertTrue(os.path.exists(decisions))
        with open(decisions) as f:
            content = f.read()
        self.assertIn("LR-OVERRIDE", content)
        self.assertIn("experimental direct falsification", content)

    def test_standalone_no_session_defaults_to_lenient_cap(self):
        """Without state.md, default cap is 10.0 — LR=5.0 should pass.

        Prior set to 0.2 so posterior stays below the Rule 5 gate (0.2*5 /
        (0.2*5 + 0.8) = 0.556 < 0.80); isolates cap behavior from gate.
        """
        # No _write_state call
        self._run_cli('add', 'Test', '--prior', '0.2')
        result = self._run_cli('update', 'H1', 'evidence', '--lr', '5.0')
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        # LR=11.0 exceeds default 10.0 → rejected by cap (cap is checked first)
        result = self._run_cli('update', 'H1', 'evidence', '--lr', '11.0')
        self.assertNotEqual(result.returncode, 0)

    def test_phase_2_allows_lr_up_to_10(self):
        """Phase 2 cap is 10.0 — LR=8.0 should pass under the cap.

        Note (plan_2026-05-28_000d7a7a/D-001): LR=8.0 from prior 0.5 produces
        posterior 0.889 which crosses the Rule 5 disconfirm-before-confirm gate.
        This test isolates the LR cap, so we opt out of the disconfirm gate via
        --override-disconfirm with a documented reason.
        """
        self._write_state("2")
        self._run_cli('add', 'Test', '--prior', '0.5')
        result = self._run_cli(
            'update', 'H1', 'evidence', '--lr', '8.0',
            '--override-disconfirm', 'isolating LR cap test from Rule 5 gate')
        self.assertEqual(result.returncode, 0, msg=result.stderr)


class TestDisconfirmGate(unittest.TestCase):
    """plan_2026-05-28_000d7a7a/D-001: SKILL.md Evidence Rule 5
    (disconfirm-before-confirm) enforcement at the CLI layer.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.hyp_file = os.path.join(self.tmpdir, "hypotheses.json")
        self.state_file = os.path.join(self.tmpdir, "state.md")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_state(self, phase):
        with open(self.state_file, "w") as f:
            f.write(f"# Current State\n## Phase: {phase}\n## Tier: STANDARD\n")

    def _run_cli(self, *args):
        import subprocess
        script = os.path.join(
            os.path.dirname(__file__), '..', 'src', 'scripts',
            'bayesian_tracker.py')
        return subprocess.run(
            ['python3', script, '--file', self.hyp_file] + list(args),
            capture_output=True, text=True)

    def test_disconfirm_gate_refuses_without_disconfirm_history(self):
        """LR=8.0 from prior 0.5 → posterior 0.889 crosses 0.80 with no prior
        disconfirm in trail → gate fires."""
        self._write_state("2")
        self._run_cli('add', 'Test', '--prior', '0.5')
        result = self._run_cli('update', 'H1', 'first confirm', '--lr', '8.0')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("posterior would cross", result.stderr)
        self.assertIn("Rule 5", result.stderr)

    def test_disconfirm_gate_passes_with_prior_disconfirm(self):
        """After applying any LR<1.0 evidence, the gate no longer fires."""
        self._write_state("2")
        self._run_cli('add', 'Test', '--prior', '0.5')
        r1 = self._run_cli('update', 'H1', 'early disconfirm', '--lr', '0.5')
        self.assertEqual(r1.returncode, 0, msg=r1.stderr)
        r2 = self._run_cli('update', 'H1', 'big confirm', '--lr', '8.0')
        self.assertEqual(r2.returncode, 0, msg=r2.stderr)
        # Posterior trajectory: 0.5 -> 0.333 -> 0.800
        self.assertIn("posterior=0.800", r2.stdout)

    def test_disconfirm_gate_bypassed_with_override(self):
        """--override-disconfirm allows the cross and writes a
        DISCONFIRM-OVERRIDE block to session decisions.md."""
        self._write_state("2")
        self._run_cli('add', 'Test', '--prior', '0.5')
        result = self._run_cli(
            'update', 'H1', 'forced confirm', '--lr', '8.0',
            '--override-disconfirm', 'analyst-approved direct falsification')
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        decisions = os.path.join(self.tmpdir, "decisions.md")
        self.assertTrue(os.path.exists(decisions))
        with open(decisions) as f:
            content = f.read()
        self.assertIn("DISCONFIRM-OVERRIDE", content)
        self.assertIn("analyst-approved direct falsification", content)

    def test_disconfirm_gate_inactive_below_threshold(self):
        """LR=1.5 from prior 0.5 → posterior 0.6 stays below 0.80 → gate does
        not fire even without prior disconfirm."""
        self._write_state("2")
        self._run_cli('add', 'Test', '--prior', '0.5')
        result = self._run_cli('update', 'H1', 'small confirm', '--lr', '1.5')
        self.assertEqual(result.returncode, 0, msg=result.stderr)

    def test_disconfirm_gate_inactive_for_disconfirming_updates(self):
        """LR<1.0 cannot cross the upper threshold by definition; gate must
        never fire on a disconfirming update."""
        self._write_state("2")
        self._run_cli('add', 'Test', '--prior', '0.5')
        result = self._run_cli('update', 'H1', 'disconfirm', '--lr', '0.5')
        self.assertEqual(result.returncode, 0, msg=result.stderr)

    def test_disconfirm_gate_inactive_when_already_above_threshold(self):
        """Once posterior is already past 0.80, subsequent confirms are not
        re-gated (the rule guards the first crossing, not all confirms)."""
        self._write_state("2")
        self._run_cli('add', 'Test', '--prior', '0.85')  # already past gate
        result = self._run_cli('update', 'H1', 'second confirm', '--lr', '2.0')
        self.assertEqual(result.returncode, 0, msg=result.stderr)


class TestValidatePriors(unittest.TestCase):
    """plan_2026-05-28_000d7a7a/D-002: SKILL.md Evidence Rule 6 cross-hypothesis
    prior sum-to-1 check via the validate-priors subcommand.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.hyp_file = os.path.join(self.tmpdir, "hypotheses.json")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run_cli(self, *args):
        import subprocess
        script = os.path.join(
            os.path.dirname(__file__), '..', 'src', 'scripts',
            'bayesian_tracker.py')
        return subprocess.run(
            ['python3', script, '--file', self.hyp_file] + list(args),
            capture_output=True, text=True)

    def test_validate_priors_pass(self):
        """Priors 0.4/0.3/0.3 sum to 1.0 → exit 0 with PASS message."""
        self._run_cli('add', 'A', '--prior', '0.4')
        self._run_cli('add', 'B', '--prior', '0.3')
        self._run_cli('add', 'C', '--prior', '0.3')
        result = self._run_cli('validate-priors',
                               '--exclusive-set', 'H1,H2,H3')
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("VALIDATE-PRIORS: PASS", result.stdout)

    def test_validate_priors_fail_sum_too_high(self):
        """Priors 0.5/0.3/0.3 sum to 1.1 → exit 1 with FAIL message."""
        self._run_cli('add', 'A', '--prior', '0.5')
        self._run_cli('add', 'B', '--prior', '0.3')
        self._run_cli('add', 'C', '--prior', '0.3')
        result = self._run_cli('validate-priors',
                               '--exclusive-set', 'H1,H2,H3')
        self.assertEqual(result.returncode, 1)
        self.assertIn("VALIDATE-PRIORS: FAIL", result.stderr)

    def test_validate_priors_unknown_hid_exit_2(self):
        """Unknown HID → exit 2 with clear error."""
        self._run_cli('add', 'A', '--prior', '0.5')
        result = self._run_cli('validate-priors',
                               '--exclusive-set', 'H1,H99')
        self.assertEqual(result.returncode, 2)
        self.assertIn("H99", result.stderr)

    def test_validate_priors_single_id_exit_2(self):
        """Degenerate single-ID set → exit 2 with usage error."""
        self._run_cli('add', 'A', '--prior', '0.5')
        result = self._run_cli('validate-priors', '--exclusive-set', 'H1')
        self.assertEqual(result.returncode, 2)
        self.assertIn("at least 2", result.stderr)

    def test_validate_priors_custom_tolerance(self):
        """Tighter tolerance flips a near-pass into a fail."""
        self._run_cli('add', 'A', '--prior', '0.5')
        self._run_cli('add', 'B', '--prior', '0.495')  # sum 0.995, dev 0.005
        r_default = self._run_cli('validate-priors',
                                  '--exclusive-set', 'H1,H2')
        self.assertEqual(r_default.returncode, 0)  # default tol 0.01
        r_strict = self._run_cli('validate-priors',
                                 '--exclusive-set', 'H1,H2',
                                 '--tolerance', '0.001')
        self.assertEqual(r_strict.returncode, 1)


if __name__ == '__main__':
    unittest.main()
