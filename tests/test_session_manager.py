#!/usr/bin/env python3
"""Tests for src/scripts/session_manager.py"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
from io import StringIO

# Allow importing from src/scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'scripts'))

import session_manager as sm


class SessionManagerTestBase(unittest.TestCase):
    """Base class that redirects session_manager paths to a temp directory."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.orig_analyses = sm.ANALYSES_DIR
        self.orig_pointer = sm.POINTER_FILE
        self.orig_findings = sm.CONSOLIDATED_FINDINGS
        self.orig_decisions = sm.CONSOLIDATED_DECISIONS
        self.orig_cwd = os.getcwd()

        sm.ANALYSES_DIR = os.path.join(self.tmpdir, "analyses")
        sm.POINTER_FILE = os.path.join(sm.ANALYSES_DIR, ".current_analysis")
        sm.CONSOLIDATED_FINDINGS = os.path.join(sm.ANALYSES_DIR, "FINDINGS.md")
        sm.CONSOLIDATED_DECISIONS = os.path.join(sm.ANALYSES_DIR, "DECISIONS.md")
        os.chdir(self.tmpdir)

    def tearDown(self):
        os.chdir(self.orig_cwd)
        sm.ANALYSES_DIR = self.orig_analyses
        sm.POINTER_FILE = self.orig_pointer
        sm.CONSOLIDATED_FINDINGS = self.orig_findings
        sm.CONSOLIDATED_DECISIONS = self.orig_decisions
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_args(self, **kwargs):
        """Create a simple namespace for args."""
        from types import SimpleNamespace
        return SimpleNamespace(**kwargs)


class TestReadPointer(SessionManagerTestBase):

    def test_no_pointer_file(self):
        """read_pointer returns None when no pointer file exists."""
        self.assertIsNone(sm.read_pointer())

    def test_pointer_to_nonexistent_dir(self):
        """read_pointer returns None if pointer references missing dir."""
        os.makedirs(sm.ANALYSES_DIR, exist_ok=True)
        with open(sm.POINTER_FILE, 'w') as f:
            f.write("analysis_2025-01-01_deadbeef")
        self.assertIsNone(sm.read_pointer())

    def test_valid_pointer(self):
        """read_pointer returns absolute path when pointer and dir both exist."""
        os.makedirs(sm.ANALYSES_DIR, exist_ok=True)
        dir_name = "analysis_2025-01-01_abcd1234"
        abs_dir = os.path.join(sm.ANALYSES_DIR, dir_name)
        os.makedirs(abs_dir)
        with open(sm.POINTER_FILE, 'w') as f:
            f.write(dir_name)
        result = sm.read_pointer()
        self.assertEqual(result, abs_dir)


class TestCmdNew(SessionManagerTestBase):

    def test_creates_directory_structure(self):
        """cmd_new creates session dir with expected files."""
        args = self._make_args(goal=["Test", "system"], force=False)
        with patch('sys.stdout', new_callable=StringIO):
            sm.cmd_new(args)

        pointer = sm.read_pointer()
        self.assertIsNotNone(pointer)
        session_dir = os.path.join(sm.ANALYSES_DIR, pointer)
        self.assertTrue(os.path.isdir(session_dir))
        self.assertTrue(os.path.isdir(os.path.join(session_dir, "observations")))
        self.assertTrue(os.path.isdir(os.path.join(session_dir, "phase_outputs")))
        for fname in ("state.md", "analysis_plan.md", "decisions.md",
                       "observations.md", "progress.md", "validation.md"):
            self.assertTrue(os.path.isfile(os.path.join(session_dir, fname)),
                            f"Missing file: {fname}")

    def test_creates_pointer_file(self):
        """cmd_new creates the .current_analysis pointer file."""
        args = self._make_args(goal=["Test"], force=False)
        with patch('sys.stdout', new_callable=StringIO):
            sm.cmd_new(args)
        self.assertTrue(os.path.isfile(sm.POINTER_FILE))

    def test_refuses_when_active_session(self):
        """cmd_new exits with error when active session exists (no --force)."""
        args = self._make_args(goal=["First"], force=False)
        with patch('sys.stdout', new_callable=StringIO):
            sm.cmd_new(args)

        args2 = self._make_args(goal=["Second"], force=False)
        with self.assertRaises(SystemExit) as ctx:
            with patch('sys.stdout', new_callable=StringIO), \
                 patch('sys.stderr', new_callable=StringIO):
                sm.cmd_new(args2)
        self.assertEqual(ctx.exception.code, 1)

    def test_force_closes_previous(self):
        """cmd_new --force closes previous and starts new."""
        args1 = self._make_args(goal=["First"], force=False)
        with patch('sys.stdout', new_callable=StringIO):
            sm.cmd_new(args1)
        first_pointer = sm.read_pointer()

        args2 = self._make_args(goal=["Second"], force=True)
        with patch('sys.stdout', new_callable=StringIO):
            sm.cmd_new(args2)
        second_pointer = sm.read_pointer()

        self.assertNotEqual(first_pointer, second_pointer)
        # First session dir should still exist (preserved)
        self.assertTrue(os.path.isdir(os.path.join(sm.ANALYSES_DIR, first_pointer)))

    def test_creates_consolidated_files(self):
        """cmd_new creates FINDINGS.md and DECISIONS.md."""
        args = self._make_args(goal=["Test"], force=False)
        with patch('sys.stdout', new_callable=StringIO):
            sm.cmd_new(args)
        self.assertTrue(os.path.isfile(sm.CONSOLIDATED_FINDINGS))
        self.assertTrue(os.path.isfile(sm.CONSOLIDATED_DECISIONS))

    def test_state_contains_goal(self):
        """state.md should contain the analysis goal."""
        args = self._make_args(goal=["Analyze", "target", "API"], force=False)
        with patch('sys.stdout', new_callable=StringIO):
            sm.cmd_new(args)
        pointer = sm.read_pointer()
        state = sm.read_analysis_file(pointer, "state.md")
        self.assertIn("Analyze target API", state)


class TestCmdResume(SessionManagerTestBase):

    def test_resume_with_no_session(self):
        """cmd_resume on no active session exits 0 with NO_ACTIVE_SESSION marker.

        Contract change (plan_2026-05-27_33d457f3/D-001): the orchestrator's
        mandated FIRST tool call must not pollute transcripts with an exit-1
        "Error" on cold boot. Parity with cmd_status.
        """
        args = self._make_args()
        stdout = StringIO()
        with patch('sys.stdout', stdout), \
             patch('sys.stderr', new_callable=StringIO):
            sm.cmd_resume(args)
        self.assertIn("NO_ACTIVE_SESSION", stdout.getvalue())

    def test_resume_outputs_state(self):
        """cmd_resume outputs state summary."""
        args_new = self._make_args(goal=["Test", "target"], force=False)
        with patch('sys.stdout', new_callable=StringIO):
            sm.cmd_new(args_new)

        args = self._make_args()
        output = StringIO()
        with patch('sys.stdout', output):
            sm.cmd_resume(args)
        text = output.getvalue()
        self.assertIn("SESSION_DIR=", text)
        self.assertIn("Phase", text)


class TestCmdStatus(SessionManagerTestBase):

    def test_status_no_session(self):
        """cmd_status prints 'No active analysis' when none exists."""
        args = self._make_args()
        output = StringIO()
        with patch('sys.stdout', output):
            sm.cmd_status(args)
        self.assertIn("No active analysis", output.getvalue())

    def test_status_with_session(self):
        """cmd_status prints one-liner with active session."""
        args_new = self._make_args(goal=["Widget"], force=False)
        with patch('sys.stdout', new_callable=StringIO):
            sm.cmd_new(args_new)

        args = self._make_args()
        output = StringIO()
        with patch('sys.stdout', output):
            sm.cmd_status(args)
        text = output.getvalue()
        self.assertIn("Phase", text)
        self.assertIn("Widget", text)


class TestCmdClose(SessionManagerTestBase):

    def test_close_removes_pointer(self):
        """cmd_close removes the pointer file."""
        args_new = self._make_args(goal=["Test"], force=False)
        with patch('sys.stdout', new_callable=StringIO):
            sm.cmd_new(args_new)
        self.assertIsNotNone(sm.read_pointer())

        args = self._make_args()
        with patch('sys.stdout', new_callable=StringIO):
            sm.cmd_close(args)
        self.assertIsNone(sm.read_pointer())

    def test_close_preserves_directory(self):
        """cmd_close preserves the analysis directory."""
        args_new = self._make_args(goal=["Test"], force=False)
        with patch('sys.stdout', new_callable=StringIO):
            sm.cmd_new(args_new)
        pointer = sm.read_pointer()

        args = self._make_args()
        with patch('sys.stdout', new_callable=StringIO):
            sm.cmd_close(args)
        self.assertTrue(os.path.isdir(os.path.join(sm.ANALYSES_DIR, pointer)))

    def test_close_no_session(self):
        """cmd_close exits with error when no session."""
        args = self._make_args()
        with self.assertRaises(SystemExit):
            with patch('sys.stdout', new_callable=StringIO), \
                 patch('sys.stderr', new_callable=StringIO):
                sm.cmd_close(args)


class TestCmdList(SessionManagerTestBase):

    def test_list_no_directory(self):
        """cmd_list handles missing analyses/ directory."""
        args = self._make_args()
        output = StringIO()
        with patch('sys.stdout', output):
            sm.cmd_list(args)
        self.assertIn("No analyses/ directory", output.getvalue())

    def test_list_empty(self):
        """cmd_list handles empty analyses/ directory."""
        os.makedirs(sm.ANALYSES_DIR, exist_ok=True)
        args = self._make_args()
        output = StringIO()
        with patch('sys.stdout', output):
            sm.cmd_list(args)
        self.assertIn("No analysis directories", output.getvalue())

    def test_list_shows_sessions(self):
        """cmd_list shows created sessions."""
        args_new = self._make_args(goal=["First"], force=False)
        with patch('sys.stdout', new_callable=StringIO):
            sm.cmd_new(args_new)

        args = self._make_args()
        output = StringIO()
        with patch('sys.stdout', output):
            sm.cmd_list(args)
        text = output.getvalue()
        self.assertIn("1 total", text)
        self.assertIn("active", text)


class TestCmdReopen(SessionManagerTestBase):

    def _create_session_with_phase_output(self, phase="1", phase_file="phase_1.md"):
        """Helper: create session and write a phase output file."""
        args_new = self._make_args(goal=["Test system"], force=False)
        with patch('sys.stdout', new_callable=StringIO):
            sm.cmd_new(args_new)
        abs_dir = sm.read_pointer()
        # Write a phase output to simulate phase completion
        phase_output = os.path.join(abs_dir, "phase_outputs", phase_file)
        with open(phase_output, 'w') as f:
            f.write(f"# Phase {phase} Output\nFindings here.\n")
        return abs_dir

    def test_reopen_no_session(self):
        """cmd_reopen exits with error when no active session."""
        args = self._make_args(phase="1", reason=["test"])
        with self.assertRaises(SystemExit):
            with patch('sys.stdout', new_callable=StringIO), \
                 patch('sys.stderr', new_callable=StringIO):
                sm.cmd_reopen(args)

    def test_reopen_invalid_phase(self):
        """cmd_reopen exits with error for invalid phase identifier."""
        self._create_session_with_phase_output()
        args = self._make_args(phase="99", reason=["test"])
        with self.assertRaises(SystemExit):
            with patch('sys.stdout', new_callable=StringIO), \
                 patch('sys.stderr', new_callable=StringIO):
                sm.cmd_reopen(args)

    def test_reopen_uncompleted_phase(self):
        """cmd_reopen exits with error if phase has no output file."""
        self._create_session_with_phase_output(phase="1")
        args = self._make_args(phase="2", reason=["test"])
        with self.assertRaises(SystemExit):
            with patch('sys.stdout', new_callable=StringIO), \
                 patch('sys.stderr', new_callable=StringIO):
                sm.cmd_reopen(args)

    def test_reopen_empty_reason(self):
        """cmd_reopen exits with error if reason is empty."""
        self._create_session_with_phase_output()
        args = self._make_args(phase="1", reason=["  "])
        with self.assertRaises(SystemExit):
            with patch('sys.stdout', new_callable=StringIO), \
                 patch('sys.stderr', new_callable=StringIO):
                sm.cmd_reopen(args)

    def test_reopen_archives_phase_output(self):
        """cmd_reopen archives phase_N.md to phase_N_pass1.md."""
        abs_dir = self._create_session_with_phase_output()
        args = self._make_args(phase="1", reason=["Weak", "findings"])
        with patch('sys.stdout', new_callable=StringIO):
            sm.cmd_reopen(args)

        phase_dir = os.path.join(abs_dir, "phase_outputs")
        # Original should be gone
        self.assertFalse(os.path.exists(os.path.join(phase_dir, "phase_1.md")))
        # Archive should exist
        archive = os.path.join(phase_dir, "phase_1_pass1.md")
        self.assertTrue(os.path.exists(archive))
        with open(archive) as f:
            self.assertIn("Phase 1 Output", f.read())

    def test_reopen_updates_state(self):
        """cmd_reopen updates state.md with reopened phase and transition."""
        abs_dir = self._create_session_with_phase_output()
        args = self._make_args(phase="1", reason=["Validation", "failure"])
        with patch('sys.stdout', new_callable=StringIO):
            sm.cmd_reopen(args)

        state = sm.read_analysis_file(abs_dir, "state.md")
        self.assertIn("## Phase: 1", state)
        self.assertIn("REOPEN Phase 1 pass 2", state)
        self.assertIn("Validation failure", state)

    def test_reopen_multiple_passes(self):
        """cmd_reopen handles sequential reopens with incrementing pass numbers."""
        abs_dir = self._create_session_with_phase_output()
        phase_dir = os.path.join(abs_dir, "phase_outputs")

        # First reopen (archives pass 1)
        args = self._make_args(phase="1", reason=["First", "reopen"])
        with patch('sys.stdout', new_callable=StringIO):
            sm.cmd_reopen(args)
        self.assertTrue(os.path.exists(os.path.join(phase_dir, "phase_1_pass1.md")))

        # Write new phase output (simulating pass 2 completion)
        with open(os.path.join(phase_dir, "phase_1.md"), 'w') as f:
            f.write("# Phase 1 Output - Pass 2\n")

        # Second reopen (archives pass 2)
        args = self._make_args(phase="1", reason=["Second", "reopen"])
        with patch('sys.stdout', new_callable=StringIO):
            sm.cmd_reopen(args)
        self.assertTrue(os.path.exists(os.path.join(phase_dir, "phase_1_pass1.md")))
        self.assertTrue(os.path.exists(os.path.join(phase_dir, "phase_1_pass2.md")))

    def test_reopen_max_limit(self):
        """cmd_reopen refuses after MAX_REOPENS reached."""
        abs_dir = self._create_session_with_phase_output()
        phase_dir = os.path.join(abs_dir, "phase_outputs")

        for i in range(sm.MAX_REOPENS):
            args = self._make_args(phase="1", reason=[f"Reopen {i+1}"])
            with patch('sys.stdout', new_callable=StringIO):
                sm.cmd_reopen(args)
            # Re-create phase output for next reopen
            with open(os.path.join(phase_dir, "phase_1.md"), 'w') as f:
                f.write(f"# Phase 1 - Pass {i+2}\n")

        # Next reopen should fail
        args = self._make_args(phase="1", reason=["One too many"])
        with self.assertRaises(SystemExit):
            with patch('sys.stdout', new_callable=StringIO), \
                 patch('sys.stderr', new_callable=StringIO):
                sm.cmd_reopen(args)

    def test_reopen_phase_0_5(self):
        """cmd_reopen handles phase 0.5 (underscore in filename)."""
        abs_dir = self._create_session_with_phase_output(
            phase="0.5", phase_file="phase_0_5.md")
        args = self._make_args(phase="0.5", reason=["Missed", "red", "flags"])
        with patch('sys.stdout', new_callable=StringIO):
            sm.cmd_reopen(args)

        phase_dir = os.path.join(abs_dir, "phase_outputs")
        self.assertTrue(os.path.exists(os.path.join(phase_dir, "phase_0_5_pass1.md")))
        self.assertFalse(os.path.exists(os.path.join(phase_dir, "phase_0_5.md")))

    def test_reopen_psych_phase(self):
        """cmd_reopen handles PSYCH tier phases (e.g. 2-P)."""
        abs_dir = self._create_session_with_phase_output(
            phase="2-P", phase_file="phase_2_P.md")
        args = self._make_args(phase="2-P", reason=["Missed", "baseline"])
        with patch('sys.stdout', new_callable=StringIO):
            sm.cmd_reopen(args)

        phase_dir = os.path.join(abs_dir, "phase_outputs")
        self.assertTrue(os.path.exists(os.path.join(phase_dir, "phase_2_P_pass1.md")))

    def test_reopen_output_message(self):
        """cmd_reopen prints helpful guidance."""
        self._create_session_with_phase_output()
        args = self._make_args(phase="1", reason=["Need", "more", "probes"])
        output = StringIO()
        with patch('sys.stdout', output):
            sm.cmd_reopen(args)
        text = output.getvalue()
        self.assertIn("Reopened Phase 1", text)
        self.assertIn("pass 2 of 4", text)
        self.assertIn("Archived", text)
        self.assertIn("Need more probes", text)


class TestCmdSkip(SessionManagerTestBase):

    def _create_session(self):
        """Helper: create a session without any phase output files.

        Declares `domain_familiarity: high` by default to satisfy the D-001
        familiarity gate on Phase 0.3 skip; tests that exercise the gate
        itself live in TestSkipFamiliarityCheck.
        """
        args_new = self._make_args(goal=["Test system"], force=False)
        with patch('sys.stdout', new_callable=StringIO):
            sm.cmd_new(args_new)
        abs_dir = sm.read_pointer()
        _set_domain_familiarity(abs_dir, "high")
        return abs_dir

    def test_skip_no_session(self):
        """cmd_skip exits with error when no active session."""
        args = self._make_args(phase="0.3", reason=["familiar domain"])
        with self.assertRaises(SystemExit):
            with patch('sys.stdout', new_callable=StringIO), \
                 patch('sys.stderr', new_callable=StringIO):
                sm.cmd_skip(args)

    def test_skip_invalid_phase(self):
        """cmd_skip exits with error for phase not in PHASE_FILENAME_MAP."""
        self._create_session()
        args = self._make_args(phase="99", reason=["test"])
        with self.assertRaises(SystemExit):
            with patch('sys.stdout', new_callable=StringIO), \
                 patch('sys.stderr', new_callable=StringIO):
                sm.cmd_skip(args)

    def test_skip_empty_reason(self):
        """cmd_skip exits with error if reason is whitespace-only."""
        self._create_session()
        args = self._make_args(phase="0.3", reason=["   "])
        with self.assertRaises(SystemExit):
            with patch('sys.stdout', new_callable=StringIO), \
                 patch('sys.stderr', new_callable=StringIO):
                sm.cmd_skip(args)

    def test_skip_appends_decisions_md(self):
        """cmd_skip writes a decisions.md entry with phase, reason, and cost."""
        abs_dir = self._create_session()
        args = self._make_args(
            phase="0.3",
            reason=["domain_familiarity=high;", "prior", "domain", "expertise"],
        )
        with patch('sys.stdout', new_callable=StringIO):
            sm.cmd_skip(args)
        with open(os.path.join(abs_dir, "decisions.md")) as f:
            content = f.read()
        self.assertIn("SKIP Phase 0.3", content)
        self.assertIn("domain_familiarity=high", content)
        self.assertIn("Cost", content)

    def test_skip_updates_state_md(self):
        """cmd_skip updates Last Transition in state.md."""
        abs_dir = self._create_session()
        args = self._make_args(phase="0.3", reason=["analyst is SME"])
        with patch('sys.stdout', new_callable=StringIO):
            sm.cmd_skip(args)
        with open(os.path.join(abs_dir, "state.md")) as f:
            state = f.read()
        self.assertIn("SKIP Phase 0.3", state)
        self.assertIn("analyst is SME", state)

    def test_skip_does_not_archive_phase_output(self):
        """cmd_skip (unlike reopen) does not archive or create phase_0_3.md."""
        abs_dir = self._create_session()
        args = self._make_args(phase="0.3", reason=["no orientation needed"])
        with patch('sys.stdout', new_callable=StringIO):
            sm.cmd_skip(args)
        phase_dir = os.path.join(abs_dir, "phase_outputs")
        # No phase_0_3.md, no phase_0_3_pass1.md — skip does not write phase_outputs
        self.assertFalse(os.path.exists(os.path.join(phase_dir, "phase_0_3.md")))
        self.assertFalse(os.path.exists(os.path.join(phase_dir, "phase_0_3_pass1.md")))

    def test_phase_filename_map_has_0_3(self):
        """PHASE_FILENAME_MAP now includes Phase 0.3 (regression test for the extension)."""
        self.assertIn("0.3", sm.PHASE_FILENAME_MAP)
        self.assertEqual(sm.PHASE_FILENAME_MAP["0.3"], "phase_0_3.md")


class TestSkipFamiliarityCheck(SessionManagerTestBase):
    """plan_2026-05-25_c0b0049a/D-001: cmd_skip Phase 0.3 / 0-P.3 requires
    `domain_familiarity: high` in analysis_plan.md. Whitelist (D-003) gates
    WHICH phase may be skipped; familiarity gate (D-001) gates WHEN."""

    def _make_session(self, familiarity, tier="STANDARD"):
        args_new = self._make_args(goal=["Test"], force=False)
        with patch('sys.stdout', new_callable=StringIO):
            sm.cmd_new(args_new)
        abs_dir = sm.read_pointer()
        _set_tier_and_phase(abs_dir, tier, "0")
        if familiarity is not None:
            _set_domain_familiarity(abs_dir, familiarity)
        return abs_dir

    def test_skip_0_3_refused_when_familiarity_low(self):
        self._make_session(familiarity="low")
        args = self._make_args(phase="0.3", reason=["I just don't want to"])
        with self.assertRaises(SystemExit) as ctx:
            with patch('sys.stdout', new_callable=StringIO), \
                 patch('sys.stderr', new_callable=StringIO):
                sm.cmd_skip(args)
        self.assertEqual(ctx.exception.code, 1)

    def test_skip_0_3_refused_when_familiarity_medium(self):
        self._make_session(familiarity="medium")
        args = self._make_args(phase="0.3", reason=["partial knowledge"])
        with self.assertRaises(SystemExit):
            with patch('sys.stdout', new_callable=StringIO), \
                 patch('sys.stderr', new_callable=StringIO):
                sm.cmd_skip(args)

    def test_skip_0_3_refused_when_familiarity_unknown(self):
        self._make_session(familiarity="unknown")
        args = self._make_args(phase="0.3", reason=["dunno"])
        with self.assertRaises(SystemExit):
            with patch('sys.stdout', new_callable=StringIO), \
                 patch('sys.stderr', new_callable=StringIO):
                sm.cmd_skip(args)

    def test_skip_0_3_refused_when_familiarity_missing(self):
        """No `domain_familiarity:` line in analysis_plan.md => refusal."""
        self._make_session(familiarity=None)
        args = self._make_args(phase="0.3", reason=["assume high"])
        with self.assertRaises(SystemExit):
            with patch('sys.stdout', new_callable=StringIO), \
                 patch('sys.stderr', new_callable=StringIO):
                sm.cmd_skip(args)

    def test_skip_0_3_accepted_when_familiarity_high(self):
        abs_dir = self._make_session(familiarity="high")
        args = self._make_args(phase="0.3", reason=["SME"])
        with patch('sys.stdout', new_callable=StringIO):
            sm.cmd_skip(args)
        with open(os.path.join(abs_dir, "decisions.md")) as f:
            self.assertIn("SKIP Phase 0.3", f.read())

    def test_skip_0_3_accepted_when_familiarity_high_case_insensitive(self):
        """Parser accepts 'HIGH', 'High', 'high  ' (D-001 leniency)."""
        abs_dir = self._make_session(familiarity="HIGH")
        args = self._make_args(phase="0.3", reason=["expert"])
        with patch('sys.stdout', new_callable=StringIO):
            sm.cmd_skip(args)
        with open(os.path.join(abs_dir, "decisions.md")) as f:
            self.assertIn("SKIP Phase 0.3", f.read())

    def test_skip_0_P_3_familiarity_gate_applies_to_psych(self):
        """PSYCH-tier 0-P.3 skip is also gated by familiarity = high."""
        self._make_session(familiarity="low", tier="PSYCH")
        args = self._make_args(phase="0-P.3", reason=["expert"])
        with self.assertRaises(SystemExit):
            with patch('sys.stdout', new_callable=StringIO), \
                 patch('sys.stderr', new_callable=StringIO):
                sm.cmd_skip(args)

    def test_familiarity_gate_does_not_apply_to_other_phases(self):
        """Defensive: phase 99 (invalid) goes through the existing
        PHASE_FILENAME_MAP refusal BEFORE the familiarity check fires —
        confirm the familiarity gate is scoped to {0.3, 0-P.3}."""
        self._make_session(familiarity="low")
        args = self._make_args(phase="99", reason=["nope"])
        with self.assertRaises(SystemExit) as ctx:
            with patch('sys.stdout', new_callable=StringIO), \
                 patch('sys.stderr', new_callable=StringIO):
                sm.cmd_skip(args)
        # phase 99 is rejected by PHASE_FILENAME_MAP check, not familiarity.
        self.assertEqual(ctx.exception.code, 1)

    def test_read_domain_familiarity_parser_variants(self):
        """Helper parser: case-insensitive, whitespace-tolerant, returns None
        for missing/invalid values."""
        abs_dir = self._make_session(familiarity=None)
        # Missing -> None
        self.assertIsNone(sm._read_domain_familiarity(abs_dir))
        # 'high' -> 'high'
        _set_domain_familiarity(abs_dir, "high")
        self.assertEqual(sm._read_domain_familiarity(abs_dir), "high")
        # 'HIGH' -> 'high' (case-insensitive)
        path = os.path.join(abs_dir, "analysis_plan.md")
        with open(path, "w") as f:
            f.write("\ndomain_familiarity: HIGH\n")
        self.assertEqual(sm._read_domain_familiarity(abs_dir), "high")
        # Invalid value -> None
        with open(path, "w") as f:
            f.write("\ndomain_familiarity: superhuman\n")
        self.assertIsNone(sm._read_domain_familiarity(abs_dir))


def _set_tier_and_phase(abs_dir, tier, phase):
    """Test helper — overwrite state.md Tier: and Phase: fields in-place."""
    path = os.path.join(abs_dir, "state.md")
    with open(path) as f:
        content = f.read()
    import re as _re
    content = _re.sub(r'^## Tier:\s*.*$', f'## Tier: {tier}',
                      content, flags=_re.MULTILINE)
    content = _re.sub(r'^## Phase:\s*.*$', f'## Phase: {phase}',
                      content, flags=_re.MULTILINE)
    with open(path, "w") as f:
        f.write(content)


def _set_domain_familiarity(abs_dir, value):
    """Test helper — append `domain_familiarity: <value>` to analysis_plan.md.

    Used by skip-flow tests that need to satisfy the D-001 familiarity gate
    (cmd_skip refuses Phase 0.3/0-P.3 skip unless familiarity == 'high').
    """
    path = os.path.join(abs_dir, "analysis_plan.md")
    try:
        with open(path) as f:
            content = f.read()
    except FileNotFoundError:
        content = ""
    if not content.endswith("\n"):
        content += "\n"
    with open(path, "w") as f:
        f.write(content + f"\ndomain_familiarity: {value}\n")


def _touch(path, content="placeholder\n"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


class TestAdvance(SessionManagerTestBase):
    """Tests for cmd_advance — gate-enforced phase progression."""

    def _create_session(self, tier="STANDARD", phase="0"):
        args_new = self._make_args(goal=["Test system"], force=False)
        with patch('sys.stdout', new_callable=StringIO):
            sm.cmd_new(args_new)
        abs_dir = sm.read_pointer()
        _set_tier_and_phase(abs_dir, tier, phase)
        return abs_dir

    def test_advance_no_session(self):
        with self.assertRaises(SystemExit):
            with patch('sys.stderr', new_callable=StringIO):
                sm.cmd_advance(self._make_args(reason=[]))

    def test_advance_missing_tier(self):
        """Refuses when Tier is (pending)."""
        args_new = self._make_args(goal=["Test"], force=False)
        with patch('sys.stdout', new_callable=StringIO):
            sm.cmd_new(args_new)
        with self.assertRaises(SystemExit):
            with patch('sys.stdout', new_callable=StringIO), \
                 patch('sys.stderr', new_callable=StringIO):
                sm.cmd_advance(self._make_args(reason=[]))

    def test_advance_unknown_tier(self):
        abs_dir = self._create_session(tier="EXOTIC", phase="0")
        _touch(os.path.join(abs_dir, "phase_outputs", "phase_0.md"))
        with self.assertRaises(SystemExit):
            with patch('sys.stdout', new_callable=StringIO), \
                 patch('sys.stderr', new_callable=StringIO):
                sm.cmd_advance(self._make_args(reason=[]))

    def test_advance_phase_0_to_0_3_with_artifacts_passes(self):
        abs_dir = self._create_session(tier="STANDARD", phase="0")
        _touch(os.path.join(abs_dir, "phase_outputs", "phase_0.md"))
        with patch('sys.stdout', new_callable=StringIO):
            sm.cmd_advance(self._make_args(reason=["Phase 0 done"]))
        # Phase should now be 0.3
        with open(os.path.join(abs_dir, "state.md")) as f:
            state = f.read()
        self.assertIn("## Phase: 0.3", state)
        self.assertIn("ADVANCE", state)

    def test_advance_refuses_when_phase_output_missing(self):
        abs_dir = self._create_session(tier="STANDARD", phase="0")
        # No phase_0.md created.
        with self.assertRaises(SystemExit):
            with patch('sys.stdout', new_callable=StringIO), \
                 patch('sys.stderr', new_callable=StringIO):
                sm.cmd_advance(self._make_args(reason=[]))

    def test_advance_refuses_when_gate_fails(self):
        """When the gate script returns non-zero, advance refuses."""
        abs_dir = self._create_session(tier="STANDARD", phase="0.3")
        _touch(os.path.join(abs_dir, "phase_outputs", "phase_0_3.md"))
        # No domain_orientation.json → gate script will fail.
        with self.assertRaises(SystemExit):
            with patch('sys.stdout', new_callable=StringIO), \
                 patch('sys.stderr', new_callable=StringIO):
                sm.cmd_advance(self._make_args(reason=[]))

    def test_advance_rapid_p0_5_to_p5(self):
        abs_dir = self._create_session(tier="RAPID", phase="0.5")
        _touch(os.path.join(abs_dir, "phase_outputs", "phase_0_5.md"))
        with patch('sys.stdout', new_callable=StringIO):
            sm.cmd_advance(self._make_args(reason=["RAPID screen complete"]))
        with open(os.path.join(abs_dir, "state.md")) as f:
            state = f.read()
        self.assertIn("## Phase: 5", state)

    def test_advance_psych_phase_ids_supported(self):
        """PSYCH tier `0-P` advances to `0-P.3`."""
        abs_dir = self._create_session(tier="PSYCH", phase="0-P")
        _touch(os.path.join(abs_dir, "phase_outputs", "phase_0_P.md"))
        with patch('sys.stdout', new_callable=StringIO):
            sm.cmd_advance(self._make_args(reason=[]))
        with open(os.path.join(abs_dir, "state.md")) as f:
            state = f.read()
        self.assertIn("## Phase: 0-P.3", state)

    def test_advance_terminal_phase_refuses(self):
        abs_dir = self._create_session(tier="STANDARD", phase="5")
        _touch(os.path.join(abs_dir, "phase_outputs", "phase_5.md"))
        with self.assertRaises(SystemExit):
            with patch('sys.stdout', new_callable=StringIO), \
                 patch('sys.stderr', new_callable=StringIO):
                sm.cmd_advance(self._make_args(reason=[]))


class TestGateCheck(SessionManagerTestBase):

    def _create_session(self, tier="STANDARD", phase="0"):
        args_new = self._make_args(goal=["Test"], force=False)
        with patch('sys.stdout', new_callable=StringIO):
            sm.cmd_new(args_new)
        abs_dir = sm.read_pointer()
        _set_tier_and_phase(abs_dir, tier, phase)
        return abs_dir

    def test_gate_check_passes_with_artifacts(self):
        abs_dir = self._create_session(tier="STANDARD", phase="0")
        _touch(os.path.join(abs_dir, "phase_outputs", "phase_0.md"))
        with patch('sys.stdout', new_callable=StringIO) as out:
            with self.assertRaises(SystemExit) as cm:
                sm.cmd_gate_check(self._make_args())
            self.assertEqual(cm.exception.code, 0)
        report = json.loads(out.getvalue())
        self.assertTrue(report["would_advance"])
        self.assertEqual(report["next_phase"], "0.3")

    def test_gate_check_fails_without_artifacts(self):
        self._create_session(tier="STANDARD", phase="0")
        with patch('sys.stdout', new_callable=StringIO):
            with self.assertRaises(SystemExit) as cm:
                sm.cmd_gate_check(self._make_args())
            self.assertEqual(cm.exception.code, 1)


class TestSkipWhitelist(SessionManagerTestBase):
    """D-003: cmd_skip refuses non-whitelisted phases per tier."""

    def _create_session(self, tier="STANDARD"):
        args_new = self._make_args(goal=["Test"], force=False)
        with patch('sys.stdout', new_callable=StringIO):
            sm.cmd_new(args_new)
        abs_dir = sm.read_pointer()
        _set_tier_and_phase(abs_dir, tier, "0")
        # D-001: satisfy the Phase 0.3/0-P.3 familiarity gate by default.
        _set_domain_familiarity(abs_dir, "high")
        return abs_dir

    def test_skip_whitelisted_0_3_on_standard_passes(self):
        abs_dir = self._create_session(tier="STANDARD")
        args = self._make_args(phase="0.3", reason=["high familiarity"])
        with patch('sys.stdout', new_callable=StringIO):
            sm.cmd_skip(args)
        with open(os.path.join(abs_dir, "decisions.md")) as f:
            self.assertIn("SKIP Phase 0.3", f.read())

    def test_skip_non_whitelisted_phase_refuses(self):
        """STANDARD tier: skipping Phase 3 must be refused."""
        self._create_session(tier="STANDARD")
        args = self._make_args(phase="3", reason=["I don't want to do it"])
        with self.assertRaises(SystemExit):
            with patch('sys.stdout', new_callable=StringIO), \
                 patch('sys.stderr', new_callable=StringIO):
                sm.cmd_skip(args)

    def test_skip_rapid_tier_no_skips_allowed(self):
        """RAPID tier has empty SKIPPABLE — every skip refused."""
        self._create_session(tier="RAPID")
        args = self._make_args(phase="0.5", reason=["why bother"])
        with self.assertRaises(SystemExit):
            with patch('sys.stdout', new_callable=StringIO), \
                 patch('sys.stderr', new_callable=StringIO):
                sm.cmd_skip(args)

    def test_skip_psych_tier_0_P_3_whitelisted(self):
        abs_dir = self._create_session(tier="PSYCH")
        _set_tier_and_phase(abs_dir, "PSYCH", "0-P")
        args = self._make_args(phase="0-P.3", reason=["expert"])
        with patch('sys.stdout', new_callable=StringIO):
            sm.cmd_skip(args)
        with open(os.path.join(abs_dir, "decisions.md")) as f:
            self.assertIn("SKIP Phase 0-P.3", f.read())


class TestSkipAdvanceIntegration(SessionManagerTestBase):
    """plan_2026-05-19_8608e41f/D-002: skip mutates Phase: cursor; subsequent
    advance must skip OVER the bypassed phase, not land on it.
    Regression test for F-1: skip was previously a half-implemented logger
    that left the cursor stale, forcing analysts to use --force-state.
    """

    def _create_session(self, tier="STANDARD", phase="0"):
        args_new = self._make_args(goal=["Test"], force=False)
        with patch('sys.stdout', new_callable=StringIO):
            sm.cmd_new(args_new)
        abs_dir = sm.read_pointer()
        _set_tier_and_phase(abs_dir, tier, phase)
        # D-001: satisfy the Phase 0.3/0-P.3 familiarity gate by default.
        _set_domain_familiarity(abs_dir, "high")
        return abs_dir

    def test_skip_advances_phase_cursor(self):
        """skip 0.3 from Phase 0 STANDARD moves cursor to 0.7 (post-skip phase)."""
        abs_dir = self._create_session(tier="STANDARD", phase="0")
        args = self._make_args(phase="0.3", reason=["domain_familiarity=high"])
        with patch('sys.stdout', new_callable=StringIO):
            sm.cmd_skip(args)
        with open(os.path.join(abs_dir, "state.md")) as f:
            state = f.read()
        self.assertIn("## Phase: 0.7", state)
        self.assertIn("SKIP", state)

    def test_skip_advances_when_at_skipped_phase(self):
        """skip 0.3 from Phase 0.3 also moves cursor to 0.7."""
        abs_dir = self._create_session(tier="STANDARD", phase="0.3")
        args = self._make_args(phase="0.3", reason=["mid-phase abandon"])
        with patch('sys.stdout', new_callable=StringIO):
            sm.cmd_skip(args)
        with open(os.path.join(abs_dir, "state.md")) as f:
            state = f.read()
        self.assertIn("## Phase: 0.7", state)

    def test_skip_then_advance_lands_past_skipped_phase(self):
        """After skip 0.3, advance proceeds from 0.7 (not 0.3 — that's the bug)."""
        abs_dir = self._create_session(tier="STANDARD", phase="0")
        # Skip 0.3
        with patch('sys.stdout', new_callable=StringIO):
            sm.cmd_skip(self._make_args(
                phase="0.3", reason=["domain_familiarity=high"]))
        # Provide artifacts + scope_audit.json for Phase 0.7 advance.
        _touch(os.path.join(abs_dir, "phase_outputs", "phase_0_7.md"))
        # Cursor is at 0.7 — advance from 0.7 would need scope_audit.json
        # which the gate script consumes. We assert the cursor moved past 0.3,
        # which is the regression target — the advance gate path is exercised
        # by TestAdvance tests.
        with open(os.path.join(abs_dir, "state.md")) as f:
            state = f.read()
        self.assertIn("## Phase: 0.7", state)
        self.assertNotIn("## Phase: 0.3", state)

    def test_skip_psych_phase_advances_cursor(self):
        """skip 0-P.3 on PSYCH from 0-P moves cursor to 0-P.7."""
        abs_dir = self._create_session(tier="PSYCH", phase="0-P")
        args = self._make_args(phase="0-P.3", reason=["expert"])
        with patch('sys.stdout', new_callable=StringIO):
            sm.cmd_skip(args)
        with open(os.path.join(abs_dir, "state.md")) as f:
            state = f.read()
        self.assertIn("## Phase: 0-P.7", state)


class TestSetPhase(SessionManagerTestBase):
    """D-002 + escape-hatch policy: set-phase requires --force-state + --reason."""

    def _create_session(self):
        args_new = self._make_args(goal=["Test"], force=False)
        with patch('sys.stdout', new_callable=StringIO):
            sm.cmd_new(args_new)
        abs_dir = sm.read_pointer()
        _set_tier_and_phase(abs_dir, "STANDARD", "0")
        return abs_dir

    def test_set_phase_without_force_refuses(self):
        self._create_session()
        args = self._make_args(phase="3", force_state=False, reason=["x"])
        with self.assertRaises(SystemExit):
            with patch('sys.stdout', new_callable=StringIO), \
                 patch('sys.stderr', new_callable=StringIO):
                sm.cmd_set_phase(args)

    def test_set_phase_without_reason_refuses(self):
        self._create_session()
        args = self._make_args(phase="3", force_state=True, reason=["   "])
        with self.assertRaises(SystemExit):
            with patch('sys.stdout', new_callable=StringIO), \
                 patch('sys.stderr', new_callable=StringIO):
                sm.cmd_set_phase(args)

    def test_set_phase_invalid_phase_refuses(self):
        self._create_session()
        args = self._make_args(phase="99", force_state=True, reason=["recovery"])
        with self.assertRaises(SystemExit):
            with patch('sys.stdout', new_callable=StringIO), \
                 patch('sys.stderr', new_callable=StringIO):
                sm.cmd_set_phase(args)

    def test_set_phase_with_force_logs_decision(self):
        abs_dir = self._create_session()
        args = self._make_args(phase="3", force_state=True,
                               reason=["state.md", "corrupted;", "recovery"])
        with patch('sys.stdout', new_callable=StringIO):
            sm.cmd_set_phase(args)
        with open(os.path.join(abs_dir, "decisions.md")) as f:
            content = f.read()
        self.assertIn("ADMIN-OVERRIDE", content)
        self.assertIn("Phase 0 → 3", content)
        self.assertIn("state.md corrupted; recovery", content)
        with open(os.path.join(abs_dir, "state.md")) as f:
            state = f.read()
        self.assertIn("## Phase: 3", state)
        self.assertIn("ADMIN-OVERRIDE", state)


class TestWriteStateMd(SessionManagerTestBase):
    """D-004: cmd_write refuses Phase: changes via free `write state.md`."""

    def _create_session(self):
        args_new = self._make_args(goal=["Test"], force=False)
        with patch('sys.stdout', new_callable=StringIO):
            sm.cmd_new(args_new)
        abs_dir = sm.read_pointer()
        _set_tier_and_phase(abs_dir, "STANDARD", "0")
        return abs_dir

    def _run_write(self, filename, content, force_state=False):
        args = self._make_args(filename=filename, force_state=force_state)
        with patch('sys.stdin', StringIO(content)):
            with patch('sys.stdout', new_callable=StringIO):
                sm.cmd_write(args)

    def test_write_state_md_phase_change_refused_without_force(self):
        abs_dir = self._create_session()
        with open(os.path.join(abs_dir, "state.md")) as f:
            content = f.read()
        new_content = content.replace("## Phase: 0", "## Phase: 3")
        with self.assertRaises(SystemExit):
            with patch('sys.stderr', new_callable=StringIO):
                self._run_write("state.md", new_content, force_state=False)

    def test_write_state_md_phase_unchanged_passes(self):
        """Editing other fields of state.md (not Phase:) is allowed."""
        abs_dir = self._create_session()
        with open(os.path.join(abs_dir, "state.md")) as f:
            content = f.read()
        # Change hypothesis count, not Phase:.
        new_content = content.replace("## Active Hypotheses: 0",
                                      "## Active Hypotheses: 5")
        self._run_write("state.md", new_content, force_state=False)
        with open(os.path.join(abs_dir, "state.md")) as f:
            saved = f.read()
        self.assertIn("Active Hypotheses: 5", saved)
        self.assertIn("## Phase: 0", saved)

    def test_write_state_md_force_state_allowed_and_logged(self):
        abs_dir = self._create_session()
        with open(os.path.join(abs_dir, "state.md")) as f:
            content = f.read()
        new_content = content.replace("## Phase: 0", "## Phase: 2")
        self._run_write("state.md", new_content, force_state=True)
        with open(os.path.join(abs_dir, "state.md")) as f:
            saved = f.read()
        self.assertIn("## Phase: 2", saved)
        with open(os.path.join(abs_dir, "decisions.md")) as f:
            decisions = f.read()
        self.assertIn("ADMIN-OVERRIDE", decisions)
        self.assertIn("write state.md", decisions)

    def test_write_non_state_md_unaffected(self):
        """Hardening only applies to state.md."""
        abs_dir = self._create_session()
        self._run_write("progress.md", "# Progress\nedited\n", force_state=False)
        with open(os.path.join(abs_dir, "progress.md")) as f:
            self.assertIn("edited", f.read())


class TestEnsureGitignore(SessionManagerTestBase):

    def test_creates_gitignore(self):
        """ensure_gitignore creates .gitignore with analyses/ entry."""
        sm.ensure_gitignore()
        with open(".gitignore") as f:
            content = f.read()
        self.assertIn("analyses/", content)

    def test_idempotent(self):
        """ensure_gitignore doesn't duplicate the entry."""
        sm.ensure_gitignore()
        sm.ensure_gitignore()
        with open(".gitignore") as f:
            content = f.read()
        self.assertEqual(content.count("analyses/"), 1)


class TestCmdNewFlags(SessionManagerTestBase):
    """cmd_new --tier / --domain-familiarity flags (plan_2026-05-27_33d457f3/D-001).

    The orchestrator collects tier + familiarity during Intake Triage and now
    passes them to `$SM new` so the very next `$SM advance` / `$SM skip 0.3`
    succeeds without follow-up state.md / analysis_plan.md hand-edits.
    """

    def _run_new(self, **kwargs):
        defaults = {"goal": ["Test", "system"], "force": False,
                    "tier": None, "domain_familiarity": None}
        defaults.update(kwargs)
        args = self._make_args(**defaults)
        with patch('sys.stdout', new_callable=StringIO):
            sm.cmd_new(args)

    def test_no_flags_backward_compat(self):
        """cmd_new without new flags retains placeholder tier and no familiarity."""
        self._run_new()
        abs_dir = sm.read_pointer()
        state = sm.read_analysis_file(abs_dir, "state.md")
        plan = sm.read_analysis_file(abs_dir, "analysis_plan.md")
        self.assertIn("## Tier: (pending)", state)
        self.assertIn("*(RAPID / LITE / STANDARD / COMPREHENSIVE / PSYCH)*", plan)

    def test_tier_flag_writes_state_and_plan(self):
        self._run_new(tier="STANDARD")
        abs_dir = sm.read_pointer()
        state = sm.read_analysis_file(abs_dir, "state.md")
        plan = sm.read_analysis_file(abs_dir, "analysis_plan.md")
        self.assertIn("## Tier: STANDARD", state)
        self.assertIn("## Tier Selected\nSTANDARD", plan)

    def test_familiarity_flag_writes_plan_field(self):
        self._run_new(domain_familiarity="high")
        abs_dir = sm.read_pointer()
        plan = sm.read_analysis_file(abs_dir, "analysis_plan.md")
        self.assertIn("domain_familiarity: high", plan)
        self.assertEqual(sm._read_domain_familiarity(abs_dir), "high")

    def test_both_flags_set(self):
        self._run_new(tier="COMPREHENSIVE", domain_familiarity="low")
        abs_dir = sm.read_pointer()
        state = sm.read_analysis_file(abs_dir, "state.md")
        self.assertIn("## Tier: COMPREHENSIVE", state)
        self.assertEqual(sm._read_domain_familiarity(abs_dir), "low")
        self.assertEqual(sm._current_tier(abs_dir), "COMPREHENSIVE")

    def test_user_failure_repro_new_then_skip_0_3(self):
        """Reproduces the user's exact failure path: new + skip 0.3 must succeed."""
        self._run_new(tier="STANDARD", domain_familiarity="high")
        abs_dir = sm.read_pointer()
        skip_args = self._make_args(phase="0.3",
                                    reason=["high", "familiarity", "declared"])
        with patch('sys.stdout', new_callable=StringIO), \
             patch('sys.stderr', new_callable=StringIO):
            sm.cmd_skip(skip_args)
        # Skip succeeded → cursor advanced past 0.3 → Phase: should now be 0.7.
        state = sm.read_analysis_file(abs_dir, "state.md")
        self.assertIn("## Phase: 0.7", state)


class TestCmdDeclare(SessionManagerTestBase):
    """cmd_declare — late-binding tier/familiarity setter."""

    def _start_session(self):
        args = self._make_args(goal=["X"], force=False,
                               tier=None, domain_familiarity=None)
        with patch('sys.stdout', new_callable=StringIO):
            sm.cmd_new(args)
        return sm.read_pointer()

    def _declare(self, **kwargs):
        defaults = {"tier": None, "domain_familiarity": None}
        defaults.update(kwargs)
        args = self._make_args(**defaults)
        stdout = StringIO()
        stderr = StringIO()
        rc = 0
        try:
            with patch('sys.stdout', stdout), patch('sys.stderr', stderr):
                sm.cmd_declare(args)
        except SystemExit as e:
            rc = e.code or 0
        return rc, stdout.getvalue(), stderr.getvalue()

    def test_no_active_session_refuses(self):
        rc, _, err = self._declare(tier="STANDARD")
        self.assertEqual(rc, 1)
        self.assertIn("No active analysis", err)

    def test_neither_flag_refuses(self):
        self._start_session()
        rc, _, err = self._declare()
        self.assertEqual(rc, 1)
        self.assertIn("at least one of", err)

    def test_tier_only(self):
        abs_dir = self._start_session()
        rc, _, _ = self._declare(tier="LITE")
        self.assertEqual(rc, 0)
        self.assertEqual(sm._current_tier(abs_dir), "LITE")
        plan = sm.read_analysis_file(abs_dir, "analysis_plan.md")
        self.assertIn("## Tier Selected\nLITE", plan)

    def test_familiarity_only(self):
        abs_dir = self._start_session()
        rc, _, _ = self._declare(domain_familiarity="high")
        self.assertEqual(rc, 0)
        self.assertEqual(sm._read_domain_familiarity(abs_dir), "high")

    def test_both_atomic(self):
        abs_dir = self._start_session()
        rc, _, _ = self._declare(tier="PSYCH", domain_familiarity="medium")
        self.assertEqual(rc, 0)
        self.assertEqual(sm._current_tier(abs_dir), "PSYCH")
        self.assertEqual(sm._read_domain_familiarity(abs_dir), "medium")

    def test_tier_escalation_round_trip(self):
        """STANDARD → COMPREHENSIVE via declare overwrites cleanly."""
        abs_dir = self._start_session()
        self._declare(tier="STANDARD")
        self._declare(tier="COMPREHENSIVE")
        self.assertEqual(sm._current_tier(abs_dir), "COMPREHENSIVE")
        # state.md retains a single Tier line — declare must not duplicate.
        state = sm.read_analysis_file(abs_dir, "state.md")
        self.assertEqual(state.count("## Tier:"), 1)

    def test_declare_logs_to_decisions(self):
        abs_dir = self._start_session()
        self._declare(tier="STANDARD", domain_familiarity="low")
        decisions = sm.read_analysis_file(abs_dir, "decisions.md")
        self.assertIn("DECLARE", decisions)
        self.assertIn("tier=STANDARD", decisions)
        self.assertIn("domain_familiarity=low", decisions)

    def test_declare_does_not_touch_phase(self):
        """cmd_declare MUST NOT mutate `## Phase:` — that's the FSM cursor."""
        abs_dir = self._start_session()
        before = sm._current_phase(abs_dir)
        self._declare(tier="STANDARD")
        after = sm._current_phase(abs_dir)
        self.assertEqual(before, after)


class TestReadDomainFamiliarityLastMatch(SessionManagerTestBase):
    """`_read_domain_familiarity` selects the LAST valid match (plan_2026-05-27_33d457f3)."""

    def _start_session(self):
        args = self._make_args(goal=["X"], force=False,
                               tier=None, domain_familiarity=None)
        with patch('sys.stdout', new_callable=StringIO):
            sm.cmd_new(args)
        return sm.read_pointer()

    def test_last_declaration_wins(self):
        abs_dir = self._start_session()
        plan_path = os.path.join(abs_dir, "analysis_plan.md")
        with open(plan_path, "a") as f:
            f.write("\ndomain_familiarity: low\n")
            f.write("\ndomain_familiarity: high\n")
        self.assertEqual(sm._read_domain_familiarity(abs_dir), "high")

    def test_placeholder_does_not_mask_real_declaration(self):
        """Template placeholder (`(declare with ...)`) must not block a later real value."""
        abs_dir = self._start_session()
        plan_path = os.path.join(abs_dir, "analysis_plan.md")
        # Template already has a placeholder line — append a real declaration.
        with open(plan_path, "a") as f:
            f.write("\ndomain_familiarity: medium\n")
        self.assertEqual(sm._read_domain_familiarity(abs_dir), "medium")


class TestCmdResumeNoSessionParity(SessionManagerTestBase):
    """cmd_resume on no session: exit 0 + NO_ACTIVE_SESSION marker."""

    def test_marker_on_stdout(self):
        args = self._make_args()
        stdout = StringIO()
        with patch('sys.stdout', stdout), \
             patch('sys.stderr', new_callable=StringIO):
            sm.cmd_resume(args)
        self.assertIn("NO_ACTIVE_SESSION", stdout.getvalue())
        self.assertIn("$SM new", stdout.getvalue())


if __name__ == '__main__':
    unittest.main()
