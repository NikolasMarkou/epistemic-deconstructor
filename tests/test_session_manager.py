#!/usr/bin/env python3
"""Tests for src/scripts/session_manager.py"""

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
        """read_pointer returns dir name when pointer and dir both exist."""
        os.makedirs(sm.ANALYSES_DIR, exist_ok=True)
        dir_name = "analysis_2025-01-01_abcd1234"
        os.makedirs(os.path.join(sm.ANALYSES_DIR, dir_name))
        with open(sm.POINTER_FILE, 'w') as f:
            f.write(dir_name)
        self.assertEqual(sm.read_pointer(), dir_name)


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
        """cmd_resume exits with error when no active session."""
        args = self._make_args()
        with self.assertRaises(SystemExit):
            with patch('sys.stdout', new_callable=StringIO), \
                 patch('sys.stderr', new_callable=StringIO):
                sm.cmd_resume(args)

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
        self.assertIn("Resuming", text)
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


if __name__ == '__main__':
    unittest.main()
