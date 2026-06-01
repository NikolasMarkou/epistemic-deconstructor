#!/usr/bin/env python3
"""CI-enforced documentation-consistency guardrail (RC1).

Invariant: the CLAUDE.md "Repository Structure" tree must list every shipped
script, test, and reference file. If a new file is added under ``src/scripts/``,
``tests/``, or ``src/references/`` without a matching entry in ``CLAUDE.md``,
this test fails loudly so the doc tree cannot silently drift from disk.

Checked file sets:
  1. ``src/scripts/*.py``   (excluding ``__init__.py``)
  2. ``tests/test_*.py``    (this file included — so CLAUDE.md must list it too)
  3. ``src/references/*.md``

Detection is by basename substring: each basename must appear somewhere in the
CLAUDE.md text. No mocks; reads files off disk. Pattern after test_doc_fences.py.
"""

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CLAUDE_MD = REPO_ROOT / "CLAUDE.md"

# Basenames that are infrastructure, not shipped tree entries.
_EXCLUDE = {"__init__.py", "conftest.py"}


def _basenames(glob_dir: Path, pattern: str):
    return sorted(
        p.name
        for p in glob_dir.glob(pattern)
        if p.is_file() and p.name not in _EXCLUDE
    )


class TestDocConsistency(unittest.TestCase):
    """Every script/test/reference basename must appear in CLAUDE.md."""

    @classmethod
    def setUpClass(cls):
        cls.claude_md_text = CLAUDE_MD.read_text(encoding="utf-8")
        cls.scripts = _basenames(REPO_ROOT / "src" / "scripts", "*.py")
        cls.tests = _basenames(REPO_ROOT / "tests", "test_*.py")
        cls.references = _basenames(REPO_ROOT / "src" / "references", "*.md")

    def test_claude_md_exists_and_nonempty(self):
        self.assertTrue(CLAUDE_MD.exists(), "CLAUDE.md not found")
        self.assertTrue(self.claude_md_text.strip(), "CLAUDE.md is empty")

    def test_all_scripts_listed(self):
        self.assertTrue(self.scripts, "no scripts gathered (scope bug)")
        for name in self.scripts:
            with self.subTest(script=name):
                self.assertIn(
                    name, self.claude_md_text,
                    f"src/scripts/{name} is missing from CLAUDE.md tree",
                )

    def test_all_tests_listed(self):
        self.assertTrue(self.tests, "no test files gathered (scope bug)")
        for name in self.tests:
            with self.subTest(test=name):
                self.assertIn(
                    name, self.claude_md_text,
                    f"tests/{name} is missing from CLAUDE.md tree",
                )

    def test_all_references_listed(self):
        self.assertTrue(self.references, "no reference files gathered (scope bug)")
        for name in self.references:
            with self.subTest(reference=name):
                self.assertIn(
                    name, self.claude_md_text,
                    f"src/references/{name} is missing from CLAUDE.md tree",
                )


if __name__ == "__main__":
    unittest.main()
