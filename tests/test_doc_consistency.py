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

Detection is by basename substring scoped to the "Repository Structure" fenced
tree block (NOT the whole file). Scoping to the tree block prevents the
prose-only false-green documented in the audit (D-08): a basename cited only in
narrative prose -- e.g. ``abductive-reasoning.md`` at CLAUDE.md:151 -- used to
satisfy a whole-file ``assertIn`` while being absent from the tree (D-07). No
mocks; reads files off disk. Pattern after test_doc_fences.py.
"""

import re
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


def _extract_tree_block(text: str) -> str:
    """Return the fenced code block under the '## Repository Structure' header.

    Captures from the first ``` fence after the header to the next ``` fence.
    Scoping the consistency check to this block (rather than the whole file) is
    the D-08 fix: a basename appearing only in prose must NOT count as listed.
    """
    header = re.search(r"^##\s+Repository Structure\s*$", text, re.MULTILINE)
    if not header:
        return ""
    rest = text[header.end():]
    open_fence = rest.find("```")
    if open_fence == -1:
        return ""
    after_open = rest[open_fence + 3:]
    close_fence = after_open.find("```")
    if close_fence == -1:
        return ""
    return after_open[:close_fence]


class TestDocConsistency(unittest.TestCase):
    """Every script/test/reference basename must appear in CLAUDE.md."""

    @classmethod
    def setUpClass(cls):
        cls.claude_md_text = CLAUDE_MD.read_text(encoding="utf-8")
        # D-08: scope matching to the Repository Structure tree block only, so a
        # basename cited solely in prose cannot mask a tree omission (D-07).
        cls.claude_md_tree = _extract_tree_block(cls.claude_md_text)
        cls.scripts = _basenames(REPO_ROOT / "src" / "scripts", "*.py")
        cls.tests = _basenames(REPO_ROOT / "tests", "test_*.py")
        cls.references = _basenames(REPO_ROOT / "src" / "references", "*.md")

    def test_claude_md_exists_and_nonempty(self):
        self.assertTrue(CLAUDE_MD.exists(), "CLAUDE.md not found")
        self.assertTrue(self.claude_md_text.strip(), "CLAUDE.md is empty")

    def test_tree_block_extracted(self):
        # Guard the D-08 scoping itself: if extraction silently returns "" the
        # three listed-checks below would vacuously pass (a new false-green).
        self.assertTrue(
            self.claude_md_tree.strip(),
            "Repository Structure tree block not extracted from CLAUDE.md",
        )

    def test_all_scripts_listed(self):
        self.assertTrue(self.scripts, "no scripts gathered (scope bug)")
        for name in self.scripts:
            with self.subTest(script=name):
                self.assertIn(
                    name, self.claude_md_tree,
                    f"src/scripts/{name} is missing from the CLAUDE.md "
                    f"Repository Structure TREE block (prose mentions do not count)",
                )

    def test_all_tests_listed(self):
        self.assertTrue(self.tests, "no test files gathered (scope bug)")
        for name in self.tests:
            with self.subTest(test=name):
                self.assertIn(
                    name, self.claude_md_tree,
                    f"tests/{name} is missing from the CLAUDE.md "
                    f"Repository Structure TREE block (prose mentions do not count)",
                )

    def test_all_references_listed(self):
        self.assertTrue(self.references, "no reference files gathered (scope bug)")
        for name in self.references:
            with self.subTest(reference=name):
                self.assertIn(
                    name, self.claude_md_tree,
                    f"src/references/{name} is missing from the CLAUDE.md "
                    f"Repository Structure TREE block (prose mentions do not count)",
                )


if __name__ == "__main__":
    unittest.main()
