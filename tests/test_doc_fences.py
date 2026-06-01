#!/usr/bin/env python3
"""CI-enforced documentation fence guardrail (S5).

Two invariants over tracked Markdown:
  1. Triple-backtick fence lines are balanced (even count per file), so a
     stray/unclosed ```` ``` ```` cannot ship and corrupt the naive `cat`
     build-combined output.
  2. Every ```` ```mermaid ```` block opens with a recognized Mermaid
     diagram-type on its first non-empty content line, so a mistyped
     diagram type is caught before it reaches a renderer.

Scope = every ``*.md`` under ``src/`` (recursive) PLUS the three root files
``README.md``, ``CLAUDE.md``, ``CHANGELOG.md``. ``plans/`` is gitignored
planner state and is NOT checked. No mocks; reads files off disk.
"""

import os
import re
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Recognized Mermaid diagram-type keywords (first non-empty line of a
# ```mermaid block must start with one of these). Mirrors the type map in
# src/references/mermaid-conventions.md.
RECOGNIZED_MERMAID_TYPES = (
    'stateDiagram-v2',
    'stateDiagram',
    'flowchart',
    'graph',
    'sequenceDiagram',
    'classDiagram',
    'erDiagram',
    'gantt',
    'pie',
    'journey',
    'gitGraph',
    'mindmap',
    'timeline',
    'quadrantChart',
    'xychart-beta',
    'requirementDiagram',
    'C4Context',
)

FENCE_RE = re.compile(r'^```')


def _gather_markdown_files():
    """Return absolute paths: every *.md under src/ plus the 3 root files."""
    paths = []
    src_dir = os.path.join(REPO_ROOT, 'src')
    for dirpath, _dirnames, filenames in os.walk(src_dir):
        for name in filenames:
            if name.endswith('.md'):
                paths.append(os.path.join(dirpath, name))
    for name in ('README.md', 'CLAUDE.md', 'CHANGELOG.md'):
        root_path = os.path.join(REPO_ROOT, name)
        if os.path.exists(root_path):
            paths.append(root_path)
    return sorted(paths)


class TestDocFences(unittest.TestCase):
    """Fence-balance + mermaid-type guardrail over tracked Markdown."""

    def setUp(self):
        self.files = _gather_markdown_files()
        self.assertTrue(self.files, "no markdown files gathered (scope bug)")

    def test_fences_balanced(self):
        """Each file must have an EVEN count of ``` fence lines."""
        for path in self.files:
            with self.subTest(path=os.path.relpath(path, REPO_ROOT)):
                with open(path, encoding='utf-8') as fh:
                    lines = fh.read().splitlines()
                n = sum(1 for ln in lines if FENCE_RE.match(ln.strip()))
                rel = os.path.relpath(path, REPO_ROOT)
                self.assertEqual(
                    n % 2, 0,
                    f"{rel}: odd fence count {n} (unbalanced ``` blocks)",
                )

    def test_mermaid_first_line_recognized(self):
        """Every ```mermaid block must open with a recognized diagram type."""
        for path in self.files:
            with self.subTest(path=os.path.relpath(path, REPO_ROOT)):
                with open(path, encoding='utf-8') as fh:
                    lines = fh.read().splitlines()
                rel = os.path.relpath(path, REPO_ROOT)
                in_fence = False
                in_mermaid = False
                awaiting_first = False
                for ln in lines:
                    stripped = ln.strip()
                    if FENCE_RE.match(stripped):
                        if not in_fence:
                            # Opening fence: capture info string after ```.
                            info = stripped[3:].strip()
                            in_fence = True
                            in_mermaid = (info == 'mermaid')
                            awaiting_first = in_mermaid
                        else:
                            # Closing fence.
                            in_fence = False
                            in_mermaid = False
                            awaiting_first = False
                        continue
                    if awaiting_first and stripped:
                        first = stripped
                        awaiting_first = False
                        ok = any(
                            first.startswith(t)
                            for t in RECOGNIZED_MERMAID_TYPES
                        )
                        self.assertTrue(
                            ok,
                            f"{rel}: mermaid block first line "
                            f"{first!r} is not a recognized diagram type",
                        )


if __name__ == '__main__':
    unittest.main()
