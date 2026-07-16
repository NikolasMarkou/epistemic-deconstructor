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
  4. ``src/agents/*.md``

Detection is by basename substring scoped to the "Repository Structure" fenced
tree block (NOT the whole file). Scoping to the tree block prevents the
prose-only false-green documented in the audit (D-08): a basename cited only in
narrative prose -- e.g. ``abductive-reasoning.md`` at CLAUDE.md:151 -- used to
satisfy a whole-file ``assertIn`` while being absent from the tree (D-07). No
mocks; reads files off disk. Pattern after test_doc_fences.py.

Agent frontmatter consistency (O2, v7.16.7): five preventive drift guards over
``src/agents/*.md`` frontmatter -- well-formedness, name==stem, orchestrator
``Agent(...)`` grant == agent-file set, SKILL.md dispatch list == grant, and
CLAUDE.md tree annotations (model / background) == frontmatter fields.
"""

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CLAUDE_MD = REPO_ROOT / "CLAUDE.md"
AGENTS_DIR = REPO_ROOT / "src" / "agents"
SKILL_MD = REPO_ROOT / "src" / "SKILL.md"

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


# Top-level frontmatter key line, e.g. ``model: sonnet`` or ``skills:``.
_FRONTMATTER_KEY = re.compile(r"^([A-Za-z][\w-]*):\s*(.*)$")
# Indented continuation of a folded/block scalar or a list item.
_FRONTMATTER_CONT = re.compile(r"^\s+\S")


# DECISION plan-2026-07-16T140352-760b7091/D-001: frontmatter is parsed with a
# stdlib two-fence scan + regex, NOT ``import yaml`` -- the CI test-stdlib job's
# pyyaml availability is unverified and no test file in this repo imports yaml.
# Do NOT "simplify" this to yaml.safe_load. See the plan's decisions.md D-001.
def _agent_frontmatter(path: Path):
    """Return (raw frontmatter lines, dict of top-level scalar keys).

    Two-fence scan: the frontmatter opens with ``---`` on line 1 and closes at
    the NEXT line that is exactly ``---``. Stopping at the second fence makes
    the scan immune to any later ``---`` used as a body <hr> (e.g. the one at
    ed-orchestrator.md:115). Values of folded/block scalars (``key: >`` /
    ``key: |``) are recorded as their marker; continuation lines stay in the
    raw block only. Returns ([], {}) when either fence is missing.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "---":
        return [], {}
    block = []
    closed = False
    for line in lines[1:]:
        if line == "---":
            closed = True
            break
        block.append(line)
    if not closed:
        return [], {}
    keys = {}
    for line in block:
        match = _FRONTMATTER_KEY.match(line)
        if match:
            keys[match.group(1)] = match.group(2).strip()
    return block, keys


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
        cls.agents = _basenames(REPO_ROOT / "src" / "agents", "*.md")
        # O2: raw frontmatter block + top-level key dict per agent file.
        cls.agent_frontmatter = {
            name: _agent_frontmatter(AGENTS_DIR / name) for name in cls.agents
        }

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

    def test_all_agents_listed(self):
        self.assertTrue(self.agents, "no agent files gathered (scope bug)")
        for name in self.agents:
            with self.subTest(agent=name):
                self.assertIn(
                    name, self.claude_md_tree,
                    f"src/agents/{name} is missing from the CLAUDE.md "
                    f"Repository Structure TREE block (prose mentions do not count)",
                )

    # ---- O2 agent frontmatter consistency guards (drift classes a-e) ----

    def _grant_set(self):
        """The agent-name set from ed-orchestrator.md's ``Agent(...)`` grant."""
        _block, keys = self.agent_frontmatter.get("ed-orchestrator.md", ([], {}))
        match = re.search(r"Agent\(([^)]+)\)", keys.get("tools", ""))
        if not match:
            return set()
        return {t.strip() for t in match.group(1).split(",") if t.strip()}

    def test_agent_frontmatter_well_formed(self):
        """Drift class (e): both fences present, every line a key line /
        indented continuation / blank, and the 4 required keys present."""
        self.assertTrue(
            self.agent_frontmatter, "no agent frontmatter gathered (scope bug)"
        )
        required = ("name", "description", "tools", "model")
        for name, (block, keys) in self.agent_frontmatter.items():
            with self.subTest(agent=name):
                self.assertTrue(
                    block,
                    f"src/agents/{name}: frontmatter not extracted -- opening "
                    f"'---' must be line 1 and a closing '---' must follow",
                )
                for lineno, line in enumerate(block, start=2):
                    self.assertTrue(
                        not line.strip()
                        or _FRONTMATTER_KEY.match(line)
                        or _FRONTMATTER_CONT.match(line),
                        f"src/agents/{name}:{lineno}: frontmatter line is "
                        f"neither a key line, an indented continuation/list "
                        f"line, nor blank: {line!r}",
                    )
                for key in required:
                    self.assertIn(
                        key, keys,
                        f"src/agents/{name}: frontmatter missing required "
                        f"key '{key}:'",
                    )

    def test_agent_name_matches_filename_stem(self):
        """Drift class (a): frontmatter ``name:`` equals the filename stem."""
        self.assertTrue(
            self.agent_frontmatter, "no agent frontmatter gathered (scope bug)"
        )
        for name, (_block, keys) in self.agent_frontmatter.items():
            with self.subTest(agent=name):
                self.assertEqual(
                    keys.get("name"), Path(name).stem,
                    f"src/agents/{name}: frontmatter name "
                    f"{keys.get('name')!r} != filename stem "
                    f"{Path(name).stem!r} (copy-paste-from-template drift)",
                )

    def test_orchestrator_grant_matches_agent_files(self):
        """Drift class (b): the ``Agent(...)`` grant in ed-orchestrator.md's
        ``tools:`` equals the on-disk agent-file set (bidirectional)."""
        grant = self._grant_set()
        self.assertTrue(
            grant,
            "no Agent(...) grant extracted from ed-orchestrator.md "
            "frontmatter 'tools:' line (extraction surface moved?)",
        )
        files = {Path(n).stem for n in self.agent_frontmatter} - {
            "ed-orchestrator"
        }
        self.assertEqual(
            grant, files,
            f"ed-orchestrator.md Agent(...) grant != src/agents/*.md file set. "
            f"granted-but-missing-file: {sorted(grant - files)}; "
            f"file-but-not-granted: {sorted(files - grant)}",
        )

    def test_skill_md_dispatch_list_matches_grant(self):
        """Drift class (d): SKILL.md's 'phase specialists (...)' dispatch list
        equals the orchestrator ``Agent(...)`` grant set."""
        text = SKILL_MD.read_text(encoding="utf-8")
        match = re.search(r"phase specialists \((.*?)\)", text, re.DOTALL)
        self.assertTrue(
            match,
            "'phase specialists (' dispatch marker not found in src/SKILL.md "
            "(Orchestrator Role Assumption prose reworded? update marker)",
        )
        dispatch = set(re.findall(r"ed-[\w-]+", match.group(1)))
        self.assertTrue(
            dispatch,
            "no ed-* tokens extracted from the SKILL.md dispatch parenthetical",
        )
        self.assertEqual(
            dispatch, self._grant_set(),
            f"SKILL.md dispatch list != ed-orchestrator.md Agent(...) grant. "
            f"dispatch-only: {sorted(dispatch - self._grant_set())}; "
            f"grant-only: {sorted(self._grant_set() - dispatch)}",
        )

    def test_claude_md_agent_annotations_match_frontmatter(self):
        """Drift class (c): each agent's CLAUDE.md tree-line comment carries
        the frontmatter ``model:`` value; if frontmatter has
        ``background: true``, the comment mentions 'background'
        (one-directional -- absent/false asserts nothing)."""
        self.assertTrue(
            self.agent_frontmatter, "no agent frontmatter gathered (scope bug)"
        )
        tree_lines = self.claude_md_tree.splitlines()
        for name, (_block, keys) in self.agent_frontmatter.items():
            with self.subTest(agent=name):
                matches = [ln for ln in tree_lines if name in ln]
                self.assertEqual(
                    len(matches), 1,
                    f"expected exactly one CLAUDE.md tree line containing "
                    f"{name}, found {len(matches)}",
                )
                line = matches[0]
                model = keys.get("model", "")
                self.assertTrue(
                    model, f"src/agents/{name}: no model: in frontmatter"
                )
                self.assertIn(
                    model, line,
                    f"CLAUDE.md tree annotation for {name} does not mention "
                    f"frontmatter model {model!r}: {line.strip()!r}",
                )
                if keys.get("background") == "true":
                    self.assertIn(
                        "background", line,
                        f"src/agents/{name} declares background: true but the "
                        f"CLAUDE.md tree annotation does not say 'background': "
                        f"{line.strip()!r}",
                    )


if __name__ == "__main__":
    unittest.main()
