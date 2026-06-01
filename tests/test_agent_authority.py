#!/usr/bin/env python3
"""CI-enforced agent-authority lint (RC5 prevention, D-005 / plan_2026-06-01_cf95b3e5/D-004).

Invariant: only ``ed-orchestrator.md`` (the FSM owner) may GRANT session-MUTATING
``session_manager.py`` commands. Every other ``src/agents/*.md`` file must NOT
present a mutating command as an executable/granted operation unless the same file
also carries an accompanying prohibition stanza.

Root cause this lint guards (D-005): ``ed-session-clerk.md`` listed ``$SM new --force``
and ``$SM skip`` in its Operations command table without any prohibition, and
``ed-domain-orienter.md`` instructed itself to "invoke ``$SM skip 0.3``" — both
contradicting their own Refusal Protocols. A clerk acting on that grant ran
``$SM new --force`` and split-brained a session during the audit.

Discrimination rule (the hard part):
  The 13 already-clean per-phase agents mention ``$SM advance`` ONLY inside
  refusal/redirect prose ("redirect to ``$SM advance``", "Direct them to
  ``$SM advance``"). Those are NOT grants and MUST PASS. A line is treated as a
  GRANT only when the mutating command appears either:
    (a) as a Markdown table row whose FIRST cell is the command
        (``| `$SM skip ...` | ... |``), i.e. an Operations/command-reference row, OR
    (b) in an imperative instruction ("invoke ``$SM skip``", "call ``$SM skip``",
        "run ``$SM skip``", "use ``$SM skip``") that is NOT itself negated
        ("do NOT call", "NEVER invoke", "never call", "MUST NOT", "should NOT").
  A file PASSES if it contains zero such grants, OR (defence in depth) if it
  contains a prohibition sentence ("NEVER" / "MUST NOT" / "do NOT" near a mutating
  command) that licenses an otherwise-flagged line — but the clean agents need no
  such license because they have zero grants.

Self-check note: this test is its OWN fixture. After Step 4's edits, the live tree
MUST yield exactly 0 violations across all 14 non-orchestrator agents. If this test
fails on any of the 13 historically-clean agents, the grant-vs-prohibition
discrimination below is WRONG (refusal text misread as a grant) — fix the heuristic,
do not edit the agents (see plan Pre-Mortem STOP-IF).

House style: unittest.TestCase, pathlib, no mocks, reads files off disk.
"""

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
AGENTS_DIR = REPO_ROOT / "src" / "agents"

# The FSM owner — exempt; it is allowed to grant every transition command.
ORCHESTRATOR = "ed-orchestrator.md"

# Session-MUTATING commands. ``$SM new`` (without --force), ``$SM close``,
# ``$SM declare``, ``$SM reopen``, ``$SM write``, ``$SM read`` are NOT here:
#   - new (no --force) / close / reopen are legitimate delegated clerk ops,
#   - write/read/declare do not mutate the ## Phase: FSM cursor.
# These five DO mutate FSM authority and are orchestrator-only:
MUTATING = [
    r"\$SM\s+new\s+--force",
    r"\$SM\s+skip\b",
    r"\$SM\s+advance\b",
    r"\$SM\s+close\b",          # listed for completeness; see CLERK_ALLOWED below
    r"\$SM\s+set-phase\b",
]
MUTATING_RX = re.compile("|".join(MUTATING))

# $SM close is a legitimate clerk-delegated operation (it merges to consolidated
# files; it is the terminal of a session the orchestrator already decided to end).
# It is NOT an FSM-cursor mutation in the way skip/advance/new --force/set-phase are,
# and ed-session-clerk.md grants it intentionally. So a bare `$SM close` table row
# is permitted. We therefore exclude `$SM close` from the GRANT scan; the other four
# are the true orchestrator-only transition verbs this lint enforces.
GRANT_SCAN = [
    r"\$SM\s+new\s+--force",
    r"\$SM\s+skip\b",
    r"\$SM\s+advance\b",
    r"\$SM\s+set-phase\b",
]
GRANT_SCAN_RX = re.compile("|".join(GRANT_SCAN))

# Imperative grant verbs directly adjacent to a backtick-wrapped mutating command:
# "<verb> `$SM <mut>...`". The verb must be within a short window of the opening
# backtick (<=24 chars, room for "the"/"a"/quoting) so a descriptive sentence that
# merely happens to contain "run"/"use" elsewhere on the line (e.g. "Morris if run,
# ... `$SM advance` exits 1") is NOT misread as an imperative grant.
IMPERATIVE_RX = re.compile(
    r"\b(invoke|call|run|use|issue|execute)\b[^`\n]{0,24}`\s*(?:" + "|".join(GRANT_SCAN) + r")",
    re.IGNORECASE,
)

# Negation markers — when present on the SAME line as the command, the line is a
# prohibition/refusal, not a grant.
NEGATION_RX = re.compile(
    r"\b(never|must\s+not|do\s+not|don't|cannot|no\s+authority|not\s+call|"
    r"refuse|redirect|owned\s+by|orchestrator(?:'s)?\s+(?:authority|responsibility))\b",
    re.IGNORECASE,
)

# A table-row grant: a Markdown table line whose FIRST non-empty cell starts with
# the mutating command (e.g. `| `$SM skip <phase>` | ... |`).
TABLE_ROW_RX = re.compile(
    r"^\s*\|\s*`?\s*(?:" + "|".join(GRANT_SCAN) + r")",
)

# A file-level prohibition stanza licenses otherwise-ambiguous lines.
def has_prohibition(text: str) -> bool:
    for line in text.splitlines():
        if NEGATION_RX.search(line) and GRANT_SCAN_RX.search(line):
            return True
    return False


def find_grants(text: str):
    """Return list of (lineno, line) that grant a mutating command."""
    grants = []
    for i, line in enumerate(text.splitlines(), start=1):
        if not GRANT_SCAN_RX.search(line):
            continue
        if NEGATION_RX.search(line):
            # refusal / redirect / prohibition prose — not a grant
            continue
        is_table_row = bool(TABLE_ROW_RX.search(line))
        is_imperative = bool(IMPERATIVE_RX.search(line))
        if is_table_row or is_imperative:
            grants.append((i, line.strip()))
    return grants


class TestAgentAuthority(unittest.TestCase):
    def _agent_files(self):
        files = sorted(AGENTS_DIR.glob("*.md"))
        self.assertTrue(files, f"no agent files found under {AGENTS_DIR}")
        return files

    def test_agents_dir_exists(self):
        self.assertTrue(AGENTS_DIR.is_dir(), f"missing {AGENTS_DIR}")

    def test_orchestrator_present_and_exempt(self):
        self.assertTrue(
            (AGENTS_DIR / ORCHESTRATOR).is_file(),
            "ed-orchestrator.md must exist (it is the exempt FSM owner)",
        )

    def test_no_unprohibited_fsm_grants(self):
        violations = {}
        for f in self._agent_files():
            if f.name == ORCHESTRATOR:
                continue  # FSM owner is allowed to grant every transition command
            text = f.read_text(encoding="utf-8")
            grants = find_grants(text)
            if grants and not has_prohibition(text):
                violations[f.name] = grants
        self.assertEqual(
            violations,
            {},
            "FSM-mutating command granted without a prohibition stanza in: "
            + "; ".join(
                f"{name} -> "
                + " / ".join(f"L{ln}: {txt}" for ln, txt in rows)
                for name, rows in violations.items()
            ),
        )

    def test_clerk_table_drops_force_and_skip(self):
        """Regression: ed-session-clerk Operations table must not GRANT new --force / skip."""
        text = (AGENTS_DIR / "ed-session-clerk.md").read_text(encoding="utf-8")
        for ln in text.splitlines():
            if TABLE_ROW_RX.search(ln):
                self.fail(f"ed-session-clerk.md still grants a mutating command in a table row: {ln.strip()}")

    def test_clerk_has_prohibition(self):
        text = (AGENTS_DIR / "ed-session-clerk.md").read_text(encoding="utf-8")
        self.assertTrue(
            has_prohibition(text),
            "ed-session-clerk.md must carry an FSM-mutation prohibition stanza",
        )

    def test_domain_orienter_does_not_self_invoke_skip(self):
        text = (AGENTS_DIR / "ed-domain-orienter.md").read_text(encoding="utf-8")
        grants = find_grants(text)
        self.assertEqual(
            grants,
            [],
            "ed-domain-orienter.md must not instruct itself to invoke $SM skip: "
            + "; ".join(f"L{ln}: {txt}" for ln, txt in grants),
        )


if __name__ == "__main__":
    unittest.main()
