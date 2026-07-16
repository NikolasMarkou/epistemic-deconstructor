"""Test-count guardrail (F2).

CI runs only `pytest -q`, so a guardrail must itself be a pytest test to be
enforced (LESSONS.md). This test makes the suite size an explicit invariant:
if a test is added or silently deleted without updating EXPECTED_COUNT, CI goes
red. It counts ``def test_`` definitions across ``tests/test_*.py`` with a pure
filesystem read -- NO ``pytest --collect-only`` subprocess (which would recurse
into a child pytest session). The regex count and ``pytest --collect-only`` were
verified equal at this file's introduction and re-verified at each count change.

When you intentionally change the test count:
  1. Update EXPECTED_COUNT below to the new number.
  2. Update the live count claims: Makefile (test echo), CLAUDE.md tree comment,
     README.md (badge + two prose mentions).
  3. Leave historical records (CHANGELOG.md, release-notes.md) untouched.
"""
import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = REPO_ROOT / "tests"

# Live test count. Includes this file's own test. Keep in sync with the live
# count claims (see module docstring).
# DECISION plan_2026-06-02_f07c6077/D-002
EXPECTED_COUNT = 880

_TEST_DEF = re.compile(r"^\s*def (test_\w+)", re.MULTILINE)


def _count_tests() -> int:
    total = 0
    for path in sorted(TESTS_DIR.glob("test_*.py")):
        total += len(_TEST_DEF.findall(path.read_text(encoding="utf-8")))
    return total


class TestTestCount(unittest.TestCase):
    def test_expected_test_count(self):
        found = _count_tests()
        self.assertEqual(
            found,
            EXPECTED_COUNT,
            f"Test count changed: found {found}, expected {EXPECTED_COUNT}. "
            f"If intentional, update EXPECTED_COUNT in {Path(__file__).name} AND the "
            f"live count claims (Makefile test echo, CLAUDE.md tree comment, "
            f"README.md badge + prose). A silent test deletion is the failure this guards.",
        )


if __name__ == "__main__":
    unittest.main()
