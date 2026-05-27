#!/usr/bin/env python3
"""
Session manager for Epistemic Deconstructor analysis sessions.

Creates and manages analysis session directories under analyses/.
Persists analysis state to filesystem so context window loss doesn't
destroy session progress.

Usage:
    python session_manager.py --base-dir /path/to/project new "System description"
    python session_manager.py --base-dir /path/to/project resume
    python session_manager.py --base-dir /path/to/project status
    python session_manager.py --base-dir /path/to/project close
    python session_manager.py --base-dir /path/to/project list

The --base-dir flag sets where analyses/ is created (default: current directory).
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone

# Used by cmd_write to route JSON session files through common.save_json so
# concurrent writers (e.g. bayesian_tracker.py) don't clobber each other via
# the sidecar .lock. Markdown/text files keep plain atomic-write semantics.
try:
    from common import save_json as _common_save_json
except ImportError:  # pragma: no cover — co-located module, should always import
    _common_save_json = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ANALYSES_DIR = "analyses"
POINTER_FILE = os.path.join(ANALYSES_DIR, ".current_analysis")
CONSOLIDATED_FINDINGS = os.path.join(ANALYSES_DIR, "FINDINGS.md")
CONSOLIDATED_DECISIONS = os.path.join(ANALYSES_DIR, "DECISIONS.md")
MAX_REOPENS = 3  # Max times a single phase can be reopened (total passes = MAX_REOPENS + 1)

# ---------------------------------------------------------------------------
# Phase FSM canonical data
# ---------------------------------------------------------------------------
# DECISION plan_2026-05-19_4fc8ec9a/D-001:
# Canonical per-tier phase ordering. The Phase: field in state.md can only
# advance from cur → PHASE_SEQUENCE[tier][cur]. The legitimate fast-path is
# choosing RAPID at session start (not skipping phases mid-session). PSYCH
# phase IDs use the `-P` suffix and have their own sequence. SKIPPABLE
# defines, per tier, the set of phases that may be passed over via
# `$SM skip <phase> "<reason>"` — anything else is refused.
# REQUIRED_ARTIFACTS lists the phase_outputs/*.md filenames that must exist
# before a phase can be marked completed via `advance`.
# Do NOT modify these dicts without updating the SKILL.md FSM and the
# corresponding tests in tests/test_session_manager.py.

PHASE_SEQUENCE = {
    "RAPID": {
        "0": "0.5",  # tolerate sessions that started at Phase 0 before tier was set
        "0.5": "5",
        "5": None,
    },
    "LITE": {
        "0": "0.3",
        "0.3": "1",
        "1": "1.5",
        "1.5": "5",
        "5": None,
    },
    "STANDARD": {
        "0": "0.3",
        "0.3": "0.7",
        "0.7": "1",
        "1": "1.5",
        "1.5": "2",
        "2": "3",
        "3": "4",
        "4": "5",
        "5": None,
    },
    "COMPREHENSIVE": {
        "0": "0.3",
        "0.3": "0.7",
        "0.7": "1",
        "1": "1.5",
        "1.5": "2",
        "2": "3",
        "3": "4",
        "4": "5",
        "5": None,
    },
    # DECISION plan_2026-05-25_cdd1f345/D-001: PSYCH FSM includes 1-P.5
    # (abductive expansion) between 1-P and 2-P, matching psych-profiler.md:6-7
    # which states the three pluggable sub-phases (0-P.3, 0-P.7, 1-P.5) are
    # orchestrator-dispatched to domain-orienter, scope-auditor, and
    # abductive-engine respectively. Sibling dicts REQUIRED_ARTIFACTS,
    # PHASE_FILENAME_MAP, PHASE_GATE_SCRIPTS updated in lockstep below.
    "PSYCH": {
        "0-P": "0-P.3",
        "0-P.3": "0-P.7",
        "0-P.7": "1-P",
        "1-P": "1-P.5",
        "1-P.5": "2-P",
        "2-P": "3-P",
        "3-P": "4-P",
        "4-P": "5-P",
        "5-P": None,
    },
}

# Whitelisted phases that may be skipped per tier. Phase 0.3 is legitimately
# skippable when domain_familiarity == high. Phase 0-P.3 mirrors that on PSYCH.
SKIPPABLE = {
    "RAPID": set(),
    "LITE": {"0.3"},
    "STANDARD": {"0.3"},
    "COMPREHENSIVE": {"0.3"},
    "PSYCH": {"0-P.3"},
}

# Per-phase required artifact files (relative to <session_dir>/phase_outputs/).
# `advance` refuses to leave a phase unless its required artifacts are present.
REQUIRED_ARTIFACTS = {
    "0": ["phase_0.md"],
    "0.3": ["phase_0_3.md"],
    "0.5": ["phase_0_5.md"],
    "0.7": [],  # Phase 0.7 is scope-interrogation; gate-script enforces structure.
    "1": ["phase_1.md"],
    "1.5": [],  # Phase 1.5 artifacts checked by abductive_engine gate.
    "2": ["phase_2.md"],
    "3": ["phase_3.md"],
    "4": ["phase_4.md"],
    "5": ["phase_5.md"],
    "0-P": ["phase_0_P.md"],
    "0-P.3": ["phase_0_3.md"],  # Phase 0.3 artifact is shared (domain_orienter output).
    "0-P.7": [],
    "1-P": ["phase_1_P.md"],
    "1-P.5": [],  # Phase 1-P.5 artifacts checked by abductive_engine gate (mirrors 1.5).
    "2-P": ["phase_2_P.md"],
    "3-P": ["phase_3_P.md"],
    "4-P": ["phase_4_P.md"],
    "5-P": ["phase_5_P.md"],
}

# Per-phase exit-gate script invocations. Each entry is
# (script_module, subcommand-args). The script path is resolved against
# the directory containing session_manager.py (the canonical install
# location). The session JSON is passed via --file <abs_path>. Phases
# without a gate script receive None → treated as no-op (returns PASS).
PHASE_GATE_SCRIPTS = {
    "0.3": ("domain_orienter.py", "domain_orientation.json"),
    "0.7": ("scope_auditor.py", "scope_audit.json"),
    "1.5": ("abductive_engine.py", "abductive_state.json"),
    # DECISION plan_2026-05-25_cdd1f345/D-001: PSYCH tier sub-phases reuse the
    # same gate scripts as non-PSYCH per psych-profiler.md:6-7. Tier-agnostic
    # by design — gate criteria do not differ between PSYCH and STANDARD for
    # these three sub-phases.
    "0-P.3": ("domain_orienter.py", "domain_orientation.json"),
    "0-P.7": ("scope_auditor.py", "scope_audit.json"),
    "1-P.5": ("abductive_engine.py", "abductive_state.json"),
}

SKILL_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PHASE_FILENAME_MAP = {
    "0": "phase_0.md", "0.3": "phase_0_3.md", "0.5": "phase_0_5.md",
    "0.7": "phase_0_7.md",
    "1": "phase_1.md", "1.5": "phase_1_5.md", "2": "phase_2.md",
    "3": "phase_3.md", "4": "phase_4.md", "5": "phase_5.md",
    "0-P": "phase_0_P.md", "0-P.3": "phase_0_3.md", "0-P.7": "phase_0_7.md",
    "1-P": "phase_1_P.md", "1-P.5": "phase_1_5.md",
    "2-P": "phase_2_P.md", "3-P": "phase_3_P.md",
    "4-P": "phase_4_P.md", "5-P": "phase_5_P.md",
}


def ensure_gitignore():
    """Ensure analyses/ is in .gitignore."""
    gitignore = ".gitignore"
    pattern = "analyses/"
    content = ""
    try:
        with open(gitignore, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        pass
    if any(line.strip() == pattern for line in content.split("\n")):
        return
    suffix = ("" if content.endswith("\n") or not content else "\n") + pattern + "\n"
    _atomic_write(gitignore, content + suffix)


def read_pointer():
    """Read the active analysis directory from pointer file.

    Handles both old format (name only, e.g. 'analysis_2026-02-20_abc123')
    and new format (absolute path). Returns the absolute path to the
    analysis directory, or None if no active analysis.
    """
    try:
        with open(POINTER_FILE, "r", encoding="utf-8") as f:
            value = f.read().strip()
        if not value:
            return None

        # New format: absolute path
        if os.path.isabs(value):
            if os.path.isdir(value):
                return value
            print(f"Warning: Active session directory no longer exists: {value}",
                  file=sys.stderr)
            return None

        # Old format: directory name relative to ANALYSES_DIR
        rel_path = os.path.join(ANALYSES_DIR, value)
        if os.path.isdir(rel_path):
            return os.path.abspath(rel_path)
        print(f"Warning: Active session directory no longer exists: {rel_path}",
              file=sys.stderr)
    except FileNotFoundError:
        pass
    return None


def read_analysis_file(analysis_dir, filename):
    """Read a file from an analysis directory.

    Args:
        analysis_dir: Absolute path to the analysis directory.
        filename: Name of the file to read.
    """
    try:
        with open(os.path.join(analysis_dir, filename), "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return None


def extract_field(content, pattern):
    """Extract a field value from markdown content."""
    if not content:
        return None
    match = re.search(pattern, content, re.MULTILINE)
    return match.group(1).strip() if match else None


def _atomic_write(filepath, content):
    """Write content atomically via unique tmp + rename.

    DECISION plan_2026-05-25_cdd1f345/D-003: uses tempfile.mkstemp to
    eliminate the named-collision race that previously existed when two
    concurrent writers picked the same `<path>.tmp` suffix. mkstemp creates
    a uniquely-named tmp in the SAME directory (so os.replace stays
    same-filesystem and atomic). No lock is added — session markdown files
    have no read-modify-write semantics, so any rename "winner" is a valid
    final state. For multi-step JSON I/O with locking semantics, see
    common.save_json().
    """
    import tempfile
    parent_dir = os.path.dirname(os.path.abspath(filepath)) or "."
    base_name = os.path.basename(filepath)
    fd, tmp = tempfile.mkstemp(
        prefix="." + base_name + ".", suffix=".tmp", dir=parent_dir,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp, filepath)
    except Exception:
        # Clean up tmp on any failure; suppress secondary errors during cleanup.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def ensure_consolidated_files():
    """Create consolidated findings/decisions files if they don't exist."""
    if not os.path.exists(CONSOLIDATED_FINDINGS):
        with open(CONSOLIDATED_FINDINGS, "w", encoding="utf-8") as f:
            f.write("# Consolidated Findings\n"
                    "*Cross-analysis findings archive. Merged on close. Newest first.*\n")
    if not os.path.exists(CONSOLIDATED_DECISIONS):
        with open(CONSOLIDATED_DECISIONS, "w", encoding="utf-8") as f:
            f.write("# Consolidated Decisions\n"
                    "*Cross-analysis decision archive. Merged on close. Newest first.*\n")


def strip_header(content):
    """Strip everything before the first ## heading."""
    match = re.search(r'^## ', content, re.MULTILINE)
    return content[match.start():] if match else content


def prepend_to_consolidated(filepath, analysis_dir_name, new_section):
    """Insert new section after header, before existing sections (newest first)."""
    existing = ""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            existing = f.read()
    except FileNotFoundError:
        pass
    idx = existing.find("\n## ")
    if idx >= 0:
        header = existing[:idx]
        body = existing[idx:]
    else:
        header = existing.rstrip()
        body = ""
    merged = header + f"\n\n## {analysis_dir_name}\n{new_section}\n" + body
    _atomic_write(filepath, merged)


def merge_to_consolidated(abs_dir):
    """Merge per-analysis findings/decisions to consolidated files."""
    name = os.path.basename(abs_dir)
    # Findings
    findings = read_analysis_file(abs_dir, "observations.md")
    if findings:
        stripped = strip_header(findings)
        stripped = re.sub(r'^## ', '### ', stripped, flags=re.MULTILINE)
        stripped = stripped.replace("(observations/", f"({name}/observations/")
        stripped = stripped.strip()
        if stripped:
            prepend_to_consolidated(CONSOLIDATED_FINDINGS, name, stripped)

    # Decisions
    decisions = read_analysis_file(abs_dir, "decisions.md")
    if decisions:
        stripped = strip_header(decisions)
        stripped = re.sub(r'^## ', '### ', stripped, flags=re.MULTILINE)
        stripped = stripped.strip()
        if stripped:
            prepend_to_consolidated(CONSOLIDATED_DECISIONS, name, stripped)


# ---------------------------------------------------------------------------
# FSM helpers (used by advance / gate-check / set-phase / skip / write)
# ---------------------------------------------------------------------------

def _current_phase(abs_dir):
    """Read the current Phase: field from state.md. Returns string or None."""
    state = read_analysis_file(abs_dir, "state.md")
    return extract_field(state, r'^## Phase:\s*(.+)$')


# DECISION plan_2026-05-25_c0b0049a/D-001:
# Phase 0.3 / 0-P.3 skip is gated by `domain_familiarity: high` per the
# SKILL.md trigger paragraph and references/domain-orientation.md. Before
# this helper, `cmd_skip` honored only the tier whitelist (D-003); a
# session with `domain_familiarity: low` could still skip 0.3 with any
# rationale string. The familiarity check makes the skip refuse unless
# the analyst has affirmatively declared `high`. Parser is case-insensitive
# and lenient about leading/trailing whitespace; missing / placeholder /
# anything-but-'high' returns a non-'high' value, which the caller treats
# as refusal grounds.
def _read_domain_familiarity(abs_dir):
    """Parse `domain_familiarity: <value>` from analysis_plan.md.

    Returns one of 'high' / 'medium' / 'low' / 'unknown' (normalized lowercase),
    or None if not declared. Matches the field whether it appears under a
    `## Domain Familiarity` heading or as a top-level `domain_familiarity: X`
    line, anywhere in the plan. Placeholder values like `*(to be filled)*` and
    `(pending)` return None.
    """
    plan = read_analysis_file(abs_dir, "analysis_plan.md")
    if not plan:
        return None
    m = re.search(
        r'^\s*domain_familiarity\s*:\s*([A-Za-z]+)\s*$',
        plan,
        re.MULTILINE | re.IGNORECASE,
    )
    if not m:
        return None
    value = m.group(1).strip().lower()
    if value in {"high", "medium", "low", "unknown"}:
        return value
    return None


def _current_tier(abs_dir):
    """Read tier from state.md `## Tier:` field; fall back to analysis_plan.md.

    Returns the tier string (e.g. 'STANDARD') or None if not declared.
    A literal '(pending)' / '(none)' value is treated as not-declared.
    """
    state = read_analysis_file(abs_dir, "state.md")
    tier = extract_field(state, r'^## Tier:\s*(.+)$')
    if tier and tier.strip().lower() not in ("(pending)", "(none)", "?", "unknown", ""):
        return tier.strip().upper()
    plan = read_analysis_file(abs_dir, "analysis_plan.md")
    tier = extract_field(plan, r'^## Tier Selected\s*\n(.+)$')
    if not tier:
        # Tier Selected section header followed by a value on the next non-blank line
        if plan:
            m = re.search(r'^## Tier Selected\s*$', plan, re.MULTILINE)
            if m:
                tail = plan[m.end():].strip().split("\n", 1)[0].strip()
                # Filter placeholders
                if tail.startswith("*") or tail.startswith("("):
                    return None
                tier = tail
    if tier:
        t = tier.strip().upper()
        if t in PHASE_SEQUENCE:
            return t
    return None


def _required_artifacts_present(abs_dir, phase):
    """Return (ok, missing_files) for the phase's required artifacts."""
    required = REQUIRED_ARTIFACTS.get(phase, [])
    missing = []
    phase_dir = os.path.join(abs_dir, "phase_outputs")
    for name in required:
        if not os.path.exists(os.path.join(phase_dir, name)):
            missing.append(name)
    return (len(missing) == 0, missing)


def _run_phase_gate(abs_dir, phase, timeout=30):
    """Run the per-phase exit gate script. Returns (passed, message).

    No gate script for the phase → returns (True, "no gate"). Subprocess
    errors (missing script, timeout) → returns (False, <error description>).
    """
    spec = PHASE_GATE_SCRIPTS.get(phase)
    if spec is None:
        return (True, "no gate")
    script_name, default_json = spec
    script_path = os.path.join(SKILL_SCRIPTS_DIR, script_name)
    if not os.path.exists(script_path):
        return (False, f"gate script unavailable: {script_path}")
    json_path = os.path.join(abs_dir, default_json)
    import subprocess
    try:
        result = subprocess.run(
            [sys.executable, script_path, "--file", json_path, "gate"],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return (False, f"gate script timed out after {timeout}s")
    except Exception as e:
        return (False, f"gate script error: {e}")
    if result.returncode == 0:
        return (True, "gate PASS")
    msg = (result.stderr or result.stdout or "").strip().splitlines()
    tail = msg[-3:] if msg else ["(no output)"]
    return (False, f"gate FAIL (exit {result.returncode}): " + " | ".join(tail))


def _state_md_phase_value(content):
    """Extract Phase: value from a raw state.md content string."""
    if not content:
        return None
    m = re.search(r'^## Phase:\s*(.+)$', content, re.MULTILINE)
    return m.group(1).strip() if m else None


def _append_state_transition(abs_dir, new_phase, kind, reason, ts):
    """Update state.md: rewrite Phase: + Last Transition + history line.

    `kind` is a short tag (ADVANCE / ADMIN-OVERRIDE / SKIP / ...). `reason`
    is appended to the history bullet.
    """
    state = read_analysis_file(abs_dir, "state.md") or ""
    state = re.sub(r'^## Phase:\s*.*$', f'## Phase: {new_phase}',
                   state, flags=re.MULTILINE)
    state = re.sub(r'^## Last Transition:\s*.*$',
                   f'## Last Transition: {kind} → Phase {new_phase} ({ts})',
                   state, flags=re.MULTILINE)
    tail = f"\n- {kind} → Phase {new_phase}: {reason} ({ts})\n"
    state = state.rstrip() + tail
    _atomic_write(os.path.join(abs_dir, "state.md"), state)


def _append_decisions(abs_dir, entry):
    """Append a markdown entry to decisions.md, creating the file if needed."""
    decisions_path = os.path.join(abs_dir, "decisions.md")
    try:
        with open(decisions_path, "r", encoding="utf-8") as f:
            decisions = f.read()
    except FileNotFoundError:
        decisions = "# Decisions\n"
    if decisions and not decisions.endswith("\n"):
        decisions += "\n"
    _atomic_write(decisions_path, decisions + entry)


# ---------------------------------------------------------------------------
# Session templates
# ---------------------------------------------------------------------------

def make_state(goal, timestamp, tier=None):
    tier_line = tier.strip().upper() if tier else "(pending)"
    return f"""# Current State
## Phase: 0
## Tier: {tier_line}
## Fidelity Target: (pending)
## System: {goal}
## Active Hypotheses: 0
## Lead Hypothesis: (none)
## Confidence: Low
## Last Transition: INIT → Phase 0 ({timestamp})
## Transition History:
- INIT → Phase 0 (session started)
"""


def make_analysis_plan(tier=None, familiarity=None):
    tier_block = (tier.strip().upper() if tier
                  else "*(RAPID / LITE / STANDARD / COMPREHENSIVE / PSYCH)*")
    fam_value = familiarity.strip().lower() if familiarity else None
    fam_line = (f"domain_familiarity: {fam_value}" if fam_value
                else "domain_familiarity: (declare with `$SM declare --domain-familiarity <high|medium|low|unknown>`)")
    return f"""# Analysis Plan
*Written during Phase 0 (Setup & Frame). This is the persistent record of the analysis setup.*

## System Description
*(to be filled)*

## Access Level
*(full source / binary only / black-box I/O)*

## Adversary Status
*(yes / no / unknown)*

## Tier Selected
{tier_block}

## Domain Familiarity
{fam_line}

## Fidelity Target
*(L1-L5)*

## Question Pyramid
*(L1 through target level)*

## Initial Hypotheses
*(H1: likely, H2: alternative, H3: adversarial/deceptive)*

## Adversarial Pre-check
*(result)*

## Cognitive Vulnerabilities Acknowledged
*(list)*
"""


def make_decisions(has_consolidated):
    note = ("\n*Cross-analysis context: see analyses/FINDINGS.md and analyses/DECISIONS.md*\n"
            if has_consolidated else "")
    return f"# Decision Log\n*Append-only. Never edit past entries.*\n{note}"


def make_observations(has_consolidated):
    note = ("\n*Cross-analysis context: see analyses/FINDINGS.md and analyses/DECISIONS.md*\n"
            if has_consolidated else "")
    return f"""# Observations
*Index of all observations. Detailed files go in observations/ directory.*
{note}
## Index
*(to be populated during analysis)*

## Key Constraints
*(to be populated during analysis)*
"""


def make_progress():
    return """# Progress

## Completed
*(nothing yet)*

## In Progress
- [ ] Phase 0: Setup & Frame

## Remaining
*(populated after Phase 0)*

## Blocked
*(nothing currently)*
"""


def make_validation():
    return """# Validation Results
*Populated during Phase 5. Records validation hierarchy, residual diagnostics, baseline comparison.*

## Validation Hierarchy
| # | Check | Method | Result | Evidence |
|---|-------|--------|--------|----------|
| *(to be populated during Phase 5)* |||||

## Verdict
*(to be completed during Phase 5)*
"""


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_new(args):
    """Create a new analysis session."""
    goal = " ".join(args.goal).strip()
    if not goal:
        print("ERROR: Goal is required (must not be whitespace-only).", file=sys.stderr)
        sys.exit(1)

    os.makedirs(ANALYSES_DIR, exist_ok=True)

    existing = read_pointer()
    if existing and not args.force:
        print(f"ERROR: Active analysis already exists: {existing}", file=sys.stderr)
        print(f"  To resume:    python {sys.argv[0]} resume", file=sys.stderr)
        print(f"  To close it:  python {sys.argv[0]} close", file=sys.stderr)
        print(f"  To force new: python {sys.argv[0]} new --force \"goal\"", file=sys.stderr)
        sys.exit(1)
    if existing and args.force:
        cmd_close_impl(silent=True)

    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    date_str = now.strftime("%Y-%m-%d")
    hex_str = os.urandom(4).hex()
    analysis_dir_name = f"analysis_{date_str}_{hex_str}"
    analysis_dir = os.path.join(ANALYSES_DIR, analysis_dir_name)

    has_consolidated = (os.path.exists(CONSOLIDATED_FINDINGS)
                        or os.path.exists(CONSOLIDATED_DECISIONS))

    try:
        os.makedirs(os.path.join(analysis_dir, "observations"), exist_ok=True)
        os.makedirs(os.path.join(analysis_dir, "phase_outputs"), exist_ok=True)

        tier_arg = getattr(args, "tier", None)
        familiarity_arg = getattr(args, "domain_familiarity", None)
        files = {
            "state.md": make_state(goal, timestamp, tier=tier_arg),
            "analysis_plan.md": make_analysis_plan(tier=tier_arg,
                                                  familiarity=familiarity_arg),
            "decisions.md": make_decisions(has_consolidated),
            "observations.md": make_observations(has_consolidated),
            "progress.md": make_progress(),
            "validation.md": make_validation(),
        }
        for name, content in files.items():
            with open(os.path.join(analysis_dir, name), "w", encoding="utf-8") as f:
                f.write(content)

        ensure_consolidated_files()
        abs_analysis_dir = os.path.abspath(analysis_dir)
        _atomic_write(POINTER_FILE, abs_analysis_dir)
        # Write .session_dir for easy agent path retrieval
        _atomic_write(os.path.join(ANALYSES_DIR, ".session_dir"), abs_analysis_dir)

    except Exception as e:
        # Cleanup on failure — remove directory, pointer file, and .session_dir
        import shutil
        try:
            shutil.rmtree(analysis_dir, ignore_errors=True)
        except Exception:
            pass
        for orphan in [POINTER_FILE, POINTER_FILE + ".tmp",
                       os.path.join(ANALYSES_DIR, ".session_dir"),
                       os.path.join(ANALYSES_DIR, ".session_dir.tmp")]:
            try:
                if os.path.exists(orphan):
                    os.unlink(orphan)
            except Exception:
                pass
        print(f"ERROR: Failed to create analysis session: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        ensure_gitignore()
    except Exception as e:
        print(f"WARNING: Session created but .gitignore update failed: {e}", file=sys.stderr)
        print("  Manually add analyses/ to .gitignore.", file=sys.stderr)

    abs_analysis_dir = os.path.abspath(analysis_dir)
    print(f"SESSION_DIR={abs_analysis_dir}")
    print(f"")
    print(f"^^^ CRITICAL: Use this absolute path for ALL file reads and writes in this session. ^^^")
    print(f"^^^ Every Read/Write/Edit call MUST use {abs_analysis_dir}/filename — never relative paths. ^^^")
    print(f"")
    print(f"  System: {goal}")
    print(f"  State: Phase 0 (Setup & Frame)")
    if tier_arg:
        print(f"  Tier:  {tier_arg.upper()} (declared at session start)")
    if familiarity_arg:
        print(f"  Domain familiarity: {familiarity_arg.lower()} (declared at session start)")
    print(f"  Pointer: analyses/.current_analysis → {abs_analysis_dir}")
    print(f"  Cross-analysis context: analyses/FINDINGS.md, analyses/DECISIONS.md")
    print(f"  Next: Fill analysis_plan.md, seed hypotheses, select tier.")
    print()
    print(f"  Session files (use absolute paths for ALL operations):")
    print(f"    {abs_analysis_dir}/state.md")
    print(f"    {abs_analysis_dir}/analysis_plan.md")
    print(f"    {abs_analysis_dir}/decisions.md")
    print(f"    {abs_analysis_dir}/observations.md")
    print(f"    {abs_analysis_dir}/progress.md")
    print(f"    {abs_analysis_dir}/phase_outputs/")
    print(f"    {abs_analysis_dir}/validation.md")
    print(f"    {abs_analysis_dir}/summary.md")
    print()
    print(f"  Tracker integration (pass --file to existing scripts):")
    print(f"    python3 scripts/bayesian_tracker.py --file {abs_analysis_dir}/hypotheses.json ...")
    print(f"    python3 scripts/belief_tracker.py --file {abs_analysis_dir}/beliefs.json ...")
    print(f"    python3 scripts/rapid_checker.py --file {abs_analysis_dir}/rapid_assessment.json ...")


def cmd_resume(args):
    """Output current analysis state for re-entry."""
    abs_dir = read_pointer()
    if not abs_dir:
        print("ERROR: No active analysis. Use `new` to create one.", file=sys.stderr)
        sys.exit(1)

    # Ensure .session_dir file is up to date
    _atomic_write(os.path.join(ANALYSES_DIR, ".session_dir"), abs_dir)

    name = os.path.basename(abs_dir)

    state = read_analysis_file(abs_dir, "state.md")
    plan = read_analysis_file(abs_dir, "analysis_plan.md")
    progress = read_analysis_file(abs_dir, "progress.md")
    decisions = read_analysis_file(abs_dir, "decisions.md")

    phase = extract_field(state, r'^## Phase:\s*(.+)$') or "?"
    tier = extract_field(state, r'^## Tier:\s*(.+)$') or "?"
    fidelity = extract_field(state, r'^## Fidelity Target:\s*(.+)$') or "?"
    system = extract_field(state, r'^## System:\s*(.+)$') or "?"
    hypotheses = extract_field(state, r'^## Active Hypotheses:\s*(.+)$') or "?"
    lead = extract_field(state, r'^## Lead Hypothesis:\s*(.+)$') or "?"
    confidence = extract_field(state, r'^## Confidence:\s*(.+)$') or "?"
    last_transition = extract_field(state, r'^## Last Transition:\s*(.+)$') or "?"

    print(f"SESSION_DIR={abs_dir}")
    print(f"")
    print(f"^^^ CRITICAL: Use this absolute path for ALL file reads and writes. Never use relative paths. ^^^")
    print(f"")
    print(f"  Phase:      {phase}")
    print(f"  Tier:       {tier}")
    print(f"  Fidelity:   {fidelity}")
    print(f"  System:     {system}")
    print(f"  Hypotheses: {hypotheses} (lead: {lead})")
    print(f"  Confidence: {confidence}")
    print(f"  Last:       {last_transition}")
    print()

    if progress:
        completed = len(re.findall(r'^- \[x\]', progress, re.MULTILINE))
        remaining = len(re.findall(r'^- \[ \]', progress, re.MULTILINE))
        print(f"  Progress:   {completed} done, {remaining} remaining")

    if decisions:
        decision_count = len(re.findall(r'^## D-\d+', decisions, re.MULTILINE))
        if decision_count > 0:
            print(f"  Decisions:  {decision_count} logged")

    # List phase outputs
    phase_dir = os.path.join(abs_dir, "phase_outputs")
    phase_files = []
    try:
        phase_files = sorted(f for f in os.listdir(phase_dir) if f.endswith(".md"))
    except FileNotFoundError:
        pass
    if phase_files:
        current_files = [f for f in phase_files if "_pass" not in f]
        pass_files = [f for f in phase_files if "_pass" in f]
        print(f"\n  Phase outputs ({len(current_files)}):")
        for pf in current_files:
            print(f"    {pf}")
        if pass_files:
            print(f"\n  Multi-pass archives ({len(pass_files)}):")
            for pf in pass_files:
                print(f"    {pf}")

    print()
    print(f"  Recovery files:")
    print(f"    state.md          → {abs_dir}/state.md")
    print(f"    analysis_plan.md  → {abs_dir}/analysis_plan.md")
    print(f"    decisions.md      → {abs_dir}/decisions.md")
    print(f"    observations.md   → {abs_dir}/observations.md")
    print(f"    progress.md       → {abs_dir}/progress.md")
    print(f"    phase_outputs/    → {abs_dir}/phase_outputs/")
    print()
    abs_consolidated = os.path.abspath(ANALYSES_DIR)
    print(f"  Consolidated context:")
    print(f"    {abs_consolidated}/FINDINGS.md  — cross-analysis findings archive")
    print(f"    {abs_consolidated}/DECISIONS.md — cross-analysis decision archive")


def cmd_status(args):
    """One-line state summary."""
    abs_dir = read_pointer()
    if not abs_dir:
        print("No active analysis.")
        return

    state = read_analysis_file(abs_dir, "state.md")
    phase = extract_field(state, r'^## Phase:\s*(.+)$') or "?"
    tier = extract_field(state, r'^## Tier:\s*(.+)$') or "?"
    system = extract_field(state, r'^## System:\s*(.+)$') or "?"
    hypotheses = extract_field(state, r'^## Active Hypotheses:\s*(.+)$') or "?"

    system_short = system[:60] if system else "?"
    print(f"[Phase {phase}] tier={tier} hypotheses={hypotheses} | {system_short} | {abs_dir}")


def cmd_close_impl(silent=False):
    """Close the active analysis (implementation)."""
    abs_dir = read_pointer()
    if not abs_dir:
        if not silent:
            print("ERROR: No active analysis to close.", file=sys.stderr)
            sys.exit(1)
        return

    try:
        ensure_consolidated_files()
        merge_to_consolidated(abs_dir)
    except Exception as e:
        if not silent:
            print(f"WARNING: Merge to consolidated files failed: {e}", file=sys.stderr)
            print(f"  Per-analysis files remain intact at {abs_dir}/", file=sys.stderr)

    try:
        os.unlink(POINTER_FILE)
    except FileNotFoundError:
        pass
    try:
        os.unlink(os.path.join(ANALYSES_DIR, ".session_dir"))
    except FileNotFoundError:
        pass

    if not silent:
        print(f"Closed analysis: {abs_dir}")
        print(f"  Pointer analyses/.current_analysis removed.")
        print(f"  Analysis directory preserved at {abs_dir}/")
        print(f"  Observations/decisions merged to analyses/FINDINGS.md and analyses/DECISIONS.md.")


def cmd_close(args):
    """Close the active analysis."""
    cmd_close_impl(silent=False)


def cmd_write(args):
    """Write content to a session file (reads from stdin)."""
    abs_dir = read_pointer()
    if not abs_dir:
        print("ERROR: No active analysis. Use `new` to create one.", file=sys.stderr)
        sys.exit(1)

    filename = args.filename
    # Security: prevent path traversal (check ".." as path component, not substring)
    parts = filename.replace("\\", "/").split("/")
    if any(part == ".." for part in parts) or filename.startswith("/"):
        print(f"ERROR: Invalid filename: {filename}", file=sys.stderr)
        sys.exit(1)

    filepath = os.path.join(abs_dir, filename)
    # Resolve symlinks and verify the target is still under abs_dir
    real_dir = os.path.realpath(abs_dir)
    real_path = os.path.realpath(filepath)
    if not real_path.startswith(real_dir + os.sep) and real_path != real_dir:
        print(f"ERROR: Path escapes session directory: {filename}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    content = sys.stdin.read()

    # DECISION plan_2026-05-19_4fc8ec9a/D-004:
    # Hardened write for state.md: if the incoming content changes the
    # `## Phase:` field value compared to the on-disk state, refuse unless
    # --force-state is set. The Phase: field is the canonical FSM cursor and
    # MUST only be advanced via `$SM advance` (gate-enforced) or rolled back
    # via `$SM set-phase --force-state` (logged). Free-write of Phase: is the
    # primary user-shortcut vector (vector C in findings/) and is now blocked.
    # All other state.md edits (hypothesis count, transition history, system
    # description) pass through unchanged.
    if filename == "state.md":
        try:
            with open(filepath, "r", encoding="utf-8") as _f:
                old_content = _f.read()
        except FileNotFoundError:
            old_content = None
        old_phase = _state_md_phase_value(old_content) if old_content else None
        new_phase = _state_md_phase_value(content)
        if old_phase is not None and new_phase is not None and old_phase != new_phase:
            if not getattr(args, "force_state", False):
                print(f"ERROR: Refusing to change Phase: '{old_phase}' → "
                      f"'{new_phase}' via free `write state.md`.",
                      file=sys.stderr)
                print(f"  Use `$SM advance` to move forward (gate-enforced) "
                      f"or `$SM set-phase {new_phase} --force-state "
                      f"--reason \"<why>\"` for an admin override (logged).",
                      file=sys.stderr)
                sys.exit(1)
            # --force-state set: log the override and proceed.
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            _append_decisions(abs_dir,
                f"\n## {ts} — ADMIN-OVERRIDE (write state.md) Phase {old_phase} → {new_phase}\n\n"
                f"**Decision**: Force-write state.md with --force-state, "
                f"changing Phase: directly.\n\n"
                f"**Reason**: free-write override via `$SM write state.md --force-state`.\n\n"
                f"**Cost**: gate checks bypassed for Phase {old_phase} exit.\n")

    if filename.endswith('.json') and _common_save_json is not None:
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"ERROR: {filename}: invalid JSON ({e})", file=sys.stderr)
            sys.exit(1)
        _common_save_json(filepath, data)
    else:
        _atomic_write(filepath, content)
    print(f"Wrote {filepath} ({len(content)} bytes)")


def cmd_read_file(args):
    """Read and output a session file."""
    abs_dir = read_pointer()
    if not abs_dir:
        print("ERROR: No active analysis. Use `new` to create one.", file=sys.stderr)
        sys.exit(1)

    filename = args.filename
    # Security: prevent path traversal (check ".." as path component, not substring)
    parts = filename.replace("\\", "/").split("/")
    if any(part == ".." for part in parts) or filename.startswith("/"):
        print(f"ERROR: Invalid filename: {filename}", file=sys.stderr)
        sys.exit(1)

    filepath = os.path.join(abs_dir, filename)
    # Resolve symlinks and verify the target is still under abs_dir
    real_dir = os.path.realpath(abs_dir)
    real_path = os.path.realpath(filepath)
    if not real_path.startswith(real_dir + os.sep) and real_path != real_dir:
        print(f"ERROR: Path escapes session directory: {filename}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(real_path):
        print(f"ERROR: File not found: {filepath}", file=sys.stderr)
        sys.exit(1)

    with open(real_path, "r", encoding="utf-8") as f:
        print(f.read(), end="")


def cmd_path(args):
    """Output the absolute path to a session file (or session dir if no filename)."""
    abs_dir = read_pointer()
    if not abs_dir:
        print("ERROR: No active analysis. Use `new` to create one.", file=sys.stderr)
        sys.exit(1)

    if args.filename:
        # Security: prevent path traversal (check ".." as path component, not substring)
        parts = args.filename.replace("\\", "/").split("/")
        if any(part == ".." for part in parts) or args.filename.startswith("/"):
            print(f"ERROR: Invalid filename: {args.filename}", file=sys.stderr)
            sys.exit(1)
        filepath = os.path.join(abs_dir, args.filename)
        # Resolve symlinks and verify the target is still under abs_dir
        real_dir = os.path.realpath(abs_dir)
        real_path = os.path.realpath(filepath)
        if not real_path.startswith(real_dir + os.sep) and real_path != real_dir:
            print(f"ERROR: Path escapes session directory: {args.filename}", file=sys.stderr)
            sys.exit(1)
        print(filepath)
    else:
        print(abs_dir)


def cmd_list(args):
    """Show all analysis directories."""
    if not os.path.isdir(ANALYSES_DIR):
        print("No analyses/ directory found.")
        return

    active_abs = read_pointer()
    active_name = os.path.basename(active_abs) if active_abs else None
    entries = sorted(
        d for d in os.listdir(ANALYSES_DIR)
        if os.path.isdir(os.path.join(ANALYSES_DIR, d)) and d.startswith("analysis_")
    )

    if not entries:
        print("No analysis directories found.")
        return

    print(f"Analysis directories in analyses/ ({len(entries)} total):")
    for name in entries:
        marker = " ← active" if name == active_name else ""
        abs_entry = os.path.abspath(os.path.join(ANALYSES_DIR, name))
        state = read_analysis_file(abs_entry, "state.md")
        phase = extract_field(state, r'^## Phase:\s*(.+)$') or "?"
        tier = extract_field(state, r'^## Tier:\s*(.+)$') or "?"
        system = extract_field(state, r'^## System:\s*(.+)$') or "?"
        system_short = (system[:50] if system else "?")
        print(f"  {name}  [Phase {phase} | {tier}] {system_short}{marker}")


def cmd_reopen(args):
    """Reopen a completed phase for multi-pass analysis."""
    abs_dir = read_pointer()
    if not abs_dir:
        print("ERROR: No active analysis. Use `new` to create one.", file=sys.stderr)
        sys.exit(1)

    phase = args.phase
    reason = " ".join(args.reason).strip()
    if not reason:
        print("ERROR: Reason is required for reopening a phase.", file=sys.stderr)
        sys.exit(1)

    # Validate phase identifier
    phase_file = PHASE_FILENAME_MAP.get(phase)
    if not phase_file:
        valid = ", ".join(sorted(PHASE_FILENAME_MAP.keys(),
                                 key=lambda x: (x.endswith("-P"), x)))
        print(f"ERROR: Invalid phase '{phase}'. Valid phases: {valid}",
              file=sys.stderr)
        sys.exit(1)

    # Phase must have been completed (output file exists)
    phase_dir = os.path.join(abs_dir, "phase_outputs")
    phase_output_path = os.path.join(phase_dir, phase_file)
    if not os.path.exists(phase_output_path):
        print(f"ERROR: Phase {phase} has no output ({phase_file}). "
              f"Cannot reopen an uncompleted phase.", file=sys.stderr)
        sys.exit(1)

    # Count existing archived passes
    base = phase_file[:-3]  # strip .md
    archived = sorted(
        f for f in os.listdir(phase_dir)
        if re.match(rf'^{re.escape(base)}_pass\d+\.md$', f)
    )
    reopen_count = len(archived)

    if reopen_count >= MAX_REOPENS:
        print(f"ERROR: Phase {phase} already reopened {MAX_REOPENS} times "
              f"(max reached). Escalate tier or change approach.",
              file=sys.stderr)
        sys.exit(1)

    # Archive current phase output: phase_N.md → phase_N_passK.md
    pass_num = reopen_count + 1
    archive_name = f"{base}_pass{pass_num}.md"
    archive_path = os.path.join(phase_dir, archive_name)
    os.replace(phase_output_path, archive_path)

    # Update state.md
    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    state = read_analysis_file(abs_dir, "state.md")
    if state:
        state = re.sub(
            r'^## Phase:\s*.*$', f'## Phase: {phase}',
            state, flags=re.MULTILINE)
        state = re.sub(
            r'^## Last Transition:\s*.*$',
            f'## Last Transition: REOPEN Phase {phase} pass {pass_num + 1} ({ts})',
            state, flags=re.MULTILINE)
        state = (state.rstrip()
                 + f"\n- REOPEN Phase {phase} pass {pass_num + 1}: "
                 + f"{reason} ({ts})\n")
        _atomic_write(os.path.join(abs_dir, "state.md"), state)

    total_passes = MAX_REOPENS + 1
    print(f"Reopened Phase {phase} (now pass {pass_num + 1} of {total_passes})")
    print(f"  Archived: phase_outputs/{archive_name}")
    print(f"  Reason: {reason}")
    print(f"  Reopens remaining: {MAX_REOPENS - reopen_count - 1}")
    print()
    print(f"  Next steps:")
    print(f"    1. Update progress.md (mark Phase {phase} in-progress)")
    print(f"    2. Update decisions.md (log reopen rationale)")
    print(f"    3. Re-execute Phase {phase} activities")
    print(f"    4. Reference phase_outputs/{archive_name} for prior findings")
    print(f"    5. Pass EXIT GATE again → write new phase_outputs/{phase_file}")


def cmd_advance(args):
    """Advance to the next phase per the canonical FSM.

    DECISION plan_2026-05-19_4fc8ec9a/D-002:
    The sole legitimate path forward through the protocol. Reads current
    Phase + Tier, computes the next legitimate phase from PHASE_SEQUENCE,
    enforces required-artifacts and per-phase exit-gate checks, then writes
    Phase: to state.md atomically. Free `$SM write state.md` of the Phase:
    field is REFUSED (see cmd_write hardening). The only escape hatch is
    `$SM set-phase --force-state` which logs an admin override.
    """
    abs_dir = read_pointer()
    if not abs_dir:
        print("ERROR: No active analysis. Use `new` to create one.", file=sys.stderr)
        sys.exit(1)

    cur = _current_phase(abs_dir)
    if not cur:
        print("ERROR: Current Phase: field is missing from state.md.", file=sys.stderr)
        sys.exit(1)

    tier = _current_tier(abs_dir)
    if not tier:
        print("ERROR: Tier not declared. Set '## Tier:' in state.md or "
              "'## Tier Selected' in analysis_plan.md to one of "
              f"{sorted(PHASE_SEQUENCE.keys())}.", file=sys.stderr)
        sys.exit(1)

    tier_seq = PHASE_SEQUENCE.get(tier)
    if tier_seq is None:
        print(f"ERROR: Unknown tier '{tier}'. Valid tiers: "
              f"{sorted(PHASE_SEQUENCE.keys())}.", file=sys.stderr)
        sys.exit(1)

    if cur not in tier_seq:
        print(f"ERROR: Phase '{cur}' is not valid for tier {tier}. "
              f"Tier {tier} sequence: {list(tier_seq.keys())}.", file=sys.stderr)
        sys.exit(1)

    next_phase = tier_seq[cur]
    if next_phase is None:
        print(f"ERROR: Phase {cur} is terminal for tier {tier}. "
              f"Use `close` to finalize.", file=sys.stderr)
        sys.exit(1)

    # Required artifacts for the CURRENT phase must be present before we leave it.
    ok, missing = _required_artifacts_present(abs_dir, cur)
    if not ok:
        print(f"ERROR: Phase {cur} required artifacts missing: {missing}. "
              f"Write them under phase_outputs/ before advancing.",
              file=sys.stderr)
        sys.exit(1)

    # Run per-phase exit gate for current phase.
    passed, msg = _run_phase_gate(abs_dir, cur)
    if not passed:
        print(f"ERROR: Phase {cur} exit gate refused advance: {msg}",
              file=sys.stderr)
        print(f"  Recovery options:", file=sys.stderr)
        print(f"    1. Address the gate failure and re-run `advance`.",
              file=sys.stderr)
        print(f"    2. Reopen the phase via `reopen {cur} \"<reason>\"`.",
              file=sys.stderr)
        print(f"    3. Admin override: `set-phase {next_phase} "
              f"--force-state --reason \"<why>\"`.", file=sys.stderr)
        sys.exit(1)

    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    reason = " ".join(args.reason).strip() if getattr(args, "reason", None) else (
        f"advance from Phase {cur}")
    _append_state_transition(abs_dir, next_phase, "ADVANCE", reason, ts)

    print(f"Advanced Phase {cur} → Phase {next_phase}  (tier={tier})")
    print(f"  Gate: {msg}")
    print(f"  Logged: state.md transition history")


def cmd_gate_check(args):
    """Read-only audit — print JSON describing whether `advance` would pass.

    Does not modify state. Exit 0 if all checks pass, 1 otherwise.
    """
    abs_dir = read_pointer()
    if not abs_dir:
        print("ERROR: No active analysis. Use `new` to create one.", file=sys.stderr)
        sys.exit(1)

    cur = _current_phase(abs_dir)
    tier = _current_tier(abs_dir)
    tier_seq = PHASE_SEQUENCE.get(tier) if tier else None
    next_phase = tier_seq.get(cur) if (tier_seq and cur in tier_seq) else None

    ok_artifacts, missing = _required_artifacts_present(abs_dir, cur) if cur else (False, [])
    ok_gate, gate_msg = _run_phase_gate(abs_dir, cur) if cur else (False, "no current phase")

    overall = bool(cur and tier and tier_seq and cur in tier_seq and
                   next_phase is not None and ok_artifacts and ok_gate)

    report = {
        "session_dir": abs_dir,
        "current_phase": cur,
        "tier": tier,
        "next_phase": next_phase,
        "required_artifacts_ok": ok_artifacts,
        "required_artifacts_missing": missing,
        "gate_passed": ok_gate,
        "gate_message": gate_msg,
        "would_advance": overall,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    sys.exit(0 if overall else 1)


def cmd_set_phase(args):
    """Admin escape hatch — set Phase: directly, bypassing gate checks.

    Requires --force-state AND --reason. Appends an ADMIN-OVERRIDE entry to
    decisions.md and state.md transition history. Use sparingly: this is the
    documented bypass for recovery scenarios (e.g. corrupted state.md, gate
    script broken). All uses are logged.
    """
    abs_dir = read_pointer()
    if not abs_dir:
        print("ERROR: No active analysis. Use `new` to create one.", file=sys.stderr)
        sys.exit(1)

    if not getattr(args, "force_state", False):
        print("ERROR: `set-phase` requires --force-state to make the admin "
              "override explicit. The legitimate path is `advance`. If you "
              "want to revisit a completed phase, use `reopen`.",
              file=sys.stderr)
        sys.exit(1)

    reason = " ".join(args.reason).strip() if getattr(args, "reason", None) else ""
    if not reason:
        print("ERROR: --reason is mandatory for set-phase --force-state. "
              "Every admin override is logged.", file=sys.stderr)
        sys.exit(1)

    new_phase = args.phase
    if new_phase not in PHASE_FILENAME_MAP:
        valid = ", ".join(sorted(PHASE_FILENAME_MAP.keys(),
                                 key=lambda x: (x.endswith("-P"), x)))
        print(f"ERROR: Invalid phase '{new_phase}'. Valid phases: {valid}",
              file=sys.stderr)
        sys.exit(1)

    cur = _current_phase(abs_dir) or "(unknown)"
    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Log to decisions.md first — if logging fails, we have not yet mutated state.
    entry = (
        f"\n## {ts} — ADMIN-OVERRIDE Phase {cur} → {new_phase}\n\n"
        f"**Decision**: Force Phase: field to {new_phase} via "
        f"`set-phase --force-state` (gate checks bypassed).\n\n"
        f"**Reason**: {reason}\n\n"
        f"**Cost**: protocol invariants potentially violated; downstream "
        f"phases proceed without exit-gate guarantees for Phase {cur}. "
        f"Analyst accepts responsibility for the bypass.\n"
    )
    _append_decisions(abs_dir, entry)
    _append_state_transition(abs_dir, new_phase, "ADMIN-OVERRIDE", reason, ts)

    print(f"ADMIN OVERRIDE: Phase {cur} → Phase {new_phase}")
    print(f"  Reason: {reason}")
    print(f"  Logged: decisions.md, state.md")
    print(f"  WARNING: gate checks were bypassed.")


def cmd_skip(args):
    """Skip a phase with logged rationale (no archival, unlike reopen).

    Intended for conditional phases such as Phase 0.3 Domain Orientation when
    the analyst has declared `domain_familiarity: high` and wishes to bypass
    the phase with a documented justification. The phase-output file is not
    written; downstream phases proceed without the artifacts this phase would
    have produced.
    """
    abs_dir = read_pointer()
    if not abs_dir:
        print("ERROR: No active analysis. Use `new` to create one.", file=sys.stderr)
        sys.exit(1)

    phase = args.phase
    reason = " ".join(args.reason).strip()
    if not reason:
        print("ERROR: Reason is required for skipping a phase.", file=sys.stderr)
        sys.exit(1)

    if phase not in PHASE_FILENAME_MAP:
        valid = ", ".join(sorted(PHASE_FILENAME_MAP.keys(),
                                 key=lambda x: (x.endswith("-P"), x)))
        print(f"ERROR: Invalid phase '{phase}'. Valid phases: {valid}",
              file=sys.stderr)
        sys.exit(1)

    # DECISION plan_2026-05-19_4fc8ec9a/D-003:
    # Per-tier whitelist of skippable phases. Anything outside the whitelist
    # is refused — `skip` is NOT a free fast-path. The legitimate fast-path
    # is choosing the RAPID tier at session start. The legitimate revisit is
    # `reopen`. The documented escape hatch is `set-phase --force-state`.
    tier = _current_tier(abs_dir)
    if tier:
        allowed = SKIPPABLE.get(tier, set())
        if phase not in allowed:
            allowed_str = ", ".join(sorted(allowed)) if allowed else "(none)"
            print(f"ERROR: Phase '{phase}' is not whitelisted for skipping on "
                  f"tier {tier}. Skippable on {tier}: {allowed_str}.",
                  file=sys.stderr)
            print(f"  The legitimate fast-path is choosing the RAPID tier at "
                  f"session start. To revisit a completed phase use `reopen`. "
                  f"To force an unsupported transition use "
                  f"`set-phase --force-state --reason \"<why>\"` (logged).",
                  file=sys.stderr)
            sys.exit(1)

    # DECISION plan_2026-05-25_c0b0049a/D-001:
    # Phase 0.3 / 0-P.3 skip additionally requires `domain_familiarity: high`
    # in analysis_plan.md. The whitelist (D-003) gates WHICH phases may be
    # skipped per tier; the familiarity check gates WHEN the skip is
    # legitimate. Together they enforce SKILL.md's trigger paragraph at the
    # CLI layer rather than only in prose.
    if phase in {"0.3", "0-P.3"}:
        familiarity = _read_domain_familiarity(abs_dir)
        if familiarity != "high":
            declared = familiarity if familiarity else "(not declared)"
            print(f"ERROR: Phase {phase} skip requires "
                  f"`domain_familiarity: high` in analysis_plan.md. "
                  f"Current value: {declared}.", file=sys.stderr)
            print(f"  Phase {phase} (Domain Orientation) exists to ground "
                  f"unfamiliar jargon; skipping it is only legitimate when "
                  f"the analyst has affirmatively self-assessed `high` "
                  f"familiarity per references/domain-orientation.md. To "
                  f"override despite low familiarity, use "
                  f"`set-phase --force-state --reason \"<why>\"` (logged).",
                  file=sys.stderr)
            sys.exit(1)

    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Append a decisions.md entry (create if missing).
    decisions_path = os.path.join(abs_dir, "decisions.md")
    try:
        with open(decisions_path, "r", encoding="utf-8") as f:
            decisions = f.read()
    except FileNotFoundError:
        decisions = "# Decisions\n"
    entry = (
        f"\n## {ts} — SKIP Phase {phase}\n\n"
        f"**Decision**: Skip Phase {phase} with logged rationale.\n\n"
        f"**Reason**: {reason}\n\n"
        f"**Cost**: downstream phases proceed without the artifacts Phase "
        f"{phase} would have produced. Trade-off accepted by analyst.\n"
    )
    if decisions and not decisions.endswith("\n"):
        decisions += "\n"
    _atomic_write(decisions_path, decisions + entry)

    # DECISION plan_2026-05-19_8608e41f/D-002:
    # Skip advances the Phase: cursor past the skipped phase when the skip
    # is contextually reachable (cur == skipped_phase, OR cur's next in the
    # tier sequence == skipped_phase). Otherwise log without moving cursor.
    # This makes skip behave per the SYSTEM.md invariant — Phase: is mutated
    # only by advance/skip/reopen/set-phase, and skip's mutation is loud
    # (writes Last Transition + history via _append_state_transition).
    cur = _current_phase(abs_dir)
    tier_seq = PHASE_SEQUENCE.get(tier) if tier else None
    post_skip = tier_seq.get(phase) if tier_seq else None
    cursor_moved = False
    if tier_seq and post_skip is not None and (
            cur == phase or tier_seq.get(cur) == phase):
        _append_state_transition(abs_dir, post_skip, "SKIP", reason, ts)
        cursor_moved = True
    else:
        # Cursor stays put; still record the skip in transition history.
        state = read_analysis_file(abs_dir, "state.md")
        if state:
            state = re.sub(
                r'^## Last Transition:\s*.*$',
                f'## Last Transition: SKIP Phase {phase} ({ts})',
                state, flags=re.MULTILINE)
            state = (state.rstrip()
                     + f"\n- SKIP Phase {phase}: {reason} ({ts})\n")
            _atomic_write(os.path.join(abs_dir, "state.md"), state)

    print(f"Skipped Phase {phase}")
    print(f"  Reason: {reason}")
    if cursor_moved:
        print(f"  Phase: {cur} → {post_skip} (cursor advanced past skipped phase)")
    else:
        print(f"  Phase cursor unchanged (cur={cur}); skip logged for the record.")
    print(f"  Logged: decisions.md, state.md")
    print()
    print(f"  Next steps:")
    if cursor_moved:
        print(f"    1. Proceed with Phase {post_skip} work, then `advance`")
    else:
        print(f"    1. Proceed to the next phase in the FSM")
    print(f"    2. If the skip alters success criteria, note in plan/phase_outputs")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Manage Epistemic Deconstructor analysis sessions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Commands:
  new "goal"              Create a new analysis session
  new --force "goal"      Close active session and create a new one
  resume                  Output current session state for re-entry
  status                  One-line state summary
  close                   Close active session (preserves directory)
  advance                 Advance to next phase (runs gate checks). The
                          ONLY legitimate path forward through the FSM.
  gate-check              Read-only audit: would `advance` succeed?
  set-phase <phase> --force-state --reason "..."
                          Admin override: force Phase: directly (logged).
  reopen <phase> "reason" Reopen completed phase for multi-pass (max 3 reopens)
  skip <phase> "reason"   Skip a whitelisted phase with logged justification
  list                    Show all analysis directories""")

    parser.add_argument("--base-dir", default=None,
                        help="Base directory for analyses/ (default: current directory)")

    sub = parser.add_subparsers(dest="command")

    p_new = sub.add_parser("new", help="Create a new analysis session")
    p_new.add_argument("--force", action="store_true", help="Close active session first")
    p_new.add_argument("--tier",
                       choices=["RAPID", "LITE", "STANDARD", "COMPREHENSIVE", "PSYCH"],
                       default=None,
                       help="Tier to declare at session start (writes state.md `## Tier:` "
                            "and analysis_plan.md `## Tier Selected`). Eliminates "
                            "the post-`new` advance refusal 'Tier not declared'.")
    p_new.add_argument("--domain-familiarity",
                       choices=["high", "medium", "low", "unknown"],
                       default=None,
                       help="Domain familiarity declaration written into analysis_plan.md. "
                            "Required for `$SM skip 0.3` to succeed without follow-up edits.")
    p_new.add_argument("goal", nargs="+", help="System description / analysis goal")

    sub.add_parser("resume", help="Output current session state for re-entry")
    sub.add_parser("status", help="One-line state summary")
    sub.add_parser("close", help="Close active session")
    sub.add_parser("list", help="Show all analysis directories")

    p_reopen = sub.add_parser("reopen", help="Reopen a completed phase for multi-pass")
    p_reopen.add_argument("phase", help="Phase to reopen (e.g. 0, 1, 2, 3, 4, 5, 0.5, 0-P)")
    p_reopen.add_argument("reason", nargs="+", help="Reason for reopening")

    p_skip = sub.add_parser("skip", help="Skip a conditional phase with logged rationale")
    p_skip.add_argument("phase", help="Phase to skip (e.g. 0.3 for Domain Orientation)")
    p_skip.add_argument("reason", nargs="+", help="Justification (will be logged to decisions.md)")

    p_adv = sub.add_parser("advance", help="Advance to next phase (gate-enforced; legitimate path forward)")
    p_adv.add_argument("reason", nargs="*", default=[],
                       help="Optional one-line reason for the advance.")

    sub.add_parser("gate-check", help="Read-only audit — would `advance` pass right now?")

    p_setp = sub.add_parser("set-phase", help="ADMIN: force Phase: directly (logged)")
    p_setp.add_argument("phase", help="Target phase (e.g. 2, 0.3, 0-P.3)")
    p_setp.add_argument("--force-state", action="store_true",
                        help="Required: makes the admin override explicit.")
    p_setp.add_argument("--reason", nargs="+", required=True,
                        help="Mandatory justification (logged to decisions.md).")

    p_write = sub.add_parser("write", help="Write stdin to a session file")
    p_write.add_argument("filename", help="File to write (e.g. state.md, observations/obs_001.md)")
    p_write.add_argument("--force-state", action="store_true",
                         help="Allow Phase: field changes via `write state.md` "
                              "(admin override; logged to decisions.md).")

    p_read = sub.add_parser("read", help="Read and output a session file")
    p_read.add_argument("filename", help="File to read (e.g. state.md, hypotheses.json)")

    p_path = sub.add_parser("path", help="Output absolute path to session dir or file")
    p_path.add_argument("filename", nargs="?", default=None, help="Optional filename")

    args = parser.parse_args()

    if args.base_dir:
        base = os.path.abspath(args.base_dir)
        os.makedirs(base, exist_ok=True)
        os.chdir(base)

    if args.command == "new":
        cmd_new(args)
    elif args.command == "resume":
        cmd_resume(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "close":
        cmd_close(args)
    elif args.command == "reopen":
        cmd_reopen(args)
    elif args.command == "skip":
        cmd_skip(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "write":
        cmd_write(args)
    elif args.command == "read":
        cmd_read_file(args)
    elif args.command == "path":
        cmd_path(args)
    elif args.command == "advance":
        cmd_advance(args)
    elif args.command == "gate-check":
        cmd_gate_check(args)
    elif args.command == "set-phase":
        cmd_set_phase(args)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
