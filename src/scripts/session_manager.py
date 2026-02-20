#!/usr/bin/env python3
"""
Session manager for Epistemic Deconstructor analysis sessions.

Creates and manages analysis session directories under analyses/ in cwd.
Persists analysis state to filesystem so context window loss doesn't
destroy session progress.

Usage:
    python session_manager.py new "System description"
    python session_manager.py resume
    python session_manager.py status
    python session_manager.py close
    python session_manager.py list
"""

import argparse
import os
import re
import sys
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ANALYSES_DIR = "analyses"
POINTER_FILE = os.path.join(ANALYSES_DIR, ".current_analysis")
CONSOLIDATED_FINDINGS = os.path.join(ANALYSES_DIR, "FINDINGS.md")
CONSOLIDATED_DECISIONS = os.path.join(ANALYSES_DIR, "DECISIONS.md")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
            return None

        # Old format: directory name relative to ANALYSES_DIR
        rel_path = os.path.join(ANALYSES_DIR, value)
        if os.path.isdir(rel_path):
            return os.path.abspath(rel_path)
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
    """Write content atomically via tmp + rename."""
    tmp = filepath + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
    os.replace(tmp, filepath)


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
# Session templates
# ---------------------------------------------------------------------------

def make_state(goal, timestamp):
    return f"""# Current State
## Phase: 0
## Tier: (pending)
## Fidelity Target: (pending)
## System: {goal}
## Active Hypotheses: 0
## Lead Hypothesis: (none)
## Confidence: Low
## Time Budget: (pending)
## Last Transition: INIT → Phase 0 ({timestamp})
## Transition History:
- INIT → Phase 0 (session started)
"""


def make_analysis_plan():
    return """# Analysis Plan
*Written during Phase 0 (Setup & Frame). This is the persistent record of the analysis setup.*

## System Description
*(to be filled)*

## Access Level
*(full source / binary only / black-box I/O)*

## Adversary Status
*(yes / no / unknown)*

## Tier Selected
*(RAPID / LITE / STANDARD / COMPREHENSIVE / PSYCH)*

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
    goal = " ".join(args.goal)
    if not goal:
        print("ERROR: Goal is required.", file=sys.stderr)
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

        files = {
            "state.md": make_state(goal, timestamp),
            "analysis_plan.md": make_analysis_plan(),
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

    except Exception as e:
        # Cleanup on failure
        import shutil
        try:
            shutil.rmtree(analysis_dir, ignore_errors=True)
        except Exception:
            pass
        try:
            if os.path.exists(POINTER_FILE + ".tmp"):
                os.unlink(POINTER_FILE + ".tmp")
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
    print(f"Initialized {abs_analysis_dir}/")
    print(f"  Pointer: analyses/.current_analysis → {abs_analysis_dir}")
    print(f"  System: {goal}")
    print(f"  State: Phase 0 (Setup & Frame)")
    print(f"  Cross-analysis context: analyses/FINDINGS.md, analyses/DECISIONS.md")
    print(f"  Next: Fill analysis_plan.md, seed hypotheses, select tier.")
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

    print(f"Resuming {abs_dir}/")
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
        print(f"\n  Phase outputs ({len(phase_files)}):")
        for pf in phase_files:
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

    if not silent:
        print(f"Closed analysis: {abs_dir}")
        print(f"  Pointer analyses/.current_analysis removed.")
        print(f"  Analysis directory preserved at {abs_dir}/")
        print(f"  Observations/decisions merged to analyses/FINDINGS.md and analyses/DECISIONS.md.")
    else:
        print(f"  Closed previous analysis: {abs_dir}")


def cmd_close(args):
    """Close the active analysis."""
    cmd_close_impl(silent=False)


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
  list                    Show all analysis directories""")

    sub = parser.add_subparsers(dest="command")

    p_new = sub.add_parser("new", help="Create a new analysis session")
    p_new.add_argument("--force", action="store_true", help="Close active session first")
    p_new.add_argument("goal", nargs="+", help="System description / analysis goal")

    sub.add_parser("resume", help="Output current session state for re-entry")
    sub.add_parser("status", help="One-line state summary")
    sub.add_parser("close", help="Close active session")
    sub.add_parser("list", help="Show all analysis directories")

    args = parser.parse_args()

    if args.command == "new":
        cmd_new(args)
    elif args.command == "resume":
        cmd_resume(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "close":
        cmd_close(args)
    elif args.command == "list":
        cmd_list(args)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
