#!/usr/bin/env python3
"""
Structural exit-gate dispatcher for Epistemic Deconstructor phases.

Resolves audit H1/H10 (plan_2026-05-28_b0332cf7): prior to v7.15.20,
session_manager.py's _run_phase_gate returned (True, "no gate") for any phase
without a PHASE_GATE_SCRIPTS entry — 13 non-terminal phases (0, 0.5, 1, 2, 3,
4, 5 in STANDARD/COMPREHENSIVE; 0-P, 1-P, 2-P, 3-P, 4-P, 5-P in PSYCH) advanced
on file-presence only.

This script provides the structural gate for those 13 phases. It is invoked
exactly like the other gate scripts:

    phase_gate.py --file <abs-path-to-phase_N.md> gate

Phase identity is inferred from the basename of --file (e.g. "phase_2_P.md" →
phase "2-P"). The dispatcher loads the phase-specific rule set and applies it.

Severity model (mirrors scope_auditor.py):
    [REQUIRED]    — failure causes overall FAIL (exit 1)
    [RECOMMENDED] — failure logged as WARN but does not affect exit code

Stdlib-only. No third-party dependencies.

Semantic gate criteria (e.g. "≥70% behaviors explained", "cross-val R² > 0.8")
remain agent-attested per references/phase-protocols.md. This script enforces
STRUCTURAL validity only: file presence, minimum content size, required marker
patterns, and sibling-file existence where relevant.
"""

import argparse
import json
import os
import re
import sys
from typing import Callable, List, Tuple


# -----------------------------------------------------------------------------
# Filename → phase identity
# -----------------------------------------------------------------------------

# Canonical filename map mirroring session_manager.PHASE_FILENAME_MAP.
# We replicate it here (instead of importing) to keep phase_gate.py independent
# of session_manager's import-time side effects.
_FILENAME_TO_PHASE = {
    "phase_0.md": "0",
    "phase_0_3.md": "0.3",     # served by domain_orienter.py, not this script
    "phase_0_5.md": "0.5",
    "phase_0_7.md": "0.7",     # served by scope_auditor.py, not this script
    "phase_1.md": "1",
    "phase_1_5.md": "1.5",     # served by abductive_engine.py, not this script
    "phase_2.md": "2",
    "phase_3.md": "3",
    "phase_4.md": "4",
    "phase_5.md": "5",
    "phase_0_P.md": "0-P",
    "phase_1_P.md": "1-P",
    "phase_2_P.md": "2-P",
    "phase_3_P.md": "3-P",
    "phase_4_P.md": "4-P",
    "phase_5_P.md": "5-P",
}


def _basename_to_phase(name: str):
    """Return the canonical phase ID for *name* (basename), or None if unknown."""
    return _FILENAME_TO_PHASE.get(name)


# -----------------------------------------------------------------------------
# Check helpers
# -----------------------------------------------------------------------------

def _min_bytes(threshold: int) -> Callable:
    def predicate(content: str, _session_dir: str, _phase: str) -> bool:
        return len(content.encode("utf-8")) >= threshold
    predicate.__name__ = f"min_bytes_{threshold}"
    return predicate


def _strip_fenced_blocks(text: str) -> str:
    """Remove ``` fenced code/diagram blocks so keyword gates ignore diagram labels.

    Keyword-class predicates must treat prose as the only valid evidence: a
    required keyword (e.g. ``causal``) hiding inside a ```mermaid node label is
    NOT prose and must not satisfy a gate. The byte-size predicate (_min_bytes)
    deliberately does NOT use this — it is a structural size check over full
    content.
    """
    return re.sub(r"```.*?```", "", text, flags=re.DOTALL)


def _contains_any(patterns: List[str], flags: int = re.IGNORECASE) -> Callable:
    """Pass if any of the regex patterns matches the file content (fences stripped)."""
    compiled = [re.compile(p, flags) for p in patterns]
    def predicate(content: str, _session_dir: str, _phase: str) -> bool:
        target = _strip_fenced_blocks(content)
        return any(p.search(target) for p in compiled)
    predicate.__name__ = "contains_any:" + "|".join(patterns)
    return predicate


def _contains_all(patterns: List[str], flags: int = re.IGNORECASE) -> Callable:
    """Pass if every regex pattern matches the file content (any order; fences stripped)."""
    compiled = [re.compile(p, flags) for p in patterns]
    def predicate(content: str, _session_dir: str, _phase: str) -> bool:
        target = _strip_fenced_blocks(content)
        return all(p.search(target) for p in compiled)
    predicate.__name__ = "contains_all:" + "|".join(patterns)
    return predicate


def _min_h_refs(n: int) -> Callable:
    """Pass if the content contains at least *n* distinct ``H\\d+`` references (fences stripped)."""
    pat = re.compile(r"\bH\d+\b")
    def predicate(content: str, _session_dir: str, _phase: str) -> bool:
        target = _strip_fenced_blocks(content)
        return len(set(pat.findall(target))) >= n
    predicate.__name__ = f"min_h_refs_{n}"
    return predicate


def _sibling_exists(*names: str) -> Callable:
    """Pass if any named sibling file exists in session_dir.

    Names are interpreted relative to session_dir. ``foo`` checks
    ``<session_dir>/foo``; ``observations.md`` checks
    ``<session_dir>/observations.md``; a glob like ``phase_outputs/obs_*.md`` is
    treated as a glob (cheap fnmatch over the directory listing).
    """
    import fnmatch

    def predicate(_content: str, session_dir: str, _phase: str) -> bool:
        if not session_dir:
            return False
        for name in names:
            if "*" in name or "?" in name:
                base = os.path.join(session_dir, os.path.dirname(name) or ".")
                pat = os.path.basename(name)
                if os.path.isdir(base):
                    for entry in os.listdir(base):
                        if fnmatch.fnmatch(entry, pat):
                            return True
            else:
                if os.path.exists(os.path.join(session_dir, name)):
                    return True
        return False

    predicate.__name__ = "sibling_exists:" + ",".join(names)
    return predicate


def _hypotheses_json_has_n_entries(n: int) -> Callable:
    """Pass if ``<session_dir>/hypotheses.json`` parses and lists >= n hypotheses."""
    def predicate(_content: str, session_dir: str, _phase: str) -> bool:
        if not session_dir:
            return False
        path = os.path.join(session_dir, "hypotheses.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return False
        return len(data.get("hypotheses", [])) >= n
    predicate.__name__ = f"hypotheses_json_has_{n}_entries"
    return predicate


# -----------------------------------------------------------------------------
# Per-phase rule sets
# -----------------------------------------------------------------------------
# Each entry: list of (severity, name, predicate). severity ∈ {"required",
# "recommended"}. Required failures cause exit 1. Recommended failures emit
# stderr warnings but do not affect exit code.

RULES = {
    # ----- STANDARD / COMPREHENSIVE main phases -----

    "0": [
        ("required", "min_size_200B", _min_bytes(200)),
        ("required", "at_least_3_H_refs", _min_h_refs(3)),
        ("required", "framing_keyword",
            _contains_any([r"hypothes", r"framing", r"hypothesis"])),
        ("recommended", "hypotheses_json_seeded",
            _hypotheses_json_has_n_entries(3)),
    ],

    "0.5": [
        ("required", "min_size_100B", _min_bytes(100)),
        ("required", "rapid_verdict_present",
            _contains_any([r"\bCREDIBLE\b", r"\bSKEPTICAL\b",
                          r"\bDOUBTFUL\b", r"\bREJECT\b"], flags=0)),
    ],

    "1": [
        ("required", "min_size_200B", _min_bytes(200)),
        ("required", "observation_reference",
            _contains_any([r"obs(_|\W)", r"observation"])),
        ("recommended", "sibling_observations_or_files",
            _sibling_exists("observations.md", "phase_outputs/obs_*.md",
                           "obs_*.md")),
    ],

    "2": [
        ("required", "min_size_300B", _min_bytes(300)),
        ("required", "causal_or_falsification_keyword",
            _contains_any([r"causal", r"falsif", r"mechanism",
                          r"refut", r"weaken"])),
        ("recommended", "at_least_1_H_ref", _min_h_refs(1)),
    ],

    "3": [
        ("required", "min_size_200B", _min_bytes(200)),
        ("required", "model_or_fit_keyword",
            _contains_any([r"\bmodel\b", r"\bfit\b", r"\bparameter",
                          r"\bforecast", r"identif", r"estimat"])),
        ("recommended", "uncertainty_keyword",
            _contains_any([r"\binterval", r"\bCI\b", r"R²", r"R\^2",
                          r"residual", r"whiteness", r"AIC", r"BIC",
                          r"uncertain", r"confidence", r"Wilson"])),
    ],

    "4": [
        ("required", "min_size_200B", _min_bytes(200)),
        ("required", "synthesis_keyword",
            _contains_any([r"archetype", r"compos", r"emergen",
                          r"\bsynth", r"sub.?model"])),
    ],

    "5": [
        ("required", "min_size_200B", _min_bytes(200)),
        ("required", "validation_keyword",
            _contains_any([r"validation", r"validat"])),
        ("required", "summary_or_verdict_keyword",
            _contains_any([r"summary", r"verdict", r"conclusion"])),
        ("recommended", "sibling_validation_md",
            _sibling_exists("validation.md")),
        ("recommended", "sibling_summary_md",
            _sibling_exists("summary.md")),
    ],

    # ----- PSYCH main phases -----

    "0-P": [
        ("required", "min_size_200B", _min_bytes(200)),
        ("required", "subject_or_baseline_keyword",
            _contains_any([r"subject", r"profile", r"baseline",
                          r"trait", r"hypothes", r"frame"])),
        ("recommended", "sibling_beliefs_json",
            _sibling_exists("beliefs.json")),
    ],

    "1-P": [
        ("required", "min_size_200B", _min_bytes(200)),
        ("required", "stimulus_or_observation_keyword",
            _contains_any([r"baseline", r"stimul", r"response",
                          r"observ", r"behav"])),
    ],

    "2-P": [
        ("required", "min_size_300B", _min_bytes(300)),
        ("required", "behavioral_or_causal_keyword",
            _contains_any([r"stimul", r"response", r"trait",
                          r"behavioral", r"causal", r"reaction"])),
    ],

    "3-P": [
        ("required", "min_size_200B", _min_bytes(200)),
        ("required", "trait_model_keyword",
            _contains_any([r"trait", r"archetype", r"OCEAN", r"profile",
                          r"\bmodel\b", r"identif"])),
    ],

    "4-P": [
        ("required", "min_size_200B", _min_bytes(200)),
        ("required", "motive_keyword",
            _contains_any([r"motiv", r"MICE", r"RASP", r"archetype",
                          r"intent"])),
    ],

    "5-P": [
        ("required", "min_size_200B", _min_bytes(200)),
        ("required", "validation_keyword",
            _contains_any([r"validation", r"validat"])),
        ("required", "summary_or_verdict_keyword",
            _contains_any([r"summary", r"verdict", r"profile",
                          r"conclusion"])),
        ("recommended", "sibling_validation_md",
            _sibling_exists("validation.md")),
        ("recommended", "sibling_summary_md",
            _sibling_exists("summary.md")),
    ],
}


# -----------------------------------------------------------------------------
# Gate runner
# -----------------------------------------------------------------------------

def run_gate(path: str) -> Tuple[int, List[Tuple[str, str, bool]], str]:
    """Run the structural gate for the phase implied by *path*'s basename.

    Returns (exit_code, results, phase).
    exit_code = 0 (PASS), 1 (FAIL), 2 (USAGE error — unknown phase).
    results is a list of (severity, name, passed) tuples.
    """
    name = os.path.basename(path)
    phase = _basename_to_phase(name)
    if phase is None:
        return 2, [], "?"
    if phase not in RULES:
        # File maps to a phase handled by a different script (e.g. 0.3, 0.7,
        # 1.5). phase_gate.py is not configured for these — treat as USAGE.
        return 2, [], phase

    # session_dir = directory containing phase_outputs/<filename>, i.e. two
    # levels up from <path> if path is <session>/phase_outputs/phase_N.md,
    # or one level up if path is directly <session>/phase_N.md.
    parent = os.path.dirname(os.path.abspath(path))
    if os.path.basename(parent) == "phase_outputs":
        session_dir = os.path.dirname(parent)
    else:
        session_dir = parent

    # Read file (presence + content). File-missing → all required FAIL.
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except (OSError, UnicodeDecodeError):
        content = None

    results: List[Tuple[str, str, bool]] = []
    has_required_fail = False
    for severity, check_name, predicate in RULES[phase]:
        if content is None:
            passed = False
        else:
            try:
                passed = bool(predicate(content, session_dir, phase))
            except Exception:
                passed = False
        results.append((severity, check_name, passed))
        if severity == "required" and not passed:
            has_required_fail = True

    return (1 if has_required_fail else 0), results, phase


def _emit_report(phase: str, results: List[Tuple[str, str, bool]],
                 exit_code: int, file_path: str) -> None:
    """Print labeled gate output mirroring scope_auditor's convention.

    Routing rule: on overall FAIL the entire report goes to stderr so that
    session_manager._run_phase_gate's tail-of-stderr diagnostic surfaces the
    REQUIRED misses to the user. On PASS, stdout is used; recommended misses
    are mirrored to stderr as advisory WARNs.
    """
    stream = sys.stderr if exit_code != 0 else sys.stdout
    print(f"Phase {phase} Structural Gate:", file=stream)
    print(f"  Target: {file_path}", file=stream)
    for severity, name, passed in results:
        tag = "REQUIRED" if severity == "required" else "RECOMMENDED"
        status = "OK" if passed else "MISS"
        line = f"  [{tag}] {name}: {status}"
        if exit_code == 0:
            # Overall PASS: print all to stdout; advisory recommended-miss also
            # echoed to stderr so analysts notice it.
            print(line)
            if not passed and severity == "recommended":
                print(line, file=sys.stderr)
        else:
            # Overall FAIL: route entire report to stderr (single stream).
            print(line, file=stream)
    overall = "PASS" if exit_code == 0 else "FAIL"
    print(f"  pass: {overall}", file=stream)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Structural exit-gate dispatcher for "
                    "Epistemic Deconstructor phases.")
    parser.add_argument("--file", required=True,
                        help="Absolute path to <session>/phase_outputs/phase_N.md")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("gate", help="Run the structural gate for this phase")
    args = parser.parse_args()

    if args.cmd != "gate":
        parser.print_help(sys.stderr)
        return 2

    exit_code, results, phase = run_gate(args.file)
    if exit_code == 2:
        print(f"Error: phase_gate.py does not handle phase '{phase}' "
              f"(file: {args.file}). Phases 0.3 / 0.7 / 1.5 (and PSYCH "
              f"counterparts) are served by domain_orienter.py / "
              f"scope_auditor.py / abductive_engine.py respectively.",
              file=sys.stderr)
        return 2
    _emit_report(phase, results, exit_code, args.file)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
