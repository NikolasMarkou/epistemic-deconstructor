#!/usr/bin/env python3
"""
Bayesian Hypothesis Tracker for Epistemic Deconstruction v6.0
Implements proper Bayesian updating with likelihood ratios.

Extended with red flag tracking, coherence checking, and verdict generation
for RAPID tier claim validation.
"""

import json
import math
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict

from common import bayesian_update, load_json, save_json

class Status(Enum):
    ACTIVE = "ACTIVE"
    WEAKENED = "WEAKENED"
    KILLED = "KILLED"
    CONFIRMED = "CONFIRMED"

class FlagCategory(Enum):
    METHODOLOGY = "methodology"
    DOCUMENTATION = "documentation"
    RESULTS = "results"
    CLAIMS = "claims"
    TOOL_WORSHIP = "tool_worship"
    STATISTICAL = "statistical"

class CoherenceStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"

class Verdict(Enum):
    CREDIBLE = "CREDIBLE"
    SKEPTICAL = "SKEPTICAL"
    DOUBTFUL = "DOUBTFUL"
    REJECT = "REJECT"

@dataclass
class RedFlag:
    id: str
    category: str
    description: str
    severity: str  # "minor", "major", "critical"
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

@dataclass
class CoherenceCheck:
    check_type: str  # e.g., "data-task-match", "metric-task-match"
    status: str  # PASS, FAIL, UNKNOWN
    notes: str = ""
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

@dataclass
class Evidence:
    description: str
    likelihood_ratio: float  # P(E|H) / P(E|¬H)
    timestamp: str
    confirms: bool

@dataclass
class Hypothesis:
    id: str
    statement: str
    phase: str
    prior: float
    posterior: float
    evidence: List[Dict] = field(default_factory=list)
    status: str = Status.ACTIVE.value
    created: str = ""
    updated: str = ""
    
    def __post_init__(self):
        if not self.created:
            self.created = datetime.now().isoformat()
        self.updated = datetime.now().isoformat()

class BayesianTracker:
    """
    Tracks hypotheses with proper Bayesian updating.

    Extended with red flag tracking, coherence checking, and verdict generation
    for RAPID tier claim validation.

    Usage:
        tracker = BayesianTracker("hypotheses.json")
        h1 = tracker.add("System uses REST API", phase="P0", prior=0.6)
        tracker.update(h1, "Found /api/v1 endpoint", likelihood_ratio=5.0)
        tracker.update(h1, "No SOAP headers in traffic", likelihood_ratio=3.0)
        print(tracker.report())

        # Red flag tracking
        tracker.add_flag("methodology", "No baseline comparison")
        tracker.add_flag("results", "Test > Train performance", severity="critical")

        # Coherence tracking
        tracker.add_coherence("data-task-match", "PASS")
        tracker.add_coherence("metric-task-match", "FAIL", "Classification metrics for regression")

        # Get verdict
        print(tracker.get_verdict())
    """

    # Likelihood ratio presets for common evidence types
    LR_PRESETS = {
        'strong_confirm': 10.0,    # Very diagnostic positive evidence
        'moderate_confirm': 3.0,   # Moderately diagnostic
        'weak_confirm': 1.5,       # Slightly favors hypothesis
        'neutral': 1.0,            # No diagnostic value
        'weak_disconfirm': 0.67,   # Slightly disfavors
        'moderate_disconfirm': 0.33,
        'strong_disconfirm': 0.1,
        'falsify': 0.0             # Logically incompatible
    }

    # Valid flag categories
    FLAG_CATEGORIES = [e.value for e in FlagCategory]

    # Valid coherence check types
    COHERENCE_TYPES = [
        'data-task-match',
        'metric-task-match',
        'internal-consistency',
        'methodology-coherence',
        'plausibility'
    ]

    def __init__(self, filepath: str = "hypotheses.json"):
        self.filepath = filepath
        self.hypotheses: Dict[str, Hypothesis] = {}
        self.red_flags: List[RedFlag] = []
        self.coherence_checks: List[CoherenceCheck] = []
        self._next_hypothesis_id: int = 1
        self._next_flag_id: int = 1
        self.load()
    
    def load(self):
        """Load hypotheses, flags, and coherence checks from JSON file."""
        data = load_json(self.filepath)
        if data is not None:
            # Handle both old format (list) and new format (dict)
            if isinstance(data, list):
                # Old format: just hypotheses
                for h in data:
                    self.hypotheses[h['id']] = Hypothesis(**h)
                self._next_hypothesis_id = max(
                    (int(h['id'][1:]) for h in data if h['id'][1:].isdigit()), default=0) + 1
            else:
                # New format: dict with hypotheses, flags, coherence
                for h in data.get('hypotheses', []):
                    self.hypotheses[h['id']] = Hypothesis(**h)
                for f in data.get('red_flags', []):
                    self.red_flags.append(RedFlag(**f))
                for c in data.get('coherence_checks', []):
                    self.coherence_checks.append(CoherenceCheck(**c))
                # Recover monotonic counters (backward-compatible with old format)
                self._next_hypothesis_id = data.get('_next_hypothesis_id',
                    max((int(h['id'][1:]) for h in data.get('hypotheses', []) if h['id'][1:].isdigit()), default=0) + 1)
                self._next_flag_id = data.get('_next_flag_id',
                    max((int(f['id'][1:]) for f in data.get('red_flags', []) if f['id'][1:].isdigit()), default=0) + 1)

    def save(self):
        """Save hypotheses, flags, and coherence checks to JSON file."""
        data = {
            'hypotheses': [asdict(h) for h in self.hypotheses.values()],
            'red_flags': [asdict(f) for f in self.red_flags],
            'coherence_checks': [asdict(c) for c in self.coherence_checks],
            '_next_hypothesis_id': self._next_hypothesis_id,
            '_next_flag_id': self._next_flag_id
        }
        save_json(self.filepath, data)
    
    def add(self, statement: str, phase: str = "P0", prior: float = 0.5) -> str:
        """
        Add a new hypothesis.
        
        Args:
            statement: The hypothesis statement
            phase: Which phase generated this hypothesis
            prior: Initial probability (0-1)
        
        Returns:
            Hypothesis ID
        """
        if not 0 < prior < 1:
            raise ValueError("Prior must be between 0 and 1 (exclusive)")
        
        hid = f"H{self._next_hypothesis_id}"
        self._next_hypothesis_id += 1
        self.hypotheses[hid] = Hypothesis(
            id=hid,
            statement=statement,
            phase=phase,
            prior=prior,
            posterior=prior
        )
        self.save()
        return hid
    
    def update(self, hid: str, evidence_desc: str, 
               likelihood_ratio: Optional[float] = None,
               preset: Optional[str] = None) -> float:
        """
        Update hypothesis with new evidence using Bayes' rule.
        
        The likelihood ratio LR = P(E|H) / P(E|¬H) determines the update:
        - LR > 1: Evidence favors hypothesis
        - LR = 1: Evidence is neutral
        - LR < 1: Evidence disfavors hypothesis
        - LR = 0: Evidence falsifies hypothesis
        
        Args:
            hid: Hypothesis ID
            evidence_desc: Description of evidence
            likelihood_ratio: P(E|H) / P(E|¬H), or use preset
            preset: One of LR_PRESETS keys
        
        Returns:
            New posterior probability
        """
        if hid not in self.hypotheses:
            raise KeyError(f"Hypothesis {hid} not found")
        if not evidence_desc or not evidence_desc.strip():
            raise ValueError("Evidence description must not be empty")

        h = self.hypotheses[hid]

        if h.status == Status.KILLED.value:
            raise ValueError(
                f"Hypothesis {hid} is KILLED and cannot be updated. "
                "Add a new hypothesis instead.")

        # Get likelihood ratio
        if preset:
            lr = self.LR_PRESETS.get(preset)
            if lr is None:
                raise ValueError(f"Unknown preset: {preset}. Use one of {list(self.LR_PRESETS.keys())}")
        elif likelihood_ratio is not None:
            lr = likelihood_ratio
        else:
            raise ValueError("Must provide either likelihood_ratio or preset")
        
        # Bayesian update using shared math (handles division-by-zero)
        if lr == 0:
            h.status = Status.KILLED.value
        new_posterior = bayesian_update(h.posterior, lr)
        
        # Record evidence
        h.evidence.append({
            'description': evidence_desc,
            'likelihood_ratio': lr,
            'prior_before': h.posterior,
            'posterior_after': new_posterior,
            'timestamp': datetime.now().isoformat(),
            'confirms': lr > 1
        })
        
        # Saturation warning
        if new_posterior >= 0.95:
            print(f"Warning: {hid} posterior={new_posterior:.3f} near saturation "
                  "— consider if evidence items are truly independent",
                  file=sys.stderr)
        elif new_posterior <= 0.05:
            print(f"Warning: {hid} posterior={new_posterior:.3f} near zero "
                  "— consider if evidence items are truly independent",
                  file=sys.stderr)

        h.posterior = new_posterior
        h.updated = datetime.now().isoformat()

        # Update status
        # Thresholds for system analysis: tighter bands than PSYCH tier
        # because deterministic I/O evidence is less noisy.
        # See CLAUDE.md "Threshold Bands" table for cross-tracker comparison.
        if new_posterior >= 0.90:
            h.status = Status.CONFIRMED.value
        elif new_posterior <= 0.05:
            h.status = Status.KILLED.value
        elif new_posterior <= 0.2:
            h.status = Status.WEAKENED.value
        else:
            h.status = Status.ACTIVE.value
        
        self.save()
        return new_posterior
    
    def compare(self, h1_id: str, h2_id: str) -> Dict:
        """
        Compute Bayes factor comparing two hypotheses.
        
        K = P(H1|D) / P(H2|D) when priors are equal
        
        Returns:
            Dict with bayes_factor, log10_K, and interpretation
        """
        h1 = self.hypotheses[h1_id]
        h2 = self.hypotheses[h2_id]
        
        if h2.posterior == 0:
            k = float('inf')
            log_k = float('inf')
        elif h1.posterior == 0:
            k = 0
            log_k = float('-inf')
        else:
            k = h1.posterior / h2.posterior
            log_k = math.log10(k)
        
        # Interpretation (Jeffreys scale)
        if log_k > 2:
            interp = "Decisive evidence for H1"
        elif log_k > 1:
            interp = "Strong evidence for H1"
        elif log_k > 0.5:
            interp = "Substantial evidence for H1"
        elif log_k > -0.5:
            interp = "Barely worth mentioning"
        elif log_k > -1:
            interp = "Substantial evidence for H2"
        elif log_k > -2:
            interp = "Strong evidence for H2"
        else:
            interp = "Decisive evidence for H2"
        
        return {
            'bayes_factor': k,
            'log10_K': log_k,
            'interpretation': interp,
            'h1_posterior': h1.posterior,
            'h2_posterior': h2.posterior
        }
    
    def report(self, verbose: bool = False) -> str:
        """Generate hypothesis registry report."""
        lines = [
            "# Hypothesis Registry",
            "",
            "| ID | Statement | Prior | Posterior | Status |",
            "|:---|:----------|------:|----------:|:-------|"
        ]
        
        for h in sorted(self.hypotheses.values(), key=lambda x: x.id):
            stmt = h.statement[:50] + "..." if len(h.statement) > 50 else h.statement
            lines.append(f"| {h.id} | {stmt} | {h.prior:.2f} | {h.posterior:.2f} | {h.status} |")
        
        if verbose:
            lines.append("")
            lines.append("## Evidence Trail")
            for h in self.hypotheses.values():
                if h.evidence:
                    lines.append(f"\n### {h.id}: {h.statement[:60]}")
                    for e in h.evidence:
                        direction = "+" if e['confirms'] else "-"
                        lines.append(f"- [{direction}] LR={e['likelihood_ratio']:.2f}: {e['description']}")
                        lines.append(f"  - {e['prior_before']:.3f} → {e['posterior_after']:.3f}")
        
        return "\n".join(lines)
    
    def get_active(self) -> List[Hypothesis]:
        """Get all active hypotheses."""
        return [h for h in self.hypotheses.values() if h.status == Status.ACTIVE.value]
    
    def get_by_phase(self, phase: str) -> List[Hypothesis]:
        """Get hypotheses from a specific phase."""
        return [h for h in self.hypotheses.values() if h.phase == phase]

    def remove(self, hid: str) -> bool:
        """
        Remove a hypothesis by ID.

        Args:
            hid: Hypothesis ID to remove

        Returns:
            True if removed, False if not found
        """
        if hid in self.hypotheses:
            del self.hypotheses[hid]
            self.save()
            return True
        return False

    # === Red Flag Methods ===

    def add_flag(self, category: str, description: str,
                 severity: str = "major") -> str:
        """
        Add a red flag.

        Args:
            category: One of methodology, documentation, results, claims, tool_worship, statistical
            description: Description of the red flag
            severity: minor, major, or critical

        Returns:
            Flag ID
        """
        if category not in self.FLAG_CATEGORIES:
            raise ValueError(f"Unknown category: {category}. Use one of {self.FLAG_CATEGORIES}")
        if severity not in ['minor', 'major', 'critical']:
            raise ValueError("Severity must be minor, major, or critical")

        fid = f"F{self._next_flag_id}"
        self._next_flag_id += 1
        self.red_flags.append(RedFlag(
            id=fid,
            category=category,
            description=description,
            severity=severity
        ))
        self.save()
        return fid

    def remove_flag(self, flag_id: str):
        """Remove a red flag by ID."""
        self.red_flags = [f for f in self.red_flags if f.id != flag_id]
        self.save()

    def get_flags_by_category(self, category: str) -> List[RedFlag]:
        """Get all flags in a category."""
        return [f for f in self.red_flags if f.category == category]

    def flag_count(self) -> Dict[str, int]:
        """Count flags by category."""
        counts = {cat: 0 for cat in self.FLAG_CATEGORIES}
        for f in self.red_flags:
            counts[f.category] = counts.get(f.category, 0) + 1
        return counts

    def flag_report(self) -> str:
        """Generate red flag report."""
        lines = [
            "# Red Flag Report",
            "",
            f"**Total Flags**: {len(self.red_flags)}",
            ""
        ]

        # Count by category
        counts = self.flag_count()
        categories_with_flags = sum(1 for c in counts.values() if c > 0)
        lines.append(f"**Categories with flags**: {categories_with_flags}")
        lines.append("")

        # List by category
        for cat in self.FLAG_CATEGORIES:
            flags = self.get_flags_by_category(cat)
            if flags:
                lines.append(f"## {cat.title()}")
                for f in flags:
                    sev = f"[{f.severity.upper()}]" if f.severity == "critical" else f"[{f.severity}]"
                    lines.append(f"- {sev} {f.description}")
                lines.append("")

        # Meta-rule check
        if categories_with_flags >= 3:
            lines.append("**WARNING**: 3+ categories with flags - treat entire work as suspect (Meta-Rule)")

        return "\n".join(lines)

    # === Coherence Check Methods ===

    def add_coherence(self, check_type: str, status: str, notes: str = "") -> None:
        """
        Record a coherence check result.

        Args:
            check_type: Type of check (e.g., data-task-match)
            status: PASS, FAIL, or UNKNOWN
            notes: Optional notes
        """
        status_upper = status.upper()
        if status_upper not in ['PASS', 'FAIL', 'UNKNOWN']:
            raise ValueError("Status must be PASS, FAIL, or UNKNOWN")

        # Remove existing check of same type
        self.coherence_checks = [c for c in self.coherence_checks if c.check_type != check_type]

        self.coherence_checks.append(CoherenceCheck(
            check_type=check_type,
            status=status_upper,
            notes=notes
        ))
        self.save()

    def coherence_summary(self) -> Dict[str, str]:
        """Get summary of coherence checks."""
        return {c.check_type: c.status for c in self.coherence_checks}

    def coherence_passed(self) -> bool:
        """Check if all coherence checks passed (no FAIL)."""
        return all(c.status != 'FAIL' for c in self.coherence_checks)

    def coherence_report(self) -> str:
        """Generate coherence check report."""
        lines = [
            "# Coherence Check Report",
            "",
            "| Check Type | Status | Notes |",
            "|------------|--------|-------|"
        ]

        for c in self.coherence_checks:
            notes = c.notes[:50] + "..." if len(c.notes) > 50 else c.notes
            lines.append(f"| {c.check_type} | {c.status} | {notes} |")

        lines.append("")
        passed = sum(1 for c in self.coherence_checks if c.status == 'PASS')
        failed = sum(1 for c in self.coherence_checks if c.status == 'FAIL')
        lines.append(f"**Summary**: {passed} PASS, {failed} FAIL")

        if failed > 0:
            lines.append("")
            lines.append("**WARNING**: Coherence failures detected - claim may be incoherent")

        return "\n".join(lines)

    # === Verdict Methods ===

    def get_verdict(self) -> Dict:
        """
        Generate overall verdict based on flags and coherence.

        Returns:
            Dict with verdict, reasoning, and details
        """
        # Check for instant rejects (critical flags or coherence failures)
        critical_flags = [f for f in self.red_flags if f.severity == "critical"]
        coherence_failures = [c for c in self.coherence_checks if c.status == "FAIL"]

        total_flags = len(self.red_flags)
        categories_with_flags = sum(1 for c in self.flag_count().values() if c > 0)

        # Determine verdict
        if critical_flags or coherence_failures:
            verdict = Verdict.REJECT.value
            reason = "Critical flag or coherence failure"
        elif total_flags > 5 or categories_with_flags >= 4:
            verdict = Verdict.REJECT.value
            reason = f"Too many red flags ({total_flags}) across {categories_with_flags} categories"
        elif total_flags >= 4 or categories_with_flags >= 3:
            verdict = Verdict.DOUBTFUL.value
            reason = f"{total_flags} red flags across {categories_with_flags} categories (Meta-Rule triggered)"
        elif total_flags >= 2:
            verdict = Verdict.SKEPTICAL.value
            reason = f"{total_flags} red flags - increased scrutiny warranted"
        else:
            verdict = Verdict.CREDIBLE.value
            reason = f"Only {total_flags} red flags, coherence checks passed"

        return {
            'verdict': verdict,
            'reason': reason,
            'total_flags': total_flags,
            'categories_with_flags': categories_with_flags,
            'critical_flags': len(critical_flags),
            'coherence_failures': len(coherence_failures),
            'coherence_passed': self.coherence_passed()
        }

    def verdict_report(self) -> str:
        """Generate full verdict report."""
        v = self.get_verdict()

        lines = [
            "# Verdict Report",
            "",
            f"## Verdict: **{v['verdict']}**",
            "",
            f"**Reason**: {v['reason']}",
            "",
            "## Summary",
            f"- Total red flags: {v['total_flags']}",
            f"- Categories with flags: {v['categories_with_flags']}",
            f"- Critical flags: {v['critical_flags']}",
            f"- Coherence failures: {v['coherence_failures']}",
            f"- Coherence passed: {v['coherence_passed']}",
            "",
            "---",
            ""
        ]

        # Include flag and coherence details
        lines.append(self.flag_report())
        lines.append("")
        lines.append(self.coherence_report())

        return "\n".join(lines)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Bayesian Hypothesis Tracker")
    parser.add_argument("--file", default="hypotheses.json", help="Hypothesis file")
    
    subparsers = parser.add_subparsers(dest="cmd", help="Commands")
    
    # Add command
    add_p = subparsers.add_parser("add", help="Add hypothesis")
    add_p.add_argument("statement", help="Hypothesis statement")
    add_p.add_argument("--phase", default="P0", help="Phase (P0-P5)")
    add_p.add_argument("--prior", type=float, default=0.5, help="Prior probability")
    
    # Update command
    upd_p = subparsers.add_parser("update", help="Update with evidence")
    upd_p.add_argument("id", help="Hypothesis ID")
    upd_p.add_argument("evidence", help="Evidence description")
    upd_p.add_argument("--lr", type=float, help="Likelihood ratio")
    upd_p.add_argument("--preset", choices=list(BayesianTracker.LR_PRESETS.keys()),
                       help="Use preset likelihood ratio")
    
    # Compare command
    cmp_p = subparsers.add_parser("compare", help="Compare two hypotheses")
    cmp_p.add_argument("h1", help="First hypothesis ID")
    cmp_p.add_argument("h2", help="Second hypothesis ID")

    # Remove command
    rm_p = subparsers.add_parser("remove", help="Remove a hypothesis")
    rm_p.add_argument("id", help="Hypothesis ID to remove")

    # Report command
    rep_p = subparsers.add_parser("report", help="Generate report")
    rep_p.add_argument("--verbose", "-v", action="store_true", help="Include evidence trail")

    # === Red Flag Commands ===
    # Flag add command
    flag_p = subparsers.add_parser("flag", help="Red flag management")
    flag_sub = flag_p.add_subparsers(dest="flag_cmd")

    flag_add = flag_sub.add_parser("add", help="Add a red flag")
    flag_add.add_argument("category", choices=BayesianTracker.FLAG_CATEGORIES,
                          help="Flag category")
    flag_add.add_argument("description", help="Flag description")
    flag_add.add_argument("--severity", choices=['minor', 'major', 'critical'],
                          default='major', help="Flag severity")

    flag_rm = flag_sub.add_parser("remove", help="Remove a red flag")
    flag_rm.add_argument("flag_id", help="Flag ID to remove")

    flag_sub.add_parser("report", help="Generate red flag report")
    flag_sub.add_parser("count", help="Count flags by category")

    # === Coherence Commands ===
    coh_p = subparsers.add_parser("coherence", help="Coherence check management")
    coh_p.add_argument("check_type", help="Type of coherence check")
    coh_p.add_argument("--pass", dest="status_pass", action="store_true",
                       help="Mark check as PASS")
    coh_p.add_argument("--fail", dest="status_fail", action="store_true",
                       help="Mark check as FAIL")
    coh_p.add_argument("--notes", default="", help="Optional notes")

    # Coherence report
    coh_rep = subparsers.add_parser("coherence-report", help="Generate coherence report")

    # === Verdict Command ===
    verdict_p = subparsers.add_parser("verdict", help="Generate verdict")
    verdict_p.add_argument("--full", action="store_true", help="Include full report")

    args = parser.parse_args()
    tracker = BayesianTracker(args.file)
    
    try:
        if args.cmd == "add":
            hid = tracker.add(args.statement, args.phase, args.prior)
            print(f"Added: {hid} (prior={args.prior})")

        elif args.cmd == "update":
            if not args.lr and not args.preset:
                print("Error: Must specify --lr or --preset")
                sys.exit(1)
            new_p = tracker.update(args.id, args.evidence,
                                   likelihood_ratio=args.lr, preset=args.preset)
            print(f"Updated {args.id}: posterior={new_p:.3f}")

        elif args.cmd == "compare":
            result = tracker.compare(args.h1, args.h2)
            print(f"Bayes Factor K = {result['bayes_factor']:.2f}")
            print(f"log₁₀(K) = {result['log10_K']:.2f}")
            print(f"Interpretation: {result['interpretation']}")

        elif args.cmd == "remove":
            if tracker.remove(args.id):
                print(f"Removed: {args.id}")
            else:
                print(f"Error: Hypothesis {args.id} not found")
                sys.exit(1)

        elif args.cmd == "report":
            print(tracker.report(verbose=args.verbose))

        # === Red Flag Commands ===
        elif args.cmd == "flag":
            if args.flag_cmd == "add":
                fid = tracker.add_flag(args.category, args.description, args.severity)
                print(f"Added flag: {fid} [{args.severity}] ({args.category})")
            elif args.flag_cmd == "remove":
                tracker.remove_flag(args.flag_id)
                print(f"Removed flag: {args.flag_id}")
            elif args.flag_cmd == "report":
                print(tracker.flag_report())
            elif args.flag_cmd == "count":
                counts = tracker.flag_count()
                total = sum(counts.values())
                print(f"Total flags: {total}")
                for cat, count in counts.items():
                    if count > 0:
                        print(f"  {cat}: {count}")
            else:
                flag_p.print_help()

        # === Coherence Commands ===
        elif args.cmd == "coherence":
            if args.status_pass:
                status = "PASS"
            elif args.status_fail:
                status = "FAIL"
            else:
                print("Error: Must specify --pass or --fail")
                sys.exit(1)
            tracker.add_coherence(args.check_type, status, args.notes)
            print(f"Recorded: {args.check_type} = {status}")

        elif args.cmd == "coherence-report":
            print(tracker.coherence_report())

        # === Verdict Command ===
        elif args.cmd == "verdict":
            if args.full:
                print(tracker.verdict_report())
            else:
                v = tracker.get_verdict()
                print(f"Verdict: {v['verdict']}")
                print(f"Reason: {v['reason']}")
                print(f"Flags: {v['total_flags']} ({v['categories_with_flags']} categories)")

        else:
            parser.print_help()

    except (KeyError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
