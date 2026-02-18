#!/usr/bin/env python3
"""
RAPID Checker for Epistemic Deconstruction v6.0
Standalone 10-minute assessment tool for RAPID tier claim validation.

Usage:
    python rapid_checker.py start "Paper: XYZ Claims"
    python rapid_checker.py coherence data-task-match --pass
    python rapid_checker.py coherence metric-task-match --fail --notes "Classification metrics for regression"
    python rapid_checker.py flag methodology "No baseline comparison"
    python rapid_checker.py calibrate accuracy 0.99 --domain ml_classification
    python rapid_checker.py verdict
    python rapid_checker.py report
"""

import json
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from common import load_json, save_json


class Verdict(Enum):
    CREDIBLE = "CREDIBLE"
    SKEPTICAL = "SKEPTICAL"
    DOUBTFUL = "DOUBTFUL"
    REJECT = "REJECT"


def load_domain_calibration() -> Dict:
    """
    Load domain calibration from config/domains.json.
    Falls back to built-in defaults if config file not found.
    """
    # Try to load from config file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, '..', 'config', 'domains.json')

    if os.path.exists(config_path):
        with open(config_path) as f:
            data = json.load(f)
            # Remove comment key and convert lists to tuples
            result = {}
            for domain, metrics in data.items():
                if domain.startswith('_'):
                    continue
                result[domain] = {}
                for metric, bounds in metrics.items():
                    result[domain][metric] = tuple(bounds)
            return result

    # Fallback to built-in defaults
    return {
        'ml_classification': {
            'accuracy': (0.99, 0.70, 0.90, 0.98),
            'f1': (0.99, 0.70, 0.90, 0.98),
            'auc': (0.999, 0.70, 0.90, 0.98),
        },
        'ml_regression': {
            'r2': (0.99, 0.70, 0.90, 0.98),
            'mape': (0.01, 0.15, 0.05, 0.02),
        },
        'financial_prediction': {
            'directional_accuracy': (0.65, 0.52, 0.58, 0.62),
            'sharpe': (3.0, 0.5, 1.5, 2.5),
            'r2_returns': (0.30, 0.01, 0.05, 0.15),
            'r2_prices': (0.95, None, None, None),
            'annual_alpha': (0.30, 0.02, 0.08, 0.15),
            'mcc': (0.60, 0.02, 0.10, 0.25),
            'max_drawdown': (0.05, 0.50, 0.20, 0.10),
            'r2_returns_train': (0.05, 0.80, 0.50, 0.30),
        },
        'engineering': {
            'r2_insample': (1.0, 0.85, 0.95, 0.99),
            'r2_outsample': (0.99, 0.80, 0.90, 0.98),
            'mape': (0.01, 0.15, 0.05, 0.02),
        },
        'medical': {
            'sensitivity': (0.99, 0.70, 0.85, 0.95),
            'specificity': (0.99, 0.70, 0.85, 0.95),
            'auc': (0.99, 0.70, 0.85, 0.95),
        },
        'organizational': {
            'employee_retention': (0.99, 0.70, 0.85, 0.95),
            'revenue_growth': (0.50, 0.02, 0.15, 0.30),
            'cost_reduction': (0.60, 0.05, 0.20, 0.40),
            'customer_satisfaction': (0.99, 0.60, 0.80, 0.95),
            'process_efficiency_gain': (0.80, 0.05, 0.25, 0.50),
        }
    }


# Domain calibration bounds: {domain: {metric: (suspicious, plausible_low, plausible_high, excellent)}}
DOMAIN_CALIBRATION = load_domain_calibration()


@dataclass
class Assessment:
    """Container for a RAPID assessment session."""
    id: str
    title: str
    created: str
    coherence_checks: Dict[str, Dict] = field(default_factory=dict)
    red_flags: List[Dict] = field(default_factory=list)
    calibrations: List[Dict] = field(default_factory=list)
    notes: str = ""
    verdict: Optional[str] = None
    verdict_reason: Optional[str] = None
    next_flag_id: int = 1


class RapidChecker:
    """
    RAPID tier assessment tool.

    Implements the 5-step RAPID workflow:
    1. Coherence Check (2 min)
    2. Verifiability Check (2 min)
    3. Red Flag Scan (3 min)
    4. Domain Calibration (3 min)
    5. Verdict
    """

    FLAG_CATEGORIES = [
        'methodology', 'documentation', 'results',
        'claims', 'tool_worship', 'statistical'
    ]

    COHERENCE_TYPES = [
        'data-task-match',
        'metric-task-match',
        'internal-consistency',
        'verifiable-data',
        'verifiable-method',
        'plausible-claims'
    ]

    def __init__(self, filepath: str = "rapid_assessment.json"):
        self.filepath = filepath
        self.assessment: Optional[Assessment] = None
        self.load()

    def load(self):
        """Load assessment from file."""
        data = load_json(self.filepath)
        if data is not None:
            # Recover next_flag_id if missing (backward-compatible with old format)
            if 'next_flag_id' not in data:
                flags = data.get('red_flags', [])
                data['next_flag_id'] = max(
                    (int(f['id'][1:]) for f in flags if f.get('id', '')[1:].isdigit()), default=0) + 1
            self.assessment = Assessment(**data)

    def save(self):
        """Save assessment to file."""
        if self.assessment:
            save_json(self.filepath, asdict(self.assessment))

    def start(self, title: str) -> str:
        """Start a new assessment session."""
        aid = f"A{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.assessment = Assessment(
            id=aid,
            title=title,
            created=datetime.now().isoformat()
        )
        self.save()
        return aid

    def require_session(self):
        """Ensure an assessment session exists."""
        if not self.assessment:
            raise RuntimeError("No active assessment. Run 'start' first.")

    # === Coherence Methods ===

    def add_coherence(self, check_type: str, passed: bool, notes: str = ""):
        """Record a coherence check result."""
        self.require_session()
        self.assessment.coherence_checks[check_type] = {
            'passed': passed,
            'notes': notes,
            'timestamp': datetime.now().isoformat()
        }
        self.save()

    def coherence_summary(self) -> Dict:
        """Get coherence check summary."""
        self.require_session()
        checks = self.assessment.coherence_checks
        passed = sum(1 for c in checks.values() if c['passed'])
        failed = sum(1 for c in checks.values() if not c['passed'])
        return {
            'total': len(checks),
            'passed': passed,
            'failed': failed,
            'all_passed': failed == 0
        }

    # === Red Flag Methods ===

    def add_flag(self, category: str, description: str, severity: str = "major"):
        """Add a red flag."""
        self.require_session()
        if category not in self.FLAG_CATEGORIES:
            raise ValueError(f"Invalid category. Use: {self.FLAG_CATEGORIES}")
        if severity not in ['minor', 'major', 'critical']:
            raise ValueError("Severity must be: minor, major, critical")

        fid = f"F{self.assessment.next_flag_id}"
        self.assessment.next_flag_id += 1
        self.assessment.red_flags.append({
            'id': fid,
            'category': category,
            'description': description,
            'severity': severity,
            'timestamp': datetime.now().isoformat()
        })
        self.save()
        return fid

    def flag_count(self) -> Dict:
        """Count flags by category and severity."""
        self.require_session()
        by_category = {cat: 0 for cat in self.FLAG_CATEGORIES}
        by_severity = {'minor': 0, 'major': 0, 'critical': 0}
        categories_with_flags = set()

        for f in self.assessment.red_flags:
            by_category[f['category']] += 1
            by_severity[f['severity']] += 1
            categories_with_flags.add(f['category'])

        return {
            'total': len(self.assessment.red_flags),
            'by_category': by_category,
            'by_severity': by_severity,
            'categories_with_flags': len(categories_with_flags)
        }

    # === Calibration Methods ===

    def calibrate(self, metric: str, value: float, domain: str) -> Dict:
        """
        Check a metric value against domain calibration bounds.

        Returns assessment: 'suspicious', 'plausible', 'excellent', or 'unknown'
        """
        self.require_session()

        if domain not in DOMAIN_CALIBRATION:
            assessment = 'unknown'
            reason = f"No calibration for domain: {domain}"
        elif metric not in DOMAIN_CALIBRATION[domain]:
            assessment = 'unknown'
            reason = f"No calibration for metric: {metric} in {domain}"
        else:
            bounds = DOMAIN_CALIBRATION[domain][metric]
            suspicious, plaus_low, plaus_high, excellent = bounds

            # Special case: r2_prices (always suspicious if high)
            if plaus_low is None:
                if value >= suspicious:
                    assessment = 'suspicious'
                    reason = f"{value} >= {suspicious} (suspicious threshold)"
                else:
                    assessment = 'plausible'
                    reason = f"{value} < {suspicious}"
            # Normal case
            elif metric in ['mape', 'max_drawdown', 'r2_returns_train']:  # Lower is better
                if value <= suspicious:
                    assessment = 'suspicious'
                    reason = f"{value} <= {suspicious} (too good)"
                elif value <= excellent:
                    assessment = 'excellent'
                    reason = f"{value} in excellent range"
                elif value <= plaus_high:
                    assessment = 'plausible'
                    reason = f"{value} in plausible range"
                else:
                    assessment = 'plausible'
                    reason = f"{value} > {plaus_high} (poor but plausible)"
            else:  # Higher is better
                if value >= suspicious:
                    assessment = 'suspicious'
                    reason = f"{value} >= {suspicious} (too good)"
                elif value >= excellent:
                    assessment = 'excellent'
                    reason = f"{value} in excellent range"
                elif value >= plaus_low:
                    assessment = 'plausible'
                    reason = f"{value} in plausible range"
                else:
                    assessment = 'plausible'
                    reason = f"{value} < {plaus_low} (poor but plausible)"

        result = {
            'metric': metric,
            'value': value,
            'domain': domain,
            'assessment': assessment,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        }

        self.assessment.calibrations.append(result)
        self.save()
        return result

    # === Verdict Methods ===

    def compute_verdict(self) -> Dict:
        """Compute the overall verdict."""
        self.require_session()

        coherence = self.coherence_summary()
        flags = self.flag_count()

        # Check for instant rejects
        critical_flags = flags['by_severity']['critical']
        coherence_failures = coherence['failed']
        suspicious_calibrations = sum(
            1 for c in self.assessment.calibrations
            if c['assessment'] == 'suspicious'
        )

        total_flags = flags['total']
        categories_with_flags = flags['categories_with_flags']

        # Determine verdict
        if critical_flags > 0 or coherence_failures > 0 or suspicious_calibrations > 0:
            verdict = Verdict.REJECT.value
            reasons = []
            if critical_flags > 0:
                reasons.append(f"{critical_flags} critical flag(s)")
            if coherence_failures > 0:
                reasons.append(f"{coherence_failures} coherence failure(s)")
            if suspicious_calibrations > 0:
                reasons.append(f"{suspicious_calibrations} suspicious calibration(s)")
            reason = "Instant reject: " + ", ".join(reasons)

        elif total_flags > 5 or categories_with_flags >= 4:
            verdict = Verdict.REJECT.value
            reason = f"{total_flags} flags across {categories_with_flags} categories"

        elif total_flags >= 4 or categories_with_flags >= 3:
            verdict = Verdict.DOUBTFUL.value
            reason = f"{total_flags} flags across {categories_with_flags} categories (Meta-Rule)"

        elif total_flags >= 2:
            verdict = Verdict.SKEPTICAL.value
            reason = f"{total_flags} flags warrant increased scrutiny"

        else:
            verdict = Verdict.CREDIBLE.value
            reason = f"Only {total_flags} flag(s), coherence passed, calibrations OK"

        # Update assessment
        self.assessment.verdict = verdict
        self.assessment.verdict_reason = reason
        self.save()

        return {
            'verdict': verdict,
            'reason': reason,
            'coherence': coherence,
            'flags': flags,
            'calibrations': {
                'total': len(self.assessment.calibrations),
                'suspicious': suspicious_calibrations
            }
        }

    # === Reporting Methods ===

    def report(self) -> str:
        """Generate full assessment report."""
        self.require_session()
        v = self.compute_verdict()

        lines = [
            f"# RAPID Assessment Report",
            f"",
            f"**Assessment ID**: {self.assessment.id}",
            f"**Title**: {self.assessment.title}",
            f"**Created**: {self.assessment.created}",
            f"",
            f"---",
            f"",
            f"## Verdict: **{v['verdict']}**",
            f"",
            f"**Reason**: {v['reason']}",
            f"",
            f"---",
            f"",
            f"## Coherence Checks",
            f""
        ]

        if self.assessment.coherence_checks:
            lines.append("| Check | Status | Notes |")
            lines.append("|-------|--------|-------|")
            for check_type, data in self.assessment.coherence_checks.items():
                status = "PASS" if data['passed'] else "FAIL"
                notes = data['notes'][:30] + "..." if len(data['notes']) > 30 else data['notes']
                lines.append(f"| {check_type} | {status} | {notes} |")
        else:
            lines.append("*No coherence checks recorded*")

        lines.extend([
            f"",
            f"**Summary**: {v['coherence']['passed']} PASS, {v['coherence']['failed']} FAIL",
            f"",
            f"---",
            f"",
            f"## Red Flags ({v['flags']['total']} total)",
            f""
        ])

        if self.assessment.red_flags:
            for f in self.assessment.red_flags:
                sev = f"**{f['severity'].upper()}**" if f['severity'] == 'critical' else f['severity']
                lines.append(f"- [{sev}] ({f['category']}) {f['description']}")
        else:
            lines.append("*No red flags recorded*")

        if v['flags']['categories_with_flags'] >= 3:
            lines.append("")
            lines.append("**WARNING**: 3+ categories with flags - Meta-Rule triggered")

        lines.extend([
            f"",
            f"---",
            f"",
            f"## Calibration Checks",
            f""
        ])

        if self.assessment.calibrations:
            lines.append("| Metric | Value | Domain | Assessment |")
            lines.append("|--------|-------|--------|------------|")
            for c in self.assessment.calibrations:
                lines.append(f"| {c['metric']} | {c['value']} | {c['domain']} | {c['assessment'].upper()} |")

            if v['calibrations']['suspicious'] > 0:
                lines.append("")
                lines.append(f"**WARNING**: {v['calibrations']['suspicious']} suspicious calibration(s)")
        else:
            lines.append("*No calibration checks recorded*")

        lines.extend([
            f"",
            f"---",
            f"",
            f"## State Block",
            f"```",
            f"[STATE: Phase 0.5 | Tier: RAPID | Coherence: {'PASS' if v['coherence']['all_passed'] else 'FAIL'} | Red Flags: {v['flags']['total']} | Verdict: {v['verdict']}]",
            f"```"
        ])

        return "\n".join(lines)

    def status(self) -> str:
        """Get quick status summary."""
        self.require_session()
        v = self.compute_verdict()
        return (
            f"Assessment: {self.assessment.title}\n"
            f"Coherence: {v['coherence']['passed']}/{v['coherence']['total']} passed\n"
            f"Red Flags: {v['flags']['total']} ({v['flags']['categories_with_flags']} categories)\n"
            f"Calibrations: {v['calibrations']['total']} ({v['calibrations']['suspicious']} suspicious)\n"
            f"Verdict: {v['verdict']}"
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="RAPID Checker - 10-minute claim assessment tool"
    )
    parser.add_argument("--file", default="rapid_assessment.json",
                        help="Assessment file")

    subparsers = parser.add_subparsers(dest="cmd", help="Commands")

    # Start command
    start_p = subparsers.add_parser("start", help="Start new assessment")
    start_p.add_argument("title", help="Assessment title")

    # Coherence command
    coh_p = subparsers.add_parser("coherence", help="Record coherence check")
    coh_p.add_argument("check_type", help="Type of check")
    coh_p.add_argument("--pass", dest="passed", action="store_true",
                       help="Mark as PASS")
    coh_p.add_argument("--fail", dest="failed", action="store_true",
                       help="Mark as FAIL")
    coh_p.add_argument("--notes", default="", help="Notes")

    # Flag command
    flag_p = subparsers.add_parser("flag", help="Add red flag")
    flag_p.add_argument("category", choices=RapidChecker.FLAG_CATEGORIES,
                        help="Flag category")
    flag_p.add_argument("description", help="Flag description")
    flag_p.add_argument("--severity", choices=['minor', 'major', 'critical'],
                        default='major', help="Severity")

    # Calibrate command
    cal_p = subparsers.add_parser("calibrate", help="Check metric calibration")
    cal_p.add_argument("metric", help="Metric name")
    cal_p.add_argument("value", type=float, help="Metric value")
    cal_p.add_argument("--domain", required=True,
                       choices=list(DOMAIN_CALIBRATION.keys()),
                       help="Domain for calibration")

    # Verdict command
    subparsers.add_parser("verdict", help="Get verdict")

    # Report command
    subparsers.add_parser("report", help="Generate full report")

    # Status command
    subparsers.add_parser("status", help="Quick status")

    # Domains command
    subparsers.add_parser("domains", help="List available domains and metrics")

    args = parser.parse_args()
    checker = RapidChecker(args.file)

    try:
        if args.cmd == "start":
            aid = checker.start(args.title)
            print(f"Started assessment: {aid}")
            print(f"Title: {args.title}")

        elif args.cmd == "coherence":
            if args.passed:
                passed = True
            elif args.failed:
                passed = False
            else:
                print("Error: Must specify --pass or --fail")
                sys.exit(1)
            checker.add_coherence(args.check_type, passed, args.notes)
            status = "PASS" if passed else "FAIL"
            print(f"Recorded: {args.check_type} = {status}")

        elif args.cmd == "flag":
            fid = checker.add_flag(args.category, args.description, args.severity)
            print(f"Added: {fid} [{args.severity}] ({args.category})")
            print(f"  {args.description}")

        elif args.cmd == "calibrate":
            result = checker.calibrate(args.metric, args.value, args.domain)
            print(f"Calibration: {result['assessment'].upper()}")
            print(f"  {result['reason']}")

        elif args.cmd == "verdict":
            v = checker.compute_verdict()
            print(f"Verdict: {v['verdict']}")
            print(f"Reason: {v['reason']}")

        elif args.cmd == "report":
            print(checker.report())

        elif args.cmd == "status":
            print(checker.status())

        elif args.cmd == "domains":
            print("Available domains and metrics:")
            for domain, metrics in DOMAIN_CALIBRATION.items():
                print(f"\n  {domain}:")
                for metric, bounds in metrics.items():
                    print(f"    - {metric}: suspicious >= {bounds[0]}")

        else:
            parser.print_help()

    except (KeyError, ValueError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
