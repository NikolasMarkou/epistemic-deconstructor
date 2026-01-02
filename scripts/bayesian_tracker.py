#!/usr/bin/env python3
"""
Bayesian Hypothesis Tracker for Epistemic Deconstruction v6.0
Implements proper Bayesian updating with likelihood ratios.
"""

import json
import math
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict

class Status(Enum):
    ACTIVE = "ACTIVE"
    WEAKENED = "WEAKENED"
    KILLED = "KILLED"
    CONFIRMED = "CONFIRMED"

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
    
    Usage:
        tracker = BayesianTracker("hypotheses.json")
        h1 = tracker.add("System uses REST API", phase="P0", prior=0.6)
        tracker.update(h1, "Found /api/v1 endpoint", likelihood_ratio=5.0)
        tracker.update(h1, "No SOAP headers in traffic", likelihood_ratio=3.0)
        print(tracker.report())
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
    
    def __init__(self, filepath: str = "hypotheses.json"):
        self.filepath = filepath
        self.hypotheses: Dict[str, Hypothesis] = {}
        self.load()
    
    def load(self):
        """Load hypotheses from JSON file."""
        if os.path.exists(self.filepath):
            with open(self.filepath) as f:
                data = json.load(f)
                for h in data:
                    self.hypotheses[h['id']] = Hypothesis(**h)
    
    def save(self):
        """Save hypotheses to JSON file."""
        with open(self.filepath, 'w') as f:
            json.dump([asdict(h) for h in self.hypotheses.values()], f, indent=2)
    
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
        
        hid = f"H{len(self.hypotheses) + 1}"
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
        
        h = self.hypotheses[hid]
        
        # Get likelihood ratio
        if preset:
            lr = self.LR_PRESETS.get(preset)
            if lr is None:
                raise ValueError(f"Unknown preset: {preset}. Use one of {list(self.LR_PRESETS.keys())}")
        elif likelihood_ratio is not None:
            lr = likelihood_ratio
        else:
            raise ValueError("Must provide either likelihood_ratio or preset")
        
        # Bayesian update: posterior_odds = prior_odds × likelihood_ratio
        # P(H|E) = P(H) × LR / (P(H) × LR + P(¬H))
        if lr == 0:
            new_posterior = 0.0
            h.status = Status.KILLED.value
        else:
            prior_odds = h.posterior / (1 - h.posterior)
            posterior_odds = prior_odds * lr
            new_posterior = posterior_odds / (1 + posterior_odds)
        
        # Record evidence
        h.evidence.append({
            'description': evidence_desc,
            'likelihood_ratio': lr,
            'prior_before': h.posterior,
            'posterior_after': new_posterior,
            'timestamp': datetime.now().isoformat(),
            'confirms': lr > 1
        })
        
        h.posterior = new_posterior
        h.updated = datetime.now().isoformat()
        
        # Update status
        if new_posterior >= 0.95:
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
    
    # Report command
    rep_p = subparsers.add_parser("report", help="Generate report")
    rep_p.add_argument("--verbose", "-v", action="store_true", help="Include evidence trail")
    
    args = parser.parse_args()
    tracker = BayesianTracker(args.file)
    
    if args.cmd == "add":
        hid = tracker.add(args.statement, args.phase, args.prior)
        print(f"Added: {hid} (prior={args.prior})")
    
    elif args.cmd == "update":
        if not args.lr and not args.preset:
            parser.error("Must specify --lr or --preset")
        new_p = tracker.update(args.id, args.evidence, 
                               likelihood_ratio=args.lr, preset=args.preset)
        print(f"Updated {args.id}: posterior={new_p:.3f}")
    
    elif args.cmd == "compare":
        result = tracker.compare(args.h1, args.h2)
        print(f"Bayes Factor K = {result['bayes_factor']:.2f}")
        print(f"log₁₀(K) = {result['log10_K']:.2f}")
        print(f"Interpretation: {result['interpretation']}")
    
    elif args.cmd == "report":
        print(tracker.report(verbose=args.verbose))
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
