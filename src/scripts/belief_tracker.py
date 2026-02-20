#!/usr/bin/env python3
"""
Belief Tracker for Psychological Profiling (PSYCH Tier)
Tracks trait confidence using Bayesian inference, baseline observations,
and deviations for psychological analysis.

Adapted from bayesian_tracker.py for human behavioral analysis.
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

# Minimum posterior score for a MICE motivation to appear in ranked display
MICE_DISPLAY_THRESHOLD = 0.3


class TraitCategory(Enum):
    """Categories of psychological traits."""
    # Big Five (OCEAN)
    OCEAN_O = "openness"
    OCEAN_C = "conscientiousness"
    OCEAN_E = "extraversion"
    OCEAN_A = "agreeableness"
    OCEAN_N = "neuroticism"

    # Dark Triad
    DARK_NARC = "narcissism"
    DARK_MACH = "machiavellianism"
    DARK_PSYCH = "psychopathy"

    # MICE Motivations
    MOTIVE_MONEY = "money"
    MOTIVE_IDEOLOGY = "ideology"
    MOTIVE_COERCION = "coercion"
    MOTIVE_EGO = "ego"

    # Other
    COGNITIVE = "cognitive_style"
    COMMUNICATION = "communication"
    CUSTOM = "custom"


class TraitLevel(Enum):
    """Trait intensity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class BaselineCategory(Enum):
    """Categories for baseline observations."""
    LINGUISTIC = "linguistic"
    EMOTIONAL = "emotional"
    TIMING = "timing"
    BEHAVIORAL = "behavioral"
    IDIOSYNCRASY = "idiosyncrasy"


@dataclass
class Evidence:
    """Evidence supporting a trait assessment."""
    description: str
    likelihood_ratio: float
    timestamp: str
    confirms: bool
    context: str = ""


@dataclass
class TraitHypothesis:
    """A hypothesis about a psychological trait."""
    id: str
    trait: str
    category: str
    polarity: str  # "high" or "low"
    prior: float
    posterior: float
    evidence: List[Dict] = field(default_factory=list)
    status: str = "ACTIVE"
    created: str = ""
    updated: str = ""

    def __post_init__(self):
        if not self.created:
            self.created = datetime.now().isoformat()
        self.updated = datetime.now().isoformat()


@dataclass
class BaselineObservation:
    """A baseline observation about the subject."""
    id: str
    category: str
    description: str
    value: str
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class DeviationRecord:
    """A recorded deviation from baseline."""
    id: str
    description: str
    baseline_reference: str  # ID of baseline being deviated from
    context: str
    significance: str  # "minor", "moderate", "major"
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class BeliefTracker:
    """
    Tracks psychological trait hypotheses with Bayesian updating.

    Extended with baseline tracking and deviation detection for
    PSYCH tier behavioral analysis.

    Usage:
        tracker = BeliefTracker("profile.json")

        # Add trait hypothesis
        t1 = tracker.add_trait("High Neuroticism", category="OCEAN_N",
                               polarity="high", prior=0.5)

        # Update with evidence
        tracker.update_trait(t1, "Catastrophizing language in email",
                            preset="strong_indicator")

        # Track baseline
        tracker.add_baseline("linguistic", "Average sentence length: 15 words")

        # Record deviation
        tracker.add_deviation("Sentence length dropped to 5 words",
                             context="Under deadline pressure",
                             significance="moderate")

        # Generate profile
        print(tracker.profile_report())
    """

    # Likelihood ratio presets for behavioral evidence
    LR_PRESETS = {
        'smoking_gun': 20.0,       # Direct admission, unambiguous evidence
        'strong_indicator': 5.0,   # Consistent pattern across contexts
        'indicator': 2.0,          # Single clear occurrence
        'weak_indicator': 1.5,     # Suggestive but not definitive
        'neutral': 1.0,            # No diagnostic value
        'weak_counter': 0.67,      # Slightly contradicts
        'counter_indicator': 0.5,  # Single clear contradiction
        'strong_counter': 0.2,     # Pattern contradicts
        'disconfirm': 0.1          # Strong evidence against
    }

    def __init__(self, filepath: str = "profile.json"):
        self.filepath = filepath
        self.traits: Dict[str, TraitHypothesis] = {}
        self.baselines: List[BaselineObservation] = []
        self.deviations: List[DeviationRecord] = []
        self.subject_name: str = "Unknown Subject"
        self.analysis_context: str = ""
        self._next_trait_id: int = 1
        self._next_baseline_id: int = 1
        self._next_deviation_id: int = 1
        self.load()

    def load(self):
        """Load profile from JSON file."""
        data = load_json(self.filepath)
        if data is not None:
            self.subject_name = data.get('subject_name', 'Unknown Subject')
            self.analysis_context = data.get('analysis_context', '')

            for t in data.get('traits', []):
                self.traits[t['id']] = TraitHypothesis(**t)
            for b in data.get('baselines', []):
                self.baselines.append(BaselineObservation(**b))
            for d in data.get('deviations', []):
                self.deviations.append(DeviationRecord(**d))
            # Recover monotonic counters (backward-compatible with old format)
            self._next_trait_id = data.get('_next_trait_id',
                max((int(t['id'][1:]) for t in data.get('traits', []) if t['id'][1:].isdigit()), default=0) + 1)
            self._next_baseline_id = data.get('_next_baseline_id',
                max((int(b['id'][1:]) for b in data.get('baselines', []) if b['id'][1:].isdigit()), default=0) + 1)
            self._next_deviation_id = data.get('_next_deviation_id',
                max((int(d['id'][1:]) for d in data.get('deviations', []) if d['id'][1:].isdigit()), default=0) + 1)

    def save(self):
        """Save profile to JSON file."""
        data = {
            'subject_name': self.subject_name,
            'analysis_context': self.analysis_context,
            'traits': [asdict(t) for t in self.traits.values()],
            'baselines': [asdict(b) for b in self.baselines],
            'deviations': [asdict(d) for d in self.deviations],
            '_next_trait_id': self._next_trait_id,
            '_next_baseline_id': self._next_baseline_id,
            '_next_deviation_id': self._next_deviation_id
        }
        save_json(self.filepath, data)

    def set_subject(self, name: str, context: str = ""):
        """Set subject name and analysis context."""
        self.subject_name = name
        self.analysis_context = context
        self.save()

    # === Trait Methods ===

    def add_trait(self, trait: str, category: str = "custom",
                  polarity: str = "high", prior: float = 0.5) -> str:
        """
        Add a trait hypothesis.

        Args:
            trait: Description of the trait (e.g., "High Neuroticism")
            category: TraitCategory value (e.g., "OCEAN_N", "DARK_NARC")
            polarity: "high" or "low"
            prior: Initial probability (0-1)

        Returns:
            Trait ID
        """
        if not 0 < prior < 1:
            raise ValueError("Prior must be between 0 and 1 (exclusive)")

        tid = f"T{self._next_trait_id}"
        self._next_trait_id += 1
        self.traits[tid] = TraitHypothesis(
            id=tid,
            trait=trait,
            category=category,
            polarity=polarity,
            prior=prior,
            posterior=prior
        )
        self.save()
        return tid

    def update_trait(self, tid: str, evidence_desc: str,
                     likelihood_ratio: Optional[float] = None,
                     preset: Optional[str] = None,
                     context: str = "") -> float:
        """
        Update trait hypothesis with new evidence.

        Args:
            tid: Trait ID
            evidence_desc: Description of behavioral evidence
            likelihood_ratio: P(E|H) / P(E|¬H), or use preset
            preset: One of LR_PRESETS keys
            context: Context where evidence was observed

        Returns:
            New posterior probability
        """
        if tid not in self.traits:
            raise KeyError(f"Trait {tid} not found")

        t = self.traits[tid]

        if t.status == "REFUTED":
            raise ValueError(
                f"Trait {tid} is REFUTED and cannot be updated. "
                "Add a new trait hypothesis instead.")

        # Get likelihood ratio
        if preset:
            lr = self.LR_PRESETS.get(preset)
            if lr is None:
                raise ValueError(f"Unknown preset: {preset}. "
                               f"Use one of {list(self.LR_PRESETS.keys())}")
        elif likelihood_ratio is not None:
            lr = likelihood_ratio
        else:
            raise ValueError("Must provide either likelihood_ratio or preset")

        # Bayesian update using shared math (handles division-by-zero)
        if lr == 0:
            t.status = "REFUTED"
        new_posterior = bayesian_update(t.posterior, lr)

        # Record evidence
        t.evidence.append({
            'description': evidence_desc,
            'likelihood_ratio': lr,
            'prior_before': t.posterior,
            'posterior_after': new_posterior,
            'timestamp': datetime.now().isoformat(),
            'confirms': lr > 1,
            'context': context
        })

        # Saturation warning
        if new_posterior >= 0.95:
            print(f"Warning: {tid} posterior={new_posterior:.3f} near saturation "
                  "— consider if evidence items are truly independent",
                  file=sys.stderr)
        elif new_posterior <= 0.05:
            print(f"Warning: {tid} posterior={new_posterior:.3f} near zero "
                  "— consider if evidence items are truly independent",
                  file=sys.stderr)

        t.posterior = new_posterior
        t.updated = datetime.now().isoformat()

        # Update status
        # Thresholds for PSYCH tier: wider bands than system analysis
        # because behavioral evidence is noisier and more ambiguous.
        # See CLAUDE.md "Threshold Bands" table for cross-tracker comparison.
        if new_posterior >= 0.90:
            t.status = "CONFIRMED"
        elif new_posterior <= 0.10:
            t.status = "REFUTED"
        elif new_posterior <= 0.30:
            t.status = "WEAKENED"
        else:
            t.status = "ACTIVE"

        self.save()
        return new_posterior

    def get_trait(self, tid: str) -> TraitHypothesis:
        """Get a trait by ID."""
        return self.traits[tid]

    def get_traits_by_category(self, category: str) -> List[TraitHypothesis]:
        """Get all traits in a category."""
        return [t for t in self.traits.values() if t.category == category]

    # === Baseline Methods ===

    def add_baseline(self, category: str, description: str,
                     value: str = "") -> str:
        """
        Add a baseline observation.

        Args:
            category: One of BaselineCategory values
            description: What was observed
            value: Specific measured value if applicable

        Returns:
            Baseline ID
        """
        bid = f"B{self._next_baseline_id}"
        self._next_baseline_id += 1
        self.baselines.append(BaselineObservation(
            id=bid,
            category=category,
            description=description,
            value=value
        ))
        self.save()
        return bid

    def get_baselines(self, category: Optional[str] = None) -> List[BaselineObservation]:
        """Get baselines, optionally filtered by category."""
        if category:
            return [b for b in self.baselines if b.category == category]
        return self.baselines

    # === Deviation Methods ===

    def add_deviation(self, description: str,
                      baseline_reference: str = "",
                      context: str = "",
                      significance: str = "moderate") -> str:
        """
        Record a deviation from baseline.

        Args:
            description: What deviated
            baseline_reference: ID of baseline being deviated from
            context: When/where deviation occurred
            significance: "minor", "moderate", or "major"

        Returns:
            Deviation ID
        """
        if significance not in ['minor', 'moderate', 'major']:
            raise ValueError("Significance must be minor, moderate, or major")

        did = f"D{self._next_deviation_id}"
        self._next_deviation_id += 1
        self.deviations.append(DeviationRecord(
            id=did,
            description=description,
            baseline_reference=baseline_reference,
            context=context,
            significance=significance
        ))
        self.save()
        return did

    # === Profile Generation ===

    def get_ocean_profile(self) -> Dict[str, Dict]:
        """Get Big Five assessment summary."""
        ocean_map = {
            'openness': 'O',
            'conscientiousness': 'C',
            'extraversion': 'E',
            'agreeableness': 'A',
            'neuroticism': 'N'
        }

        profile = {}
        for cat, abbrev in ocean_map.items():
            traits = self.get_traits_by_category(cat)
            if traits:
                # Take the highest confidence trait for this category
                best = max(traits, key=lambda t: t.posterior if t.status != 'REFUTED' else 0)
                profile[abbrev] = {
                    'level': best.polarity,
                    'confidence': best.posterior,
                    'status': best.status
                }
            else:
                profile[abbrev] = {'level': 'unknown', 'confidence': 0, 'status': 'UNASSESSED'}

        return profile

    def get_dark_triad_profile(self) -> Dict[str, Dict]:
        """Get Dark Triad assessment summary."""
        dt_map = {
            'narcissism': 'N',
            'machiavellianism': 'M',
            'psychopathy': 'P'
        }

        profile = {}
        for cat, abbrev in dt_map.items():
            traits = self.get_traits_by_category(cat)
            if traits:
                best = max(traits, key=lambda t: t.posterior if t.status != 'REFUTED' else 0)
                profile[abbrev] = {
                    'level': best.polarity,
                    'confidence': best.posterior,
                    'status': best.status
                }
            else:
                profile[abbrev] = {'level': 'unknown', 'confidence': 0, 'status': 'UNASSESSED'}

        return profile

    def get_mice_profile(self) -> Dict[str, Dict]:
        """Get MICE motivation assessment."""
        mice_map = {
            'money': 'Money',
            'ideology': 'Ideology',
            'coercion': 'Coercion',
            'ego': 'Ego'
        }

        profile = {}
        for cat, name in mice_map.items():
            traits = self.get_traits_by_category(cat)
            if traits:
                best = max(traits, key=lambda t: t.posterior if t.status != 'REFUTED' else 0)
                profile[name] = {
                    'score': best.posterior,
                    'status': best.status
                }
            else:
                profile[name] = {'score': 0, 'status': 'UNASSESSED'}

        return profile

    def calculate_dt_risk(self) -> float:
        """Calculate Dark Triad risk score (0-1)."""
        dt = self.get_dark_triad_profile()

        def _dt_component(entry):
            # Unassessed traits contribute 0 risk (no data = no signal)
            if entry['level'] == 'unknown':
                return 0.0
            return entry['confidence'] if entry['level'] == 'high' else 1 - entry['confidence']

        # Weight: N=0.3, M=0.4, P=0.3
        n_score = _dt_component(dt['N'])
        m_score = _dt_component(dt['M'])
        p_score = _dt_component(dt['P'])

        return (n_score * 0.3) + (m_score * 0.4) + (p_score * 0.3)

    # === Reports ===

    def trait_report(self, verbose: bool = False) -> str:
        """Generate trait assessment report."""
        lines = [
            f"# Trait Assessment: {self.subject_name}",
            "",
            "| ID | Trait | Category | Prior | Posterior | Status |",
            "|:---|:------|:---------|------:|----------:|:-------|"
        ]

        for t in sorted(self.traits.values(), key=lambda x: x.id):
            trait_desc = t.trait[:50] + "..." if len(t.trait) > 50 else t.trait
            lines.append(f"| {t.id} | {trait_desc} | {t.category} | "
                        f"{t.prior:.2f} | {t.posterior:.2f} | {t.status} |")

        if verbose and any(t.evidence for t in self.traits.values()):
            lines.append("")
            lines.append("## Evidence Trail")
            for t in self.traits.values():
                if t.evidence:
                    lines.append(f"\n### {t.id}: {t.trait[:60]}")
                    for e in t.evidence:
                        direction = "+" if e['confirms'] else "-"
                        lines.append(f"- [{direction}] LR={e['likelihood_ratio']:.2f}: {e['description']}")
                        if e.get('context'):
                            lines.append(f"  - Context: {e['context']}")

        return "\n".join(lines)

    def baseline_report(self) -> str:
        """Generate baseline observation report."""
        lines = [
            f"# Baseline Profile: {self.subject_name}",
            "",
        ]

        # Group by category
        categories = set(b.category for b in self.baselines)
        for cat in sorted(categories):
            lines.append(f"## {cat.title()}")
            for b in self.baselines:
                if b.category == cat:
                    value_str = f" = {b.value}" if b.value else ""
                    lines.append(f"- [{b.id}] {b.description}{value_str}")
            lines.append("")

        if self.deviations:
            lines.append("## Deviations Recorded")
            for d in self.deviations:
                ref = f" (from {d.baseline_reference})" if d.baseline_reference else ""
                lines.append(f"- [{d.significance.upper()}] {d.description}{ref}")
                if d.context:
                    lines.append(f"  - Context: {d.context}")

        return "\n".join(lines)

    def profile_report(self) -> str:
        """Generate unified profile report."""
        ocean = self.get_ocean_profile()
        dt = self.get_dark_triad_profile()
        mice = self.get_mice_profile()
        dt_risk = self.calculate_dt_risk()

        # Format OCEAN string
        ocean_str = "".join([
            f"O[{ocean['O']['level'][0].upper() if ocean['O']['level'] != 'unknown' else '?'}] "
            f"C[{ocean['C']['level'][0].upper() if ocean['C']['level'] != 'unknown' else '?'}] "
            f"E[{ocean['E']['level'][0].upper() if ocean['E']['level'] != 'unknown' else '?'}] "
            f"A[{ocean['A']['level'][0].upper() if ocean['A']['level'] != 'unknown' else '?'}] "
            f"N[{ocean['N']['level'][0].upper() if ocean['N']['level'] != 'unknown' else '?'}]"
        ])

        # Format Dark Triad string
        dt_str = (f"N[{dt['N']['level'][0].upper() if dt['N']['level'] != 'unknown' else '?'}] "
                  f"M[{dt['M']['level'][0].upper() if dt['M']['level'] != 'unknown' else '?'}] "
                  f"P[{dt['P']['level'][0].upper() if dt['P']['level'] != 'unknown' else '?'}]")

        # Format MICE ranking
        mice_sorted = sorted(mice.items(), key=lambda x: x[1]['score'], reverse=True)
        mice_str = " > ".join([f"{k}" for k, v in mice_sorted if v['score'] > MICE_DISPLAY_THRESHOLD])
        if not mice_str:
            mice_str = "Not assessed"

        # Determine overall confidence
        assessed_traits = [t for t in self.traits.values() if t.status != 'UNASSESSED']
        if not assessed_traits:
            confidence = "No data"
        else:
            avg_conf = sum(t.posterior for t in assessed_traits) / len(assessed_traits)
            confidence = "High" if avg_conf > 0.7 else "Medium" if avg_conf > 0.4 else "Low"

        lines = [
            f"# Psychological Profile: {self.subject_name}",
            "",
            f"**Context**: {self.analysis_context or 'Not specified'}",
            "",
            "## Summary",
            "",
            f"**OCEAN**: {ocean_str}",
            f"**Dark Triad**: {dt_str} (Risk: {dt_risk:.2f})",
            f"**MICE**: {mice_str}",
            f"**Confidence**: {confidence}",
            "",
            "## Detailed Assessments",
            "",
            "### Big Five (OCEAN)",
            "| Trait | Level | Confidence |",
            "|-------|-------|------------|",
        ]

        for trait, abbrev in [('Openness', 'O'), ('Conscientiousness', 'C'),
                              ('Extraversion', 'E'), ('Agreeableness', 'A'),
                              ('Neuroticism', 'N')]:
            level = ocean[abbrev]['level'].title()
            conf = f"{ocean[abbrev]['confidence']:.0%}" if ocean[abbrev]['confidence'] > 0 else "N/A"
            lines.append(f"| {trait} | {level} | {conf} |")

        lines.extend([
            "",
            "### Dark Triad",
            "| Trait | Level | Confidence |",
            "|-------|-------|------------|",
        ])

        for trait, abbrev in [('Narcissism', 'N'), ('Machiavellianism', 'M'),
                              ('Psychopathy', 'P')]:
            level = dt[abbrev]['level'].title()
            conf = f"{dt[abbrev]['confidence']:.0%}" if dt[abbrev]['confidence'] > 0 else "N/A"
            lines.append(f"| {trait} | {level} | {conf} |")

        lines.extend([
            "",
            f"**Dark Triad Risk Score**: {dt_risk:.2f} "
            f"({'Low' if dt_risk < 0.3 else 'Moderate' if dt_risk < 0.6 else 'High'})",
            "",
            "### MICE Motivations",
            "| Driver | Score | Status |",
            "|--------|-------|--------|",
        ])

        for driver, data in mice_sorted:
            score = f"{data['score']:.0%}" if data['score'] > 0 else "N/A"
            lines.append(f"| {driver} | {score} | {data['status']} |")

        # Data quality section
        lines.extend([
            "",
            "## Data Quality",
            "",
            f"- Traits assessed: {len(self.traits)}",
            f"- Baseline observations: {len(self.baselines)}",
            f"- Deviations recorded: {len(self.deviations)}",
            f"- Evidence items: {sum(len(t.evidence) for t in self.traits.values())}",
        ])

        return "\n".join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Belief Tracker for Psychological Profiling")
    parser.add_argument("--file", default="profile.json", help="Profile file")

    subparsers = parser.add_subparsers(dest="cmd", help="Commands")

    # Subject command
    subj_p = subparsers.add_parser("subject", help="Set subject info")
    subj_p.add_argument("name", help="Subject name")
    subj_p.add_argument("--context", default="", help="Analysis context")

    # Add trait command
    add_p = subparsers.add_parser("add", help="Add trait hypothesis")
    add_p.add_argument("trait", help="Trait description")
    add_p.add_argument("--category", default="custom",
                       help="Trait category (e.g., openness, narcissism, money)")
    add_p.add_argument("--polarity", choices=['high', 'low'], default='high',
                       help="Trait polarity")
    add_p.add_argument("--prior", type=float, default=0.5, help="Prior probability")

    # Update trait command
    upd_p = subparsers.add_parser("update", help="Update trait with evidence")
    upd_p.add_argument("id", help="Trait ID")
    upd_p.add_argument("evidence", help="Evidence description")
    upd_p.add_argument("--lr", type=float, help="Likelihood ratio")
    upd_p.add_argument("--preset", choices=list(BeliefTracker.LR_PRESETS.keys()),
                       help="Use preset likelihood ratio")
    upd_p.add_argument("--context", default="", help="Context of observation")

    # Baseline commands
    base_p = subparsers.add_parser("baseline", help="Baseline management")
    base_sub = base_p.add_subparsers(dest="base_cmd")

    base_add = base_sub.add_parser("add", help="Add baseline observation")
    base_add.add_argument("description", help="Observation description")
    base_add.add_argument("--category", default="behavioral",
                          choices=['linguistic', 'emotional', 'timing',
                                   'behavioral', 'idiosyncrasy'],
                          help="Baseline category")
    base_add.add_argument("--value", default="", help="Measured value")

    base_sub.add_parser("list", help="List baselines")

    # Deviation command
    dev_p = subparsers.add_parser("deviation", help="Record deviation")
    dev_p.add_argument("description", help="Deviation description")
    dev_p.add_argument("--baseline", default="", help="Baseline ID being deviated from")
    dev_p.add_argument("--context", default="", help="Context of deviation")
    dev_p.add_argument("--significance", choices=['minor', 'moderate', 'major'],
                       default='moderate', help="Significance level")

    # Report commands
    traits_p = subparsers.add_parser("traits", help="Show trait report")
    traits_p.add_argument("--verbose", "-v", action="store_true",
                          help="Include evidence trail")
    subparsers.add_parser("baselines", help="Show baseline report")

    profile_p = subparsers.add_parser("profile", help="Generate unified profile")

    report_p = subparsers.add_parser("report", help="Generate full report")
    report_p.add_argument("--verbose", "-v", action="store_true",
                          help="Include evidence trail")

    args = parser.parse_args()
    tracker = BeliefTracker(args.file)

    try:
        if args.cmd == "subject":
            tracker.set_subject(args.name, args.context)
            print(f"Subject set: {args.name}")

        elif args.cmd == "add":
            tid = tracker.add_trait(args.trait, args.category,
                                    args.polarity, args.prior)
            print(f"Added trait: {tid} (prior={args.prior})")

        elif args.cmd == "update":
            if not args.lr and not args.preset:
                print("Error: Must specify --lr or --preset")
                sys.exit(1)
            new_p = tracker.update_trait(args.id, args.evidence,
                                         likelihood_ratio=args.lr,
                                         preset=args.preset,
                                         context=args.context)
            print(f"Updated {args.id}: posterior={new_p:.3f}")

        elif args.cmd == "baseline":
            if args.base_cmd == "add":
                bid = tracker.add_baseline(args.category, args.description, args.value)
                print(f"Added baseline: {bid}")
            elif args.base_cmd == "list":
                print(tracker.baseline_report())
            else:
                base_p.print_help()

        elif args.cmd == "deviation":
            did = tracker.add_deviation(args.description, args.baseline,
                                        args.context, args.significance)
            print(f"Recorded deviation: {did}")

        elif args.cmd == "traits":
            print(tracker.trait_report(verbose=args.verbose))

        elif args.cmd == "baselines":
            print(tracker.baseline_report())

        elif args.cmd == "profile":
            print(tracker.profile_report())

        elif args.cmd == "report":
            print(tracker.profile_report())
            print("\n" + "="*60 + "\n")
            print(tracker.trait_report(verbose=args.verbose))
            print("\n" + "="*60 + "\n")
            print(tracker.baseline_report())

        else:
            parser.print_help()

    except (KeyError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
