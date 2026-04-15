#!/usr/bin/env python3
"""
Scope Auditor for Epistemic Deconstruction v7.12.1

Implements Phase 0.7 Scope Interrogation mechanisms:
  M1 Flow Tracing           -> `trace`
  M2 Archetype Accomplices  -> `enumerate`
  M3 Residual Matching      -> `residual-match`

Persists state to scope_audit.json (dataclass container). Loads the
archetype library from src/config/archetypes.json using the same
script-dir-relative pattern as rapid_checker.py.

Usage:
    python3 scope_auditor.py enumerate --archetype speculative_asset_market
    python3 scope_auditor.py trace --inputs "capital,labor" --outputs "housing,taxes"
    python3 scope_auditor.py residual-match --residuals residuals.csv --indices-dir ./indices
    python3 scope_auditor.py list-archetypes
    python3 scope_auditor.py report
"""

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass, field, asdict, fields as dataclass_fields
from datetime import datetime
from typing import Dict, List, Optional, Tuple

try:
    from common import load_json, save_json
except ImportError:  # allow running as standalone script
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in sys.path:
        sys.path.insert(0, _here)
    from common import load_json, save_json


# ---------------------------------------------------------------------------
# Archetype library loader
# ---------------------------------------------------------------------------

def load_archetype_library(config_path: Optional[str] = None) -> Dict:
    """
    Load archetype library from src/config/archetypes.json.

    Falls back to a minimal built-in library if the config file is
    missing. Keys starting with '_' (comment/schema) are filtered out.
    """
    if config_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, '..', 'config', 'archetypes.json')

    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Failed to parse archetype library at {config_path}: {e}"
            )
        return {k: v for k, v in data.items() if not k.startswith('_')}

    # Minimal fallback — allows --help and smoke tests to work even if
    # config file is missing.
    return {
        'generic_system': {
            'name': 'Generic system (fallback)',
            'description': 'Fallback archetype — install config/archetypes.json.',
            'examples': [],
            'accomplices': [
                {'domain': 'external environment', 'mechanism': 'unmodeled forcing',
                 'prior': 0.2}
            ],
        }
    }


# ---------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------

@dataclass
class ScopeAudit:
    """Container for a scope interrogation session."""
    id: str
    target: str
    created: str
    archetypes_queried: List[str] = field(default_factory=list)
    candidates: List[Dict] = field(default_factory=list)  # {id, domain, mechanism, prior, source}
    traces: List[Dict] = field(default_factory=list)  # {direction, channel, neighbor, in_scope, source}
    steelman: List[Dict] = field(default_factory=list)  # {persona, domain, mechanism}
    residual_matches: List[Dict] = field(default_factory=list)  # {index, r, p, flagged}
    notes: str = ""
    next_candidate_id: int = 1


class ScopeAuditor:
    """
    Scope interrogation state + operations. Parallels RapidChecker.
    """

    def __init__(self, filepath: str = "scope_audit.json"):
        self.filepath = filepath
        self.audit: Optional[ScopeAudit] = None
        self.load()

    # --- persistence -------------------------------------------------------

    def load(self):
        data = load_json(self.filepath)
        if data is not None:
            if 'next_candidate_id' not in data:
                cands = data.get('candidates', [])
                ids = [int(c['id'][1:]) for c in cands
                       if isinstance(c.get('id'), str) and c['id'][1:].isdigit()]
                data['next_candidate_id'] = (max(ids) if ids else 0) + 1
            known = {f.name for f in dataclass_fields(ScopeAudit)}
            filtered = {k: v for k, v in data.items() if k in known}
            self.audit = ScopeAudit(**filtered)

    def save(self):
        if self.audit:
            save_json(self.filepath, asdict(self.audit))

    def start(self, target: str, force: bool = False) -> str:
        if self.audit and self.audit.id and not force:
            raise RuntimeError(
                f"Scope audit for '{self.audit.target}' already exists (ID: {self.audit.id}). "
                f"Use --force to overwrite."
            )
        aid = f"SA{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.audit = ScopeAudit(
            id=aid,
            target=target,
            created=datetime.now().isoformat(),
        )
        self.save()
        return aid

    def require_session(self):
        if not self.audit:
            raise RuntimeError("No active scope audit. Run 'start' first.")

    # --- candidate management ---------------------------------------------

    def add_candidate(self, domain: str, mechanism: str, prior: float,
                      source: str) -> str:
        self.require_session()
        cid = f"C{self.audit.next_candidate_id}"
        self.audit.next_candidate_id += 1
        entry = {
            'id': cid,
            'domain': domain,
            'mechanism': mechanism,
            'prior': prior,
            'source': source,
            'logged': datetime.now().isoformat(),
        }
        self.audit.candidates.append(entry)
        self.save()
        return cid

    def dedupe_candidates(self) -> int:
        """Remove duplicate candidates by (domain) keeping first occurrence.

        Returns count of removed entries.
        """
        self.require_session()
        seen = set()
        deduped = []
        removed = 0
        for c in self.audit.candidates:
            key = c['domain'].strip().lower()
            if key in seen:
                removed += 1
                continue
            seen.add(key)
            deduped.append(c)
        self.audit.candidates = deduped
        self.save()
        return removed

    # --- M2 enumerate ------------------------------------------------------

    def enumerate_archetype(self, archetype_id: str,
                            library: Optional[Dict] = None) -> List[Dict]:
        """Return accomplice list for archetype. Raises KeyError if missing."""
        if library is None:
            library = load_archetype_library()
        if archetype_id not in library:
            raise KeyError(
                f"Archetype '{archetype_id}' not found. Known: "
                f"{sorted(library.keys())}"
            )
        entry = library[archetype_id]
        accomplices = entry.get('accomplices', [])
        if self.audit is not None:
            if archetype_id not in self.audit.archetypes_queried:
                self.audit.archetypes_queried.append(archetype_id)
            for acc in accomplices:
                self.add_candidate(
                    domain=acc['domain'],
                    mechanism=acc['mechanism'],
                    prior=float(acc.get('prior', 0.15)),
                    source=f"M2:archetype={archetype_id}",
                )
        return list(accomplices)

    # --- M1 trace ----------------------------------------------------------

    def trace_flows(self, inputs: List[str], outputs: List[str]) -> List[Dict]:
        """Record M1 flow-trace entries and return the checklist items.

        The checklist asks the analyst to fill in the upstream generator
        (for inputs) or downstream consumer (for outputs) for each
        channel. `in_scope` defaults to False — the analyst must
        confirm true later via `trace-resolve`.
        """
        self.require_session()
        entries = []
        for channel in inputs:
            e = {
                'direction': 'input',
                'channel': channel.strip(),
                'neighbor': None,  # to be filled by analyst
                'in_scope': False,
                'source': 'M1:flow_trace',
                'logged': datetime.now().isoformat(),
            }
            self.audit.traces.append(e)
            entries.append(e)
        for channel in outputs:
            e = {
                'direction': 'output',
                'channel': channel.strip(),
                'neighbor': None,
                'in_scope': False,
                'source': 'M1:flow_trace',
                'logged': datetime.now().isoformat(),
            }
            self.audit.traces.append(e)
            entries.append(e)
        self.save()
        return entries

    # --- M4 steelman -------------------------------------------------------

    def add_steelman(self, persona: str, domain: str, mechanism: str) -> None:
        self.require_session()
        entry = {
            'persona': persona,
            'domain': domain,
            'mechanism': mechanism,
            'logged': datetime.now().isoformat(),
        }
        self.audit.steelman.append(entry)
        # Also log as candidate with prior 0.15 per protocol
        self.add_candidate(
            domain=domain,
            mechanism=mechanism,
            prior=0.15,
            source=f"M4:steelman:{persona}",
        )

    # --- M3 residual match -------------------------------------------------

    def residual_match(self, residuals: List[float],
                       indices: Dict[str, List[float]],
                       r_threshold: float = 0.3,
                       p_threshold: float = 0.05) -> List[Dict]:
        """Compute Pearson correlation between residuals and each index.

        Index series are aligned by truncation to the common length.
        Flags indices with |r| >= r_threshold AND p <= p_threshold.
        """
        self.require_session()
        matches = []
        for name, series in indices.items():
            n = min(len(residuals), len(series))
            if n < 3:
                matches.append({
                    'index': name, 'r': 0.0, 'p': 1.0,
                    'n': n, 'flagged': False, 'note': 'series too short',
                })
                continue
            r = pearson_correlation(residuals[:n], series[:n])
            p = pearson_pvalue(r, n)
            flagged = abs(r) >= r_threshold and p <= p_threshold
            matches.append({
                'index': name, 'r': r, 'p': p,
                'n': n, 'flagged': flagged,
            })
            if flagged:
                self.add_candidate(
                    domain=name,
                    mechanism=f"residual correlation r={r:.2f} p={p:.3f}",
                    prior=min(0.30, max(0.10, abs(r))),
                    source=f"M3:residual_match",
                )
        self.audit.residual_matches.extend(matches)
        self.save()
        return matches

    # --- reporting ---------------------------------------------------------

    def report(self, verbose: bool = False) -> str:
        self.require_session()
        a = self.audit
        lines = []
        lines.append(f"# Scope Audit — {a.target}")
        lines.append(f"")
        lines.append(f"- Session: {a.id}  (created {a.created})")
        lines.append(f"- Archetypes queried: {', '.join(a.archetypes_queried) or '(none)'}")
        lines.append(f"- Flow traces: {len(a.traces)}")
        lines.append(f"- Steelman critiques: {len(a.steelman)}")
        lines.append(f"- Residual matches: {len(a.residual_matches)}"
                     f" (flagged: {sum(1 for m in a.residual_matches if m.get('flagged'))})")
        lines.append(f"- Exogeneity candidates: {len(a.candidates)}")
        lines.append("")
        lines.append("## Candidates")
        lines.append("")
        lines.append("| ID | Domain | Mechanism | Prior | Source |")
        lines.append("|---|---|---|---:|---|")
        for c in a.candidates:
            lines.append(
                f"| {c['id']} | {c['domain']} | {c['mechanism']} | "
                f"{c['prior']:.2f} | {c['source']} |"
            )
        if verbose:
            if a.steelman:
                lines.append("")
                lines.append("## Steelman critiques")
                lines.append("")
                for s in a.steelman:
                    lines.append(f"- **{s['persona']}** → domain: {s['domain']}, "
                                 f"mechanism: {s['mechanism']}")
            if a.residual_matches:
                lines.append("")
                lines.append("## Residual matches")
                lines.append("")
                lines.append("| Index | r | p | n | flagged |")
                lines.append("|---|---:|---:|---:|:---:|")
                for m in a.residual_matches:
                    lines.append(
                        f"| {m['index']} | {m['r']:.3f} | {m['p']:.3f} | "
                        f"{m['n']} | {'YES' if m.get('flagged') else ''} |"
                    )
        return "\n".join(lines)

    def gate_status(self) -> Dict:
        """Compute Phase 0.7 exit gate status."""
        self.require_session()
        n_unique_candidates = len({c['domain'].strip().lower()
                                    for c in self.audit.candidates})
        gate = {
            'candidates_unique': n_unique_candidates,
            'min_required': 3,
            'has_steelman': len(self.audit.steelman) >= 3,
            'has_archetype_query': len(self.audit.archetypes_queried) >= 1,
            'has_traces': len(self.audit.traces) >= 1,
        }
        gate['pass'] = (
            gate['candidates_unique'] >= gate['min_required']
            and gate['has_archetype_query']
        )
        return gate


# ---------------------------------------------------------------------------
# Pearson correlation with stdlib p-value (no scipy dependency)
# ---------------------------------------------------------------------------

def pearson_correlation(x: List[float], y: List[float]) -> float:
    """Pearson product-moment correlation. Returns 0.0 for degenerate input."""
    n = len(x)
    if n != len(y) or n < 2:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    denx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    deny = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if denx == 0 or deny == 0:
        return 0.0
    return num / (denx * deny)


def pearson_pvalue(r: float, n: int) -> float:
    """
    Two-sided p-value for Pearson r under the null hypothesis r=0.

    Uses the t-statistic t = r * sqrt((n-2) / (1 - r^2)) which follows
    a Student-t distribution with n-2 degrees of freedom under the null.

    We approximate the two-sided p-value via the Student-t survival
    function using a series expansion. Falls back to scipy if available
    for higher accuracy on small n.
    """
    if n < 3:
        return 1.0
    if abs(r) >= 1.0:
        return 0.0
    df = n - 2
    t = abs(r) * math.sqrt(df / max(1e-12, (1 - r * r)))
    # Try scipy for accuracy; fall back to stdlib approximation.
    try:
        from scipy import stats  # type: ignore
        return 2.0 * float(stats.t.sf(t, df))
    except Exception:
        return 2.0 * _student_t_sf_stdlib(t, df)


def _student_t_sf_stdlib(t: float, df: int) -> float:
    """
    Stdlib-only approximation of the Student-t survival function.

    Uses the relationship P(T > t) = 0.5 * I_{df/(df+t^2)}(df/2, 1/2)
    where I is the regularized incomplete beta function. We compute
    I via a continued fraction expansion (Numerical Recipes style).
    Accurate to ~1e-6 for df >= 3; sufficient for flagging thresholds.
    """
    if t <= 0:
        return 0.5
    x = df / (df + t * t)
    # Regularized incomplete beta I_x(a, b) where a = df/2, b = 0.5
    a = df / 2.0
    b = 0.5
    # Use symmetry to ensure continued fraction converges
    if x > (a + 1.0) / (a + b + 2.0):
        # Use identity I_x(a,b) = 1 - I_{1-x}(b,a)
        return 0.5 * (1.0 - _incomplete_beta(1 - x, b, a))
    return 0.5 * _incomplete_beta(x, a, b)


def _incomplete_beta(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta I_x(a, b) via Lentz's continued fraction."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    # Prefactor: x^a * (1-x)^b / (a * B(a, b))
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b + lbeta) / a
    # Continued fraction
    MAX_ITER = 200
    EPS = 1e-12
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, MAX_ITER + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return front * h


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def read_csv_column(path: str, column: Optional[str] = None) -> List[float]:
    """Read a CSV file and return the named column (or first numeric column).

    Skips non-numeric rows silently.
    """
    values: List[float] = []
    with open(path, 'r', newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return values
    header = rows[0]
    try:
        float(header[0])
        # No header
        start = 0
        col_idx = 0
    except (ValueError, IndexError):
        start = 1
        if column is not None and column in header:
            col_idx = header.index(column)
        else:
            # Default to last column
            col_idx = len(header) - 1
    for row in rows[start:]:
        if col_idx >= len(row):
            continue
        try:
            values.append(float(row[col_idx]))
        except ValueError:
            continue
    return values


def load_indices_dir(directory: str) -> Dict[str, List[float]]:
    """Load all CSV files in directory as indices (filename stem -> series)."""
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Indices directory not found: {directory}")
    indices = {}
    for fname in sorted(os.listdir(directory)):
        if not fname.lower().endswith('.csv'):
            continue
        stem = os.path.splitext(fname)[0]
        path = os.path.join(directory, fname)
        indices[stem] = read_csv_column(path)
    return indices


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='scope_auditor.py',
        description='Scope Interrogation tool (Phase 0.7 mechanisms M1-M4).',
    )
    parser.add_argument(
        '--file', default='scope_audit.json',
        help='Path to scope_audit.json state file (default: ./scope_audit.json)',
    )
    parser.add_argument(
        '--config', default=None,
        help='Path to archetypes.json (default: ../config/archetypes.json)',
    )
    sub = parser.add_subparsers(dest='cmd', required=False)

    p_start = sub.add_parser('start', help='Start a new scope audit session')
    p_start.add_argument('target', help='Target system description')
    p_start.add_argument('--force', action='store_true',
                         help='Overwrite existing session')

    p_list = sub.add_parser('list-archetypes', help='List known archetypes')
    p_list.add_argument('--json', action='store_true', help='Output as JSON')

    p_enum = sub.add_parser('enumerate', help='M2: enumerate archetype accomplices')
    p_enum.add_argument('--archetype', required=True, help='Archetype ID')
    p_enum.add_argument('--json', action='store_true', help='Output as JSON')

    p_trace = sub.add_parser('trace', help='M1: flow-trace checklist')
    p_trace.add_argument('--inputs', default='', help='Comma-separated input channels')
    p_trace.add_argument('--outputs', default='', help='Comma-separated output channels')

    p_steel = sub.add_parser('steelman', help='M4: record a steelman critique')
    p_steel.add_argument('--persona', required=True,
                         choices=['outsider', 'journalist', 'regulator'],
                         help='Steelman persona')
    p_steel.add_argument('--domain', required=True, help='Named excluded domain')
    p_steel.add_argument('--mechanism', required=True, help='One-line mechanism')

    p_resid = sub.add_parser('residual-match', help='M3: residual signature matching')
    p_resid.add_argument('--residuals', required=True, help='CSV file with residuals')
    p_resid.add_argument('--residuals-column', default=None,
                         help='Column name (default: last)')
    p_resid.add_argument('--indices-dir', required=True,
                         help='Directory of external index CSVs')
    p_resid.add_argument('--r-threshold', type=float, default=0.3)
    p_resid.add_argument('--p-threshold', type=float, default=0.05)

    p_report = sub.add_parser('report', help='Print scope audit report')
    p_report.add_argument('--verbose', action='store_true')

    sub.add_parser('gate', help='Print Phase 0.7 exit gate status')

    p_dedupe = sub.add_parser('dedupe', help='Deduplicate candidates by domain')

    p_cand = sub.add_parser('add-candidate', help='Manually add an exogeneity candidate')
    p_cand.add_argument('--domain', required=True)
    p_cand.add_argument('--mechanism', required=True)
    p_cand.add_argument('--prior', type=float, default=0.15)
    p_cand.add_argument('--source', default='manual')

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd is None:
        parser.print_help()
        return 0

    # Commands that do not require a session
    if args.cmd == 'list-archetypes':
        library = load_archetype_library(args.config)
        if args.json:
            print(json.dumps(sorted(library.keys())))
        else:
            print("Available archetypes:")
            for aid in sorted(library.keys()):
                name = library[aid].get('name', aid)
                print(f"  {aid}  — {name}")
        return 0

    if args.cmd == 'enumerate' and not os.path.exists(args.file):
        # Allow dry-run enumerate without a session (prints accomplices only)
        library = load_archetype_library(args.config)
        if args.archetype not in library:
            print(f"ERROR: archetype '{args.archetype}' not found. "
                  f"Known: {sorted(library.keys())}", file=sys.stderr)
            return 2
        accomplices = library[args.archetype].get('accomplices', [])
        if args.json:
            print(json.dumps(accomplices, indent=2))
        else:
            print(f"# Accomplices for {args.archetype}")
            print(f"# {library[args.archetype].get('name', args.archetype)}")
            print("")
            print("| Domain | Mechanism | Prior |")
            print("|---|---|---:|")
            for a in accomplices:
                print(f"| {a['domain']} | {a['mechanism']} | {a.get('prior', 0.15):.2f} |")
        return 0

    auditor = ScopeAuditor(args.file)

    if args.cmd == 'start':
        aid = auditor.start(args.target, force=args.force)
        print(f"Started scope audit {aid} for: {args.target}")
        return 0

    # Auto-start session with target="(unspecified)" if absent, so the
    # other commands work out of the box. The analyst is expected to
    # call `start` explicitly in production use.
    if auditor.audit is None:
        auditor.start('(unspecified target)')

    if args.cmd == 'enumerate':
        try:
            accomplices = auditor.enumerate_archetype(
                args.archetype,
                library=load_archetype_library(args.config),
            )
        except KeyError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps(accomplices, indent=2))
        else:
            print(f"Added {len(accomplices)} candidates from archetype "
                  f"'{args.archetype}'")
            for a in accomplices:
                print(f"  - {a['domain']}: {a['mechanism']} (prior={a.get('prior', 0.15):.2f})")
        return 0

    if args.cmd == 'trace':
        inputs = [s.strip() for s in args.inputs.split(',') if s.strip()]
        outputs = [s.strip() for s in args.outputs.split(',') if s.strip()]
        if not inputs and not outputs:
            print("ERROR: at least one of --inputs or --outputs must be non-empty",
                  file=sys.stderr)
            return 2
        entries = auditor.trace_flows(inputs, outputs)
        print(f"Recorded {len(entries)} flow-trace entries.")
        print("For each entry, identify the immediate upstream generator")
        print("(for inputs) or downstream consumer (for outputs):")
        for e in entries:
            print(f"  [{e['direction']:7s}] {e['channel']}  "
                  f"→ neighbor: ??  in_scope: ??")
        return 0

    if args.cmd == 'steelman':
        auditor.add_steelman(args.persona, args.domain, args.mechanism)
        print(f"Recorded steelman critique ({args.persona}): "
              f"{args.domain} — {args.mechanism}")
        return 0

    if args.cmd == 'residual-match':
        residuals = read_csv_column(args.residuals, args.residuals_column)
        if len(residuals) < 3:
            print(f"ERROR: residuals file has fewer than 3 values", file=sys.stderr)
            return 2
        try:
            indices = load_indices_dir(args.indices_dir)
        except FileNotFoundError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2
        if not indices:
            print(f"WARNING: no CSV files in {args.indices_dir}", file=sys.stderr)
        matches = auditor.residual_match(
            residuals, indices,
            r_threshold=args.r_threshold,
            p_threshold=args.p_threshold,
        )
        print(f"Tested {len(matches)} indices against residuals.")
        print("| Index | r | p | n | flagged |")
        print("|---|---:|---:|---:|:---:|")
        for m in matches:
            print(f"| {m['index']} | {m['r']:.3f} | {m['p']:.3f} | "
                  f"{m['n']} | {'YES' if m.get('flagged') else ''} |")
        return 0

    if args.cmd == 'report':
        print(auditor.report(verbose=args.verbose))
        return 0

    if args.cmd == 'gate':
        gate = auditor.gate_status()
        print("Phase 0.7 Exit Gate:")
        for k, v in gate.items():
            print(f"  {k}: {v}")
        return 0 if gate.get('pass') else 1

    if args.cmd == 'dedupe':
        removed = auditor.dedupe_candidates()
        print(f"Removed {removed} duplicate candidates.")
        return 0

    if args.cmd == 'add-candidate':
        cid = auditor.add_candidate(args.domain, args.mechanism, args.prior, args.source)
        print(f"Added candidate {cid}: {args.domain}")
        return 0

    parser.print_help()
    return 0


if __name__ == '__main__':
    sys.exit(main())
