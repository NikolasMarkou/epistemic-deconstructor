#!/usr/bin/env python3
"""
Domain Orienter for Epistemic Deconstruction v7.15.0

Implements Phase 0.3 Domain Orientation operators:
  TE — Term Extraction    -> `extract`
  TG — Term Grounding     -> `ground`
  MM — Metrics Mapping    -> `add-metric`
  AM — Alias Map          -> `alias`
  CS — Canonical Sources  -> `source` / `verify`

Also provides:
  candidates {list|promote}       — staging and verification gate
  glossary render                 — emit domain_glossary.md
  metrics  render                 — emit domain_metrics.json
  sources  render                 — emit domain_sources.md
  gate                            — check Phase 0.3 exit gate
  report [--verbose]              — human-readable status
  skip --reason <r>               — print a decisions.md block

State is persisted to a domain_orientation.json file (dataclass container)
via common.load_json / save_json (atomic write + sidecar .lock).

Provenance discipline: every term, metric, and source carries a `source`
field, one of {library, analyst, llm_parametric, chain_derived}. Hard caps
mirror Phase 1.5 and are enforced in code, not documentation.

Stdlib-only. No numpy/scipy. No network I/O — the `verify` subcommand
accepts a pre-fetched HTTP status so the sub-agent owns WebFetch calls.

Usage:
    python3 domain_orienter.py --file state.json start --tier STANDARD --domain credit_derivatives
    python3 domain_orienter.py --file state.json extract --input doc.md
    python3 domain_orienter.py --file state.json ground --term "CDS" --definition "..." --source library --url https://...
    python3 domain_orienter.py --file state.json add-metric --name cs01 --units USD/bp --higher-is-better false --plausibility null,100,10000,null --source library --url ...
    python3 domain_orienter.py --file state.json source --title "ISDA 2014" --category standard --url ...
    python3 domain_orienter.py --file state.json verify --source-id SID-001 --http-status 200
    python3 domain_orienter.py --file state.json glossary render --output glossary.md
    python3 domain_orienter.py --file state.json gate
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import uuid
from dataclasses import dataclass, field, asdict, fields as dataclass_fields
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    from common import load_json, save_json
except ImportError:  # allow running as standalone script
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in sys.path:
        sys.path.insert(0, _here)
    from common import load_json, save_json


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# DECISION D-006: valid provenance sources — mirrors P1.5 (D-003)
# Changing this set requires updating both SKILL.md Evidence Rule 8
# and references/domain-orientation.md Provenance and Hard Caps.
VALID_SOURCES = {"library", "analyst", "llm_parametric", "chain_derived"}

# DECISION D-007: LLM-parametric hard caps — extends P1.5 (D-004) to terminology
# Raising caps silently erodes provenance discipline. Any change requires a
# decisions.md entry describing why the existing cap failed.
LLM_PARAMETRIC_CONFIDENCE_CAP = 0.60
ANALYST_CONFIDENCE_CAP = 0.80
CHAIN_DERIVED_CONFIDENCE_CAP = 0.90

# DECISION D-008: Phase 0.3 exit-gate thresholds
# Lowering these thresholds is a protocol change, not a tool change.
# See references/domain-orientation.md for the protocol context.
GATE_MIN_TERMS_STANDARD = 10
GATE_MIN_TERMS_LITE = 5
GATE_MIN_METRICS = 3
GATE_MIN_VERIFIED_SOURCES = 2
GATE_MIN_LIBRARY_FRACTION = 0.30

VALID_TIERS = ("STANDARD", "COMPREHENSIVE", "LITE", "PSYCH")
VALID_SOURCE_CATEGORIES = (
    "textbook", "regulator", "standard", "seminal_paper", "benchmark_dataset",
)

# Regex patterns for TE operator
_RE_ACRONYM = re.compile(r"\b[A-Z]{2,}\b")
_RE_ALNUM = re.compile(r"\b[A-Za-z]+\d+[A-Za-z\d]*\b")
_RE_CAPWORDS = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CandidateTerm:
    """A term surfaced by TE, not yet grounded."""
    id: str
    text: str
    frequency: int = 1
    source_ref: str = ""
    logged: str = ""


@dataclass
class GroundedTerm:
    """A term grounded by TG — has definition, source, confidence."""
    id: str
    text: str
    definition: str
    source: str
    confidence: float
    url: Optional[str] = None
    source_id: Optional[str] = None
    grounded_at: str = ""


@dataclass
class Metric:
    """A metric registered by MM. Staged (promoted=False) until analyst promotes."""
    id: str
    name: str
    units: str
    higher_is_better: bool
    plausibility: List[Optional[float]] = field(default_factory=list)
    source: str = "analyst"
    url: Optional[str] = None
    source_id: Optional[str] = None
    domain: Optional[str] = None
    promoted: bool = False
    logged: str = ""


@dataclass
class Alias:
    """An alias-group registered by AM."""
    id: str
    canonical_id: str
    canonical_text: str
    aliases: List[str] = field(default_factory=list)
    region: Optional[str] = None
    source: Optional[str] = None
    logged: str = ""


@dataclass
class Source:
    """A canonical source registered by CS."""
    id: str
    title: str
    category: str
    url: Optional[str] = None
    authors: Optional[str] = None
    year: Optional[int] = None
    coverage: Optional[str] = None
    verified: bool = False
    verified_at: Optional[str] = None
    http_status: Optional[int] = None
    verified_by: Optional[str] = None
    promoted: bool = False
    logged: str = ""


@dataclass
class DomainOrientationState:
    """Container for a Phase 0.3 Domain Orientation session."""
    run_id: str
    created_at: str
    tier: str = "STANDARD"
    domain_declared: str = ""
    candidate_terms: List[Dict] = field(default_factory=list)
    grounded_terms: List[Dict] = field(default_factory=list)
    metrics: List[Dict] = field(default_factory=list)
    aliases: List[Dict] = field(default_factory=list)
    sources: List[Dict] = field(default_factory=list)
    gate_last_checked: Optional[str] = None
    gate_status: Optional[str] = None
    next_term_id: int = 1
    next_metric_id: int = 1
    next_alias_id: int = 1
    next_source_id: int = 1


# ---------------------------------------------------------------------------
# DomainOrienter
# ---------------------------------------------------------------------------

class DomainOrienter:
    """
    Phase 0.3 state + operators.

    Parallels AbductiveEngine. Persists state to a single JSON file via
    common.load_json / save_json (atomic + locked).
    """

    def __init__(self, file_path: str = "domain_orientation.json"):
        self.file_path = file_path
        self.state: Optional[DomainOrientationState] = None
        self.load()

    # --- persistence -------------------------------------------------------

    def load(self):
        data = load_json(self.file_path)
        if data is not None:
            known = {f.name for f in dataclass_fields(DomainOrientationState)}
            filtered = {k: v for k, v in data.items() if k in known}
            # Recover monotonic ID counters from existing entries if legacy.
            if "next_term_id" not in filtered:
                filtered["next_term_id"] = _recover_counter(
                    filtered.get("candidate_terms", []) +
                    filtered.get("grounded_terms", []),
                    prefix="TERM-",
                )
            if "next_metric_id" not in filtered:
                filtered["next_metric_id"] = _recover_counter(
                    filtered.get("metrics", []), prefix="MET-",
                )
            if "next_alias_id" not in filtered:
                filtered["next_alias_id"] = _recover_counter(
                    filtered.get("aliases", []), prefix="ALIAS-",
                )
            if "next_source_id" not in filtered:
                filtered["next_source_id"] = _recover_counter(
                    filtered.get("sources", []), prefix="SID-",
                )
            self.state = DomainOrientationState(**filtered)

    def save(self):
        if self.state is not None:
            save_json(self.file_path, asdict(self.state))

    def start(self, tier: str = "STANDARD", domain: str = "",
              force: bool = False) -> str:
        """Initialize a new domain-orientation session."""
        if tier not in VALID_TIERS:
            raise ValueError(
                f"Invalid tier '{tier}'. Must be one of {VALID_TIERS}."
            )
        if self.state is not None and self.state.run_id and not force:
            raise FileExistsError(
                f"Domain orientation session '{self.state.run_id}' already "
                f"exists at {self.file_path}. Use --force to overwrite."
            )
        run_id = f"domain-{uuid.uuid4().hex[:12]}"
        self.state = DomainOrientationState(
            run_id=run_id,
            created_at=datetime.utcnow().isoformat() + "Z",
            tier=tier,
            domain_declared=domain,
        )
        self.save()
        return run_id

    def require_session(self):
        if self.state is None:
            raise RuntimeError(
                "No active domain-orientation session. Run 'start' first."
            )

    # --- TE Term Extraction ------------------------------------------------

    def extract(self, text_or_path: str) -> List[CandidateTerm]:
        """
        Extract candidate terms from a file path or raw text.

        Uses stdlib regex heuristics: acronyms (`[A-Z]{2,}`), multi-word
        capitalized phrases, alphanumeric tokens (e.g. OD600, CS01).
        Dedupes case-insensitively (preserves first casing) and ranks by
        frequency.
        """
        self.require_session()
        text, source_ref = _read_input(text_or_path)
        if not text:
            return []

        # Collect all matches with original casing
        counts: Dict[str, int] = {}
        first_cased: Dict[str, str] = {}

        def record(token: str):
            if not token or not token.strip():
                return
            token = token.strip()
            key = token.lower()
            if key not in first_cased:
                first_cased[key] = token
            counts[key] = counts.get(key, 0) + 1

        for m in _RE_ACRONYM.finditer(text):
            record(m.group(0))
        for m in _RE_ALNUM.finditer(text):
            record(m.group(0))
        for m in _RE_CAPWORDS.finditer(text):
            record(m.group(0))

        # Skip tokens already registered (candidates or grounded).
        existing_lower = {
            c["text"].lower() for c in self.state.candidate_terms
        } | {
            g["text"].lower() for g in self.state.grounded_terms
        }

        # Build ranked new candidates
        ranked: List[Tuple[str, int]] = sorted(
            counts.items(), key=lambda kv: (-kv[1], kv[0]),
        )
        produced: List[CandidateTerm] = []
        for key, freq in ranked:
            if key in existing_lower:
                continue
            tid = f"TERM-{self.state.next_term_id:03d}"
            self.state.next_term_id += 1
            entry = CandidateTerm(
                id=tid,
                text=first_cased[key],
                frequency=freq,
                source_ref=source_ref,
                logged=datetime.utcnow().isoformat() + "Z",
            )
            self.state.candidate_terms.append(asdict(entry))
            produced.append(entry)
        self.save()
        return produced

    # --- TG Term Grounding -------------------------------------------------

    def ground(self, term: str, definition: str, source: str,
               url: Optional[str] = None,
               confidence: Optional[float] = None,
               allow_novel: bool = False) -> GroundedTerm:
        """
        Ground a candidate term with a definition, source, and confidence.

        Provenance discipline (DECISION D-007):
          - library:        confidence fixed at 1.00 unless explicitly set (rejected if != 1.00).
          - analyst:        confidence capped at ANALYST_CONFIDENCE_CAP (0.80).
          - llm_parametric: confidence capped at LLM_PARAMETRIC_CONFIDENCE_CAP (0.60).
          - chain_derived:  confidence capped at CHAIN_DERIVED_CONFIDENCE_CAP (0.90).

        `term` must already appear in candidate_terms OR the caller must
        pass allow_novel=True (typically the `analyst` source) to register
        a novel term without prior extraction.
        """
        self.require_session()
        if source not in VALID_SOURCES:
            raise ValueError(
                f"Invalid source '{source}'. Must be one of {sorted(VALID_SOURCES)}."
            )
        if not term or not term.strip():
            raise ValueError("term must be a non-empty string")
        if not definition or not definition.strip():
            raise ValueError("definition must be a non-empty string")

        # Resolve confidence per source.
        if source == "library":
            if confidence is not None and abs(confidence - 1.0) > 1e-9:
                raise ValueError(
                    "library confidence is fixed at 1.00 by construction; "
                    f"cannot set confidence={confidence}."
                )
            conf = 1.0
        elif source == "analyst":
            conf = 0.70 if confidence is None else float(confidence)
            if conf > ANALYST_CONFIDENCE_CAP + 1e-9:
                raise ValueError(
                    f"analyst confidence cap is {ANALYST_CONFIDENCE_CAP:.2f}; "
                    f"requested {conf:.2f}. Upgrade source to chain_derived "
                    f"with a logged inference chain to lift."
                )
        elif source == "llm_parametric":
            conf = 0.50 if confidence is None else float(confidence)
            # DECISION D-007: LLM-parametric confidence cap is hard code.
            if conf > LLM_PARAMETRIC_CONFIDENCE_CAP + 1e-9:
                raise ValueError(
                    f"llm_parametric confidence cap is "
                    f"{LLM_PARAMETRIC_CONFIDENCE_CAP:.2f}; requested "
                    f"{conf:.2f}. Upgrade source (library/analyst/"
                    f"chain_derived) before lifting."
                )
        elif source == "chain_derived":
            conf = 0.75 if confidence is None else float(confidence)
            if conf > CHAIN_DERIVED_CONFIDENCE_CAP + 1e-9:
                raise ValueError(
                    f"chain_derived confidence cap is "
                    f"{CHAIN_DERIVED_CONFIDENCE_CAP:.2f}; requested "
                    f"{conf:.2f}."
                )
        else:  # defensive
            raise ValueError(f"Unhandled source '{source}'")

        if not 0.0 < conf <= 1.0:
            raise ValueError(f"confidence must be in (0, 1], got {conf}")

        # Find candidate by text (case-insensitive) or create novel.
        key = term.strip().lower()
        candidate = None
        for c in self.state.candidate_terms:
            if c["text"].lower() == key:
                candidate = c
                break

        # Reject duplicate grounding unless we upgrade the existing.
        for g in self.state.grounded_terms:
            if g["text"].lower() == key:
                raise ValueError(
                    f"Term '{term}' is already grounded as {g['id']}. "
                    f"Remove the existing grounding before re-grounding."
                )

        if candidate is None and not allow_novel:
            raise ValueError(
                f"Term '{term}' not found in candidate_terms. Run "
                f"`extract` first, or pass --allow-novel to register a "
                f"term without prior extraction."
            )

        if candidate is not None:
            tid = candidate["id"]
            # Remove from candidate list
            self.state.candidate_terms = [
                c for c in self.state.candidate_terms if c["id"] != tid
            ]
            text_canonical = candidate["text"]
        else:
            tid = f"TERM-{self.state.next_term_id:03d}"
            self.state.next_term_id += 1
            text_canonical = term.strip()

        entry = GroundedTerm(
            id=tid,
            text=text_canonical,
            definition=definition.strip(),
            source=source,
            confidence=conf,
            url=url,
            source_id=None,
            grounded_at=datetime.utcnow().isoformat() + "Z",
        )
        self.state.grounded_terms.append(asdict(entry))
        self.save()
        return entry

    # --- MM Metrics Mapping ------------------------------------------------

    def add_metric(self, name: str, units: str, higher_is_better: bool,
                   plausibility: Tuple[Optional[float], Optional[float],
                                       Optional[float], Optional[float]],
                   source: str, url: Optional[str] = None,
                   domain: Optional[str] = None,
                   promoted: bool = False) -> Metric:
        """
        Register a metric with its plausibility tuple.

        Plausibility is a 4-tuple `[suspicious, plausible_low, plausible_high,
        excellent]` matching `src/config/domains.json`. `null` permitted for
        unbounded sides.

        Metrics are always staged (promoted=False). Promotion to
        domain_metrics.json happens via `candidates promote` and is blocked
        for `llm_parametric` metrics (DECISION D-007).
        """
        self.require_session()
        if source not in VALID_SOURCES:
            raise ValueError(
                f"Invalid source '{source}'. Must be one of {sorted(VALID_SOURCES)}."
            )
        if not name or not name.strip():
            raise ValueError("metric name must be a non-empty string")
        if not units or not units.strip():
            raise ValueError("metric units must be a non-empty string")

        if len(plausibility) != 4:
            raise ValueError(
                f"plausibility must be a 4-tuple, got length "
                f"{len(plausibility)}."
            )
        # DECISION D-007: caller cannot bypass promotion gate at add-time for
        # llm_parametric metrics.
        if promoted and source == "llm_parametric":
            raise ValueError(
                "llm_parametric metrics cannot be promoted directly; they "
                "must be grounded against a verified library source first."
            )

        # Every non-analyst metric must have a URL or source_id linkage.
        if source in ("library", "llm_parametric", "chain_derived") and not url:
            raise ValueError(
                f"metric source='{source}' requires --url (or a linked "
                f"verified source). analyst is the only exception."
            )

        # Duplicate name within domain rejected.
        for m in self.state.metrics:
            if m["name"] == name.strip() and m.get("domain") == domain:
                raise ValueError(
                    f"metric '{name}' already registered for domain "
                    f"'{domain}' as {m['id']}."
                )

        mid = f"MET-{self.state.next_metric_id:03d}"
        self.state.next_metric_id += 1
        entry = Metric(
            id=mid,
            name=name.strip(),
            units=units.strip(),
            higher_is_better=bool(higher_is_better),
            plausibility=list(plausibility),
            source=source,
            url=url,
            domain=(domain or self.state.domain_declared or None),
            promoted=bool(promoted),
            logged=datetime.utcnow().isoformat() + "Z",
        )
        self.state.metrics.append(asdict(entry))
        self.save()
        return entry

    # --- AM Alias Map ------------------------------------------------------

    def alias(self, canonical: str, aliases_list: List[str],
              region: Optional[str] = None,
              source: Optional[str] = None) -> Alias:
        """
        Register aliases for a canonical grounded term.

        Canonical term MUST already exist in grounded_terms; if not, raises
        ValueError (the analyst cannot alias a term that hasn't been
        grounded — it has no definition to anchor the aliases to).
        """
        self.require_session()
        if not aliases_list:
            raise ValueError("aliases list must contain at least one entry")
        if source is not None and source not in VALID_SOURCES:
            raise ValueError(
                f"Invalid source '{source}'. Must be one of {sorted(VALID_SOURCES)}."
            )

        key = canonical.strip().lower()
        canonical_entry = None
        for g in self.state.grounded_terms:
            if g["text"].lower() == key:
                canonical_entry = g
                break
        if canonical_entry is None:
            raise ValueError(
                f"Canonical term '{canonical}' not found in grounded_terms. "
                f"Ground the term first via `ground` before aliasing."
            )

        aid = f"ALIAS-{self.state.next_alias_id:03d}"
        self.state.next_alias_id += 1
        clean_aliases = [a.strip() for a in aliases_list if a and a.strip()]
        entry = Alias(
            id=aid,
            canonical_id=canonical_entry["id"],
            canonical_text=canonical_entry["text"],
            aliases=clean_aliases,
            region=region,
            source=source,
            logged=datetime.utcnow().isoformat() + "Z",
        )
        self.state.aliases.append(asdict(entry))
        self.save()
        return entry

    # --- CS Canonical Sources ---------------------------------------------

    def add_source(self, title: str, category: str,
                   url: Optional[str] = None,
                   authors: Optional[str] = None,
                   year: Optional[int] = None,
                   coverage: Optional[str] = None) -> Source:
        """Register a candidate canonical source (unverified by default)."""
        self.require_session()
        if not title or not title.strip():
            raise ValueError("source title must be a non-empty string")
        if category not in VALID_SOURCE_CATEGORIES:
            raise ValueError(
                f"Invalid category '{category}'. Must be one of "
                f"{VALID_SOURCE_CATEGORIES}."
            )
        # Dedup by exact title.
        for s in self.state.sources:
            if s["title"].strip().lower() == title.strip().lower():
                raise ValueError(
                    f"Source with title '{title}' already registered as "
                    f"{s['id']}."
                )
        sid = f"SID-{self.state.next_source_id:03d}"
        self.state.next_source_id += 1
        entry = Source(
            id=sid,
            title=title.strip(),
            category=category,
            url=url,
            authors=authors,
            year=year,
            coverage=coverage,
            verified=False,
            logged=datetime.utcnow().isoformat() + "Z",
        )
        self.state.sources.append(asdict(entry))
        self.save()
        return entry

    def verify_source(self, source_id: str,
                      http_status: Optional[int] = None,
                      verified_by: Optional[str] = None) -> Source:
        """
        Mark a source verified.

        Verification is accepted when either:
          - http_status == 200 (a real WebFetch succeeded), OR
          - verified_by == 'citation' (analyst confirmed DOI/ISBN offline).

        Otherwise the source stays unverified and cannot be promoted.
        """
        self.require_session()
        for s in self.state.sources:
            if s["id"] == source_id:
                accepted = (
                    (http_status is not None and int(http_status) == 200)
                    or (verified_by == "citation")
                    or (verified_by == "fetched" and http_status is not None
                        and int(http_status) == 200)
                )
                if not accepted:
                    raise ValueError(
                        f"Source {source_id} not verified: http_status="
                        f"{http_status}, verified_by={verified_by}. "
                        f"Require http_status=200 OR verified_by=citation."
                    )
                s["verified"] = True
                s["verified_at"] = datetime.utcnow().isoformat() + "Z"
                s["http_status"] = http_status
                s["verified_by"] = verified_by or "fetched"
                self.save()
                # Return a Source dataclass instance for consistency
                return Source(**{
                    k: v for k, v in s.items()
                    if k in {f.name for f in dataclass_fields(Source)}
                })
        raise KeyError(f"Source {source_id} not found")

    # --- Candidates (terms, metrics, sources) -----------------------------

    def list_candidates(self, kind: str) -> List[Dict]:
        """List pending candidate entries of the given kind."""
        self.require_session()
        if kind == "terms":
            return list(self.state.candidate_terms)
        if kind == "metrics":
            return [m for m in self.state.metrics if not m.get("promoted")]
        if kind == "sources":
            return [s for s in self.state.sources if not s.get("promoted")]
        raise ValueError(
            f"Invalid kind '{kind}'. Must be one of terms|metrics|sources."
        )

    def promote_candidate(self, candidate_id: str) -> Dict:
        """
        Promote a staged candidate (metric or source) to canonical status.

        Terms are promoted by calling `ground`, not this method.

        Enforces (DECISION D-007):
          - llm_parametric metrics cannot be promoted (RuntimeError).
          - Unverified sources cannot be promoted (RuntimeError).
        """
        self.require_session()
        # Metric path
        if candidate_id.startswith("MET-"):
            for m in self.state.metrics:
                if m["id"] == candidate_id:
                    if m.get("promoted"):
                        raise RuntimeError(
                            f"Metric {candidate_id} already promoted."
                        )
                    # DECISION D-007: llm_parametric metrics are blocked.
                    if m["source"] == "llm_parametric":
                        raise RuntimeError(
                            f"Metric {candidate_id} source=llm_parametric "
                            f"cannot be promoted. Ground it against a "
                            f"verified library source first."
                        )
                    m["promoted"] = True
                    self.save()
                    return dict(m)
            raise KeyError(f"Metric {candidate_id} not found")

        # Source path
        if candidate_id.startswith("SID-"):
            for s in self.state.sources:
                if s["id"] == candidate_id:
                    if s.get("promoted"):
                        raise RuntimeError(
                            f"Source {candidate_id} already promoted."
                        )
                    # DECISION D-007: canonical source must be verified
                    # before it can be cited downstream.
                    if not s.get("verified"):
                        raise RuntimeError(
                            f"Source {candidate_id} not verified; run "
                            f"`verify --source-id {candidate_id} "
                            f"--http-status 200` (or "
                            f"--verified-by citation) before promoting."
                        )
                    s["promoted"] = True
                    self.save()
                    return dict(s)
            raise KeyError(f"Source {candidate_id} not found")

        raise ValueError(
            f"Cannot promote '{candidate_id}': prefix must be MET- "
            f"(metric) or SID- (source). Terms are promoted via `ground`."
        )

    # --- Rendering --------------------------------------------------------

    def render_glossary(self) -> str:
        """Render grounded terms as domain_glossary.md markdown."""
        self.require_session()
        lines: List[str] = []
        lines.append(f"# Domain Glossary — {self.state.domain_declared or '(unspecified)'}")
        lines.append("")
        lines.append(f"Tier: {self.state.tier}")
        lines.append(f"Run ID: {self.state.run_id}")
        lines.append(f"Created: {self.state.created_at}")
        lines.append(f"Grounded terms: {len(self.state.grounded_terms)}")
        lines.append("")
        if not self.state.grounded_terms:
            lines.append("*No grounded terms yet.*")
            return "\n".join(lines) + "\n"

        # Source lookup for citations
        src_lookup = {s["id"]: s for s in self.state.sources}
        # Alias lookup by canonical_id
        alias_lookup: Dict[str, List[str]] = {}
        for a in self.state.aliases:
            alias_lookup.setdefault(a["canonical_id"], []).extend(a["aliases"])

        # Group by source provenance
        groups: Dict[str, List[Dict]] = {}
        for g in self.state.grounded_terms:
            groups.setdefault(g["source"], []).append(g)

        order = ("library", "chain_derived", "analyst", "llm_parametric")
        for group in order:
            if group not in groups:
                continue
            header = {
                "library": "Library-grounded",
                "chain_derived": "Chain-derived",
                "analyst": "Analyst-grounded",
                "llm_parametric": "LLM-parametric (capped)",
            }[group]
            lines.append(f"## {header}")
            lines.append("")
            for g in sorted(groups[group], key=lambda x: x["text"].lower()):
                lines.append(f"### {g['text']}")
                lines.append("")
                lines.append(f"**Definition**: {g['definition']}")
                lines.append("")
                sid = g.get("source_id")
                src_line = f"**Source**: {g['source']}"
                if sid and sid in src_lookup:
                    s = src_lookup[sid]
                    verified = "verified" if s.get("verified") else "unverified"
                    src_line += f" — {s['title']} [{s['category']}, {verified}]"
                if g.get("url"):
                    src_line += f" — {g['url']}"
                lines.append(src_line)
                aliases = alias_lookup.get(g["id"], [])
                if aliases:
                    lines.append(f"**Aliases**: {', '.join(aliases)}")
                lines.append(f"**Confidence**: {g['confidence']:.2f}")
                lines.append(f"**Term ID**: {g['id']}")
                lines.append("")
        return "\n".join(lines) + "\n"

    def render_metrics(self) -> Dict[str, Dict[str, List]]:
        """
        Render promoted metrics as a nested dict:
            {domain: {metric_name: plausibility_list}}
        Matches the shape of `src/config/domains.json` for session-local merge.
        """
        self.require_session()
        out: Dict[str, Dict[str, List]] = {}
        for m in self.state.metrics:
            if not m.get("promoted"):
                continue
            domain = m.get("domain") or self.state.domain_declared or "generic"
            out.setdefault(domain, {})[m["name"]] = list(m["plausibility"])
        return out

    def render_sources(self) -> str:
        """Render registered sources as domain_sources.md markdown, grouped by category."""
        self.require_session()
        lines: List[str] = []
        lines.append(f"# Domain Sources — {self.state.domain_declared or '(unspecified)'}")
        lines.append("")
        lines.append(f"Run ID: {self.state.run_id}")
        lines.append(f"Sources: {len(self.state.sources)} "
                     f"(verified: {sum(1 for s in self.state.sources if s.get('verified'))})")
        lines.append("")
        if not self.state.sources:
            lines.append("*No sources registered yet.*")
            return "\n".join(lines) + "\n"

        groups: Dict[str, List[Dict]] = {}
        for s in self.state.sources:
            groups.setdefault(s["category"], []).append(s)
        header_map = {
            "textbook": "Textbooks",
            "regulator": "Regulators",
            "standard": "Standards",
            "seminal_paper": "Seminal Papers",
            "benchmark_dataset": "Benchmark Datasets",
        }
        for cat in VALID_SOURCE_CATEGORIES:
            if cat not in groups:
                continue
            lines.append(f"## {header_map[cat]}")
            lines.append("")
            for s in sorted(groups[cat], key=lambda x: x["title"].lower()):
                lines.append(f"### {s['title']}")
                lines.append("")
                lines.append(f"- **Category**: {s['category']}")
                if s.get("authors"):
                    lines.append(f"- **Authors/Publisher**: {s['authors']}")
                if s.get("year"):
                    lines.append(f"- **Year**: {s['year']}")
                if s.get("url"):
                    lines.append(f"- **URL**: {s['url']}")
                if s.get("verified"):
                    vline = (f"- **Verified**: {s.get('verified_at', 'unknown')}")
                    if s.get("http_status"):
                        vline += f" (HTTP {s['http_status']})"
                    elif s.get("verified_by") == "citation":
                        vline += " (citation)"
                    lines.append(vline)
                else:
                    lines.append("- **Verified**: NO — not citable downstream")
                if s.get("coverage"):
                    lines.append(f"- **Coverage**: {s['coverage']}")
                lines.append(f"- **Source ID**: {s['id']}")
                lines.append("")
        return "\n".join(lines) + "\n"

    # --- Exit Gate --------------------------------------------------------

    def gate(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Phase 0.3 exit-gate evaluation.

        Applies DECISION D-008 thresholds (tier-dependent for term floor).
        Returns (pass_bool, details_dict).
        """
        self.require_session()
        s = self.state
        n_terms = len(s.grounded_terms)
        n_lib = sum(1 for g in s.grounded_terms if g.get("source") == "library")
        lib_fraction = (n_lib / n_terms) if n_terms > 0 else 0.0
        n_metrics_promoted = sum(1 for m in s.metrics if m.get("promoted"))
        n_verified = sum(1 for src in s.sources if src.get("verified"))
        n_aliases = len(s.aliases)
        n_candidates_seen = n_terms + len(s.candidate_terms)

        tier = s.tier
        is_lite = (tier == "LITE")
        min_terms = GATE_MIN_TERMS_LITE if is_lite else GATE_MIN_TERMS_STANDARD

        checks = {
            "tier": tier,
            "grounded_terms": n_terms,
            "min_terms": min_terms,
            "library_sourced_fraction": round(lib_fraction, 3),
            "min_library_fraction": GATE_MIN_LIBRARY_FRACTION,
            "metrics_promoted": n_metrics_promoted,
            "min_metrics": 0 if is_lite else GATE_MIN_METRICS,
            "verified_sources": n_verified,
            "min_verified_sources": GATE_MIN_VERIFIED_SOURCES,
            "alias_map_present": n_aliases > 0,
            "alias_required": not is_lite,
            "extract_run": n_candidates_seen > 0,
        }

        failures: List[str] = []
        if not checks["extract_run"]:
            failures.append("extract has not been run (no candidate_terms)")
        if n_terms < min_terms:
            failures.append(
                f"grounded_terms={n_terms} < floor={min_terms} ({tier})"
            )
        if lib_fraction < GATE_MIN_LIBRARY_FRACTION and n_terms > 0:
            failures.append(
                f"library_sourced_fraction={lib_fraction:.2f} < "
                f"{GATE_MIN_LIBRARY_FRACTION:.2f}"
            )
        if not is_lite and n_metrics_promoted < GATE_MIN_METRICS:
            failures.append(
                f"metrics_promoted={n_metrics_promoted} < {GATE_MIN_METRICS}"
            )
        if n_verified < GATE_MIN_VERIFIED_SOURCES:
            failures.append(
                f"verified_sources={n_verified} < {GATE_MIN_VERIFIED_SOURCES}"
            )
        if not is_lite and not checks["alias_map_present"]:
            failures.append(
                "alias_map missing (log attestation in decisions.md or "
                "run `alias`)"
            )

        passed = len(failures) == 0
        checks["failures"] = failures
        checks["pass"] = passed

        # Persist last-checked
        s.gate_last_checked = datetime.utcnow().isoformat() + "Z"
        s.gate_status = "PASS" if passed else "FAIL"
        self.save()
        return passed, checks

    # --- Reporting --------------------------------------------------------

    def report(self, verbose: bool = False) -> str:
        """Markdown report of current state."""
        self.require_session()
        s = self.state
        lines: List[str] = []
        lines.append(f"# Domain Orientation Report — {s.run_id}")
        lines.append("")
        lines.append(f"- Tier: {s.tier}")
        lines.append(f"- Domain declared: {s.domain_declared or '(unspecified)'}")
        lines.append(f"- Created: {s.created_at}")
        lines.append(f"- Candidate terms: {len(s.candidate_terms)}")
        lines.append(f"- Grounded terms: {len(s.grounded_terms)}")
        n_lib = sum(1 for g in s.grounded_terms if g.get("source") == "library")
        lines.append(f"  - library-sourced: {n_lib}")
        lines.append(f"- Metrics: {len(s.metrics)} "
                     f"(promoted: {sum(1 for m in s.metrics if m.get('promoted'))})")
        lines.append(f"- Aliases: {len(s.aliases)}")
        lines.append(f"- Sources: {len(s.sources)} "
                     f"(verified: {sum(1 for src in s.sources if src.get('verified'))})")
        lines.append(f"- Last gate: {s.gate_status or '(not checked)'} "
                     f"at {s.gate_last_checked or '—'}")
        lines.append("")

        if verbose:
            lines.append("## Candidate Terms (pending grounding)")
            lines.append("")
            if not s.candidate_terms:
                lines.append("*none*")
            else:
                lines.append("| ID | Text | Freq | Source Ref |")
                lines.append("|---|---|---:|---|")
                for c in s.candidate_terms[:50]:
                    lines.append(
                        f"| {c['id']} | {c['text']} | {c.get('frequency', 1)} "
                        f"| {c.get('source_ref', '')} |"
                    )
            lines.append("")
            lines.append("## Grounded Terms")
            lines.append("")
            if not s.grounded_terms:
                lines.append("*none*")
            else:
                lines.append("| ID | Text | Source | Confidence |")
                lines.append("|---|---|---|---:|")
                for g in s.grounded_terms:
                    lines.append(
                        f"| {g['id']} | {g['text']} | {g['source']} "
                        f"| {g['confidence']:.2f} |"
                    )
            lines.append("")
            lines.append("## Metrics")
            lines.append("")
            if not s.metrics:
                lines.append("*none*")
            else:
                lines.append("| ID | Name | Units | Source | Promoted |")
                lines.append("|---|---|---|---|:---:|")
                for m in s.metrics:
                    lines.append(
                        f"| {m['id']} | {m['name']} | {m['units']} "
                        f"| {m['source']} | "
                        f"{'YES' if m.get('promoted') else ''} |"
                    )
            lines.append("")
            lines.append("## Sources")
            lines.append("")
            if not s.sources:
                lines.append("*none*")
            else:
                lines.append("| ID | Title | Category | Verified |")
                lines.append("|---|---|---|:---:|")
                for src in s.sources:
                    lines.append(
                        f"| {src['id']} | {src['title'][:60]} "
                        f"| {src['category']} | "
                        f"{'YES' if src.get('verified') else ''} |"
                    )
        return "\n".join(lines) + "\n"

    # --- Skip helper ------------------------------------------------------

    def skip_block(self, reason: str) -> str:
        """
        Return a decisions.md block the analyst can pipe into the session's
        decisions.md. Does NOT mutate state — `session_manager.py skip` is
        the authoritative way to flip phase state.
        """
        if not reason or not reason.strip():
            raise ValueError("skip requires a --reason argument")
        now = datetime.utcnow().isoformat() + "Z"
        tier = self.state.tier if self.state else "(no session)"
        domain = self.state.domain_declared if self.state else ""
        lines = [
            f"## {now} — Phase 0.3 skipped",
            "",
            f"**Tier**: {tier}",
            f"**Domain declared**: {domain or '(none)'}",
            "",
            f"**Reason**: {reason.strip()}",
            "",
            "**Attestation**: analyst declared `domain_familiarity=high`. "
            "Phase 0.3 operators (TE/TG/MM/AM/CS) NOT run. If a downstream "
            "phase surfaces a framing error traceable to vocabulary, this "
            "decision owns the cost.",
            "",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _recover_counter(entries: List[Dict], prefix: str) -> int:
    """Recover monotonic counter from existing IDs with the given prefix."""
    nums: List[int] = []
    for e in entries:
        eid = e.get("id") if isinstance(e, dict) else None
        if isinstance(eid, str) and eid.startswith(prefix):
            tail = eid[len(prefix):]
            if tail.isdigit():
                nums.append(int(tail))
    return (max(nums) if nums else 0) + 1


def _read_input(text_or_path: str) -> Tuple[str, str]:
    """
    Resolve the `--input` argument.

    Accepts a path to a readable file, '-' for stdin, or raw text. Returns
    (content, source_ref).
    """
    if text_or_path == "-":
        return sys.stdin.read(), "<stdin>"
    if os.path.isfile(text_or_path):
        with open(text_or_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read(), os.path.basename(text_or_path)
    # Treat as raw text literal.
    return text_or_path, "<inline>"


def _parse_bool(value: str) -> bool:
    v = str(value).strip().lower()
    if v in ("true", "1", "yes", "y", "t"):
        return True
    if v in ("false", "0", "no", "n", "f"):
        return False
    raise ValueError(f"Expected boolean (true/false), got '{value}'")


def _parse_plausibility(value: str) -> Tuple[Optional[float], Optional[float],
                                             Optional[float], Optional[float]]:
    """Parse 'null,100,10000,null' into a 4-tuple of floats/None."""
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 4:
        raise ValueError(
            f"--plausibility expects 4 comma-separated values "
            f"(suspicious,plausible_low,plausible_high,excellent); got "
            f"{len(parts)} ('{value}')."
        )
    out: List[Optional[float]] = []
    for p in parts:
        if p.lower() in ("null", "none", ""):
            out.append(None)
        else:
            try:
                out.append(float(p))
            except ValueError as e:
                raise ValueError(
                    f"--plausibility entry '{p}' is not a number or null"
                ) from e
    return tuple(out)  # type: ignore[return-value]


def _parse_aliases(value: str) -> List[str]:
    return [a.strip() for a in value.split(",") if a and a.strip()]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="domain_orienter.py",
        description="Domain Orientation tool (Phase 0.3).",
    )
    parser.add_argument(
        "--file", default="domain_orientation.json",
        help="Path to domain_orientation.json (default: ./domain_orientation.json)",
    )
    sub = parser.add_subparsers(dest="cmd", required=False)

    # start
    p_start = sub.add_parser("start", help="Start a new Phase 0.3 session")
    p_start.add_argument("--tier", default="STANDARD", choices=list(VALID_TIERS))
    p_start.add_argument("--domain", default="", help="Domain declared")
    p_start.add_argument("--force", action="store_true",
                         help="Overwrite existing session")

    # extract (TE)
    p_ext = sub.add_parser("extract", help="TE: extract candidate terms")
    p_ext.add_argument("--input", required=True,
                       help="Path to input file, '-' for stdin, or inline text")
    p_ext.add_argument("--allow-novel", action="store_true",
                       help="(accepted for symmetry with `ground`; no effect)")

    # ground (TG)
    p_gr = sub.add_parser("ground", help="TG: ground a candidate term")
    p_gr.add_argument("--term", required=True)
    p_gr.add_argument("--definition", required=True)
    p_gr.add_argument("--source", required=True, choices=sorted(VALID_SOURCES))
    p_gr.add_argument("--url", default=None)
    p_gr.add_argument("--confidence", type=float, default=None)
    p_gr.add_argument("--allow-novel", action="store_true",
                      help="Accept a term not in candidate_terms")

    # add-metric (MM)
    p_mm = sub.add_parser("add-metric", help="MM: register a metric")
    p_mm.add_argument("--name", required=True)
    p_mm.add_argument("--units", required=True)
    p_mm.add_argument("--higher-is-better", required=True)
    p_mm.add_argument("--plausibility", required=True,
                      help="'suspicious,plausible_low,plausible_high,excellent'; "
                           "use 'null' for unbounded")
    p_mm.add_argument("--source", required=True, choices=sorted(VALID_SOURCES))
    p_mm.add_argument("--url", default=None)
    p_mm.add_argument("--domain", default=None)

    # alias (AM)
    p_al = sub.add_parser("alias", help="AM: register aliases for a grounded term")
    p_al.add_argument("--canonical", required=True)
    p_al.add_argument("--aliases", required=True,
                      help="Comma-separated alias list")
    p_al.add_argument("--region", default=None)
    p_al.add_argument("--source", default=None, choices=sorted(VALID_SOURCES))

    # source (CS — add)
    p_src = sub.add_parser("source", help="CS: register a canonical source")
    p_src.add_argument("--title", required=True)
    p_src.add_argument("--category", required=True,
                       choices=list(VALID_SOURCE_CATEGORIES))
    p_src.add_argument("--url", default=None)
    p_src.add_argument("--authors", default=None)
    p_src.add_argument("--year", type=int, default=None)
    p_src.add_argument("--coverage", default=None)

    # verify
    p_ver = sub.add_parser("verify", help="CS: mark a source verified")
    p_ver.add_argument("--source-id", required=True)
    p_ver.add_argument("--http-status", type=int, default=None)
    p_ver.add_argument("--verified-by", default=None,
                       choices=["citation", "fetched"])

    # candidates (compound)
    p_cand = sub.add_parser("candidates", help="Candidate management")
    cand_sub = p_cand.add_subparsers(dest="cand_cmd", required=True)
    p_cl = cand_sub.add_parser("list", help="List pending candidates")
    p_cl.add_argument("--kind", required=True,
                      choices=["terms", "metrics", "sources"])
    p_cp = cand_sub.add_parser("promote", help="Promote a candidate")
    p_cp.add_argument("--id", required=True,
                      help="MET-NNN (metric) or SID-NNN (source)")

    # glossary render
    p_gl = sub.add_parser("glossary", help="Glossary management")
    gl_sub = p_gl.add_subparsers(dest="gl_cmd", required=True)
    p_glr = gl_sub.add_parser("render", help="Render domain_glossary.md")
    p_glr.add_argument("--output", default=None, help="Write to file")

    # metrics render
    p_me = sub.add_parser("metrics", help="Metrics management")
    me_sub = p_me.add_subparsers(dest="me_cmd", required=True)
    p_mer = me_sub.add_parser("render", help="Render domain_metrics.json")
    p_mer.add_argument("--output", default=None, help="Write to file")

    # sources render
    p_so = sub.add_parser("sources", help="Sources management")
    so_sub = p_so.add_subparsers(dest="so_cmd", required=True)
    p_sor = so_sub.add_parser("render", help="Render domain_sources.md")
    p_sor.add_argument("--output", default=None, help="Write to file")

    # gate
    p_gate = sub.add_parser("gate", help="Phase 0.3 exit gate check")
    p_gate.add_argument("--json", action="store_true",
                        help="Emit machine-readable JSON")

    # report
    p_rep = sub.add_parser("report", help="Human-readable status")
    p_rep.add_argument("--verbose", action="store_true")

    # skip
    p_skip = sub.add_parser("skip", help="Print a decisions.md skip block")
    p_skip.add_argument("--reason", required=True)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd is None:
        parser.print_help()
        return 0

    orienter = DomainOrienter(args.file)

    # start is the one command that creates state
    if args.cmd == "start":
        try:
            run_id = orienter.start(
                tier=args.tier, domain=args.domain, force=args.force,
            )
            print(f"Started domain-orientation session {run_id} "
                  f"(tier={args.tier}, domain={args.domain or '(unspecified)'})")
            return 0
        except (ValueError, FileExistsError) as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2

    # Auto-start for convenience so operators work out of the box.
    if orienter.state is None:
        # `skip` may be called on an empty session — permit it.
        if args.cmd != "skip":
            orienter.start()

    try:
        if args.cmd == "extract":
            produced = orienter.extract(args.input)
            print(f"TE produced {len(produced)} candidate terms:")
            for c in produced[:30]:
                print(f"  - {c.id} [{c.frequency}x]: {c.text}")
            if len(produced) > 30:
                print(f"  ... ({len(produced) - 30} more)")
            return 0

        if args.cmd == "ground":
            g = orienter.ground(
                term=args.term,
                definition=args.definition,
                source=args.source,
                url=args.url,
                confidence=args.confidence,
                allow_novel=args.allow_novel,
            )
            print(f"Grounded {g.id} '{g.text}' "
                  f"[{g.source}, conf={g.confidence:.2f}]")
            return 0

        if args.cmd == "add-metric":
            higher = _parse_bool(args.higher_is_better)
            plaus = _parse_plausibility(args.plausibility)
            m = orienter.add_metric(
                name=args.name,
                units=args.units,
                higher_is_better=higher,
                plausibility=plaus,
                source=args.source,
                url=args.url,
                domain=args.domain,
            )
            print(f"Registered metric {m.id} '{m.name}' "
                  f"({m.units}) [source={m.source}, staged]")
            return 0

        if args.cmd == "alias":
            aliases_list = _parse_aliases(args.aliases)
            a = orienter.alias(
                canonical=args.canonical,
                aliases_list=aliases_list,
                region=args.region,
                source=args.source,
            )
            print(f"Registered {a.id} for {a.canonical_id} "
                  f"'{a.canonical_text}' — {len(a.aliases)} aliases")
            return 0

        if args.cmd == "source":
            s = orienter.add_source(
                title=args.title,
                category=args.category,
                url=args.url,
                authors=args.authors,
                year=args.year,
                coverage=args.coverage,
            )
            print(f"Registered source {s.id} '{s.title}' "
                  f"[category={s.category}, verified=NO]")
            return 0

        if args.cmd == "verify":
            s = orienter.verify_source(
                source_id=args.source_id,
                http_status=args.http_status,
                verified_by=args.verified_by,
            )
            print(f"Verified {s.id} '{s.title}' "
                  f"(http={s.http_status}, by={s.verified_by})")
            return 0

        if args.cmd == "candidates":
            if args.cand_cmd == "list":
                items = orienter.list_candidates(kind=args.kind)
                print(f"{len(items)} pending {args.kind}:")
                for it in items[:50]:
                    if args.kind == "terms":
                        print(f"  - {it['id']} [{it.get('frequency', 1)}x]: "
                              f"{it['text']}")
                    elif args.kind == "metrics":
                        print(f"  - {it['id']} {it['name']} ({it['units']}) "
                              f"[source={it['source']}]")
                    else:  # sources
                        vflag = "verified" if it.get("verified") else "UNVERIFIED"
                        print(f"  - {it['id']} [{vflag}] "
                              f"{it['category']}: {it['title']}")
                if len(items) > 50:
                    print(f"  ... ({len(items) - 50} more)")
                return 0
            if args.cand_cmd == "promote":
                promoted = orienter.promote_candidate(args.id)
                print(f"Promoted {promoted['id']} ({promoted.get('name') or promoted.get('title', '')[:40]})")
                return 0

        if args.cmd == "glossary":
            if args.gl_cmd == "render":
                md = orienter.render_glossary()
                if args.output:
                    with open(args.output, "w", encoding="utf-8") as f:
                        f.write(md)
                    print(f"Wrote glossary to {args.output}")
                else:
                    print(md)
                return 0

        if args.cmd == "metrics":
            if args.me_cmd == "render":
                data = orienter.render_metrics()
                blob = json.dumps(data, indent=2, sort_keys=True)
                if args.output:
                    with open(args.output, "w", encoding="utf-8") as f:
                        f.write(blob + "\n")
                    print(f"Wrote metrics to {args.output}")
                else:
                    print(blob)
                return 0

        if args.cmd == "sources":
            if args.so_cmd == "render":
                md = orienter.render_sources()
                if args.output:
                    with open(args.output, "w", encoding="utf-8") as f:
                        f.write(md)
                    print(f"Wrote sources to {args.output}")
                else:
                    print(md)
                return 0

        if args.cmd == "gate":
            passed, details = orienter.gate()
            if args.json:
                print(json.dumps(details, indent=2, sort_keys=True))
            else:
                print("Phase 0.3 Exit Gate:")
                for k, v in details.items():
                    if k == "failures":
                        continue
                    print(f"  {k}: {v}")
                if details.get("failures"):
                    print("  failures:")
                    for f in details["failures"]:
                        print(f"    - {f}")
            return 0 if passed else 1

        if args.cmd == "report":
            print(orienter.report(verbose=args.verbose))
            return 0

        if args.cmd == "skip":
            block = orienter.skip_block(args.reason)
            print(block)
            return 0

    except (KeyError, ValueError, RuntimeError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
