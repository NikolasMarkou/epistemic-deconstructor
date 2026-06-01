#!/usr/bin/env python3
"""Deterministic, stdlib-only Mermaid emitters for the Epistemic Deconstructor.

This module is the single home for building Mermaid diagram strings from the
project's existing data structures (PHASE_SEQUENCE, inference chains,
candidate<->observation coverage, causal graphs, ABM adjacency). It is a pure
library: every emit function takes plain Python data and returns a ``str``
holding a complete Mermaid code-block *body* (no surrounding ```` ```mermaid ````
fence; callers add the fence, or use :func:`fence`).

Determinism contract (see ``src/references/mermaid-conventions.md`` §6 and
``plans/plan_2026-06-01_a27c8aac/decisions.md`` D-001):

* node IDs are a pure, sanitized function of the source content;
* iteration is sorted by a stable key wherever input order is not defined;
* NO ``datetime``, NO ``random``, NO PIDs — same input yields byte-identical
  output so snapshot tests stay stable.

STDLIB-ONLY by hard invariant (INV-2): this module imports only ``re``,
``argparse``, ``sys``, and ``typing``. It must NEVER import ``simulator.py``,
``numpy``, ``scipy``, or ``pandas``.
"""

import argparse
import re
import sys
from typing import Dict, List, Optional, Union

__all__ = [
    "fence",
    "phase_sequence_to_mermaid",
    "inference_chain_to_mermaid",
    "coverage_graph_to_mermaid",
    "causal_graph_to_mermaid",
    "adjacency_to_mermaid",
]


# ---------------------------------------------------------------------------
# Escaping / ID helpers
# ---------------------------------------------------------------------------

# DECISION plan_2026-06-01_a27c8aac/D-001: the node-ID scheme below is the
# anchored design choice. IDs are a *pure, content-derived* function — lowercase,
# non-alphanumerics collapsed to single underscores, digit-leading IDs prefixed
# with `n`. Do NOT switch to insertion-order counters (e.g. n0, n1, ...) or
# object id()/hash()-based IDs: those make output depend on iteration order or
# hash seeding and break the byte-identical-snapshot contract (mermaid-conventions
# §6). Do NOT leave Unicode/operators in the ID — Mermaid only accepts ASCII
# `[A-Za-z_][A-Za-z0-9_]*` left of the `["label"]`; such characters belong in the
# quoted label via `_escape_label`. See decisions.md D-001.
def _sanitize_id(s: str) -> str:
    """Return a deterministic ASCII-safe Mermaid node id for ``s``.

    Lowercases, replaces every run of non-``[a-z0-9]`` characters with a single
    underscore, strips leading/trailing underscores, and prefixes ``n`` when the
    result is empty or starts with a digit. Output always matches
    ``^[a-z_][a-z0-9_]*$``.
    """
    lowered = str(s).lower()
    cleaned = re.sub(r"[^a-z0-9]+", "_", lowered)
    cleaned = cleaned.strip("_")
    if not cleaned:
        return "n"
    if cleaned[0].isdigit():
        cleaned = "n" + cleaned
    return cleaned


# Transliteration table for common non-ASCII glyphs that show up in labels so
# diagrams stay readable without raw Unicode tripping a parser.
_LABEL_TRANSLITERATE = {
    "→": "->",   # →
    "←": "<-",   # ←
    "↔": "<->",  # ↔
    "⇒": "=>",   # ⇒
    "⊕": "(+)",  # ⊕
    "⊖": "(-)",  # ⊖
    "≤": "<=",   # ≤
    "≥": ">=",   # ≥
    "≠": "!=",   # ≠
    "×": "x",    # ×
}


def _escape_label(s: str) -> str:
    """Return ``s`` made safe to place inside a Mermaid ``["..."]`` label.

    Double quotes become single quotes (Mermaid label delimiter is ``"``), any
    newline/tab/carriage-return collapses to a single space, square brackets and
    curly braces are softened to parens, and a small set of common Unicode
    operators is transliterated to ASCII. Whitespace runs collapse so output is
    stable.
    """
    text = str(s)
    for glyph, repl in _LABEL_TRANSLITERATE.items():
        text = text.replace(glyph, repl)
    text = text.replace('"', "'")
    text = text.replace("[", "(").replace("]", ")")
    text = text.replace("{", "(").replace("}", ")")
    # Collapse all whitespace (incl. raw newlines) to single spaces.
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _fmt_lr(value) -> str:
    """Format a likelihood ratio for an edge label deterministically.

    Integers render without a trailing ``.0``; floats keep up to 3 significant
    decimals with trailing zeros trimmed.
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        return _escape_label(str(value))
    if f == int(f):
        return str(int(f))
    return ("%.3f" % f).rstrip("0").rstrip(".")


def fence(body: str, kind: str = "mermaid") -> str:
    """Wrap ``body`` in a balanced fenced code block.

    Returns ``"```<kind>\\n<body>\\n```"`` with no trailing newline. ``body`` is
    emitted verbatim (already a complete diagram body from one of the emitters).
    """
    return "```" + kind + "\n" + body + "\n```"


# ---------------------------------------------------------------------------
# Phase-sequence FSM  ->  stateDiagram-v2
# ---------------------------------------------------------------------------

def phase_sequence_to_mermaid(
    phase_sequence: Dict,
    tier: Optional[str] = None,
    current: Optional[str] = None,
    labels: Optional[Dict[str, str]] = None,
) -> str:
    """Render a per-tier phase map as a Mermaid ``stateDiagram-v2`` body.

    ``phase_sequence`` is either a single-tier map ``{phase: next_or_None}`` or
    the full multi-tier dict ``{tier: {phase: next}}``. When ``tier`` is given
    and the selected values are dicts, ``phase_sequence[tier]`` is used.

    The first phase (one that is never a transition target) gets a ``[*] -->``
    start edge; phases whose ``next`` is ``None`` get a ``--> [*]`` terminal
    edge. ``current`` (if set and present) is styled via a ``current`` classDef.
    ``labels`` maps phase -> display label, emitted as ``state "label" as id``.
    """
    seq = phase_sequence
    if tier is not None and isinstance(seq, dict) and seq:
        sample = next(iter(seq.values()))
        if isinstance(sample, dict):
            if tier not in seq:
                raise KeyError("tier %r not in phase_sequence" % (tier,))
            seq = seq[tier]
    if not isinstance(seq, dict):
        raise TypeError("phase_sequence (or its tier slice) must be a dict")

    labels = labels or {}
    phases = sorted(seq.keys())
    targets = {nxt for nxt in seq.values() if nxt is not None}
    starts = sorted(p for p in phases if p not in targets)

    id_of = {p: _sanitize_id(p) for p in phases}
    # Targets may name a phase not present as a key; sanitize those too.
    for nxt in seq.values():
        if nxt is not None and nxt not in id_of:
            id_of[nxt] = _sanitize_id(nxt)

    lines: List[str] = ["stateDiagram-v2"]

    # Declarations for labelled states (deterministic by sanitized id).
    for phase in sorted(labels.keys()):
        if phase in id_of:
            lines.append(
                '    state "%s" as %s' % (_escape_label(labels[phase]), id_of[phase])
            )

    for start in starts:
        lines.append("    [*] --> %s" % id_of[start])

    for phase in phases:
        nxt = seq[phase]
        if nxt is None:
            lines.append("    %s --> [*]" % id_of[phase])
        else:
            lines.append("    %s --> %s" % (id_of[phase], id_of[nxt]))

    if current is not None and current in id_of:
        lines.append(
            "    classDef current fill:#cde4ff,stroke:#1c5d99,stroke-width:2px;"
        )
        lines.append("    class %s current" % id_of[current])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Abductive inference chain  ->  flowchart LR
# ---------------------------------------------------------------------------

def inference_chain_to_mermaid(chain: Dict) -> str:
    """Render one abductive inference chain as a ``flowchart LR`` body.

    ``chain`` carries ``premise``, an ordered ``steps`` list (each with ``idx``,
    ``claim``, ``lr``), a ``status``, and ``final_posterior``. Emits:
    ``premise --> step1 --> step2 ... --> close`` where each edge is labelled with
    the *target* step's ``lr`` and the terminal ``close`` node shows the final
    posterior when ``status == "closed"``.
    """
    if not isinstance(chain, dict):
        raise TypeError("chain must be a dict")

    cid = chain.get("id", "chain")
    base = _sanitize_id(cid)
    premise_id = base + "_premise"
    premise_text = _escape_label(chain.get("premise", "(premise)"))

    lines: List[str] = ["flowchart LR"]
    lines.append('    %s["Premise: %s"]' % (premise_id, premise_text))

    steps = chain.get("steps") or []
    # Steps are an ordered list; preserve given order (idx is authoritative).
    prev_id = premise_id
    for i, step in enumerate(steps):
        idx = step.get("idx", i + 1)
        step_id = "%s_s%s" % (base, _sanitize_id(str(idx)))
        claim = _escape_label(step.get("claim", "(claim)"))
        lr_label = _fmt_lr(step.get("lr", ""))
        lines.append('    %s["Step %s: %s"]' % (step_id, _escape_label(str(idx)), claim))
        lines.append("    %s -->|LR %s| %s" % (prev_id, lr_label, step_id))
        prev_id = step_id

    if chain.get("status") == "closed":
        close_id = base + "_close"
        posterior = chain.get("final_posterior")
        if posterior is None:
            post_label = "closed"
        else:
            post_label = "posterior %s" % _fmt_lr(posterior)
        lines.append('    %s(["%s"])' % (close_id, _escape_label(post_label)))
        lines.append("    %s --> %s" % (prev_id, close_id))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Candidate <-> observation coverage  ->  bipartite graph LR
# ---------------------------------------------------------------------------

def coverage_graph_to_mermaid(
    observations: List[Dict],
    candidates: List[Dict],
) -> str:
    """Render candidate<->observation coverage as a bipartite ``graph LR`` body.

    Each observation and candidate becomes a node (id derived from its ``id``).
    For every id in a candidate's ``observations_explained`` (and any
    ``obs['explained_by']`` back-link) an edge ``candidate --> observation`` is
    drawn. The two node classes are distinguished via ``classDef``.
    """
    obs_list = list(observations or [])
    cand_list = list(candidates or [])

    obs_id_of: Dict[str, str] = {}
    for o in obs_list:
        raw = o["id"] if isinstance(o, dict) else o
        obs_id_of[str(raw)] = "obs_" + _sanitize_id(str(raw))
    cand_id_of: Dict[str, str] = {}
    for c in cand_list:
        raw = c["id"] if isinstance(c, dict) else c
        cand_id_of[str(raw)] = "cand_" + _sanitize_id(str(raw))

    lines: List[str] = ["graph LR"]

    # Observation nodes (sorted by raw id for stability).
    for o in sorted(obs_list, key=lambda x: str(x["id"] if isinstance(x, dict) else x)):
        raw = str(o["id"] if isinstance(o, dict) else o)
        text = o.get("text", raw) if isinstance(o, dict) else raw
        lines.append('    %s["%s"]' % (obs_id_of[raw], _escape_label(text)))

    # Candidate nodes.
    for c in sorted(cand_list, key=lambda x: str(x["id"] if isinstance(x, dict) else x)):
        raw = str(c["id"] if isinstance(c, dict) else c)
        text = c.get("cause", raw) if isinstance(c, dict) else raw
        lines.append('    %s("%s")' % (cand_id_of[raw], _escape_label(text)))

    # Edges: collect (cand_raw, obs_raw) pairs from both directions, dedupe, sort.
    pairs = set()
    for c in cand_list:
        if not isinstance(c, dict):
            continue
        craw = str(c["id"])
        for oid in c.get("observations_explained", []) or []:
            pairs.add((craw, str(oid)))
    for o in obs_list:
        if not isinstance(o, dict):
            continue
        oraw = str(o["id"])
        for cid in o.get("explained_by", []) or []:
            pairs.add((str(cid), oraw))

    for craw, oraw in sorted(pairs):
        c_node = cand_id_of.get(craw, "cand_" + _sanitize_id(craw))
        o_node = obs_id_of.get(oraw, "obs_" + _sanitize_id(oraw))
        lines.append("    %s --> %s" % (c_node, o_node))

    lines.append("    classDef observation fill:#e6f0ff,stroke:#1c5d99;")
    lines.append("    classDef candidate fill:#fff2cc,stroke:#b8860b;")
    obs_nodes = sorted(obs_id_of.values())
    cand_nodes = sorted(cand_id_of.values())
    if obs_nodes:
        lines.append("    class %s observation" % ",".join(obs_nodes))
    if cand_nodes:
        lines.append("    class %s candidate" % ",".join(cand_nodes))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Causal graph  ->  flowchart TD
# ---------------------------------------------------------------------------

def _node_id_name(node) -> tuple:
    """Return ``(raw_id, display_name)`` for a node given as dict or string."""
    if isinstance(node, dict):
        raw = node.get("id", node.get("name"))
        name = node.get("name", raw)
        return str(raw), str(name)
    return str(node), str(node)


def causal_graph_to_mermaid(nodes: List, edges: List) -> str:
    """Render a signed causal graph as a ``flowchart TD`` body.

    ``nodes`` is a list of ``{id, name}`` dicts (or bare strings). ``edges`` is a
    list of ``{source, target, sign}`` where ``sign`` is ``+``/``-`` (a ``type``
    key is accepted as a fallback). Edges are labelled with the sign; ``+`` edges
    are styled ``reinforcing`` and ``-`` edges ``balancing`` where derivable.
    """
    node_list = list(nodes or [])
    edge_list = list(edges or [])

    id_of: Dict[str, str] = {}
    name_of: Dict[str, str] = {}
    for n in node_list:
        raw, name = _node_id_name(n)
        id_of[raw] = _sanitize_id(raw)
        name_of[raw] = name

    def resolve(raw) -> str:
        raw = str(raw)
        if raw not in id_of:
            id_of[raw] = _sanitize_id(raw)
            name_of[raw] = raw
        return id_of[raw]

    lines: List[str] = ["flowchart TD"]
    for raw in sorted(id_of.keys()):
        lines.append('    %s["%s"]' % (id_of[raw], _escape_label(name_of.get(raw, raw))))

    # Normalize and sort edges deterministically.
    norm_edges = []
    for e in edge_list:
        src = str(e.get("source"))
        tgt = str(e.get("target"))
        sign = e.get("sign", e.get("type", ""))
        sign = "" if sign is None else str(sign)
        norm_edges.append((src, tgt, sign))
    for src, tgt, sign in sorted(norm_edges):
        s_node = resolve(src)
        t_node = resolve(tgt)
        label = _escape_label(sign) if sign else ""
        if label:
            lines.append("    %s -->|%s| %s" % (s_node, label, t_node))
        else:
            lines.append("    %s --> %s" % (s_node, t_node))

    if norm_edges:
        lines.append("    classDef reinforcing fill:#fff2cc,stroke:#b8860b;")
        lines.append("    classDef balancing fill:#e6e6fa,stroke:#5b4b8a;")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ABM adjacency  ->  undirected-style graph LR
# ---------------------------------------------------------------------------

def adjacency_to_mermaid(
    adj: Dict,
    labels: Optional[Dict[str, str]] = None,
) -> str:
    """Render an adjacency map (ABM topology) as a ``graph LR`` body.

    ``adj`` maps node -> iterable of neighbours. Edges are treated as undirected:
    symmetric pairs are deduped deterministically by sorting each endpoint pair.
    ``labels`` maps node -> display label.
    """
    adj = adj or {}
    labels = labels or {}

    nodes = set()
    for src, neighbours in adj.items():
        nodes.add(str(src))
        for dst in (neighbours or []):
            nodes.add(str(dst))

    id_of = {n: _sanitize_id(n) for n in nodes}

    lines: List[str] = ["graph LR"]
    for n in sorted(nodes):
        label = labels.get(n, n)
        lines.append('    %s["%s"]' % (id_of[n], _escape_label(label)))

    pairs = set()
    for src, neighbours in adj.items():
        for dst in (neighbours or []):
            a, b = str(src), str(dst)
            if a == b:
                pairs.add((a, b))
            else:
                pairs.add(tuple(sorted((a, b))))

    for a, b in sorted(pairs):
        lines.append("    %s --- %s" % (id_of[a], id_of[b]))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Thin CLI (library is the primary use; this exists for the --help smoke loop)
# ---------------------------------------------------------------------------

def _demo() -> str:
    """Return a small, self-contained example diagram (fenced)."""
    seq = {"A": "B", "B": "C", "C": None}
    body = phase_sequence_to_mermaid(seq, current="B")
    return fence(body)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mermaid_render.py",
        description=(
            "Deterministic stdlib-only Mermaid emitters. Primary use is as an "
            "importable library; this CLI exists for smoke-testing."
        ),
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("demo", help="Print a small example Mermaid diagram block.")

    args = parser.parse_args(argv)

    if args.command == "demo":
        print(_demo())
        return 0

    # No subcommand: print help to stdout and succeed (smoke-friendly).
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
