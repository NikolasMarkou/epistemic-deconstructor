#!/usr/bin/env python3
"""
Abductive Engine for Epistemic Deconstruction v7.14.1

Implements Phase 1.5 Abductive Expansion operators:
  TI — Trace Inversion      -> `invert`
  AA — Absence Audit        -> `absence-audit`
  SA — Surplus Audit        -> `surplus-audit`
  AR — Analogical Retrieval -> `analogize`
  IC — Inference Chains     -> `chain {start|step|close|audit}`

Also provides:
  catalog {bootstrap|review}  — LLM-bootstrapped trace catalog workflow
  candidates {list|promote}   — staging and coverage-weighted promotion
  report                      — full audit report

State is persisted to an abductive_state.json file (dataclass container).
Candidates live in a staging area (hypothesis_candidates.json) until they
pass the coverage-weighted promotion gate. Promotion writes the candidate
into the session's hypotheses.json via a subprocess call to
bayesian_tracker.py.

Provenance discipline: every candidate cause carries a `source` field,
one of {library, llm_parametric, analyst, chain_derived}. LLM-parametric
candidates are hard-capped at prior 0.30 and LR 2.0 until independent
evidence lifts them (source upgrade).

Stdlib-only. Numpy/scipy are not required.

Usage:
    python3 abductive_engine.py --file state.json start
    python3 abductive_engine.py --file state.json invert --obs-id O1 --text "..." --category timing
    python3 abductive_engine.py --file state.json absence-audit --hypothesis H1 --predictions "A;B;C"
    python3 abductive_engine.py --file state.json surplus-audit
    python3 abductive_engine.py --file state.json analogize --signature "..."
    python3 abductive_engine.py --file state.json chain start --target H1 --premise "..."
    python3 abductive_engine.py --file state.json candidates promote --id CAND1 --threshold 0.5
    python3 abductive_engine.py --file state.json report
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field, asdict, fields as dataclass_fields
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

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

# DECISION D-003: Provenance enum is load-bearing, not decorative.
# Every candidate/step carries source in this tuple; LR/prior caps below
# depend on this field being set correctly. Do NOT add an 'unknown' or
# default source — missing provenance is a protocol bug, not a state.
# See plan_2026-04-15_015cc878/decisions.md D-003.
VALID_SOURCES = ('library', 'llm_parametric', 'analyst', 'chain_derived')

# DECISION D-004: LLM-parametric caps are hard code, not docs.
# These bounds enforce Evidence Rule 8 (SKILL.md) against LLM-anchoring
# bias. Do NOT raise these without first logging a pivot in decisions.md
# and updating SKILL.md Evidence Rule 8. The override path is:
# upgrade the candidate's source to library/analyst/chain_derived by
# recording independent evidence — never by loosening the cap.
# See plan_2026-04-15_015cc878/decisions.md D-004.
LLM_PARAMETRIC_MAX_PRIOR = 0.30
LLM_PARAMETRIC_MAX_LR = 2.0

# DECISION D-005: Coverage-weighted selection is the primary mitigation
# for hypothesis explosion (Phase 1.5's main failure mode). Threshold is
# tunable per-call, but the default must stay above "anything goes".
# See plan_2026-04-15_015cc878/decisions.md D-005.
DEFAULT_COVERAGE_THRESHOLD = 0.30

# Default complexity penalty per extra mechanism unit (parsimony weight).
DEFAULT_COMPLEXITY_PENALTY = 1.0


# ---------------------------------------------------------------------------
# Catalog loader
# ---------------------------------------------------------------------------

def load_trace_catalog(config_path: Optional[str] = None) -> Dict:
    """
    Load the trace catalog from src/config/trace_catalog.json.

    The catalog is keyed by observation-category (e.g. 'timing',
    'resource', 'output_anomaly') and maps each category to a list of
    candidate causes with mechanism and prior.

    Entries prefixed with '_' (comment/schema) are filtered out.
    Falls back to a minimal built-in catalog if the config file is
    missing.
    """
    if config_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, '..', 'config', 'trace_catalog.json')

    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Failed to parse trace catalog at {config_path}: {e}"
            )
        return {k: v for k, v in data.items() if not k.startswith('_')}

    # Minimal fallback so --help and smoke tests work without config.
    return {
        'generic': {
            'description': 'Fallback category — install config/trace_catalog.json.',
            'candidates': [
                {
                    'cause': 'unobserved upstream input',
                    'mechanism': 'input variation not recorded drives output variation',
                    'prior': 0.2,
                    'source': 'library',
                }
            ],
        }
    }


def load_archetype_library_for_analogy(config_path: Optional[str] = None) -> Dict:
    """
    Load archetypes.json and return only entries that carry a
    `trace_signatures` field. Used by the AR operator for bidirectional
    archetype <-> signature matching.
    """
    if config_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, '..', 'config', 'archetypes.json')

    if not os.path.exists(config_path):
        return {}
    try:
        with open(config_path) as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return {}
    return {
        k: v for k, v in data.items()
        if not k.startswith('_') and isinstance(v, dict) and 'trace_signatures' in v
    }


# ---------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------

@dataclass
class AbductiveState:
    """Container for a Phase 1.5 Abductive Expansion session."""
    id: str
    created: str
    observations: List[Dict] = field(default_factory=list)
    candidates: List[Dict] = field(default_factory=list)
    pending_predictions: List[Dict] = field(default_factory=list)
    inference_chains: List[Dict] = field(default_factory=list)
    surplus_log: List[Dict] = field(default_factory=list)
    analogies: List[Dict] = field(default_factory=list)
    notes: str = ""
    next_candidate_id: int = 1
    next_chain_id: int = 1
    next_prediction_id: int = 1


class AbductiveEngine:
    """
    Phase 1.5 state + operators.

    Parallels ScopeAuditor. Persists state to a single JSON file via
    common.load_json / save_json (atomic + locked).
    """

    def __init__(self, filepath: str = "abductive_state.json"):
        self.filepath = filepath
        self.state: Optional[AbductiveState] = None
        self.load()

    # --- persistence -------------------------------------------------------

    def load(self):
        data = load_json(self.filepath)
        if data is not None:
            known = {f.name for f in dataclass_fields(AbductiveState)}
            filtered = {k: v for k, v in data.items() if k in known}
            # Recover monotonic counters if legacy.
            if 'next_candidate_id' not in filtered:
                cands = filtered.get('candidates', [])
                ids = [int(c['id'][4:]) for c in cands
                       if isinstance(c.get('id'), str) and c['id'][4:].isdigit()]
                filtered['next_candidate_id'] = (max(ids) if ids else 0) + 1
            self.state = AbductiveState(**filtered)

    def save(self):
        if self.state is not None:
            save_json(self.filepath, asdict(self.state))

    def start(self, force: bool = False) -> str:
        if self.state is not None and self.state.id and not force:
            raise RuntimeError(
                f"Abductive session '{self.state.id}' already exists. "
                f"Use force=True to overwrite."
            )
        sid = f"AE{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.state = AbductiveState(
            id=sid,
            created=datetime.now().isoformat(),
        )
        self.save()
        return sid

    def require_session(self):
        if self.state is None:
            raise RuntimeError("No active abductive session. Run 'start' first.")

    # --- observation intake -----------------------------------------------

    def add_observation(self, obs_id: str, text: str, category: str = 'generic') -> Dict:
        """Record an observation. Idempotent by obs_id."""
        self.require_session()
        for o in self.state.observations:
            if o['id'] == obs_id:
                return o
        entry = {
            'id': obs_id,
            'text': text,
            'category': category,
            'logged': datetime.now().isoformat(),
            'explained_by': [],  # list of candidate IDs that claim to explain
        }
        self.state.observations.append(entry)
        self.save()
        return entry

    # --- candidate management ---------------------------------------------

    def add_candidate(self, cause: str, mechanism: str, prior: float,
                      source: str, observation_ids: Optional[List[str]] = None,
                      complexity: float = 1.0,
                      provenance_note: str = "") -> str:
        """
        Stage a candidate cause. Returns the candidate ID.

        Provenance discipline: source must be one of VALID_SOURCES.
        LLM-parametric candidates are hard-capped at prior 0.30.
        """
        self.require_session()
        if source not in VALID_SOURCES:
            raise ValueError(
                f"Invalid source '{source}'. Must be one of {VALID_SOURCES}."
            )
        if not 0.0 < prior < 1.0:
            raise ValueError(f"Prior must be in (0, 1), got {prior}")

        # DECISION D-004: prior cap enforced at insertion (see constants).
        if source == 'llm_parametric' and prior > LLM_PARAMETRIC_MAX_PRIOR:
            raise ValueError(
                f"LLM-parametric candidates are capped at prior "
                f"{LLM_PARAMETRIC_MAX_PRIOR}, got {prior}. Upgrade the "
                f"source (library/analyst/chain_derived) before lifting."
            )

        if complexity <= 0:
            raise ValueError(f"Complexity must be > 0, got {complexity}")

        cid = f"CAND{self.state.next_candidate_id}"
        self.state.next_candidate_id += 1
        entry = {
            'id': cid,
            'cause': cause,
            'mechanism': mechanism,
            'prior': float(prior),
            'source': source,
            'complexity': float(complexity),
            'observations_explained': list(observation_ids or []),
            'provenance_note': provenance_note,
            'promoted': False,
            'promoted_to': None,
            'logged': datetime.now().isoformat(),
        }
        self.state.candidates.append(entry)
        # Update observation explained_by index
        for oid in entry['observations_explained']:
            for o in self.state.observations:
                if o['id'] == oid and cid not in o['explained_by']:
                    o['explained_by'].append(cid)
        self.save()
        return cid

    def get_candidate(self, cid: str) -> Dict:
        self.require_session()
        for c in self.state.candidates:
            if c['id'] == cid:
                return c
        raise KeyError(f"Candidate {cid} not found")

    # --- TI Trace Inversion -----------------------------------------------

    def invert(self, obs_id: str, text: str, category: str = 'generic',
               catalog: Optional[Dict] = None,
               llm_parametric_candidates: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Trace Inversion (TI).

        Given an observation, produce candidate causes from (1) the trace
        catalog (source=library) and (2) optional LLM-parametric
        suggestions (source=llm_parametric, capped). Returns the list of
        newly-staged candidate dicts.
        """
        self.require_session()
        self.add_observation(obs_id, text, category)

        if catalog is None:
            catalog = load_trace_catalog()

        produced: List[Dict] = []

        # Catalog lookup
        cat_entry = catalog.get(category) or catalog.get('generic', {})
        for cand in cat_entry.get('candidates', []):
            cid = self.add_candidate(
                cause=cand['cause'],
                mechanism=cand['mechanism'],
                prior=float(cand.get('prior', 0.15)),
                source=cand.get('source', 'library'),
                observation_ids=[obs_id],
                complexity=float(cand.get('complexity', 1.0)),
                provenance_note=f"TI catalog={category}",
            )
            produced.append(self.get_candidate(cid))

        # LLM-parametric candidates (capped)
        if llm_parametric_candidates:
            for cand in llm_parametric_candidates:
                prior = min(LLM_PARAMETRIC_MAX_PRIOR,
                            float(cand.get('prior', 0.15)))
                cid = self.add_candidate(
                    cause=cand['cause'],
                    mechanism=cand['mechanism'],
                    prior=prior,
                    source='llm_parametric',
                    observation_ids=[obs_id],
                    complexity=float(cand.get('complexity', 1.0)),
                    provenance_note="TI llm_parametric (capped)",
                )
                produced.append(self.get_candidate(cid))

        return produced

    # --- AA Absence Audit --------------------------------------------------

    def absence_audit(self, hypothesis_id: str,
                      predictions: List[str]) -> List[Dict]:
        """
        Absence Audit (AA).

        Given a hypothesis and a list of what-should-be-observed-if-true
        predictions, queue them as pending predictions. Closing the audit
        (separately) applies evidence to the hypothesis in the tracker.
        """
        self.require_session()
        if not predictions:
            raise ValueError("absence_audit requires at least one prediction")
        recorded = []
        for pred in predictions:
            if not pred or not pred.strip():
                continue
            pid = f"PP{self.state.next_prediction_id}"
            self.state.next_prediction_id += 1
            entry = {
                'id': pid,
                'hypothesis_id': hypothesis_id,
                'prediction': pred.strip(),
                'status': 'pending',  # pending | observed | absent
                'logged': datetime.now().isoformat(),
                'resolved_at': None,
                'note': '',
            }
            self.state.pending_predictions.append(entry)
            recorded.append(entry)
        self.save()
        return recorded

    def close_prediction(self, prediction_id: str, outcome: str,
                         note: str = "") -> Dict:
        """
        Close a pending prediction. outcome must be 'observed' or 'absent'.
        Returns the updated prediction entry.
        """
        self.require_session()
        if outcome not in ('observed', 'absent'):
            raise ValueError(
                f"outcome must be 'observed' or 'absent', got {outcome}"
            )
        for p in self.state.pending_predictions:
            if p['id'] == prediction_id:
                p['status'] = outcome
                p['resolved_at'] = datetime.now().isoformat()
                p['note'] = note
                self.save()
                return p
        raise KeyError(f"Prediction {prediction_id} not found")

    # --- SA Surplus Audit --------------------------------------------------

    def surplus_audit(self) -> List[Dict]:
        """
        Surplus Audit (SA).

        Diff the observation record against the union of candidate
        explanations. Observations not claimed by any candidate become
        unexplained surplus entries and candidates for new hypotheses.

        Returns the list of unexplained observations.
        """
        self.require_session()
        unexplained = []
        for o in self.state.observations:
            if not o.get('explained_by'):
                entry = {
                    'obs_id': o['id'],
                    'text': o['text'],
                    'category': o.get('category', 'generic'),
                    'flagged': datetime.now().isoformat(),
                }
                unexplained.append(entry)
        # Merge into persistent surplus_log (dedupe by obs_id)
        existing = {s['obs_id'] for s in self.state.surplus_log}
        for u in unexplained:
            if u['obs_id'] not in existing:
                self.state.surplus_log.append(u)
        self.save()
        return unexplained

    # --- AR Analogical Retrieval ------------------------------------------

    def analogize(self, signature: str,
                  archetype_library: Optional[Dict] = None) -> List[Dict]:
        """
        Analogical Retrieval (AR).

        Match a case signature against the archetype library. Uses
        simple token overlap as the similarity measure (stdlib only).
        Returns the list of matching archetype summaries, each with a
        similarity score.
        """
        self.require_session()
        if archetype_library is None:
            archetype_library = load_archetype_library_for_analogy()

        sig_tokens = _tokenize(signature)
        matches = []
        for aid, entry in archetype_library.items():
            best_score = 0.0
            best_sig = ""
            for trace_sig in entry.get('trace_signatures', []) or []:
                score = _token_overlap(sig_tokens, _tokenize(trace_sig))
                if score > best_score:
                    best_score = score
                    best_sig = trace_sig
            if best_score > 0:
                matches.append({
                    'archetype_id': aid,
                    'name': entry.get('name', aid),
                    'similarity': best_score,
                    'matched_signature': best_sig,
                })
        matches.sort(key=lambda m: m['similarity'], reverse=True)
        # Persist
        rec = {
            'query': signature,
            'matches': matches,
            'logged': datetime.now().isoformat(),
        }
        self.state.analogies.append(rec)
        self.save()
        return matches

    # --- IC Inference Chains ----------------------------------------------

    def chain_start(self, target: str, premise: str) -> str:
        """
        Start a new inference chain.

        target: what the chain is inferring toward (typically a candidate
                or hypothesis ID)
        premise: initial premise statement

        Returns the chain ID.
        """
        self.require_session()
        cid = f"IC{self.state.next_chain_id}"
        self.state.next_chain_id += 1
        chain = {
            'id': cid,
            'target': target,
            'premise': premise,
            'steps': [],
            'status': 'open',
            'started': datetime.now().isoformat(),
            'closed_at': None,
            'final_posterior': None,
        }
        self.state.inference_chains.append(chain)
        self.save()
        return cid

    def chain_step(self, chain_id: str, claim: str, lr: float,
                   source: str = 'analyst',
                   references: Optional[List[str]] = None) -> Dict:
        """
        Append a micro-inference step to a chain.

        Each step is a structured JSON record with claim, small LR, and
        provenance tag. LLM-parametric steps are hard-capped at LR 2.0.
        """
        self.require_session()
        if source not in VALID_SOURCES:
            raise ValueError(
                f"Invalid source '{source}'. Must be one of {VALID_SOURCES}."
            )
        if lr < 0:
            raise ValueError(f"LR must be >= 0, got {lr}")
        # DECISION D-004: LR cap enforced per chain step (see constants).
        if source == 'llm_parametric' and lr > LLM_PARAMETRIC_MAX_LR:
            raise ValueError(
                f"LLM-parametric steps are capped at LR "
                f"{LLM_PARAMETRIC_MAX_LR}, got {lr}. Upgrade the source "
                f"(library/analyst/chain_derived) before lifting."
            )
        for chain in self.state.inference_chains:
            if chain['id'] == chain_id:
                if chain['status'] != 'open':
                    raise RuntimeError(
                        f"Chain {chain_id} is {chain['status']}, cannot append."
                    )
                step = {
                    'idx': len(chain['steps']) + 1,
                    'claim': claim,
                    'lr': float(lr),
                    'source': source,
                    'references': list(references or []),
                    'logged': datetime.now().isoformat(),
                }
                chain['steps'].append(step)
                self.save()
                return step
        raise KeyError(f"Chain {chain_id} not found")

    def chain_close(self, chain_id: str,
                    seed_prior: float = 0.5) -> Dict:
        """
        Close a chain. Composes its steps into a final posterior starting
        from seed_prior, applying each step LR multiplicatively via
        Bayesian odds update.

        Returns the closed chain dict.
        """
        self.require_session()
        if not 0.0 < seed_prior < 1.0:
            raise ValueError(f"seed_prior must be in (0, 1), got {seed_prior}")
        for chain in self.state.inference_chains:
            if chain['id'] == chain_id:
                if chain['status'] != 'open':
                    raise RuntimeError(
                        f"Chain {chain_id} is already {chain['status']}."
                    )
                if len(chain['steps']) < 1:
                    raise RuntimeError(
                        f"Chain {chain_id} has no steps — cannot close."
                    )
                posterior = seed_prior
                for step in chain['steps']:
                    posterior = _odds_update(posterior, step['lr'])
                chain['status'] = 'closed'
                chain['closed_at'] = datetime.now().isoformat()
                chain['final_posterior'] = posterior
                self.save()
                return chain
        raise KeyError(f"Chain {chain_id} not found")

    def chain_audit(self, chain_id: str) -> Dict:
        """
        Audit a chain for gaps. Currently checks:
          - step index monotonicity
          - no LR of exactly 0 mid-chain without immediate close
          - every step has a references field (may be empty list)
          - at least 2 steps for a closed chain

        Returns a dict {chain_id, gaps: [list of gap descriptions],
                        ok: bool}.
        """
        self.require_session()
        for chain in self.state.inference_chains:
            if chain['id'] == chain_id:
                gaps: List[str] = []
                if chain['status'] == 'closed' and len(chain['steps']) < 2:
                    gaps.append(
                        f"closed chain has {len(chain['steps'])} steps; "
                        f"minimum 2 required"
                    )
                for i, step in enumerate(chain['steps']):
                    if step.get('idx') != i + 1:
                        gaps.append(
                            f"step {i} has idx={step.get('idx')} "
                            f"(expected {i + 1}) — possible ordering gap"
                        )
                    if 'references' not in step:
                        gaps.append(f"step {i+1} missing references field")
                    if step.get('lr', 1.0) == 0.0 and i < len(chain['steps']) - 1:
                        gaps.append(
                            f"step {i+1} has LR=0 (falsifying) but more "
                            f"steps follow — chain should have been closed"
                        )
                return {
                    'chain_id': chain_id,
                    'gaps': gaps,
                    'ok': len(gaps) == 0,
                }
        raise KeyError(f"Chain {chain_id} not found")

    # --- Coverage-weighted selection + promotion --------------------------

    def coverage_score(self, candidate_id: str) -> float:
        """
        Parsimony-weighted coverage for a candidate.

        coverage = observations_explained / max(1, total_observations)
        score    = coverage / complexity

        Returns a float in [0, 1/complexity]. Higher is better.
        """
        c = self.get_candidate(candidate_id)
        n_total = max(1, len(self.state.observations))
        n_explained = len(c.get('observations_explained', []))
        coverage = n_explained / n_total
        complexity = max(0.001, float(c.get('complexity', 1.0)))
        return coverage / complexity

    def list_candidates(self, include_promoted: bool = False) -> List[Dict]:
        """List staged candidates with coverage scores attached."""
        self.require_session()
        out = []
        for c in self.state.candidates:
            if not include_promoted and c.get('promoted'):
                continue
            entry = dict(c)
            entry['coverage_score'] = self.coverage_score(c['id'])
            out.append(entry)
        out.sort(key=lambda e: e['coverage_score'], reverse=True)
        return out

    def promote(self, candidate_id: str,
                threshold: float = DEFAULT_COVERAGE_THRESHOLD,
                tracker_path: Optional[str] = None,
                phase: str = 'P1_5') -> Dict:
        """
        Promote a candidate from staging to the tracked hypothesis set.

        Enforces the coverage gate: candidate must have
        coverage_score >= threshold. Enforces provenance caps:
        llm_parametric candidates cannot be promoted with prior > 0.30
        (this is also checked at add-time but re-checked here in case of
        post-hoc source modification).

        If tracker_path is supplied, attempts to write the hypothesis to
        the bayesian_tracker.py state file. If not, returns the hypothesis
        statement for the caller to handle.

        Returns a dict {candidate_id, hypothesis_id, hypothesis_statement,
                        prior, coverage_score, promoted}.
        """
        c = self.get_candidate(candidate_id)
        if c.get('promoted'):
            raise RuntimeError(
                f"Candidate {candidate_id} already promoted "
                f"(to {c.get('promoted_to')})."
            )

        # DECISION D-005: coverage gate is the promotion barrier, not a
        # warning. Candidates below threshold CANNOT reach the tracked
        # hypothesis set. This is the structural defense against
        # hypothesis explosion; bypassing it defeats Phase 1.5's purpose.
        score = self.coverage_score(candidate_id)
        if score < threshold:
            raise RuntimeError(
                f"Candidate {candidate_id} coverage score {score:.3f} "
                f"below promotion threshold {threshold:.3f}. "
                f"Hypothesis explosion mitigation: only high-coverage "
                f"candidates may be promoted. Increase observation coverage "
                f"or lower complexity before retrying."
            )

        # DECISION D-004: re-check prior cap at promotion to catch
        # post-hoc source modification (e.g. analyst edited the staging
        # file after add_candidate succeeded).
        if c['source'] == 'llm_parametric' and c['prior'] > LLM_PARAMETRIC_MAX_PRIOR:
            raise RuntimeError(
                f"Candidate {candidate_id} has source=llm_parametric and "
                f"prior={c['prior']:.2f} > cap {LLM_PARAMETRIC_MAX_PRIOR}. "
                f"Upgrade source before promotion."
            )

        statement = f"[H_ABDUCT_{candidate_id}] {c['cause']} — {c['mechanism']}"
        hypothesis_id: Optional[str] = None

        if tracker_path:
            import subprocess
            script_dir = os.path.dirname(os.path.abspath(__file__))
            tracker_script = os.path.join(script_dir, 'bayesian_tracker.py')
            cmd = [
                sys.executable, tracker_script,
                '--file', tracker_path,
                'add', statement,
                '--phase', phase,
                '--prior', str(c['prior']),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Tracker add failed: {result.stderr or result.stdout}"
                )
            # Output format: "Added: H<N> (prior=...)"
            out = result.stdout.strip()
            for token in out.split():
                if token.startswith('H') and token[1:].isdigit():
                    hypothesis_id = token
                    break

        c['promoted'] = True
        c['promoted_to'] = hypothesis_id or 'pending'
        self.save()

        return {
            'candidate_id': candidate_id,
            'hypothesis_id': hypothesis_id,
            'hypothesis_statement': statement,
            'prior': c['prior'],
            'coverage_score': score,
            'promoted': True,
        }

    # --- Catalog bootstrap/review -----------------------------------------

    @staticmethod
    def bootstrap_prompt(category: str) -> str:
        """
        Emit an LLM prompt template + JSON schema for bootstrapping a
        trace catalog entry for a given observation category. The
        resulting file is loaded via `catalog review` and marked
        pending-review before use.
        """
        schema = {
            'category': category,
            'description': '<one-line description>',
            'candidates': [
                {
                    'cause': '<candidate cause label>',
                    'mechanism': '<one-line mechanism>',
                    'prior': 0.15,
                    'source': 'llm_parametric',
                    'complexity': 1.0,
                    'pending_review': True,
                }
            ],
        }
        prompt = (
            f"Produce 5-10 candidate causes for observations in the "
            f"category '{category}'. For each candidate, write:\n"
            f"  - cause: a short label\n"
            f"  - mechanism: one-line mechanism\n"
            f"  - prior: a number in (0, 0.30]\n"
            f"  - complexity: a positive float (1.0 = one mechanism unit)\n"
            f"Return a JSON object matching this schema:\n\n"
            f"{json.dumps(schema, indent=2)}\n\n"
            f"CRITICAL: all entries must have source='llm_parametric' "
            f"and pending_review=true. The analyst will review before any "
            f"candidate enters the live catalog."
        )
        return prompt

    def catalog_review(self, path: str) -> Dict:
        """
        Load a JSON file produced by an offline LLM catalog bootstrap and
        stage the candidates it contains into the current session as
        llm_parametric candidates. Each candidate receives a synthetic
        observation id 'BOOT' so it is visible in surplus_audit until an
        analyst links it to a real observation.

        Returns a summary dict.
        """
        self.require_session()
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path) as f:
            data = json.load(f)
        category = data.get('category', 'generic')
        candidates = data.get('candidates', [])
        staged = 0
        for cand in candidates:
            prior = min(LLM_PARAMETRIC_MAX_PRIOR, float(cand.get('prior', 0.15)))
            self.add_candidate(
                cause=cand['cause'],
                mechanism=cand['mechanism'],
                prior=prior,
                source='llm_parametric',
                complexity=float(cand.get('complexity', 1.0)),
                provenance_note=f"catalog bootstrap category={category} "
                                f"pending_review={cand.get('pending_review', True)}",
            )
            staged += 1
        return {
            'category': category,
            'staged': staged,
        }

    # --- Reporting ---------------------------------------------------------

    def report(self, verbose: bool = False) -> str:
        self.require_session()
        s = self.state
        lines: List[str] = []
        lines.append(f"# Abductive Expansion Report — {s.id}")
        lines.append(f"")
        lines.append(f"- Created: {s.created}")
        lines.append(f"- Observations: {len(s.observations)}")
        lines.append(f"- Candidates staged: {len(s.candidates)}")
        lines.append(f"- Candidates promoted: "
                     f"{sum(1 for c in s.candidates if c.get('promoted'))}")
        lines.append(f"- Pending predictions: {len(s.pending_predictions)}")
        lines.append(f"- Inference chains: {len(s.inference_chains)}")
        lines.append(f"- Surplus entries: {len(s.surplus_log)}")
        lines.append(f"- Analogies recorded: {len(s.analogies)}")
        lines.append("")
        lines.append("## Candidates")
        lines.append("")
        lines.append("| ID | Cause | Source | Prior | Coverage | Promoted |")
        lines.append("|---|---|---|---:|---:|:---:|")
        for c in s.candidates:
            score = self.coverage_score(c['id'])
            promoted = 'YES' if c.get('promoted') else ''
            cause = c['cause'][:50]
            lines.append(
                f"| {c['id']} | {cause} | {c['source']} | "
                f"{c['prior']:.2f} | {score:.2f} | {promoted} |"
            )
        if verbose:
            if s.inference_chains:
                lines.append("")
                lines.append("## Inference Chains")
                for chain in s.inference_chains:
                    lines.append(
                        f"- **{chain['id']}** ({chain['status']}) → "
                        f"target: {chain['target']}, "
                        f"steps: {len(chain['steps'])}, "
                        f"posterior: {chain.get('final_posterior')}"
                    )
            if s.surplus_log:
                lines.append("")
                lines.append("## Surplus (unexplained observations)")
                for u in s.surplus_log:
                    lines.append(f"- {u['obs_id']}: {u['text'][:80]}")
            if s.pending_predictions:
                lines.append("")
                lines.append("## Pending Predictions")
                for p in s.pending_predictions:
                    lines.append(
                        f"- {p['id']} [{p['status']}] → {p['hypothesis_id']}: "
                        f"{p['prediction'][:80]}"
                    )
        return "\n".join(lines)

    def gate_status(self) -> Dict:
        """Phase 1.5 exit gate status."""
        self.require_session()
        s = self.state
        n_inverted = sum(1 for o in s.observations
                         if any(c for c in s.candidates
                                if o['id'] in c.get('observations_explained', [])))
        n_promoted = sum(1 for c in s.candidates if c.get('promoted'))
        n_chains = sum(1 for ch in s.inference_chains if ch['status'] == 'closed')
        gate = {
            'observations_inverted': n_inverted,
            'min_observations_inverted': 3,
            'surplus_audit_run': len(s.surplus_log) > 0 or
                                 any(o.get('explained_by') for o in s.observations),
            'promoted_or_attested': n_promoted > 0,
            'closed_chains': n_chains,
        }
        gate['pass'] = (
            gate['observations_inverted'] >= gate['min_observations_inverted']
            and gate['surplus_audit_run']
        )
        return gate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _odds_update(prior: float, lr: float) -> float:
    """Bayesian update via odds (local copy — avoid common import cycle)."""
    if lr <= 0:
        return 0.001
    prior = max(0.001, min(0.999, prior))
    prior_odds = prior / (1 - prior)
    posterior_odds = prior_odds * lr
    return max(0.001, min(0.999, posterior_odds / (1 + posterior_odds)))


def _tokenize(text: str) -> set:
    if not text:
        return set()
    return {t.strip().lower() for t in text.replace(',', ' ').split() if t.strip()}


def _token_overlap(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='abductive_engine.py',
        description='Abductive Expansion engine (Phase 1.5).',
    )
    parser.add_argument(
        '--file', default='abductive_state.json',
        help='Path to abductive_state.json (default: ./abductive_state.json)',
    )
    parser.add_argument(
        '--config', default=None,
        help='Path to trace_catalog.json (default: ../config/trace_catalog.json)',
    )
    sub = parser.add_subparsers(dest='cmd', required=False)

    sub.add_parser('start', help='Start a new abductive session')

    p_inv = sub.add_parser('invert', help='TI: invert observation into candidate causes')
    p_inv.add_argument('--obs-id', required=True)
    p_inv.add_argument('--text', required=True)
    p_inv.add_argument('--category', default='generic')

    p_aa = sub.add_parser('absence-audit',
                          help='AA: enumerate what-should-be-observed predictions')
    p_aa.add_argument('--hypothesis', required=True, help='Hypothesis ID')
    p_aa.add_argument('--predictions', required=True,
                      help='Semicolon-separated prediction list')

    p_closep = sub.add_parser('close-prediction',
                              help='Close a pending prediction (observed|absent)')
    p_closep.add_argument('--id', required=True)
    p_closep.add_argument('--outcome', required=True, choices=['observed', 'absent'])
    p_closep.add_argument('--note', default='')

    sub.add_parser('surplus-audit', help='SA: diff observations against candidate coverage')

    p_an = sub.add_parser('analogize', help='AR: match signature against archetype library')
    p_an.add_argument('--signature', required=True)

    p_chain = sub.add_parser('chain', help='IC: inference chain commands')
    chain_sub = p_chain.add_subparsers(dest='chain_cmd', required=True)

    p_cs = chain_sub.add_parser('start', help='Start a new chain')
    p_cs.add_argument('--target', required=True)
    p_cs.add_argument('--premise', required=True)

    p_cstep = chain_sub.add_parser('step', help='Append a micro-inference step')
    p_cstep.add_argument('--id', required=True)
    p_cstep.add_argument('--claim', required=True)
    p_cstep.add_argument('--lr', type=float, required=True)
    p_cstep.add_argument('--source', default='analyst', choices=list(VALID_SOURCES))
    p_cstep.add_argument('--refs', default='', help='Comma-separated references')

    p_cclose = chain_sub.add_parser('close', help='Close a chain')
    p_cclose.add_argument('--id', required=True)
    p_cclose.add_argument('--seed-prior', type=float, default=0.5)

    p_caud = chain_sub.add_parser('audit', help='Audit a chain for gaps')
    p_caud.add_argument('--id', required=True)

    p_cat = sub.add_parser('catalog', help='Trace catalog management')
    cat_sub = p_cat.add_subparsers(dest='catalog_cmd', required=True)

    p_boot = cat_sub.add_parser('bootstrap',
                                help='Emit LLM prompt template for a category')
    p_boot.add_argument('--category', required=True)
    p_boot.add_argument('--output', default=None, help='Write prompt to file')

    p_rev = cat_sub.add_parser('review',
                               help='Load reviewed LLM-bootstrapped catalog JSON')
    p_rev.add_argument('--path', required=True)

    p_cand = sub.add_parser('candidates', help='Candidate management')
    cand_sub = p_cand.add_subparsers(dest='cand_cmd', required=True)

    p_cl = cand_sub.add_parser('list', help='List staged candidates')
    p_cl.add_argument('--all', action='store_true',
                      help='Include promoted candidates')

    p_pr = cand_sub.add_parser('promote', help='Promote a candidate to hypotheses.json')
    p_pr.add_argument('--id', required=True)
    p_pr.add_argument('--threshold', type=float, default=DEFAULT_COVERAGE_THRESHOLD)
    p_pr.add_argument('--tracker-path', default=None,
                      help='Path to hypotheses.json (enables tracker write)')
    p_pr.add_argument('--phase', default='P1_5')

    p_rep = sub.add_parser('report', help='Print abductive report')
    p_rep.add_argument('--verbose', action='store_true')

    sub.add_parser('gate', help='Print Phase 1.5 exit gate status')

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd is None:
        parser.print_help()
        return 0

    # `catalog bootstrap` does not need a session
    if args.cmd == 'catalog' and args.catalog_cmd == 'bootstrap':
        prompt = AbductiveEngine.bootstrap_prompt(args.category)
        if args.output:
            with open(args.output, 'w') as f:
                f.write(prompt)
            print(f"Wrote bootstrap prompt to {args.output}")
        else:
            print(prompt)
        return 0

    engine = AbductiveEngine(args.file)

    if args.cmd == 'start':
        sid = engine.start()
        print(f"Started abductive session {sid}")
        return 0

    # Auto-start for convenience so operators work out of the box
    if engine.state is None:
        engine.start()

    try:
        if args.cmd == 'invert':
            catalog = load_trace_catalog(args.config)
            produced = engine.invert(args.obs_id, args.text, args.category,
                                     catalog=catalog)
            print(f"TI produced {len(produced)} candidates for {args.obs_id}:")
            for c in produced:
                print(f"  - {c['id']} [{c['source']}] prior={c['prior']:.2f}: "
                      f"{c['cause']}")
            return 0

        if args.cmd == 'absence-audit':
            preds = [p for p in args.predictions.split(';') if p.strip()]
            entries = engine.absence_audit(args.hypothesis, preds)
            print(f"AA queued {len(entries)} predictions for {args.hypothesis}:")
            for e in entries:
                print(f"  - {e['id']}: {e['prediction']}")
            return 0

        if args.cmd == 'close-prediction':
            entry = engine.close_prediction(args.id, args.outcome, args.note)
            print(f"Closed {entry['id']} as {entry['status']}")
            return 0

        if args.cmd == 'surplus-audit':
            unexplained = engine.surplus_audit()
            print(f"SA found {len(unexplained)} unexplained observations:")
            for u in unexplained:
                print(f"  - {u['obs_id']} [{u['category']}]: {u['text'][:80]}")
            return 0

        if args.cmd == 'analogize':
            matches = engine.analogize(args.signature)
            print(f"AR found {len(matches)} archetype matches:")
            for m in matches:
                print(f"  - {m['archetype_id']} ({m['similarity']:.2f}): "
                      f"{m['name']}")
            return 0

        if args.cmd == 'chain':
            if args.chain_cmd == 'start':
                cid = engine.chain_start(args.target, args.premise)
                print(f"Started chain {cid}")
                return 0
            if args.chain_cmd == 'step':
                refs = [r.strip() for r in args.refs.split(',') if r.strip()]
                step = engine.chain_step(args.id, args.claim, args.lr,
                                         args.source, references=refs)
                print(f"Appended step {step['idx']} to {args.id} (LR={step['lr']})")
                return 0
            if args.chain_cmd == 'close':
                chain = engine.chain_close(args.id, args.seed_prior)
                print(f"Closed {chain['id']}: final_posterior="
                      f"{chain['final_posterior']:.3f}")
                return 0
            if args.chain_cmd == 'audit':
                audit = engine.chain_audit(args.id)
                print(f"Audit {audit['chain_id']}: "
                      f"{'OK' if audit['ok'] else 'GAPS'}")
                for gap in audit['gaps']:
                    print(f"  - {gap}")
                return 0 if audit['ok'] else 1

        if args.cmd == 'catalog' and args.catalog_cmd == 'review':
            summary = engine.catalog_review(args.path)
            print(f"Staged {summary['staged']} candidates from "
                  f"category={summary['category']}")
            return 0

        if args.cmd == 'candidates':
            if args.cand_cmd == 'list':
                items = engine.list_candidates(include_promoted=args.all)
                print(f"{len(items)} candidates:")
                for it in items:
                    print(f"  - {it['id']} [{it['source']}] "
                          f"prior={it['prior']:.2f} "
                          f"score={it['coverage_score']:.2f}: {it['cause']}")
                return 0
            if args.cand_cmd == 'promote':
                result = engine.promote(
                    args.id, threshold=args.threshold,
                    tracker_path=args.tracker_path, phase=args.phase,
                )
                print(f"Promoted {result['candidate_id']} → "
                      f"{result['hypothesis_id'] or 'pending'} "
                      f"(score={result['coverage_score']:.2f})")
                return 0

        if args.cmd == 'report':
            print(engine.report(verbose=args.verbose))
            return 0

        if args.cmd == 'gate':
            gate = engine.gate_status()
            print("Phase 1.5 Exit Gate:")
            for k, v in gate.items():
                print(f"  {k}: {v}")
            return 0 if gate.get('pass') else 1

    except (KeyError, ValueError, RuntimeError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    parser.print_help()
    return 0


if __name__ == '__main__':
    sys.exit(main())
