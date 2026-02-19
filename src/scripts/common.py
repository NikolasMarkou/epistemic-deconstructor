#!/usr/bin/env python3
"""
Shared utilities for Epistemic Deconstructor tracker scripts.

Provides:
- Bayesian update math with division-by-zero protection
- JSON load/save with file locking for concurrency safety
"""

import json
import os
import sys

# Epsilon to prevent posterior from reaching exactly 0.0 or 1.0,
# which would make further Bayesian updates degenerate (division by zero).
POSTERIOR_EPSILON = 1e-9


def clamp_probability(p, eps=POSTERIOR_EPSILON):
    """Clamp a probability to (eps, 1 - eps)."""
    return max(eps, min(1 - eps, p))


def bayesian_update(prior, likelihood_ratio, eps=POSTERIOR_EPSILON):
    """
    Compute Bayesian posterior from prior and likelihood ratio.

    Args:
        prior: Current probability P(H).
        likelihood_ratio: P(E|H) / P(E|~H).  LR > 1 confirms, LR < 1 disconfirms, LR = 0 falsifies.
        eps: Epsilon for clamping to avoid degenerate probabilities.

    Returns:
        New posterior probability.
    """
    if likelihood_ratio == 0:
        return 0.0

    p = clamp_probability(prior, eps)
    prior_odds = p / (1 - p)
    posterior_odds = prior_odds * likelihood_ratio
    return posterior_odds / (1 + posterior_odds)


# ---------------------------------------------------------------------------
# Platform-aware file locking (stdlib only)
# ---------------------------------------------------------------------------

def _lock_file(f, exclusive=True):
    """Acquire a file lock (shared or exclusive)."""
    if sys.platform == 'win32':
        import msvcrt
        # Note: On Windows, LK_RLCK is identical to LK_LOCK (both exclusive).
        # msvcrt does not support true shared/reader locks.
        mode = msvcrt.LK_LOCK if exclusive else msvcrt.LK_RLCK
        # Lock from current position to end; seek to start first
        f.seek(0)
        msvcrt.locking(f.fileno(), mode, max(os.fstat(f.fileno()).st_size, 1))
    else:
        import fcntl
        flag = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fcntl.flock(f.fileno(), flag)


def _unlock_file(f):
    """Release a file lock."""
    if sys.platform == 'win32':
        import msvcrt
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, max(os.fstat(f.fileno()).st_size, 1))
    else:
        import fcntl
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def load_json(filepath):
    """
    Load JSON from *filepath* with a shared file lock.

    Returns:
        Parsed data, or None if the file does not exist.
    """
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r') as f:
        _lock_file(f, exclusive=False)
        try:
            return json.load(f)
        finally:
            _unlock_file(f)


def save_json(filepath, data):
    """
    Save *data* as JSON to *filepath* with an exclusive file lock.
    """
    with open(filepath, 'w') as f:
        _lock_file(f, exclusive=True)
        try:
            json.dump(data, f, indent=2)
        finally:
            _unlock_file(f)
