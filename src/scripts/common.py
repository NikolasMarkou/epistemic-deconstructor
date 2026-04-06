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
import tempfile

# Epsilon to prevent posterior from reaching degenerate values.
# 1e-3 caps posteriors at (0.001, 0.999) — high enough to prevent
# saturation where further evidence updates have no practical effect,
# while still allowing strong confidence levels.
POSTERIOR_EPSILON = 1e-3


def clamp_probability(p, eps=POSTERIOR_EPSILON):
    """Clamp a probability to (eps, 1 - eps).

    Raises ValueError for NaN or infinite inputs rather than silently
    clamping them to a boundary value.
    """
    import math
    if math.isnan(p) or math.isinf(p):
        raise ValueError(f"Probability must be finite, got {p}")
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
    if likelihood_ratio < 0:
        raise ValueError(f"Likelihood ratio must be >= 0, got {likelihood_ratio}")
    if likelihood_ratio == 0:
        return clamp_probability(0.0, eps)

    if not 0 < prior < 1:
        raise ValueError(f"Prior must be in open interval (0, 1), got {prior}")
    p = clamp_probability(prior, eps)
    prior_odds = p / (1 - p)
    posterior_odds = prior_odds * likelihood_ratio
    return clamp_probability(posterior_odds / (1 + posterior_odds), eps)


# ---------------------------------------------------------------------------
# Platform-aware file locking (stdlib only)
# ---------------------------------------------------------------------------

# Fixed byte range for Windows msvcrt locking.  msvcrt.locking operates on a
# byte range, so we must lock/unlock the *same* number of bytes.  Using a
# constant avoids the bug where the file size changes between lock and unlock.
_WIN_LOCK_LEN = 1 << 30  # 1 GiB — larger than any realistic JSON file


def _lock_file(f, exclusive=True):
    """Acquire a file lock (shared or exclusive)."""
    if sys.platform == 'win32':
        import msvcrt
        # Note: On Windows, LK_RLCK is identical to LK_LOCK (both exclusive).
        # msvcrt does not support true shared/reader locks.
        mode = msvcrt.LK_LOCK if exclusive else msvcrt.LK_RLCK
        f.seek(0)
        msvcrt.locking(f.fileno(), mode, _WIN_LOCK_LEN)
    else:
        import fcntl
        flag = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fcntl.flock(f.fileno(), flag)


def _unlock_file(f):
    """Release a file lock."""
    if sys.platform == 'win32':
        import msvcrt
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, _WIN_LOCK_LEN)
    else:
        import fcntl
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def load_json(filepath):
    """
    Load JSON from *filepath* with a shared file lock.

    Returns:
        Parsed data, or None if the file does not exist.
    """
    try:
        with open(filepath, 'r') as f:
            locked = False
            try:
                _lock_file(f, exclusive=False)
                locked = True
                content = f.read()
                if not content.strip():
                    return None
                return json.loads(content)
            except json.JSONDecodeError:
                print(f"Warning: Corrupt JSON in {filepath}, treating as empty",
                      file=sys.stderr)
                return None
            finally:
                if locked:
                    _unlock_file(f)
    except FileNotFoundError:
        return None


def save_json(filepath, data):
    """
    Save *data* as JSON to *filepath* atomically.

    Writes to a temporary file first, then does an atomic rename.
    This prevents data loss if serialization fails or the process crashes,
    and eliminates the race window where concurrent readers see empty data.
    """
    abs_path = os.path.abspath(filepath)
    dir_path = os.path.dirname(abs_path)
    fd = None
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix='.tmp')
        with os.fdopen(fd, 'w') as f:
            fd = None  # os.fdopen takes ownership
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, abs_path)  # atomic on POSIX
        tmp_path = None  # successfully replaced
    except Exception:
        if fd is not None:
            os.close(fd)
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise
