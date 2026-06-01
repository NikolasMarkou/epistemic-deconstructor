#!/usr/bin/env python3
"""
Shared utilities for Epistemic Deconstructor tracker scripts.

Provides:
- Bayesian update math with division-by-zero protection
- JSON load/save with file locking for concurrency safety
- Runtime Python version guard (D-002)
"""

import sys

# DECISION plan_2026-05-25_c0b0049a/D-002:
# Declared Python floor is 3.8 (README + CLAUDE.md). Effective hard floor
# is 3.7 — older interpreters fail to parse `from __future__ import
# annotations` + `@dataclass` usage in several scripts. We hard-error on
# < 3.7 so the user gets a clear message instead of an opaque SyntaxError
# from a script that imports us. We warn once on < 3.8 to nudge upgrades
# without breaking sessions on 3.7 (still in the wild on long-lived OS).
# Guard runs at import time, before any non-stdlib import below, so every
# script that touches common.py inherits it.
if sys.version_info < (3, 7):
    sys.stderr.write(
        "ERROR: Epistemic Deconstructor requires Python >= 3.7 (declared "
        ">= 3.8). Detected: {0}.{1}.{2}. Upgrade Python and retry.\n".format(
            *sys.version_info[:3]))
    sys.exit(1)
if sys.version_info < (3, 8):
    sys.stderr.write(
        "WARNING: Epistemic Deconstructor declared Python >= 3.8 (detected "
        "{0}.{1}.{2}). Scripts may work on 3.7 but it is not part of the "
        "supported matrix.\n".format(*sys.version_info[:3]))

import contextlib
import json
import os
import tempfile
import threading


# DECISION plan_2026-05-28_9d761933/D-002:
# Cross-process load→modify→save transactional locking. fcntl.flock treats
# multiple file descriptors on the same file as independent — same-process
# re-acquisition deadlocks (see Linux flock(2) man page). When a tracker
# mutator wraps its load+modify+save sequence in `transactional_json` and
# then calls `save_json` inside, the inner save_json would open the lockfile
# a second time and block forever. To avoid this, transactional_json
# registers the abs-path in a thread-local set; save_json consults the
# registry and skips re-acquisition when the path is already held. Across
# different processes the registry is independent, so cross-process flock
# serialization remains intact. Across threads the registry is per-thread,
# so transactional_json on the same path from two threads in one process
# would deadlock — this is acceptable because all tracker mutations are
# called from the CLI mainline of a single-threaded subprocess.
_lock_registry = threading.local()


def _registry_paths():
    """Return the thread-local set of abs-paths currently inside a transaction."""
    if not hasattr(_lock_registry, 'paths'):
        _lock_registry.paths = set()
    return _lock_registry.paths

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


class JSONCorruptError(Exception):
    """Raised by load_json when the target file contains malformed JSON.

    Previously load_json printed a warning and returned None on decode
    failure, making corrupt session state indistinguishable from a
    missing file. The next save would then silently overwrite the
    corrupt state with a freshly-initialized structure, losing the
    analyst's prior work. Raising instead keeps corruption visible at
    the CLI boundary so the analyst can recover rather than lose data.
    """


def load_json(filepath):
    """
    Load JSON from *filepath* with a shared file lock.

    Returns:
        Parsed data, or None if the file does not exist or is empty.

    Raises:
        JSONCorruptError: if the file exists and contains non-empty
            content that fails JSON decoding. Missing and empty files
            still return None — only malformed content raises.
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
                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    raise JSONCorruptError(
                        f"Corrupt JSON in {filepath}: {e}"
                    ) from e
            finally:
                if locked:
                    _unlock_file(f)
    except FileNotFoundError:
        return None


# DECISION plan_2026-06-01_cf95b3e5/D-001:
# Construct a state-container dataclass ONLY after validating that every
# required (no-default) field is present in *filtered*. A raw {} / corrupt /
# truncated state file must NOT reach `Dataclass(**filtered)` — that raises a
# bare TypeError and dumps a Python traceback at the CLI boundary. Instead we
# print one human-readable ERROR line and sys.exit(1) (clean exit-1).
# Earned abstraction: this guard is identical at 4 call sites
# (scope_auditor / abductive_engine / domain_orienter / rapid_checker), each
# owning a container dataclass with required positional fields; 4 verbatim
# inline copies would be the duplication smell. Do NOT inline a try/except
# TypeError instead — a TypeError can also come from a genuinely buggy field
# value, which we do NOT want to swallow as "missing fields".
def build_dataclass_or_exit(cls, filtered, path):
    """Build *cls* from *filtered* after asserting required fields are present.

    Required fields are the dataclass fields with no default and no
    default_factory. If any are missing from *filtered* (e.g. an empty {} or
    truncated state file), print a clean ERROR to stderr and ``sys.exit(1)``
    rather than letting ``cls(**filtered)`` raise a TypeError traceback.

    Returns the constructed instance on success.
    """
    from dataclasses import MISSING, fields as _dc_fields
    required = [
        f.name for f in _dc_fields(cls)
        if f.default is MISSING and f.default_factory is MISSING  # type: ignore[misc]
    ]
    missing = [name for name in required if name not in filtered]
    if missing:
        sys.stderr.write(
            "ERROR: {0} is missing required fields (corrupt or empty state "
            "file): {1}\n".format(path, ", ".join(missing)))
        sys.exit(1)
    return cls(**filtered)


def _write_atomic(abs_path, data, default=None):
    """Internal: write *data* as JSON to *abs_path* via tempfile + atomic rename.

    Caller is responsible for holding the appropriate lock. Used by both
    ``save_json`` (which acquires the lock itself) and by code already inside
    ``transactional_json`` (which has already acquired it).

    *default*, if given, is passed through to ``json.dump`` for non-serialisable
    objects (datetimes, numpy scalars, etc.).
    """
    dir_path = os.path.dirname(abs_path) or '.'
    os.makedirs(dir_path, exist_ok=True)
    fd = None
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix='.tmp')
        with os.fdopen(fd, 'w') as f:
            fd = None  # os.fdopen takes ownership
            json.dump(data, f, indent=2, default=default)
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


@contextlib.contextmanager
def transactional_json(filepath):
    """Hold an exclusive cross-process lock for a load→modify→save sequence.

    Use this context manager around any code that reads a JSON state file,
    modifies it in memory, and writes it back. Without serialization at this
    granularity, two concurrent processes can both load the same state, each
    apply distinct mutations, each call ``save_json``, and the later writer
    silently overwrites the earlier writer's changes (audit OOS-2).

    Usage::

        with transactional_json(path):
            data = load_json(path) or {}
            data['key'] = 'value'
            save_json(path, data)

    Inside the ``with`` block, ``save_json`` recognises the in-transaction
    state via the thread-local registry and skips re-acquiring the lockfile
    (avoiding fcntl re-entry deadlock per D-002 above).

    Re-entry semantics: nesting ``transactional_json(path)`` on the same path
    is idempotent — the inner call is a no-op. Different paths nest freely.

    Scope: fcntl is process-level on POSIX (msvcrt on Windows). Across
    processes, the lockfile coordinates correctly. Across threads in the same
    process, the registry is per-thread (threading.local) — two threads
    transactionally locking the same path will deadlock. This is acceptable
    because tracker mutations run from CLI mainlines of single-threaded
    subprocesses.
    """
    abs_path = os.path.abspath(filepath)
    paths = _registry_paths()
    if abs_path in paths:
        # Re-entrant: already inside a transaction for this path.
        yield
        return

    dir_path = os.path.dirname(abs_path) or '.'
    os.makedirs(dir_path, exist_ok=True)
    lock_path = abs_path + '.lock'

    lock_fd = open(lock_path, 'a+')
    locked = False
    try:
        _lock_file(lock_fd, exclusive=True)
        locked = True
        paths.add(abs_path)
        try:
            yield
        finally:
            paths.discard(abs_path)
    finally:
        if locked:
            try:
                _unlock_file(lock_fd)
            except Exception:
                pass
        lock_fd.close()


def save_json(filepath, data, default=None):
    """
    Save *data* as JSON to *filepath* atomically and under an exclusive lock.

    Writes to a temporary file first, then does an atomic rename.  An
    exclusive lock is held on a sidecar ``<filepath>.lock`` file for the
    duration of the write, so concurrent ``save_json`` calls serialize
    rather than silently clobbering each other (the earlier writer's data
    would otherwise be lost when the later ``os.replace`` wins).

    The sidecar lock file is created on first use and left in place —
    deleting it would defeat the locking invariant.

    If the calling code is already inside a ``transactional_json(filepath)``
    block (same path, same thread), the lock has already been acquired and
    this function skips its own acquisition to avoid fcntl re-entry deadlock
    (see D-002 above). External serialization is unaffected.

    *default*, when provided, is passed to ``json.dump`` to handle types
    that are not JSON-serialisable by default (datetimes, numpy scalars,
    custom dataclasses).
    """
    abs_path = os.path.abspath(filepath)

    # Fast path: caller is already inside a transactional_json for this path
    # (same thread). Skip re-acquiring the lockfile.
    if abs_path in _registry_paths():
        _write_atomic(abs_path, data, default=default)
        return

    dir_path = os.path.dirname(abs_path) or '.'
    os.makedirs(dir_path, exist_ok=True)
    lock_path = abs_path + '.lock'

    lock_fd = open(lock_path, 'a+')
    locked = False
    try:
        _lock_file(lock_fd, exclusive=True)
        locked = True
        _write_atomic(abs_path, data, default=default)
    finally:
        if locked:
            try:
                _unlock_file(lock_fd)
            except Exception:
                pass
        lock_fd.close()
