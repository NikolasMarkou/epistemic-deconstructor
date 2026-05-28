"""Process-level concurrency tests for the tracker scripts.

plan_2026-05-28_9d761933/D-002+D-003: closes audit OOS-2 (cross-process
load→modify→save TOCTOU) and OOS-3 (concurrency test suite). These tests
spawn REAL parallel subprocesses (not threads) to actually exercise the
fcntl-based serialization plumbed into ``common.transactional_json``.

Why subprocess and not threading: fcntl.flock is process-level on POSIX.
A multi-threaded test in one process would not exercise the inter-process
locking guarantee that the audit's Compound 2 finding (parallel ed-* agents
writing to hypotheses.json) actually depends on.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ProcessPoolExecutor, as_completed


SCRIPTS_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'src', 'scripts')
BAYES_SCRIPT = os.path.join(SCRIPTS_DIR, 'bayesian_tracker.py')
BELIEF_SCRIPT = os.path.join(SCRIPTS_DIR, 'belief_tracker.py')


def _bayes(hyp_file, *args):
    """Invoke bayesian_tracker.py with --file <hyp_file>."""
    return subprocess.run(
        [sys.executable, BAYES_SCRIPT, '--file', hyp_file] + list(args),
        capture_output=True, text=True, timeout=30)


def _belief(prof_file, *args):
    return subprocess.run(
        [sys.executable, BELIEF_SCRIPT, '--file', prof_file] + list(args),
        capture_output=True, text=True, timeout=30)


# Helper invoked in subprocesses by ProcessPoolExecutor. Must be defined at
# module level so it is picklable on platforms that fork-launch workers.
def _run_bayes_update(args_tuple):
    hyp_file, hid, evidence, lr = args_tuple
    return _bayes(hyp_file, 'update', hid, evidence, '--lr', str(lr),
                  '--override-disconfirm', 'concurrency test')


def _run_belief_update(args_tuple):
    prof_file, tid, evidence, lr = args_tuple
    return _belief(prof_file, 'update', tid, evidence, '--lr', str(lr))


def _run_bayes_add(args_tuple):
    """Top-level helper for ProcessPoolExecutor (local lambdas/closures aren't
    picklable on fork-launched workers)."""
    hyp_file, statement, prior = args_tuple
    return _bayes(hyp_file, 'add', statement, '--prior', str(prior))


class _ConcurrencyBase(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.hyp_file = os.path.join(self.tmpdir, 'hypotheses.json')
        self.prof_file = os.path.join(self.tmpdir, 'beliefs.json')
        # Write state.md so phase detection works for LR cap (phase 2 → cap 10).
        with open(os.path.join(self.tmpdir, 'state.md'), 'w') as f:
            f.write("# State\n## Phase: 2\n## Tier: STANDARD\n")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _load_hyp(self):
        with open(self.hyp_file) as f:
            return json.load(f)

    def _load_prof(self):
        with open(self.prof_file) as f:
            return json.load(f)


class TestBayesianTrackerConcurrency(_ConcurrencyBase):

    def test_two_parallel_updates_distinct_hids_both_visible(self):
        """Two writers updating two different hypotheses concurrently — both
        evidence entries land in the file (no silent overwrite)."""
        _bayes(self.hyp_file, 'add', 'A', '--prior', '0.5')
        _bayes(self.hyp_file, 'add', 'B', '--prior', '0.5')

        tasks = [(self.hyp_file, 'H1', 'evA', 2.0),
                 (self.hyp_file, 'H2', 'evB', 2.0)]
        with ProcessPoolExecutor(max_workers=2) as ex:
            results = list(ex.map(_run_bayes_update, tasks))
        for r in results:
            self.assertEqual(r.returncode, 0, msg=r.stderr)

        data = self._load_hyp()
        hyps = {h['id']: h for h in data['hypotheses']}
        # Both hypotheses present, each with exactly one evidence entry.
        self.assertEqual(len(hyps['H1']['evidence']), 1)
        self.assertEqual(hyps['H1']['evidence'][0]['description'], 'evA')
        self.assertEqual(len(hyps['H2']['evidence']), 1)
        self.assertEqual(hyps['H2']['evidence'][0]['description'], 'evB')

    def test_two_parallel_updates_same_hid_both_evidence_entries_present(self):
        """Two writers updating the SAME hypothesis concurrently — without
        transactional_json this is the canonical evidence-loss scenario.
        With it: both evidence entries must be present in the trail."""
        _bayes(self.hyp_file, 'add', 'shared', '--prior', '0.5')

        tasks = [(self.hyp_file, 'H1', 'ev_alpha', 2.0),
                 (self.hyp_file, 'H1', 'ev_beta', 2.0)]
        with ProcessPoolExecutor(max_workers=2) as ex:
            results = list(ex.map(_run_bayes_update, tasks))
        for r in results:
            self.assertEqual(r.returncode, 0, msg=r.stderr)

        data = self._load_hyp()
        h1 = data['hypotheses'][0]
        descs = {e['description'] for e in h1['evidence']}
        self.assertEqual(descs, {'ev_alpha', 'ev_beta'},
                         "Both evidence entries must survive concurrent writes; "
                         "if one is missing, transactional_json is not serialising "
                         "the load→modify→save sequence correctly.")
        self.assertEqual(len(h1['evidence']), 2)

    def test_n_parallel_writers_evidence_count_matches(self):
        """N=5 parallel writers against one hypothesis → exactly 5 evidence
        entries (stress test, surfaces non-deterministic loss if any)."""
        _bayes(self.hyp_file, 'add', 'stress', '--prior', '0.5')
        n = 5
        tasks = [(self.hyp_file, 'H1', f'writer_{i}', 1.5) for i in range(n)]
        with ProcessPoolExecutor(max_workers=n) as ex:
            results = list(ex.map(_run_bayes_update, tasks))
        for r in results:
            self.assertEqual(r.returncode, 0, msg=r.stderr)

        data = self._load_hyp()
        h1 = data['hypotheses'][0]
        descs = sorted(e['description'] for e in h1['evidence'])
        expected = sorted(f'writer_{i}' for i in range(n))
        self.assertEqual(descs, expected,
                         f"All {n} parallel writers must be visible in the "
                         f"evidence trail. Got: {descs}")

    def test_parallel_add_then_update_no_overwrite(self):
        """One writer ADDs a hypothesis while another UPDATEs an existing one —
        neither operation clobbers the other."""
        _bayes(self.hyp_file, 'add', 'pre-existing', '--prior', '0.5')

        with ProcessPoolExecutor(max_workers=2) as ex:
            futures = [
                ex.submit(_run_bayes_add,
                          (self.hyp_file, 'added-concurrently', 0.5)),
                ex.submit(_run_bayes_update,
                          (self.hyp_file, 'H1', 'concurrent_evidence', 2.0)),
            ]
            results = [f.result() for f in as_completed(futures)]
        for r in results:
            self.assertEqual(r.returncode, 0, msg=r.stderr)

        data = self._load_hyp()
        hids = {h['id'] for h in data['hypotheses']}
        self.assertEqual(hids, {'H1', 'H2'})
        h1 = next(h for h in data['hypotheses'] if h['id'] == 'H1')
        self.assertEqual(len(h1['evidence']), 1)
        self.assertEqual(h1['evidence'][0]['description'], 'concurrent_evidence')

    def test_serial_baseline_for_comparison(self):
        """Sanity baseline: serial updates produce N evidence entries.
        Catches regressions where the lock acquisition itself silently fails."""
        _bayes(self.hyp_file, 'add', 'serial', '--prior', '0.5')
        for i in range(3):
            r = _bayes(self.hyp_file, 'update', 'H1', f's_{i}', '--lr', '1.5',
                       '--override-disconfirm', 'serial baseline')
            self.assertEqual(r.returncode, 0, msg=r.stderr)
        data = self._load_hyp()
        h1 = data['hypotheses'][0]
        self.assertEqual(len(h1['evidence']), 3)


class TestBeliefTrackerConcurrency(_ConcurrencyBase):

    def test_belief_parallel_updates_no_loss(self):
        """Mirror of BayesianTracker test_two_parallel_updates_same_hid for
        the PSYCH tracker — both evidence entries must survive."""
        _belief(self.prof_file, 'add', 'High Neuroticism',
                '--category', 'neuroticism', '--prior', '0.5')

        tasks = [(self.prof_file, 'T1', 'observation_A', 2.0),
                 (self.prof_file, 'T1', 'observation_B', 2.0)]
        with ProcessPoolExecutor(max_workers=2) as ex:
            results = list(ex.map(_run_belief_update, tasks))
        for r in results:
            self.assertEqual(r.returncode, 0, msg=r.stderr)

        data = self._load_prof()
        t1 = data['traits'][0]
        descs = {e['description'] for e in t1['evidence']}
        self.assertEqual(descs, {'observation_A', 'observation_B'})
        self.assertEqual(len(t1['evidence']), 2)


class TestTransactionalJsonReentry(unittest.TestCase):
    """In-process re-entry semantics for the context manager — same-thread
    nested transactional_json on the same path must NOT deadlock."""

    def test_reentry_is_idempotent(self):
        # Imported here so a broken common.py surfaces as an ImportError on
        # this single test rather than at module load.
        sys.path.insert(0, SCRIPTS_DIR)
        from common import transactional_json, save_json, load_json

        with tempfile.TemporaryDirectory() as tmpd:
            path = os.path.join(tmpd, 'x.json')
            # Double-nested transactional + save_json inside → all succeed.
            with transactional_json(path):
                save_json(path, {'depth': 0})
                with transactional_json(path):
                    save_json(path, {'depth': 1})
                    with transactional_json(path):
                        save_json(path, {'depth': 2})
            self.assertEqual(load_json(path), {'depth': 2})

    def test_save_json_inside_transaction_skips_relock(self):
        """save_json called inside transactional_json must complete without
        attempting to acquire the lockfile a second time (re-entry deadlock
        would cause this test to hang and the per-test timeout to fire)."""
        sys.path.insert(0, SCRIPTS_DIR)
        from common import transactional_json, save_json, load_json

        with tempfile.TemporaryDirectory() as tmpd:
            path = os.path.join(tmpd, 'y.json')
            with transactional_json(path):
                # If re-locking happens, this hangs forever.
                save_json(path, {'k': 'v'})
            self.assertEqual(load_json(path), {'k': 'v'})


# --------------------------------------------------------------------------
# Audit H3 — session_manager state.md RMW race (plan_2026-05-28_ad87937f/D-002)
# --------------------------------------------------------------------------
SM_SCRIPT = os.path.join(SCRIPTS_DIR, 'session_manager.py')


def _sm(base_dir, *args):
    """Invoke session_manager.py with --base-dir <base_dir>."""
    return subprocess.run(
        [sys.executable, SM_SCRIPT, '--base-dir', base_dir] + list(args),
        capture_output=True, text=True, timeout=30,
    )


def _run_sm_advance(args_tuple):
    base_dir, reason = args_tuple
    return _sm(base_dir, 'advance', reason)


class TestSessionManagerAdvanceRace(unittest.TestCase):
    """N parallel `advance` invocations must produce exactly one transition,
    not N. Without transactional_json on state.md, concurrent advances each
    read pre-mutation state, each pass the gate, each append a history entry —
    audit H3."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # Create a RAPID-tier session via subprocess (matches real CLI use).
        r = _sm(self.tmpdir, 'new', '--tier', 'RAPID', 'race', 'test')
        self.assertEqual(r.returncode, 0, msg=r.stderr)
        # Locate the analysis dir.
        with open(os.path.join(self.tmpdir, 'analyses', '.current_analysis')) as f:
            self.analysis_dir = os.path.join(
                self.tmpdir, 'analyses', f.read().strip())
        # Set Phase to 0.5 so RAPID's 0.5 → 5 advance is what races.
        state_path = os.path.join(self.analysis_dir, 'state.md')
        with open(state_path) as f:
            state = f.read()
        # Set Phase line.
        import re as _re
        if _re.search(r'^## Phase:', state, _re.MULTILINE):
            state = _re.sub(r'^## Phase:.*$', '## Phase: 0.5', state, flags=_re.MULTILINE)
        else:
            state = '## Phase: 0.5\n' + state
        with open(state_path, 'w') as f:
            f.write(state)
        # Write gate-passing phase_0_5.md.
        phase_out = os.path.join(self.analysis_dir, 'phase_outputs')
        os.makedirs(phase_out, exist_ok=True)
        with open(os.path.join(phase_out, 'phase_0_5.md'), 'w') as f:
            f.write("# Phase 0.5 RAPID\nCoherence checks all pass.\n"
                    "Verdict: CREDIBLE.\n" + "x" * 200 + "\n")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_n_parallel_advance_produces_exactly_one_transition(self):
        n = 4
        tasks = [(self.tmpdir, f'race-{i}') for i in range(n)]
        with ProcessPoolExecutor(max_workers=n) as ex:
            results = list(ex.map(_run_sm_advance, tasks))
        successes = [r for r in results if r.returncode == 0]
        # Exactly one process should have legitimately advanced.
        self.assertEqual(
            len(successes), 1,
            msg=f"Expected 1 success, got {len(successes)}. "
                f"stdouts: {[r.stdout for r in results]} "
                f"stderrs: {[r.stderr for r in results]}",
        )

        # state.md must have exactly one ADVANCE history line.
        with open(os.path.join(self.analysis_dir, 'state.md')) as f:
            state = f.read()
        advance_lines = [
            line for line in state.splitlines() if line.startswith('- ADVANCE')
        ]
        self.assertEqual(
            len(advance_lines), 1,
            msg=f"Expected exactly 1 ADVANCE entry in state.md, got "
                f"{len(advance_lines)}. Lines: {advance_lines}",
        )
        # Final Phase should be 5 (RAPID terminus).
        self.assertIn('## Phase: 5', state)


if __name__ == '__main__':
    unittest.main()
