"""Forced-failure tests of the tracker's committed-state handling."""
import numpy as np

from engine import Config, make_system
from mbtrack import Tracker, apply_path

S = make_system(Config())
TRUE = [(2.0, 0.7), (5.0, -1.3)]
H = np.array([0.8, 0.5j])


def frame(B, seed=0):
    rng = np.random.default_rng(seed)
    X = np.exp(1j * np.pi / 4 * (2 * rng.integers(0, 4, (B, S.N)) + 1))
    X[0, S.zero_set] = 0
    X[0, S.m0] = np.sqrt(S.Ep)
    Y = sum(g * apply_path(S, l, k, X, np.arange(B)) for (l, k), g in zip(TRUE, H))
    Y = Y + 0.01 * (rng.standard_normal(Y.shape) + 1j * rng.standard_normal(Y.shape))
    return X, Y


def run(T, X, Y, kinds, fail, passes_after=()):
    """CRC passes (true symbols) except on blocks in `fail`; a block in
    `passes_after` passes from its second decoding on. While a failing block is
    processed, re-acquisition proposes a spurious component."""
    seen = {}

    def codec(kind, zu, v):
        b = T.current_block
        seen[b] = seen.get(b, 0) + 1
        ok = b not in fail or (b in passes_after and seen[b] > 1)
        idx = S.data_idx if kind == "P" else np.arange(S.N)
        return np.zeros(1, int), (X[b][idx] if ok else None)
    orig = T.reacquire

    def spurious_reacquire(Yw, Xw, blocks, paths, h, sigma2, M=64):
        if T.current_block in fail and T.current_block not in passes_after and len(paths) < 4:
            return paths + [(6.0, 2.5, 0.0)]
        return orig(Yw, Xw, blocks, paths, h, sigma2, M)
    T.reacquire = spurious_reacquire
    return T.run_full(Y, kinds, list(TRUE), H, 1e-4, codec=codec)


def test_rollback_removes_retry_model_changes():
    X, Y = frame(5)
    kinds = "PDDDD"
    T = Tracker(S, trust="strict", reacq_every=1, P_cap=8, policy="rollback", split=False)
    run(T, X, Y, kinds, fail={2})
    assert not T.trusted[2]
    assert len(T.final[0]) == 2, T.final[0]               # spurious component rolled back
    L = Tracker(S, trust="strict", reacq_every=1, P_cap=8, policy="legacy", split=False)
    run(L, X, Y, kinds, fail={2})
    assert len(L.final[0]) > 2                            # legacy keeps it


def test_empty_trusted_window_keeps_committed_state():
    X, Y = frame(4)
    T = Tracker(S, trust="strict", reacq_every=1, P_cap=8, window=2, policy="rollback", split=False)
    run(T, X, Y, "PDDD", fail={2, 3})
    assert T.degraded[3] and not T.trusted[2] and not T.trusted[3]
    assert len(T.final[0]) == 2


def test_refit_after_successful_retry():
    X, Y = frame(4)
    T = Tracker(S, trust="strict", reacq_every=0, P_cap=8, policy="rollback", split=False)
    calls = []
    orig = T.refit

    def logging_refit(Yb, Xb, blocks, paths, iters=None, rows=None):
        calls.append(np.array(blocks).tolist())
        return orig(Yb, Xb, blocks, paths, iters=iters, rows=rows)
    T.refit = logging_refit
    run(T, X, Y, "PDDD", fail={2}, passes_after={2})
    assert T.crc_ok[2] and T.trusted[2]
    # the last refit while processing block 2 includes block 2 with verified symbols
    assert any(c[-1] == 2 for c in calls)


def test_split_recovers_merged_pair():
    """Two equal-delay paths 0.15 apart, initialized as one merged component."""
    rng = np.random.default_rng(3)
    B = 16
    X = np.exp(1j * np.pi / 4 * (2 * rng.integers(0, 4, (B, S.N)) + 1))
    true = [(3.0, 0.40), (3.0, 0.55)]
    g = np.array([0.7, 0.7j])
    Y = sum(gg * apply_path(S, l, k, X, np.arange(B)) for (l, k), gg in zip(true, g))
    Y = Y + 0.03 * (rng.standard_normal(Y.shape) + 1j * rng.standard_normal(Y.shape))
    T = Tracker(S, P_cap=8)
    merged = [(3.0, 0.475, 0.0)]
    _, h, _ = T.refit(Y, X, np.arange(B), merged)
    # re-acquisition excludes +-1/(8 beta) around 0.475, so neither true Doppler can be proposed
    new = T.reacquire(Y[-8:], X[-8:], np.arange(8, 16), merged, h, 0.03 ** 2 * 2)
    assert all(abs(p[1] - 0.40) > 0.05 and abs(p[1] - 0.55) > 0.05 for p in new[1:])
    paths, h, ok = T.split_test(Y, X, np.arange(B), merged, h, 0.03 ** 2 * 2)
    assert ok and len(paths) == 2
    ks = sorted(p[1] for p in paths)
    assert abs(ks[0] - 0.40) < 0.01 and abs(ks[1] - 0.55) < 0.01


def test_refit_after_pass_on_first_redetection():
    """A block failing its first decoding but passing the attempt-0 re-detection is
    refit with its verified symbols before the model is committed."""
    X, Y = frame(4)
    T = Tracker(S, trust="strict", reacq_every=0, P_cap=8, policy="keep", split=False)
    log = []
    orig = T.refit

    def logging_refit(Yb, Xb, blocks, paths, iters=None, rows=None):
        blocks = np.atleast_1d(blocks)
        if blocks[-1] == 2:
            log.append(np.array_equal(Xb[-1], X[2]))        # verified (exactly the true) symbols?
        return orig(Yb, Xb, blocks, paths, iters=iters, rows=rows)
    T.refit = logging_refit
    run(T, X, Y, "PDDD", fail={2}, passes_after={2})
    assert T.crc_ok[2] and log and log[-1]


def test_empty_trusted_window_keep_policy():
    X, Y = frame(4)
    T = Tracker(S, trust="strict", reacq_every=1, P_cap=8, window=2, policy="keep", split=False)
    run(T, X, Y, "PDDD", fail={2, 3})
    assert T.degraded[3] and not T.trusted[2] and not T.trusted[3]
    # no trusted block in the window at block 3: the committed model that predicted
    # block 3 is kept unchanged (block 3's tentative proposals are discarded)
    committed_paths, committed_h = T.pred[3]
    assert T.final[0] == committed_paths and np.allclose(T.final[1], committed_h)
