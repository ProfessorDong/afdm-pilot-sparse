"""Receivers evaluated on one simulated frame. Each returns hard decisions per block.

All receivers see the same received frame Y (B, N) of known block kinds and
noise variance; only the conventional receiver uses a different frame (pilot in
every block), passed separately.
"""
from __future__ import annotations

import numpy as np

from mbafdm import MBAFDM
from mbest import ChirpDomainEstimator
from mbtrack import Tracker, channel_matrix, lmmse_soft


def _detect_with(S, Y, kinds, paths_per_block, sigma2):
    pil = np.zeros(S.N, bool); pil[S.zero_set] = True
    xpil = np.zeros(S.N, complex); xpil[S.m0] = np.sqrt(S.Ep)
    out = []
    for b, k in enumerate(kinds):
        ells, kaps, h = paths_per_block(b)
        H = channel_matrix(S, b, ells, kaps, h)
        if k == "P":
            hd = lmmse_soft(H, Y[b], pil, xpil, sigma2, S.data_idx)[0]
            x = np.zeros(S.N, complex); x[S.data_idx] = hd
        else:
            x = lmmse_soft(H, Y[b], np.zeros(S.N, bool), xpil, sigma2, np.arange(S.N))[0]
        out.append(x)
    return out


def genie(S, Y, kinds, sigma2, ch):
    return _detect_with(S, Y, kinds, lambda b: (ch["tau"], ch["kappa"], ch["h"]), sigma2)


def initial_estimate(S, E, Y, kinds, P_max, mode="vern"):
    pb = [b for b, k in enumerate(kinds) if k == "P"]
    YW = Y[pb][:, S.W]
    paths, g, _ = E.run(YW, P_max, mode if len(pb) > 1 else "sb")
    return paths, np.asarray(g) / np.sqrt(S.Ep)


def per_block_conventional(S, E, Y, sigma2, P_max):
    """Every block has pilot+guard; each block estimated on its own."""
    kinds = "P" * Y.shape[0]
    est = []
    for b in range(Y.shape[0]):
        paths, g, _ = E.run(Y[b:b + 1, S.W], P_max, "sb")
        kk = np.array([p[1] for p in paths])
        h = np.asarray(g) / np.sqrt(S.Ep) * np.exp(-1j * 2 * np.pi * kk * b * S.beta)
        est.append(([p[0] for p in paths], kk, h))
    return _detect_with(S, Y, kinds, lambda b: est[b], sigma2)


def predict_only(S, E, Y, kinds, sigma2, P_max):
    paths, h = initial_estimate(S, E, Y, kinds, P_max)
    e = ([p[0] for p in paths], [p[1] for p in paths], h)
    return _detect_with(S, Y, kinds, lambda b: e, sigma2)


def tracked(S, E, Y, kinds, sigma2, P_max, **kw):
    paths, h = initial_estimate(S, E, Y, kinds, P_max)
    T = Tracker(S, **kw)
    hard, traj = T.run(Y, kinds, paths, h, sigma2)
    return hard, traj
