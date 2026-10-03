"""Per-block AFDM channel matrices and LMMSE detection for the multi-block frame."""
from __future__ import annotations

import numpy as np

from mbafdm import MBAFDM


class BlockChannel:
    def __init__(self, S: MBAFDM):
        self.S = S
        I = np.eye(S.N)
        self.FA = S.daft(I).T            # y = FA @ r   (DAFT matrix)
        self.FAH = self.FA.conj().T      # IDAFT matrix

    def H(self, b, ells, kappas, h):
        """Exact DAFT-domain matrix of block b (prefix >= max delay, 2Nc1 integer)."""
        S = self.S
        n = np.arange(S.N)
        G = np.zeros((S.N, S.N), complex)
        t0 = b * (S.N + S.Ncp) + S.Ncp
        for l, k, g in zip(ells, kappas, h):
            l = int(l)
            ph = g * np.exp(1j * 2 * np.pi * k * (t0 + n) / S.N)
            G[n, (n - l) % S.N] += ph
        return self.FA @ G @ self.FAH

    def lmmse(self, Hb, yb, xb_known, sigma2):
        """Detect QPSK data on S.data_idx; pilot/guard entries of xb_known are used as known."""
        d = self.S.data_idx
        known = np.ones(self.S.N, bool)
        known[d] = False
        r = yb - Hb[:, known] @ xb_known[known]
        Hd = Hb[:, d]
        z = np.linalg.solve(Hd.conj().T @ Hd + sigma2 * np.eye(d.size), Hd.conj().T @ r)
        return z

    @staticmethod
    def qpsk_hard(z):
        return np.exp(1j * (np.pi / 4 + np.pi / 2 * np.round((np.angle(z) - np.pi / 4) / (np.pi / 2))))


def apply_path(S: MBAFDM, l, k, X, b0=0):
    """Fast unit-gain path response for blocks b0..b0+B-1: rows of X are x_b.

    Equals BlockChannel.H(b, [l], [k], [1]) @ X[b - b0] for every row, in O(B N log N).
    """
    B = X.shape[0]
    s = S.idaft(X)
    s = np.roll(s, int(l), axis=-1)
    n = np.arange(S.N)
    t0 = (np.arange(b0, b0 + B) * (S.N + S.Ncp) + S.Ncp)[:, None]
    return S.daft(s * np.exp(1j * 2 * np.pi * k * (t0 + n) / S.N))
