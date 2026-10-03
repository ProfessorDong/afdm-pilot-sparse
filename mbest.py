"""Multi-block chirp-domain path estimation for embedded-pilot AFDM.

Model on the pilot window W (block b = 0..B-1):
    y_b = sum_p g_p exp(j2pi kappa_p b beta) a(ell_p, kappa_p) + data leakage + noise
with a(.) the block-0 pilot atom (MBAFDM.atoms). g_p absorbs pilot amplitude and
the block-0 phase reference.

Estimators share one greedy loop and differ only in how kappa is fitted:
  'sb'   : block 0 only (single-block chirp-domain estimator)
  'nc'   : non-coherent multi-block (per-block free gains -> MMV / D-GESBL-like)
  'vern' : coarse cell from 'nc', then coherent slow-time fit inside the cell
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from mbafdm import MBAFDM


class ChirpDomainEstimator:
    def __init__(self, S: MBAFDM, kstep=0.02, kpad=0.6):
        self.S = S
        self.L = S.ell_max + 1
        self.KG = np.arange(-S.alpha_max - kpad, S.alpha_max + kpad + 1e-9, kstep)
        A = S.atoms(range(self.L), self.KG)
        self.An = A / np.linalg.norm(A, axis=-1, keepdims=True)
        self.kstep = kstep

    # ---------- atoms and slow-time steering ----------
    def atom(self, l, k):
        return self.S.atoms([l], [k])[0, 0]

    def ph(self, k, B):
        return np.exp(1j * 2 * np.pi * k * np.arange(B) * self.S.beta)

    def design(self, paths, B, mode):
        """Columns of the stacked (B*|W|) regression for a set of paths.

        'nc' gives each path a free gain per block (B columns per path);
        the coherent modes give one gain per path with the deterministic phase.
        """
        cols = []
        for l, k in paths:
            a = self.atom(l, k)
            if mode == "nc":
                for b in range(B):
                    c = np.zeros((B, a.size), complex)
                    c[b] = a
                    cols.append(c.reshape(-1))
            else:
                cols.append(np.outer(self.ph(k, B), a).reshape(-1))
        return np.stack(cols, 1) if cols else np.zeros((B * self.S.W.size, 0), complex)

    def ls(self, y, paths, mode):
        B = y.shape[0]
        D = self.design(paths, B, mode)
        g, *_ = np.linalg.lstsq(D, y.reshape(-1), rcond=None)
        return g, (y.reshape(-1) - D @ g).reshape(y.shape)

    # ---------- per-path 1-D scores on a residual ----------
    def s_nc(self, r, l, k):
        a = self.atom(l, k)
        return np.sum(np.abs(r @ np.conj(a)) ** 2) / np.vdot(a, a).real

    def s_coh(self, r, l, k):
        a = self.atom(l, k)
        return np.abs(np.conj(self.ph(k, r.shape[0])) @ (r @ np.conj(a))) ** 2 / np.vdot(a, a).real

    @staticmethod
    def _max1d(f, lo, hi, n):
        kk = np.linspace(lo, hi, n)
        v = [f(k) for k in kk]
        i = int(np.argmax(v))
        h = (hi - lo) / (n - 1)
        r = minimize_scalar(lambda k: -f(k), bounds=(kk[i] - h, kk[i] + h),
                            method="bounded", options={"xatol": 1e-7})
        return r.x

    def fit_kappa(self, r, l, k0, mode):
        B = r.shape[0]
        if mode == "sb":
            return self._max1d(lambda k: self.s_nc(r[:1], l, k), k0 - self.kstep, k0 + self.kstep, 5)
        kc = self._max1d(lambda k: self.s_nc(r, l, k), k0 - self.kstep, k0 + self.kstep, 5)
        if mode == "nc":
            return kc
        half = 0.5 / self.S.beta
        n = int(np.ceil(2 * half * B * self.S.beta * 8)) + 1   # ~8 points per main lobe
        return self._max1d(lambda k: self.s_coh(r, l, k), kc - half, kc + half, n)

    # ---------- greedy multi-path extraction ----------
    def run(self, y, P_max, mode, sweeps=2, return_coarse=False):
        if mode == "sb":
            y = y[:1]
        B = y.shape[0]
        paths, coarse = [], []
        r = y.copy()
        for _ in range(P_max):
            C = np.einsum("lkw,bw->lkb", np.conj(self.An), r)
            s = np.sum(np.abs(C) ** 2, -1)
            l, i = np.unravel_index(np.argmax(s), s.shape)
            k = self.fit_kappa(r, l, self.KG[i], mode)
            paths.append((int(l), float(k)))
            coarse.append(self.KG[i])
            _, r = self.ls(y, paths, "nc" if mode == "nc" else "coh")
        # coordinate refinement: refit each path on the residual of the others
        for _ in range(sweeps):
            for p in range(len(paths)):
                others = paths[:p] + paths[p + 1:]
                _, rp = self.ls(y, others, "nc" if mode == "nc" else "coh")
                l = paths[p][0]
                paths[p] = (l, float(self.fit_kappa(rp, l, paths[p][1], mode)))
        g, r = self.ls(y, paths, "nc" if mode == "nc" else "coh")
        return (paths, g, r, coarse) if return_coarse else (paths, g, r)
