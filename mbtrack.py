"""Soft LMMSE detection and parametric decision-directed slow-time tracking.

Per-path parametric model over the blocks b of an aperture A:
    y_b = sum_p h_p Phi_b(ell_p, kappa_p, rho_p) x_b + w_b,
    Phi_b = F_A  Delta_b(kappa, rho)  Pi(ell)  F_A^H,
with absolute-time Doppler phase (so the inter-block phase exp(j2pi kappa b beta)
is built in), Pi(ell) the band-limited circular (fractional) delay, and h_p
shared by all blocks in A.

`Tracker.refit` solves  min sum_{b in A} ||y_b - sum_p h_p Phi_b x_b||^2  by
variable projection (h eliminated by least squares) with Levenberg-damped
Gauss-Newton steps on the real per-path parameters (kappa; optionally ell and
rho), using the Kaufman Jacobian. x_b are known pilots/training and soft data
estimates E[x|y].
"""
from __future__ import annotations

import numpy as np

from mbafdm import MBAFDM


def _kt(N):
    k = np.arange(N)
    return np.where(k < N // 2, k, k - N)   # symmetric frequency index


def _shift(S, s, l, deriv=False):
    """Band-limited circular delay of s by l samples (integer -> exact roll)."""
    if not deriv and abs(l - round(l)) < 1e-12:
        return np.roll(s, int(round(l)), axis=-1)
    F = np.fft.fft(s, axis=-1)
    e = np.exp(-1j * 2 * np.pi * _kt(S.N) * l / S.N)
    out = np.fft.ifft(F * e, axis=-1)
    if not deriv:
        return out
    return out, np.fft.ifft(F * e * (-1j * 2 * np.pi * _kt(S.N) / S.N), axis=-1)


def apply_path(S: MBAFDM, l, k, X, blocks, rho=0.0, deriv=()):
    """Unit-gain path response for absolute block indices `blocks`.

    X: (len(blocks), N). deriv: subset of ('k', 'l', 'rho'); returns
    (y, {name: dy}) when deriv is non-empty. O(len(blocks) N log N).
    """
    s0 = S.idaft(X)
    if "l" in deriv:
        s, ds = _shift(S, s0, l, deriv=True)
    else:
        s = _shift(S, s0, l)
    Tb = S.N + S.Ncp
    n = np.arange(S.N)
    t = (np.asarray(blocks) * Tb + S.Ncp)[:, None] + n
    e = np.exp(1j * 2 * np.pi * (k * t + 0.5 * rho * t * t / Tb) / S.N)
    y = S.daft(s * e)
    if not deriv:
        return y
    d = {}
    if "k" in deriv:
        d["k"] = S.daft(s * e * (1j * 2 * np.pi * t / S.N))
    if "rho" in deriv:
        d["rho"] = S.daft(s * e * (1j * 2 * np.pi * 0.5 * t * t / (Tb * S.N)))
    if "l" in deriv:
        d["l"] = S.daft(ds * e)
    return y, d


_FA_CACHE: dict = {}


def _fa(S):
    key = (S.N, S.c1, S.c2)
    if key not in _FA_CACHE:
        _FA_CACHE[key] = S.daft(np.eye(S.N)).T
    return _FA_CACHE[key]


def channel_matrix_fft(S: MBAFDM, b, ells, kappas, h, rhos=None):
    """Same matrix built column-wise with the fast path operator (tests only;
    slower than BLAS at N=512)."""
    rhos = np.zeros(len(ells)) if rhos is None else rhos
    I = np.eye(S.N, dtype=complex)
    blocks = np.full(S.N, b)
    H = np.zeros((S.N, S.N), complex)
    for l, k, g, r in zip(ells, kappas, h, rhos):
        H += g * apply_path(S, l, k, I, blocks, rho=r)   # row m = response to e_m
    return H.T


def channel_matrix(S: MBAFDM, b, ells, kappas, h, rhos=None):
    """Dense DAFT-domain matrix of block b under the parametric model
    (explicit F_A G F_A^H; fastest at N=512 with BLAS)."""
    Tb = S.N + S.Ncp
    n = np.arange(S.N)
    t = b * Tb + S.Ncp + n
    rhos = np.zeros(len(ells)) if rhos is None else rhos
    G = np.zeros((S.N, S.N), complex)
    I = np.eye(S.N)
    for l, k, g, r in zip(ells, kappas, h, rhos):
        ph = g * np.exp(1j * 2 * np.pi * (k * t + 0.5 * r * t * t / Tb) / S.N)
        G += ph[:, None] * _shift(S, I.T, l).T
    FA = _fa(S)
    return FA @ G @ FA.conj().T


def lmmse_soft(H, y, known_mask, x_known, sigma2, idx):
    """LMMSE on positions idx after removing known entries.

    Returns (hard, soft_mean, soft_var, post_sinr, unbiased_z, unbiased_var)
    for unit-energy QPSK.
    """
    r = y - H[:, known_mask] @ x_known[known_mask] if known_mask is not None and known_mask.any() else y
    Hd = H[:, idx]
    G = Hd.conj().T @ Hd
    Minv = np.linalg.inv(G + sigma2 * np.eye(idx.size))
    z = Minv @ (Hd.conj().T @ r)
    mu = np.clip(np.real(np.einsum("ij,ji->i", Minv, G)), 1e-6, 1 - 1e-9)
    zu = z / mu
    v = (1 - mu) / mu
    a = np.sqrt(2) / v
    xm = (np.tanh(a * zu.real) + 1j * np.tanh(a * zu.imag)) / np.sqrt(2)
    xv = 1 - np.abs(xm) ** 2
    hard = (np.where(zu.real >= 0, 1, -1) + 1j * np.where(zu.imag >= 0, 1, -1)) / np.sqrt(2)
    return hard, xm, xv, mu / (1 - mu), zu, v


class Tracker:
    def __init__(self, S: MBAFDM, soft=True, window=None, gn_iters=3, redetect=True,
                 est_delay=False, model_rho=False, reacq_every=0, reacq_window=8,
                 reacq_pfa=1e-3, kmax=None, P_cap=8, retries=2, aperture_res=True, reacq_level="residual"):
        self.S, self.soft, self.window = S, soft, window
        self.gn_iters, self.redetect = gn_iters, redetect
        self.est_delay, self.model_rho = est_delay, model_rho
        self.reacq_every, self.reacq_window, self.reacq_pfa = reacq_every, reacq_window, reacq_pfa
        self.kmax = S.alpha_max + 0.5 if kmax is None else kmax
        self.P_cap = P_cap
        self.retries = retries
        self.merge_dl, self.merge_dk = 0.6, 0.3          # resolution-cell merge thresholds
        self.aperture_res = aperture_res                 # shrink Doppler thresholds with the aperture
        self.reacq_level = reacq_level                   # 'noise': sigma^2; 'residual': measured residual power

    # ---------- data-aided re-acquisition ----------
    def reacquire(self, Y, X, blocks, paths, h, sigma2, M=64):
        """Search the residual of the tracked model for one new path with the
        coherent multi-block cross-ambiguity, using X (pilots + decoded data) as
        a known signal. Zero-padded FFTs evaluate the Doppler axis on a 1/M grid.
        Returns the augmented path list (unchanged if no detection)."""
        S = self.S
        if len(paths) >= self.P_cap:
            return paths
        R = Y.copy()
        for (l, k, r), g in zip(paths, h):
            R -= g * apply_path(S, l, k, X, blocks, rho=r)
        rt = S.idaft(R)                      # time-domain residual (unitary)
        st = S.idaft(X)
        Tb = S.N + S.Ncp
        Nf = S.N * M
        kg = np.fft.fftfreq(Nf, d=1.0 / S.N)  # kappa grid (subcarrier spacings)
        sel = np.abs(kg) <= self.kmax
        best = (0.0, None)
        E = np.sum(np.abs(st) ** 2)           # energy of the known signal over the window
        for l in range(S.ell_max + 1):
            q = np.conj(np.roll(st, l, axis=-1)) * rt            # (Bw, N)
            Fq = np.fft.fft(q, n=Nf, axis=-1)                    # sum_n q e^{-j2pi kappa n/N}
            t0 = (np.asarray(blocks) * Tb + S.Ncp)[:, None]
            c = np.sum(Fq * np.exp(-1j * 2 * np.pi * kg[None, :] * t0 / S.N), axis=0)
            sc = np.abs(c) ** 2 / E
            sc[~sel] = 0
            i = int(np.argmax(sc))
            if sc[i] > best[0]:
                best = (sc[i], (float(l), float(kg[i])))
        cells = (S.ell_max + 1) * sel.sum() / M               # ~independent cells
        lvl = sigma2 if self.reacq_level == "noise" else max(sigma2, float(np.mean(np.abs(R) ** 2)))
        thr = lvl * -np.log(self.reacq_pfa / cells)          # exp(1) tail of |c|^2/E
        if best[0] < thr:
            return paths
        l, k = best[1]
        # reject a duplicate of an existing path
        dk_dup = 0.5 / (len(np.atleast_1d(blocks)) * S.beta) if self.aperture_res else 0.5
        if any(abs(l - p[0]) < 1.0 and abs(k - p[1]) < dk_dup for p in paths):
            return paths
        return paths + [(l, k, 0.0)]

    # ---------- parametric refit ----------
    def refit(self, Y, X, blocks, paths, iters=None, rows=None):
        """paths: list of (ell, kappa[, rho]). rows: optional chirp indices to
        regress on (e.g. the pilot window when X holds only the pilot).
        Returns (paths, h, mse)."""
        S = self.S
        P = len(paths)
        ell = np.array([p[0] for p in paths], float)
        kap = np.array([p[1] for p in paths], float)
        rho = np.array([p[2] if len(p) > 2 else 0.0 for p in paths], float)
        names = ["k"] + (["l"] if self.est_delay else []) + (["rho"] if self.model_rho else [])
        sel = slice(None) if rows is None else rows
        y = Y[:, sel].reshape(-1)
        it = self.gn_iters if iters is None else iters

        def build(ell, kap, rho):
            cols, dcols = [], {nm: [] for nm in names}
            for p in range(P):
                c, d = apply_path(S, ell[p], kap[p], X, blocks, rho=rho[p], deriv=names)
                cols.append(c[:, sel].reshape(-1))
                for nm in names:
                    dcols[nm].append(d[nm][:, sel].reshape(-1))
            return np.stack(cols, 1), {nm: np.stack(v, 1) for nm, v in dcols.items()}

        def solve(D):
            g, *_ = np.linalg.lstsq(D, y, rcond=None)
            r = y - D @ g
            return g, r, np.vdot(r, r).real

        D, dD = build(ell, kap, rho)
        g, res, cost = solve(D)
        lam = 1e-3
        for _ in range(it):
            Q, _ = np.linalg.qr(D)
            J = np.concatenate([dD[nm] * g[None, :] for nm in names], 1)
            J = J - Q @ (Q.conj().T @ J)
            Jr = np.concatenate([J.real, J.imag])
            rr = np.concatenate([res.real, res.imag])
            JTJ = Jr.T @ Jr
            grad = Jr.T @ rr
            improved = False
            while lam < 1e8:
                step = np.linalg.solve(JTJ + lam * np.diag(np.diag(JTJ) + 1e-12), grad)
                st = {nm: step[i * P:(i + 1) * P] for i, nm in enumerate(names)}
                kn = kap + st["k"]
                ln = np.clip(ell + st.get("l", 0.0), 0, S.Ncp)
                rn_ = rho + st.get("rho", 0.0)
                Dn, dDn = build(ln, kn, rn_)
                gn, rn, cn = solve(Dn)
                if cn < cost:
                    kap, ell, rho, D, dD, g, res, cost = kn, ln, rn_, Dn, dDn, gn, rn, cn
                    lam = max(lam / 3, 1e-9)
                    improved = True
                    break
                lam *= 5
            if not improved or np.max(np.abs(step)) < 1e-8:
                break
        out = [(float(ell[p]), float(kap[p]), float(rho[p])) for p in range(P)]
        # Two estimates inside one resolution cell (sub-sample delay, a fraction of a
        # Doppler bin) describe one path; keeping both makes the LS ill-conditioned
        # with large cancelling gains. Merge: keep the stronger and re-solve.
        if P > 1:
            # Doppler resolution sharpens with the aperture: ~1/(|A| beta) subcarrier spacings
            dk = self.merge_dk / max(1, len(np.atleast_1d(blocks))) if self.aperture_res else self.merge_dk
            for i in range(P):
                for j in range(i + 1, P):
                    if abs(ell[i] - ell[j]) < self.merge_dl and abs(kap[i] - kap[j]) < dk:
                        drop = i if abs(g[i]) < abs(g[j]) else j
                        keep = [out[q] for q in range(P) if q != drop]
                        return self.refit(Y, X, blocks, keep, iters=iters, rows=rows)
        return out, g, cost / y.size

    # ---------- sequential receiver over a frame ----------
    def run_full(self, Y, kinds, paths, h, sigma2, xknown=None, codec=None):
        """kinds per block: 'P' pilot(+guard)+data, 'T' fully known training,
        'D' data on all chirps. Pilot/training blocks are detected with the
        acquisition estimate; each data block is predicted, detected, appended
        to the aperture, refit, and re-detected. Returns (dets, traj) with
        dets[b] = (hard, unbiased_z, unbiased_var, idx) or None for 'T'.

        codec(kind, zu, v) -> (info_bits, reencoded_symbols): if given, the
        regression uses re-encoded decoder output (decoder-aided tracking) and
        dets[b] gains the decoded bits as a 5th element."""
        S = self.S
        B = len(kinds)
        N = S.N
        pil = np.zeros(N, bool); pil[S.zero_set] = True
        xp = np.zeros(N, complex); xp[S.m0] = np.sqrt(S.Ep)
        Xs = np.zeros((B, N), complex)
        dets = [None] * B
        traj = []
        paths = [tuple(p) + ((0.0,) if len(p) == 2 else ()) for p in paths]
        h = np.asarray(h, complex)

        def detect(b):
            H = channel_matrix(S, b, [p[0] for p in paths], [p[1] for p in paths], h,
                               [p[2] for p in paths])
            if kinds[b] == "T":
                return None, xknown[b]
            if kinds[b] == "P":
                idx = S.data_idx
                hd, xm, xv, _, zu, v = lmmse_soft(H, Y[b], pil, xp, sigma2, idx)
                xs = xp.copy()
            else:
                idx = np.arange(N)
                hd, xm, xv, _, zu, v = lmmse_soft(H, Y[b], None, None, sigma2, idx)
                xs = np.zeros(N, complex)
            if codec is not None:
                bits, xr = codec(kinds[b], zu, v)
                ok = xr is not None
                xs[idx] = xr if ok else (xm if self.soft else hd)
                self.crc_ok[b] = ok
                return (hd, zu, v, idx, bits), xs
            xs[idx] = xm if self.soft else hd
            return (hd, zu, v, idx), xs

        n_known = sum(1 for k in kinds if k in "PT")
        self.crc_ok = np.ones(B, bool)

        def reacq_loop(b, lo, blocks):
            nonlocal paths, h
            wl = max(0, b + 1 - self.reacq_window)
            wb = np.arange(wl, b + 1)
            for _ in range(self.P_cap):
                new = self.reacquire(Y[wl:b + 1], Xs[wl:b + 1], wb, paths, h, sigma2)
                if len(new) == len(paths):
                    break
                paths, h, _ = self.refit(Y[lo:b + 1], Xs[lo:b + 1], blocks, new)

        for b in range(B):
            dets[b], Xs[b] = detect(b)
            if b + 1 < n_known:
                continue
            lo = 0 if self.window is None else max(0, b + 1 - self.window)
            blocks = np.arange(lo, b + 1)
            last_known = b + 1 == n_known
            periodic = self.reacq_every and kinds[b] == "D" and (b + 1 - n_known) % self.reacq_every == 0
            for attempt in range(1 + (self.retries if codec is not None else 0)):
                paths, h, _ = self.refit(Y[lo:b + 1], Xs[lo:b + 1], blocks, paths)
                # data-aided re-acquisition: after the pilot block(s) are decoded (finds
                # weak paths the single-pilot search missed), periodically, and on CRC failure
                if self.reacq_every and (last_known or periodic or not self.crc_ok[b]):
                    reacq_loop(b, lo, blocks)
                if kinds[b] == "T":
                    break
                if (self.redetect and kinds[b] == "D") or not self.crc_ok[b] or last_known:
                    for bb in (range(lo, b + 1) if last_known else (b,)):
                        if kinds[bb] != "T":
                            dets[bb], Xs[bb] = detect(bb)
                if self.crc_ok[b]:
                    break
            traj.append((b, [p[1] for p in paths], h.copy()))
        return dets, traj
