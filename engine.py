"""Simulation engine for pilot-sparse multi-block AFDM (TVT study).

One call to `simulate(cfg, seed)` draws a channel, transmits every waveform
under comparison through the SAME physical channel realization (with
independent data and noise), runs every requested receiver, and returns
per-block symbol errors and GMI. Block energy is N for every block of every
waveform, so all comparisons are at equal transmit power.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict

import numpy as np
from scipy.stats import chi2

from mbafdm import MBAFDM
from mbtrack import Tracker, apply_path, channel_matrix, lmmse_soft
from coding import BlockCode


# ----------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------
@dataclass
class Config:
    N: int = 512
    Ncp: int = 8
    alpha_max: int = 3
    xi: int = 2
    ell_max: int = 7
    P: int = 4
    kappa_max: float = 3.0          # max |kappa| of any path
    doppler: str = "jakes"          # 'jakes': kappa_max cos(theta), theta~U; 'uniform'
    pdp: str = "exp"                # 'exp': E|h|^2 ~ exp(-tau/tau_rms); 'flat'
    tau_rms: float = 3.0
    frac_delay: bool = False
    rho_max: float = 0.0            # |Doppler rate| bound, kappa units per block
    pn_var: float = 0.0             # Wiener phase-noise increment per sample (rad^2)
    cfo: float = 0.0
    born_frac: float = 0.0          # fraction of paths born mid-frame
    snr_db: float = 15.0
    B: int = 16
    Bp: int = 1
    P_max: int | None = None        # None -> true P (oracle order); int -> CFAR order
    sp_eps: float = 0.2             # superimposed-pilot energy fraction
    sp_iters: int = 3
    window: int | None = None
    model_rho: bool = False
    reacq_every: int = 0            # tracker data-aided re-acquisition period (blocks)
    c2: float | None = None         # second chirp parameter; None -> 1/(2N)
    sp_acq_blocks: int = 1          # superimposed pilot + tracker: blocks used for acquisition
    aperture_res: bool = True       # Doppler resolution thresholds shrink with the aperture
    reacq_level: str = "residual"   # re-acquisition CFAR level: 'noise' or 'residual'
    receivers: tuple = ("genie", "conv", "sp", "openloop", "track")
    ofdm: bool = False              # include OFDM same-architecture baseline
    coded: bool = False             # rate-1/2 K=7 conv. code per block
    decision: str = "soft"          # tracker regression symbols: soft | hard | decoded


def make_system(cfg: Config) -> MBAFDM:
    return MBAFDM(N=cfg.N, Ncp=cfg.Ncp, alpha_max=cfg.alpha_max, xi=cfg.xi, ell_max=cfg.ell_max, c2=cfg.c2)


def ofdm_system(cfg: Config) -> MBAFDM:
    S = make_system(cfg)
    S.c1, S.c2 = 0.0, 0.0
    n = np.arange(S.N)
    S.dc1 = np.ones(S.N, complex)
    S.dc2 = np.ones(S.N, complex)
    return S


def draw_channel(cfg: Config, rng):
    P = cfg.P
    if cfg.frac_delay:
        tau = np.sort(rng.uniform(0, cfg.ell_max - 0.5, P))
    else:
        tau = rng.choice(cfg.ell_max + 1, P, replace=False).astype(float)
    if cfg.doppler == "jakes":
        kap = cfg.kappa_max * np.cos(rng.uniform(0, 2 * np.pi, P))
    else:
        kap = rng.uniform(-cfg.kappa_max, cfg.kappa_max, P)
    pw = np.exp(-tau / cfg.tau_rms) if cfg.pdp == "exp" else np.ones(P)
    pw = pw / pw.sum()
    h = np.sqrt(pw / 2) * (rng.standard_normal(P) + 1j * rng.standard_normal(P))
    rho = rng.uniform(-cfg.rho_max, cfg.rho_max, P) if cfg.rho_max > 0 else np.zeros(P)
    born = np.zeros(P, int)
    if cfg.born_frac > 0:
        nb = int(round(cfg.born_frac * P))
        born[rng.choice(P, nb, replace=False)] = rng.integers(cfg.Bp + 1, cfg.B, nb)
    return dict(tau=tau, kappa=kap, h=h, rho=rho, born=born)


class Payload:
    """Builds data symbols (coded or uncoded) for a block and remembers the bits."""

    def __init__(self, coded, rng):
        self.coded, self.rng = coded, rng
        self.bits = {}

    def symbols(self, key, n):
        if not self.coded:
            return MBAFDM.qpsk(self.rng, n)
        C = BlockCode(n)
        u = C.attach_crc(self.rng.integers(0, 2, C.k_payload))
        self.bits[key] = u
        return C.modulate(u)


def build_frame(S, kinds, pay, tag):
    x = np.zeros((len(kinds), S.N), complex)
    for b, k in enumerate(kinds):
        if k == "P":
            x[b, S.data_idx] = pay.symbols((tag, b), S.data_idx.size)
            x[b, S.m0] = np.sqrt(S.Ep)
        else:
            x[b] = pay.symbols((tag, b), S.N)
    return x


def make_codec():
    """Decoder-aided decisions with CRC gating: a block whose CRC passes feeds
    back re-encoded symbols; a failed block feeds back None (the tracker then
    uses its soft symbols and flags the block)."""
    def codec(kind, zu, v):
        C = BlockCode(zu.size)
        u = C.demod_decode(zu, v)
        return u, (C.modulate(u) if C.check(u) else None)
    return codec


def awgn(rng, shape, sigma2):
    return np.sqrt(sigma2 / 2) * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------
def gmi_qpsk(zu, v, x):
    """Empirical BICM GMI (bits) per symbol for unit-energy QPSK with the
    receiver's own Gaussian metric (mismatched decoding)."""
    llr_r = 2 * np.sqrt(2) * zu.real / v
    llr_i = 2 * np.sqrt(2) * zu.imag / v
    sr = np.sign(x.real); si = np.sign(x.imag)
    t = np.logaddexp(0, -sr * llr_r) + np.logaddexp(0, -si * llr_i)
    return 2.0 - np.sum(t) / np.log(2) / x.size


def score(xh, zu, v, x, idx):
    err = int(np.sum(np.abs(xh - x[idx]) > 1e-6))
    g = max(gmi_qpsk(zu, v, x[idx]), 0.0)
    return err, g * idx.size, idx.size


# ----------------------------------------------------------------------------
# estimators shared by receivers
# ----------------------------------------------------------------------------
class Acquirer:
    """Chirp-domain multi-block acquisition on the pilot window W (AFDM)."""

    def __init__(self, S: MBAFDM, kstep=0.02, kpad=0.6):
        self.S = S
        self.L = S.ell_max + 1
        self.KG = np.arange(-S.alpha_max - kpad, S.alpha_max + kpad + 1e-9, kstep)
        A = S.atoms(range(self.L), self.KG)
        self.An = A / np.linalg.norm(A, axis=-1, keepdims=True)
        self.kstep = kstep
        # guard chirps outside W: noise (+ tiny leakage) only -> noise estimate
        self.G = np.setdiff1d(S.zero_set, S.W)
        self.G = self.G[self.G != S.m0]

    def atom(self, l, k):
        return self.S.atoms([l], [k])[0, 0]

    def ph(self, k, B):
        return np.exp(1j * 2 * np.pi * k * np.arange(B) * self.S.beta)

    def design(self, paths, B):
        return np.stack([np.outer(self.ph(k, B), self.atom(l, k)).reshape(-1) for l, k in paths], 1)

    def ls(self, y, paths):
        D = self.design(paths, y.shape[0])
        g, *_ = np.linalg.lstsq(D, y.reshape(-1), rcond=None)
        return g, (y.reshape(-1) - D @ g).reshape(y.shape)

    def s_nc(self, r, l, k):
        a = self.atom(l, k)
        return np.sum(np.abs(r @ np.conj(a)) ** 2) / np.vdot(a, a).real

    def s_coh(self, r, l, k):
        a = self.atom(l, k)
        return np.abs(np.conj(self.ph(k, r.shape[0])) @ (r @ np.conj(a))) ** 2 / np.vdot(a, a).real

    @staticmethod
    def _max1d(f, lo, hi, n):
        from scipy.optimize import minimize_scalar
        kk = np.linspace(lo, hi, n)
        v = [f(k) for k in kk]
        i = int(np.argmax(v))
        h = (hi - lo) / (n - 1)
        return minimize_scalar(lambda k: -f(k), bounds=(kk[i] - h, kk[i] + h), method="bounded",
                               options={"xatol": 1e-7}).x

    def fit_kappa(self, r, l, k0):
        B = r.shape[0]
        kc = self._max1d(lambda k: self.s_nc(r, l, k), k0 - self.kstep, k0 + self.kstep, 5)
        if B == 1:
            return kc
        half = 0.5 / self.S.beta
        n = int(np.ceil(2 * half * B * self.S.beta * 8)) + 1
        return self._max1d(lambda k: self.s_coh(r, l, k), kc - half, kc + half, n)

    def noise_var(self, Yfull):
        return float(np.mean(np.abs(Yfull[:, self.G]) ** 2))

    def run(self, Yfull, P_max, P_known=None, pfa=1e-3, sweeps=2, sigma2=None):
        """Greedy extraction; order = P_known if given, else CFAR on the
        non-coherent map with the guard-based noise estimate."""
        y = Yfull[:, self.S.W]
        B = y.shape[0]
        s2 = self.noise_var(Yfull) if sigma2 is None else sigma2
        cells = self.L * len(self.KG)
        thr = s2 * chi2.isf(pfa / cells, 2 * B) / 2
        paths = []
        r = y.copy()
        cap = P_known if P_known is not None else P_max
        for _ in range(cap):
            C = np.einsum("lkw,bw->lkb", np.conj(self.An), r)
            s = np.sum(np.abs(C) ** 2, -1)
            l, i = np.unravel_index(np.argmax(s), s.shape)
            if P_known is None and s[l, i] < thr and paths:
                break                              # (the strongest candidate is always kept)
            k = self.fit_kappa(r, l, self.KG[i])
            paths.append((int(l), float(k)))
            _, r = self.ls(y, paths)
        if not paths:
            return [], np.zeros(0, complex)
        for _ in range(sweeps):
            for p in range(len(paths)):
                others = paths[:p] + paths[p + 1:]
                rp = self.ls(y, others)[1] if others else y
                l = paths[p][0]
                paths[p] = (l, float(self.fit_kappa(rp, l, paths[p][1])))
        g, _ = self.ls(y, paths)
        return paths, g / np.sqrt(self.S.Ep)


def known_block_acquire(S: MBAFDM, Y, X, blocks, P, kmax, kstep=0.02, sigma2=None, pfa=1e-3):
    """Waveform-agnostic acquisition from fully known blocks (OFDM training):
    greedy matched filter over (integer delay, Doppler) with LS refits, stopped
    by a CFAR test against the noise level (the strongest path is always kept)."""
    KG = np.arange(-kmax - 0.6, kmax + 0.6 + 1e-9, kstep)
    y = Y.reshape(-1)
    paths = []
    r = y.copy()
    for _ in range(P):
        best = (-1, None)
        for l in range(S.ell_max + 1):
            # vectorised over Doppler: correlate r with the block responses
            sc = []
            for k in KG:
                c = apply_path(S, l, k, X, blocks).reshape(-1)
                sc.append(np.abs(np.vdot(c, r)) ** 2 / np.vdot(c, c).real)
            i = int(np.argmax(sc))
            if sc[i] > best[0]:
                best = (sc[i], (l, KG[i]))
        cells = (S.ell_max + 1) * len(KG) * kstep          # ~independent cells (1-bin resolution)
        if sigma2 is not None and paths and best[0] < sigma2 * -np.log(pfa / cells):
            break
        paths.append((int(best[1][0]), float(best[1][1])))
        D = np.stack([apply_path(S, l, k, X, blocks).reshape(-1) for l, k in paths], 1)
        g, *_ = np.linalg.lstsq(D, y, rcond=None)
        r = y - D @ g
    return paths


# ----------------------------------------------------------------------------
# receivers
# ----------------------------------------------------------------------------
def detect_frame(S, Y, kinds, est_of_block, sigma2, xknown=None):
    """LMMSE-detect each block with the channel est_of_block(b) -> (ells, kaps, h)."""
    pil = np.zeros(S.N, bool); pil[S.zero_set] = True
    xp = np.zeros(S.N, complex); xp[S.m0] = np.sqrt(S.Ep)
    out = []
    for b, k in enumerate(kinds):
        ells, kaps, h = est_of_block(b)
        H = channel_matrix(S, b, ells, kaps, h)
        if k == "T":
            out.append(None); continue
        if k == "P":
            idx = S.data_idx
            hd, _, _, _, zu, v = lmmse_soft(H, Y[b], pil, xp, sigma2, idx)
        else:
            idx = np.arange(S.N)
            hd, _, _, _, zu, v = lmmse_soft(H, Y[b], None, None, sigma2, idx)
        out.append((hd, zu, v, idx))
    return out


def simulate(cfg: Config, seed: int):
    rng = np.random.default_rng(seed)
    S = make_system(cfg)
    sigma2 = 10 ** (-cfg.snr_db / 10)
    ch = draw_channel(cfg, rng)
    chan = lambda st: S.channel(st, ch["tau"], ch["kappa"], ch["h"], rho=ch["rho"], born=ch["born"],
                                cfo=cfg.cfo, pn_var=cfg.pn_var, rng=rng)
    B, Bp = cfg.B, cfg.Bp
    res = {"seed": seed}
    timing = {}

    def tally(name, dets, x, t0, tag="A"):
        e, g, n, bl, ib = [], [], [], [], []
        for b, d in enumerate(dets):
            if d is None:
                e.append(0); g.append(0.0); n.append(0); bl.append(0); ib.append(0); continue
            hd, zu, v, idx = d[:4]
            eb, gb, nb = score(hd, zu, v, x[b], idx)
            e.append(eb); g.append(gb); n.append(nb)
            if cfg.coded:
                u = pay.bits[(tag, b)]
                uh = d[4] if len(d) > 4 else BlockCode(idx.size).demod_decode(zu, v)
                ok = bool(np.array_equal(uh, u))
                bl.append(0 if ok else 1); ib.append(u.size - 16 if ok else 0)
        res[name] = {"err": e, "gmi": g, "n": n}
        if cfg.coded:
            res[name].update({"blerr": bl, "goodbits": ib})
        timing[name] = time.time() - t0

    kinds = "P" * Bp + "D" * (B - Bp)
    pay = Payload(cfg.coded, rng)
    x = build_frame(S, kinds, pay, "A")
    Y = S.receive(chan(S.transmit(x)), B)
    Y = Y + awgn(rng, Y.shape, sigma2)
    acq = Acquirer(S)
    P_known = cfg.P if cfg.P_max is None else None
    P_cap = cfg.P if cfg.P_max is None else cfg.P_max

    if "genie" in cfg.receivers:
        t0 = time.time()
        # genie uses the true channel at every block (incl. drift via block-centred kappa)
        if cfg.frac_delay or cfg.rho_max > 0 or cfg.pn_var > 0 or cfg.born_frac > 0 or cfg.cfo:
            dets = _genie_dense(S, ch, cfg, Y, kinds, sigma2)
        else:
            dets = detect_frame(S, Y, kinds, lambda b: (ch["tau"], ch["kappa"], ch["h"]), sigma2)
        tally("genie", dets, x, t0)

    if "conv" in cfg.receivers:
        t0 = time.time()
        xc = build_frame(S, "P" * B, pay, "C")
        Yc = S.receive(chan(S.transmit(xc)), B) + awgn(rng, (B, S.N), sigma2)
        ests = []
        Tpol = Tracker(S, est_delay=True)
        xpil = np.zeros((1, S.N), complex); xpil[0, S.m0] = np.sqrt(S.Ep)
        for b in range(B):
            paths, h = acq.run(Yc[b:b + 1], P_cap, P_known, sigma2=sigma2)
            if cfg.frac_delay and paths:
                pr, g, _ = Tpol.refit(Yc[b:b + 1], xpil, [b], paths, rows=S.W)
                ests.append(([p[0] for p in pr], [p[1] for p in pr], g))
                continue
            kk = np.array([p[1] for p in paths])
            ests.append(([p[0] for p in paths], kk, h * np.exp(-1j * 2 * np.pi * kk * b * S.beta)))
        tally("conv", detect_frame(S, Yc, "P" * B, lambda b: ests[b], sigma2), xc, t0, "C")

    if "sp" in cfg.receivers:
        t0 = time.time()
        dsp, xsp = _superimposed(S, cfg, chan, rng, sigma2, acq, P_cap, P_known, pay)
        tally("sp", dsp, xsp, t0, "S")

    if "openloop" in cfg.receivers or "track" in cfg.receivers:
        t0 = time.time()
        paths, h = acq.run(Y[:Bp], P_cap, P_known, sigma2=sigma2)
        t_acq = time.time() - t0
        mk = lambda: Tracker(S, soft=True, window=cfg.window, redetect=True,
                             est_delay=cfg.frac_delay, model_rho=cfg.model_rho,
                             reacq_every=cfg.reacq_every, kmax=cfg.kappa_max + 0.5,
                             P_cap=P_cap, aperture_res=cfg.aperture_res, reacq_level=cfg.reacq_level)
        if "openloop" in cfg.receivers:
            t1 = time.time()
            # same acquisition, one refit over the pilot blocks (pilots + soft data),
            # then pure phase extrapolation: the AFDM transplant of open-loop
            # inter-frame prediction (cf. Zak-OTFS prediction, Ubadah & Mohammed 2026)
            T = mk()
            pk = "P" * Bp
            d0, _ = T.run_full(Y[:Bp], pk, paths, h, sigma2)   # pilot blocks only
            Xs = np.zeros((Bp, S.N), complex)
            for b in range(Bp):
                Xs[b, S.m0] = np.sqrt(S.Ep)
                Xs[b, S.data_idx] = d0[b][0]
            pr, hr, _ = T.refit(Y[:Bp], Xs, np.arange(Bp), paths)
            e = ([p[0] for p in pr], [p[1] for p in pr], hr)
            dets = detect_frame(S, Y, kinds, lambda b: e, sigma2)
            tally("openloop", dets, x, t1)
            timing["openloop"] += t_acq
        if "track" in cfg.receivers:
            t1 = time.time()
            T = mk()
            if cfg.decision == "hard":
                T.soft = False
            codec = make_codec() if (cfg.coded and cfg.decision == "decoded") else None
            dets, traj = T.run_full(Y, kinds, paths, h, sigma2, codec=codec)
            tally("track", dets, x, t1)
            timing["track"] += t_acq
            res["traj"] = [(b, list(k), list(np.abs(g))) for b, k, g in traj]

    mkT = lambda: Tracker(S, soft=True, window=cfg.window, redetect=True, est_delay=cfg.frac_delay,
                          model_rho=cfg.model_rho, reacq_every=cfg.reacq_every, kmax=cfg.kappa_max + 0.5,
                          P_cap=P_cap, aperture_res=cfg.aperture_res, reacq_level=cfg.reacq_level)
    codec_da = make_codec() if cfg.coded else None

    if "conv-da" in cfg.receivers or "genie-conv" in cfg.receivers:
        # conventional frame (pilot + guard in every block), same seeds -> same channel
        t0 = time.time()
        xc2 = build_frame(S, "P" * B, pay, "C2")
        Yc2 = S.receive(chan(S.transmit(xc2)), B) + awgn(rng, (B, S.N), sigma2)
        if "genie-conv" in cfg.receivers:
            tally("genie-conv", _genie_dense(S, ch, cfg, Yc2, "P" * B, sigma2), xc2, t0, "C2")
        if "conv-da" in cfg.receivers:
            t0 = time.time()
            dets = []
            for b in range(B):
                # each block on its own (block-relative time), with the same data-aided
                # iterations as every other receiver: decode, re-acquire, refit, re-detect
                paths, h = acq.run(Yc2[b:b + 1], P_cap, P_known, sigma2=sigma2)
                d, _ = mkT().run_full(Yc2[b:b + 1], "P", paths, h, sigma2, codec=codec_da)
                dets.append(d[0])
            tally("conv-da", dets, xc2, t0, "C2")

    if "sp-track" in cfg.receivers:
        # superimposed pilot in every block, processed by the same multi-block tracker
        t0 = time.time()
        eps = cfg.sp_eps
        a_d = np.sqrt(1 - eps)
        d = np.stack([pay.symbols(("ST", b), S.N) for b in range(B)])
        xs = a_d * d
        xs[:, S.m0] += np.sqrt(eps * S.N)
        Ys = S.receive(chan(S.transmit(xs)), B) + awgn(rng, (B, S.N), sigma2)
        Ep0 = S.Ep; S.Ep = eps * S.N
        K = max(1, int(cfg.sp_acq_blocks))                 # every block carries the pilot:
        paths, h = acq.run(Ys[:K], P_cap, P_known)          # acquire over the first K (data + noise level)
        S.Ep = Ep0
        dets, _ = mkT().run_full(Ys, "S" * B, paths, h, sigma2, codec=codec_da, sp_eps=eps, n_acq=K)
        tally("sp-track", dets, d, t0, "ST")

    for K in (4, 8):
        name = f"interp{K}"
        if name not in cfg.receivers:
            continue
        # periodic pilot blocks every K blocks; parameters fit jointly on all pilot blocks
        # (tracked across them), then every data block is detected with that fit:
        # non-causal parametric interpolation, requiring a frame of buffering
        t0 = time.time()
        ik = "".join("P" if b % K == 0 else "D" for b in range(B))
        xi = build_frame(S, ik, pay, name)
        Yi = S.receive(chan(S.transmit(xi)), B) + awgn(rng, (B, S.N), sigma2)
        pb = np.array([b for b in range(B) if ik[b] == "P"])
        paths, h = acq.run(Yi[:1], P_cap, P_known, sigma2=sigma2)
        T = mkT()
        dp, _ = T.run_full(Yi[pb], "P" * len(pb), paths, h, sigma2, codec=codec_da, blocks_abs=pb, n_acq=1)
        fp, fh = T.final
        e = ([p[0] for p in fp], [p[1] for p in fp], fh, [p[2] for p in fp])
        dd = detect_frame(S, Yi, ik, lambda b: e[:3], sigma2)
        dets = []
        for b in range(B):
            if ik[b] == "P":
                dets.append(dp[int(np.where(pb == b)[0][0])])
            else:
                hd, zu, v, idx = dd[b]
                dets.append((hd, zu, v, idx) + ((codec_da("D", zu, v)[0],) if codec_da else ()))
        tally(name, dets, xi, t0, name)

    if cfg.ofdm:
        t0 = time.time()
        So = ofdm_system(cfg)
        okinds = "T" * Bp + "D" * (B - Bp)
        xo = np.zeros((B, So.N), complex)
        for b in range(B):
            xo[b] = So.data_block(rng) if b < Bp else pay.symbols(("O", b), So.N)
        Yo = So.receive(chan(So.transmit(xo)), B) + awgn(rng, (B, So.N), sigma2)
        tb = np.arange(Bp)
        paths = known_block_acquire(So, Yo[:Bp], xo[:Bp], tb, P_cap, cfg.kappa_max, sigma2=sigma2)
        T = Tracker(So, soft=True, window=cfg.window, redetect=True,
                    est_delay=cfg.frac_delay, model_rho=cfg.model_rho,
                    reacq_every=cfg.reacq_every, kmax=cfg.kappa_max + 0.5, P_cap=P_cap,
                    aperture_res=cfg.aperture_res, reacq_level=cfg.reacq_level)
        paths, g, _ = T.refit(Yo[:Bp], xo[:Bp], tb, paths)
        codec = make_codec() if (cfg.coded and cfg.decision == "decoded") else None
        if cfg.decision == "hard":
            T.soft = False
        dets, _ = T.run_full(Yo, okinds, paths, g, sigma2, xknown=xo, codec=codec)
        tally("ofdm-track", dets, xo, t0, "O")

    res["timing"] = timing
    res["true"] = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in ch.items()
                   if k in ("tau", "kappa", "rho", "born")}
    res["true"]["h_abs"] = np.abs(ch["h"]).tolist()
    return res


def _block_matrix_exact(S, ch, b, cfo=0.0, taps=24):
    """Exact DAFT-domain matrix of block b under the physical channel of
    mbafdm.MBAFDM.channel (same interpolation kernel, Doppler rate, births, CFO),
    built directly: CP insertion -> per-path delay filter -> absolute-time phase ->
    CP removal. Equals probing the channel with all N unit chirp vectors."""
    N, Ncp = S.N, S.Ncp
    Tb = N + Ncp
    L = Tb
    n_abs = b * Tb + np.arange(L)                       # absolute indices of block b (incl. CP)
    C = np.zeros((L, N)); C[np.arange(L), (np.arange(L) - Ncp) % N] = 1.0   # CP insertion
    G = np.zeros((L, L), complex)
    for p in range(len(ch["tau"])):
        if ch["born"][p] > b:
            continue
        t = float(ch["tau"][p])
        D = np.zeros((L, L))
        if abs(t - round(t)) < 1e-12:
            d = int(round(t))
            idx = np.arange(d, L)
            D[idx, idx - d] = 1.0
        else:
            k = np.arange(-taps, taps + 1) + int(np.floor(t))
            w = np.kaiser(2 * taps + 1, 8.0)
            gk = np.sinc(k - t) * w
            for kk, gg in zip(k, gk):
                if kk >= 0:
                    idx = np.arange(kk, L); D[idx, idx - kk] += gg
                else:
                    idx = np.arange(0, L + kk); D[idx, idx - kk] += gg
        ph = 2 * np.pi * (ch["kappa"][p] * n_abs + 0.5 * ch["rho"][p] * n_abs * n_abs / Tb) / N
        G += ch["h"][p] * (np.exp(1j * ph)[:, None] * D)
    if cfo:
        G = np.exp(1j * 2 * np.pi * cfo * n_abs / N)[:, None] * G
    Tm = G[Ncp:, :] @ C                                 # N x N time-domain block map
    FA = S.daft(np.eye(N)).T
    return FA @ Tm @ FA.conj().T


def _genie_dense(S, ch, cfg, Y, kinds, sigma2):
    """Genie for any channel: block b's exact matrix (deterministic impairments
    included; random phase noise is not known to the genie)."""
    out = []
    pil = np.zeros(S.N, bool); pil[S.zero_set] = True
    xp = np.zeros(S.N, complex); xp[S.m0] = np.sqrt(S.Ep)
    for b, k in enumerate(kinds):
        H = _block_matrix_exact(S, ch, b, cfo=cfg.cfo)
        if k == "P":
            idx = S.data_idx
            hd, _, _, _, zu, v = lmmse_soft(H, Y[b], pil, xp, sigma2, idx)
        else:
            idx = np.arange(S.N)
            hd, _, _, _, zu, v = lmmse_soft(H, Y[b], None, None, sigma2, idx)
        out.append((hd, zu, v, idx))
    return out


def _genie_probe_matrix(S, ch, cfg, b):
    """Reference: probe the simulated channel with every unit chirp vector (tests)."""
    s = S.idaft(np.eye(S.N, dtype=complex))
    probes = np.concatenate([s[:, -S.Ncp:], s], 1)
    r = S.channel(probes, ch["tau"], ch["kappa"], ch["h"], rho=ch["rho"], born=ch["born"],
                  cfo=cfg.cfo, n0=b * (S.N + S.Ncp))
    return S.receive(r, 1)[:, 0, :].T


def _superimposed(S, cfg, chan, rng, sigma2, acq, P_cap, P_known, pay):
    """Superimposed-pilot AFDM, every block (adaptation of Zheng et al., TVT 2025,
    and of the data-aided loop of D-GESBL, Luo et al., TCOM 2026).

    Data on all N chirps scaled by sqrt(1-eps) plus a pilot chirp m0 of energy
    eps*N. Per block: pilot-window acquisition under data interference, then
    `sp_iters` rounds of [LMMSE detection -> (decoding, CRC) -> data-aided
    re-acquisition of missed paths -> full-block parametric refit], using the
    same estimation machinery as the proposed tracker, so the comparison isolates
    the frame design rather than the estimator quality."""
    from mbtrack import apply_path
    B = cfg.B
    eps = cfg.sp_eps
    a_d = np.sqrt(1 - eps)
    Ep = eps * S.N
    d = np.stack([pay.symbols(("S", b), S.N) for b in range(B)])
    x = a_d * d
    x[:, S.m0] += np.sqrt(Ep)
    Y = S.receive(chan(S.transmit(x)), B) + awgn(rng, (B, S.N), sigma2)
    codec = make_codec() if cfg.coded else None
    T = Tracker(S, est_delay=cfg.frac_delay, gn_iters=4, kmax=cfg.kappa_max + 0.5, P_cap=P_cap)
    idx = np.arange(S.N)
    dets = []
    for b in range(B):
        Ep0 = S.Ep
        S.Ep = Ep
        paths, h = acq.run(Y[b:b + 1], P_cap, P_known)          # CFAR level measured: data + noise
        S.Ep = Ep0
        kk = np.array([p[1] for p in paths]) if paths else np.zeros(0)
        g = np.asarray(h) * np.exp(-1j * 2 * np.pi * kk * b * S.beta)   # absolute-time gains
        paths = [(p[0], p[1], 0.0) for p in paths]
        for it in range(cfg.sp_iters + 1):
            H = channel_matrix(S, b, [p[0] for p in paths], [p[1] for p in paths], g)
            r = Y[b] - H[:, S.m0] * np.sqrt(Ep)
            hd, xm, xv, _, zu, v = lmmse_soft(H * a_d, r, None, None, sigma2, idx)
            if it == cfg.sp_iters:
                det = (hd, zu, v, idx)
                if codec is not None:
                    det = det + (codec("D", zu, v)[0],)
                dets.append(det)
                break
            xs = xm
            if codec is not None:
                _, xr = codec("D", zu, v)
                if xr is not None:
                    xs = xr
            X = (a_d * xs)[None, :].astype(complex)
            X[0, S.m0] += np.sqrt(Ep)
            for _ in range(P_cap):                      # add paths the pilot window missed
                res = Y[b:b + 1] - sum(gg * apply_path(S, p[0], p[1], X, [b]) for p, gg in zip(paths, g)) \
                    if paths else Y[b:b + 1]
                lvl = float(np.mean(np.abs(res) ** 2))
                new = T.reacquire(Y[b:b + 1], X, [b], paths, g, lvl)
                if len(new) == len(paths):
                    break
                paths, g, _ = T.refit(Y[b:b + 1], X, [b], new)
            if paths:
                paths, g, _ = T.refit(Y[b:b + 1], X, [b], paths)
    return dets, d
