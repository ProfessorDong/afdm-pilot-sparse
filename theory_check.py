"""Numerical validation of the closed forms used in the paper (single path).

E1  grating-lobe level of the multi-block chirp-domain ambiguity at m/beta:
      |chi_1(m/beta)| = |sin(pi m Ncp/(N+Ncp))| / (N sin(pi m/(N+Ncp)))   (exact,
      full-block atom) versus the atom correlation measured on the pilot window.
E2  correct-cell probability of acquisition from Bp pilot blocks:
      P_cell ~ P_global(gamma_1, Bp, M) x [1 - 2 Q( 1/(2 beta sigma_nc) )],
      sigma_nc^2 = 3 / (2 pi^2 Bp gamma_1), M = independent search cells,
      gamma_1 = Ep |h|^2 / sigma^2 (per-block pilot SNR).
E3  prediction NMSE of the path coefficient h exp(j 2pi kappa t) over block b':
      NMSE(b') = (1/(2 gamma)) [ 2 + E_{t in b'}(t - tbar)^2 / sigma_t^2 ],
      gamma, tbar, sigma_t^2 = energy-weighted SNR, mean and variance of the
      known-sample times (t in units of N samples).
    (a) open loop from Bp pilot blocks; (b) tracking with correct decisions,
      one-step-ahead prediction after each appended data block.

Outputs runs/theory_check.json.
"""
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from scipy.stats import norm  # noqa: E402

from mbafdm import MBAFDM  # noqa: E402
from mbtrack import Tracker, apply_path  # noqa: E402
from engine import Acquirer  # noqa: E402

OUT = Path(__file__).resolve().parent / "runs"


# ---------------------------------------------------------------- E1
def e1():
    rows = []
    for N in (256, 512, 1024):
        for Ncp in (4, 8, 16, 32, 64):
            S = MBAFDM(N=N, Ncp=Ncp, ell_max=min(7, Ncp), alpha_max=3, xi=2)
            for m in (1, 2, 3):
                d = m / S.beta
                exact = abs(np.sin(np.pi * m * Ncp / (N + Ncp))) / (N * np.sin(np.pi * m / (N + Ncp)))
                # full-block atom (all N chirp bins) and pilot-window atom
                e = np.zeros(N, complex); e[S.m0] = 1
                s = np.roll(S.idaft(e), 3)
                n = np.arange(Ncp, Ncp + N)
                full = lambda k: S.daft(s * np.exp(1j * 2 * np.pi * k * n / N))
                a0, a1 = full(0.3), full(0.3 + d)
                meas_full = abs(np.vdot(a0, a1)) / np.vdot(a0, a0).real
                w0, w1 = a0[S.W], a1[S.W]
                meas_win = abs(np.vdot(w0, w1)) / np.sqrt(np.vdot(w0, w0).real * np.vdot(w1, w1).real)
                rows.append(dict(N=N, Ncp=Ncp, m=m, exact=exact, ratio=Ncp / N,
                                 meas_full=meas_full, meas_win=meas_win))
    return rows


# ---------------------------------------------------------------- E2
S0 = MBAFDM()
ACQ = Acquirer(S0)


def e2_trial(args):
    gamma1_db, Bp, seed = args
    rng = np.random.default_rng(seed)
    S = S0
    l = int(rng.integers(0, S.ell_max + 1))
    k = rng.uniform(-S.alpha_max, S.alpha_max)
    h = np.exp(1j * rng.uniform(0, 2 * np.pi))
    sigma2 = S.Ep / 10 ** (gamma1_db / 10)
    x = np.zeros((Bp, S.N), complex); x[:, S.m0] = np.sqrt(S.Ep)       # pilot only
    Y = S.receive(S.channel(S.transmit(x), [l], [k], [h]), Bp)
    Y = Y + np.sqrt(sigma2 / 2) * (rng.standard_normal(Y.shape) + 1j * rng.standard_normal(Y.shape))
    y = Y[:, S.W]
    # non-coherent coarse estimate (the stage whose cell decides acquisition)
    C = np.einsum("lkw,bw->lkb", np.conj(ACQ.An), y)
    sc = np.sum(np.abs(C) ** 2, -1)
    li, i = np.unravel_index(np.argmax(sc), sc.shape)
    kc = ACQ._max1d(lambda kk: ACQ.s_nc(y, li, kk), ACQ.KG[i] - ACQ.kstep, ACQ.KG[i] + ACQ.kstep, 5)
    kf = ACQ.fit_kappa(y, li, ACQ.KG[i])
    return dict(g=gamma1_db, Bp=Bp, delay_ok=bool(li == l), cell=bool(li == l and abs(kc - k) < 0.5 / S.beta),
                final_err=float(kf - k) if li == l else None, nc_err=float(kc - k) if li == l else None)


# independent search cells: integer delays x Doppler span of the grid (1-bin resolution)
M_EFF = (S0.ell_max + 1) * 2 * (S0.alpha_max + 0.6)


def p_global(gam, B, M=M_EFF):
    """Pr(true-cell non-coherent score exceeds the largest of M noise cells):
    true ~ noncentral chi2(2B, 2B gam)/2, noise ~ Gamma(B, 1), in units of sigma^2."""
    from scipy import integrate, stats
    f = lambda s: stats.ncx2.pdf(2 * s, 2 * B, 2 * B * gam) * 2 * stats.gamma.cdf(s, B) ** M
    mu, sd = B * (1 + gam), np.sqrt(B * (1 + 2 * gam))       # moments of the true-cell score
    lo, hi = max(0.0, mu - 12 * sd), mu + 12 * sd             # the integrand is negligible outside
    return integrate.quad(f, lo, hi, limit=400, points=[mu])[0]


def e2(trials=600):
    jobs = [(g, Bp, 31 * t + 7 * Bp + 1000 * int(g + 20)) for g in (-6, -3, 0, 3, 6, 9, 12)
            for Bp in (1, 2, 4) for t in range(trials)]
    with Pool(18) as p:
        res = p.map(e2_trial, jobs, chunksize=8)
    rows = []
    for g in sorted({r["g"] for r in res}):
        for Bp in (1, 2, 4):
            rr = [r for r in res if r["g"] == g and r["Bp"] == Bp]
            gam = 10 ** (g / 10)
            snc = np.sqrt(3 / (2 * np.pi ** 2 * Bp * gam))
            loc = 1 - 2 * norm.sf(0.5 / (S0.beta * snc))
            pred = p_global(gam, Bp) * loc
            ok = [r["nc_err"] for r in rr if r["nc_err"] is not None]
            fe = [r["final_err"] for r in rr if r["cell"]]
            rows.append(dict(gamma1_db=g, Bp=Bp, p_cell=float(np.mean([r["cell"] for r in rr])),
                             p_delay=float(np.mean([r["delay_ok"] for r in rr])), p_cell_formula=float(pred),
                             p_cell_local=float(loc), M_eff=M_EFF,
                             sigma_nc_formula=float(snc), sigma_nc_meas=float(np.std(ok)) if ok else None,
                             rmse_final_in_cell=float(np.sqrt(np.mean(np.square(fe)))) if fe else None,
                             trials=len(rr)))
    return rows


# ---------------------------------------------------------------- E3
def times_and_weights(S, X, blocks):
    """Known-sample times (units of N samples) and energy weights |s_n|^2."""
    s = S.idaft(X)
    Tb = S.N + S.Ncp
    t = (np.asarray(blocks) * Tb + S.Ncp)[:, None] + np.arange(S.N)
    return t.reshape(-1) / S.N, np.abs(s.reshape(-1)) ** 2


def nmse_formula(S, X, blocks, h, sigma2, bnext):
    t, w = times_and_weights(S, X, blocks)
    gamma = np.sum(w) * abs(h) ** 2 / sigma2
    tbar = np.sum(w * t) / np.sum(w)
    st2 = np.sum(w * (t - tbar) ** 2) / np.sum(w)
    tt = (bnext * (S.N + S.Ncp) + S.Ncp + np.arange(S.N)) / S.N
    return (2 + np.mean((tt - tbar) ** 2) / st2) / (2 * gamma)


def nmse_emp(S, l, k, h, kh, hh, bnext):
    tt = (bnext * (S.N + S.Ncp) + S.Ncp + np.arange(S.N)) / S.N
    c = h * np.exp(1j * 2 * np.pi * k * tt)
    ch = hh * np.exp(1j * 2 * np.pi * kh * tt)
    return float(np.mean(np.abs(ch - c) ** 2) / abs(h) ** 2)


def e3_trial(args):
    gamma1_db, Bp, seed, horizon, track_blocks = args
    rng = np.random.default_rng(seed)
    S = S0
    l = int(rng.integers(0, S.ell_max + 1))
    k = rng.uniform(-S.alpha_max, S.alpha_max)
    h = np.exp(1j * rng.uniform(0, 2 * np.pi))
    sigma2 = S.Ep / 10 ** (gamma1_db / 10)
    Btot = Bp + max(horizon, track_blocks)
    x = np.zeros((Btot, S.N), complex)
    x[:Bp, S.m0] = np.sqrt(S.Ep)                                   # pilot-only pilot blocks
    x[Bp:] = MBAFDM.qpsk(rng, (Btot - Bp, S.N))                    # data blocks (known: correct decisions)
    Y = S.receive(S.channel(S.transmit(x), [l], [k], [h]), Btot)
    Y = Y + np.sqrt(sigma2 / 2) * (rng.standard_normal(Y.shape) + 1j * rng.standard_normal(Y.shape))
    T = Tracker(S, gn_iters=6)
    pb = np.arange(Bp)
    paths, g = ACQ.run(Y[:Bp], 1, 1, sigma2=sigma2)
    if not paths or paths[0][0] != l or abs(paths[0][1] - k) > 0.5 / S.beta:
        return None                                                 # outside the local (in-cell) regime
    pr, gg, _ = T.refit(Y[:Bp], x[:Bp], pb, paths, rows=S.W)
    ol = [(b, nmse_emp(S, l, k, h, pr[0][1], gg[0], b), nmse_formula(S, x[:Bp], pb, h, sigma2, b))
          for b in range(Bp, Bp + horizon)]
    tr = []
    pc = pr
    for b in range(Bp, Bp + track_blocks):
        blocks = np.arange(b)                                       # aperture before block b
        # rows=None uses every chirp; pilot blocks contribute only their pilot (data positions zero)
        pc, gc, _ = T.refit(Y[:b], x[:b], blocks, pc)
        tr.append((b, nmse_emp(S, l, k, h, pc[0][1], gc[0], b), nmse_formula(S, x[:b], blocks, h, sigma2, b)))
    return dict(g=gamma1_db, Bp=Bp, ol=ol, tr=tr)


def e3(trials=300, horizon=24, track_blocks=24):
    jobs = [(g, Bp, 97 * t + 11 * Bp + 5000 * int(g + 20), horizon, track_blocks)
            for g in (10, 20) for Bp in (1, 2, 4) for t in range(trials)]
    with Pool(18) as p:
        res = [r for r in p.map(e3_trial, jobs, chunksize=4) if r is not None]
    rows = []
    for g in (10, 20):
        for Bp in (1, 2, 4):
            rr = [r for r in res if r["g"] == g and r["Bp"] == Bp]
            if not rr:
                continue
            ol = np.array([[e for _, e, _ in r["ol"]] for r in rr])
            olf = np.array([[f for _, _, f in r["ol"]] for r in rr])
            tr = np.array([[e for _, e, _ in r["tr"]] for r in rr])
            trf = np.array([[f for _, _, f in r["tr"]] for r in rr])
            rows.append(dict(gamma1_db=g, Bp=Bp, trials=len(rr),
                             lag=[b for b, _, _ in rr[0]["ol"]],
                             ol_emp=ol.mean(0).tolist(), ol_formula=olf.mean(0).tolist(),
                             tr_block=[b for b, _, _ in rr[0]["tr"]],
                             tr_emp=tr.mean(0).tolist(), tr_formula=trf.mean(0).tolist()))
    return rows


# ---------------------------------------------------------------- E4 / E5
# E4: a weak path next to a strong one: is it RETAINED by the CFAR stopping rule
#     AND in the correct cell?  Approximation (A4): the global integral starts at
#     the detection threshold T (noise-variance units) instead of zero.
# E5: false insertions of the acquisition under noise only and under one strong
#     path (the first candidate is always kept: a nonempty channel is assumed).
CELLS = S0.ell_max + 1, len(ACQ.KG)


def p_retained(gam, B, T, M=M_EFF):
    from scipy import integrate, stats
    f = lambda s: stats.ncx2.pdf(2 * s, 2 * B, 2 * B * gam) * 2 * stats.gamma.cdf(s, B) ** M
    mu, sd = B * (1 + gam), np.sqrt(B * (1 + 2 * gam))
    lo, hi = max(T, mu - 12 * sd), max(T, mu + 12 * sd)
    return integrate.quad(f, lo, hi, limit=400)[0] if hi > lo else 0.0


def acq_threshold(B, pfa=1e-3):
    from scipy.stats import chi2
    return chi2.isf(pfa / (CELLS[0] * CELLS[1]), 2 * B) / 2


def e4_trial(args):
    gamma1_db, seed = args
    rng = np.random.default_rng(seed)
    S = S0
    ls, lw = rng.choice(S.ell_max + 1, 2, replace=False)
    ks, kw = rng.uniform(-S.alpha_max, S.alpha_max, 2)
    sigma2 = 1.0
    hs = np.sqrt(10 ** (25 / 10) * sigma2 / S.Ep) * np.exp(1j * rng.uniform(0, 2 * np.pi))
    hw = np.sqrt(10 ** (gamma1_db / 10) * sigma2 / S.Ep) * np.exp(1j * rng.uniform(0, 2 * np.pi))
    x = np.zeros((1, S.N), complex); x[0, S.m0] = np.sqrt(S.Ep)
    Y = S.receive(S.channel(S.transmit(x), [ls, lw], [ks, kw], [hs, hw]), 1)
    Y = Y + np.sqrt(sigma2 / 2) * (rng.standard_normal(Y.shape) + 1j * rng.standard_normal(Y.shape))
    paths, _ = ACQ.run(Y, 8, None, sigma2=sigma2)
    ok = any(l == lw and abs(k - kw) < 0.5 / S.beta for l, k in paths)
    extra = sum(1 for l, k in paths if not ((l == lw and abs(k - kw) < 0.5 / S.beta) or
                                           (l == ls and abs(k - ks) < 0.5 / S.beta)))
    return dict(g=gamma1_db, ret=ok, extra=extra)


def e5_trial(args):
    kind, seed = args
    rng = np.random.default_rng(seed)
    S = S0
    x = np.zeros((1, S.N), complex); x[0, S.m0] = np.sqrt(S.Ep)
    if kind == "noise":
        Y = np.zeros((1, S.N), complex)
    else:
        l, k = int(rng.integers(0, S.ell_max + 1)), rng.uniform(-S.alpha_max, S.alpha_max)
        Y = S.receive(S.channel(S.transmit(x), [l], [k], [np.sqrt(10 ** 2.5 / S.Ep)]), 1)
    Y = Y + np.sqrt(0.5) * (rng.standard_normal(Y.shape) + 1j * rng.standard_normal(Y.shape))
    paths, _ = ACQ.run(Y, 8, None, sigma2=1.0)
    return dict(kind=kind, n=len(paths))


def e4(trials=1000):
    from scipy.special import erf
    jobs = [(g, 77 * t + 100000 * int(g + 20)) for g in (9, 12, 15, 18, 21) for t in range(trials)]
    with Pool(18) as p:
        res = p.map(e4_trial, jobs, chunksize=16)
    T = acq_threshold(1)
    rows = []
    for g in sorted({r["g"] for r in res}):
        rr = [r for r in res if r["g"] == g]
        gam = 10 ** (g / 10)
        snc = np.sqrt(3 / (2 * np.pi ** 2 * gam))
        loc = float(erf(1 / (2 * np.sqrt(2) * S0.beta * snc)))
        rows.append(dict(gamma1_db=g, trials=len(rr), p_retained=float(np.mean([r["ret"] for r in rr])),
                         p_retained_formula=p_retained(gam, 1, T) * loc,
                         p_cell_formula=p_global(gam, 1) * loc,
                         extra_paths_mean=float(np.mean([r["extra"] for r in rr]))))
    return dict(threshold=float(T), rows=rows)


def e5(trials=4000):
    jobs = [(k, 13 * t + (0 if k == "noise" else 999999)) for k in ("noise", "one") for t in range(trials)]
    with Pool(18) as p:
        res = p.map(e5_trial, jobs, chunksize=32)
    out = {}
    for k, forced in (("noise", 1), ("one", 1)):
        n = np.array([r["n"] for r in res if r["kind"] == k])
        out[k] = dict(trials=int(n.size), p_extra=float(np.mean(n > forced)), mean_extra=float(np.mean(n - forced)))
    return out


if __name__ == "__main__":
    which = sys.argv[1:] or ["e1", "e2", "e3"]
    path = OUT / "theory_check.json"
    res = json.load(open(path)) if path.exists() else {}
    for w in which:
        res[w] = {"e1": e1, "e2": e2, "e3": e3, "e4": e4, "e5": e5}[w]()
        json.dump(res, open(path, "w"), indent=1)
        print(f"{w} done", flush=True)
