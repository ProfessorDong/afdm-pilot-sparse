"""Numerical checks behind statements of the paper that are not Monte Carlo
throughput results. Writes runs/checks.json.

K1  Kaufman Jacobian versus the exact derivative of the projected residual
    (finite differences), near and away from the fit.
K2  Data leakage into the pilot window on the evaluation ensemble (fractional
    delays, Jakes Doppler, exponential power-delay profile): distribution of the
    leakage-to-pilot ratio, and leakage relative to the noise at 20 dB.
K3  Multi-block ambiguity: full-length versus pilot-window (restricted) atom,
    integer and fractional delays; grating-region peak sidelobe versus Bp,
    including the floor epsilon_1 at large Bp.
K4  Guard fraction versus N at a fixed sample rate and fixed physical delay and
    Doppler spread (alpha_max grows with N), and the feasibility conditions.
K5  Realized RMS delay spread of the evaluation ensemble.
K6  Joint (two-path) Cramer-Rao benchmark of the predicted received block versus
    the decoupled per-path benchmark, as a function of the Doppler separation.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

from engine import Config, draw_channel, make_system  # noqa: E402
from mbafdm import MBAFDM  # noqa: E402
from mbtrack import Tracker, apply_path  # noqa: E402

OUT = Path(__file__).resolve().parent / "runs" / "checks.json"


def qpsk(rng, shape):
    return np.exp(1j * (np.pi / 4 + np.pi / 2 * rng.integers(0, 4, shape)))


# ------------------------------------------------------------------ K1
def k1():
    S = make_system(Config())
    rng = np.random.default_rng(1)
    B = 3
    X = qpsk(rng, (B, S.N))
    bl = np.arange(B)
    true = [(1.3, 0.7), (4.6, -1.2), (2.2, 2.4)]
    h = np.array([0.8, 0.5j, -0.4])
    Y = sum(g * apply_path(S, l, k, X, bl) for (l, k), g in zip(true, h))
    Y = Y + 0.02 * (rng.standard_normal(Y.shape) + 1j * rng.standard_normal(Y.shape))
    y = Y.reshape(-1)

    def proj_res(th):
        D = np.stack([apply_path(S, th[2 * p + 1], th[2 * p], X, bl).reshape(-1) for p in range(3)], 1)
        g, *_ = np.linalg.lstsq(D, y, rcond=None)
        return y - D @ g, D, g

    out = []
    for name, off in (("at fit", 0.0), ("away", 0.05)):
        th = np.array([v for (l, k) in true for v in (k + off, l + off)])
        r, D, g = proj_res(th)
        Q, _ = np.linalg.qr(D)
        J = []
        for p in range(3):
            _, d = apply_path(S, th[2 * p + 1], th[2 * p], X, bl, deriv=("k", "l"))
            for nm in ("k", "l"):
                col = np.zeros((y.size, 3), complex)
                col[:, p] = d[nm].reshape(-1)
                v = col @ g
                J.append(-(v - Q @ (Q.conj().T @ v)))
        J = np.stack(J, 1)
        Jfd = []
        for i in range(th.size):
            e = np.zeros(th.size); e[i] = 1e-6
            Jfd.append((proj_res(th + e)[0] - proj_res(th - e)[0]) / 2e-6)
        Jfd = np.stack(Jfd, 1)
        out.append(dict(point=name, rel_err=float(np.linalg.norm(J - Jfd) / np.linalg.norm(Jfd)),
                        residual_rms=float(np.sqrt(np.mean(np.abs(r) ** 2)))))
    return out


# ------------------------------------------------------------------ K2
def k2(draws=300):
    cfg = Config(frac_delay=True)
    S = make_system(cfg)
    rng = np.random.default_rng(2)
    B = 4
    ratio, rel_noise = [], []
    for _ in range(draws):
        ch = draw_channel(cfg, rng)
        xd = np.zeros((B, S.N), complex); xd[:, S.data_idx] = qpsk(rng, (B, S.data_idx.size))
        xp = np.zeros((B, S.N), complex); xp[:, S.m0] = np.sqrt(S.Ep)
        run = lambda x: S.receive(S.channel(S.transmit(x), ch["tau"], ch["kappa"], ch["h"]), B)[1:, S.W]
        lk = np.sum(np.abs(run(xd)) ** 2)
        pl = np.sum(np.abs(run(xp)) ** 2)
        ratio.append(lk / pl)
        rel_noise.append(lk / ((B - 1) * len(S.W)) / 10 ** (-20 / 10))   # per-chirp leakage / sigma^2 at 20 dB
    ratio = 10 * np.log10(np.array(ratio))
    rel_noise = 10 * np.log10(np.array(rel_noise))
    return dict(draws=draws, leak_to_pilot_db_mean=float(10 * np.log10(np.mean(10 ** (ratio / 10)))),
                leak_to_pilot_db_median=float(np.median(ratio)),
                leak_to_pilot_db_p95=float(np.percentile(ratio, 95)),
                leak_to_noise20_db_median=float(np.median(rel_noise)),
                leak_to_noise20_db_p95=float(np.percentile(rel_noise, 95)))


# ------------------------------------------------------------------ K3
def k3():
    S = MBAFDM()
    e = np.zeros(S.N, complex); e[S.m0] = 1
    s0 = S.idaft(e)
    n = np.arange(S.Ncp, S.Ncp + S.N)

    def atom(l, k, restrict):
        if abs(l - round(l)) < 1e-12:
            s = np.roll(s0, int(round(l)))
        else:
            F = np.fft.fft(s0)
            kk = np.fft.fftfreq(S.N) * S.N
            s = np.fft.ifft(F * np.exp(-1j * 2 * np.pi * kk * l / S.N))
        a = S.daft(s * np.exp(1j * 2 * np.pi * k * n / S.N))
        return a[S.W] if restrict else a

    def chi(Bp, d, l, restrict, k0=0.3):
        a0, a1 = atom(l, k0, restrict), atom(l, k0 + d, restrict)
        inner = abs(np.vdot(a0, a1)) / np.sqrt(np.vdot(a0, a0).real * np.vdot(a1, a1).real)
        A = abs(np.sin(np.pi * Bp * S.beta * d) / (Bp * np.sin(np.pi * S.beta * d))) if d else 1.0
        return inner * A

    rows = []
    for Bp in (2, 4, 16, 128, 512):
        dd = np.union1d(np.linspace(0.6, 1.4, 801), [1 / S.beta])        # grating region (as grating_peak.json)
        for l, restrict in ((3, False), (3, True), (3.5, True)):
            if Bp > 16 and l != 3:
                continue
            pk = max(chi(Bp, d, l, restrict) for d in dd)
            rows.append(dict(Bp=Bp, ell=l, restricted=restrict, peak_db=float(20 * np.log10(pk))))
    eps1 = abs(np.sin(np.pi * S.Ncp / (S.N + S.Ncp))) / (S.N * np.sin(np.pi / (S.N + S.Ncp)))
    return dict(rows=rows, eps1_db=float(20 * np.log10(eps1)))


# ------------------------------------------------------------------ K4
def k4():
    rows = []
    ell, xi, kap512 = 7, 2, 3.0          # fixed sample rate: delay in samples fixed, kappa_max ~ N
    for N in (512, 1024, 2048):
        amax = int(np.ceil(kap512 * N / 512))
        Q = 2 * (amax + xi) + 1
        MW = (ell + 1) * Q
        Z = 2 * MW - 1
        rows.append(dict(N=N, alpha_max=amax, Q=Q, MW=MW, Z=Z, frac=Z / N,
                         separable=MW <= N, payload_feasible=Z < N))
    # same channel in normalized units (fixed ell_max and alpha_max), for contrast
    norm_rows = [dict(N=N, frac=(2 * (ell + 1) * 11 - 1) / N) for N in (512, 1024, 2048)]
    return dict(fixed_rate=rows, fixed_normalized=norm_rows)


# ------------------------------------------------------------------ K5
def k5(draws=20000):
    cfg = Config(frac_delay=True)
    rng = np.random.default_rng(5)
    rms = []
    for _ in range(draws):
        ch = draw_channel(cfg, rng)
        p = np.abs(ch["h"]) ** 2
        p = p / p.sum()
        m = np.sum(p * ch["tau"])
        rms.append(np.sqrt(np.sum(p * (ch["tau"] - m) ** 2)))
    rms = np.array(rms)
    return dict(median=float(np.median(rms)), p90=float(np.percentile(rms, 90)), mean=float(rms.mean()))


# ------------------------------------------------------------------ K6
def joint_rho(S, paths, h, L, b_next, rng, draws=4):
    """Local Cramer-Rao benchmark of the predicted noise-free block b_next, divided by
    N sigma^2 (sigma^2-free), for an aperture of L known full-energy blocks, under the
    assumptions of Theorem 1: known integer delays, parameters (Re h, Im h, kappa) per
    path. Joint: tr(D J^{-1} D^H)/N with J = (2/sigma^2) Re(G^H G) of all paths;
    decoupled: the same with each path's block of J alone (others known), i.e. eq. (10).
    Medians over random-data draws (J can be ill-conditioned for close paths)."""
    vals, dec, conds = [], [], []
    for _ in range(draws):
        X = qpsk(rng, (L, S.N))
        Xn = qpsk(rng, (1, S.N))
        bl = np.arange(L)
        G, D = [], []
        for (l, k), g in zip(paths, h):
            a, d = apply_path(S, l, k, X, bl, deriv=("k",))
            an, dn = apply_path(S, l, k, Xn, [b_next], deriv=("k",))
            for col, coln in ((a, an), (1j * a, 1j * an), (g * d["k"], g * dn["k"])):
                G.append(col.reshape(-1)); D.append(coln.reshape(-1))
        G = np.stack(G, 1); D = np.stack(D, 1)
        J = 2 * np.real(G.conj().T @ G)                   # sigma^2 = 1
        conds.append(float(np.linalg.cond(J)))
        vals.append(float(np.real(np.trace(D @ np.linalg.solve(J, D.conj().T))) / S.N))
        s = 0.0
        for p in range(len(paths)):
            sl = slice(3 * p, 3 * p + 3)
            s += float(np.real(np.trace(D[:, sl] @ np.linalg.solve(J[sl, sl], D[:, sl].conj().T))) / S.N)
        dec.append(s)
    return vals, dec, conds


def k6(draws=400):
    """Joint versus decoupled benchmark for two paths. Every draw takes an independent
    geometry (integer delay, first Doppler shift, relative phase) and training data, so
    the median is over geometries as well as data."""
    S = make_system(Config())
    rng = np.random.default_rng(6)
    rows = []
    for dl in (0, 1):
        for L in (1, 2, 4, 8):
            for dk in np.round(np.logspace(np.log10(0.02), np.log10(2.0), 13), 4):
                rj, rd, cc = [], [], []
                for _ in range(draws):
                    l0 = int(rng.integers(0, S.ell_max + 1 - dl))
                    k0 = rng.uniform(-S.alpha_max, S.alpha_max - dk)
                    h = np.array([1, np.exp(1j * rng.uniform(0, 2 * np.pi))]) / np.sqrt(2)
                    a, b, c = joint_rho(S, [(l0, k0), (l0 + dl, k0 + dk)], h, L, L, rng, draws=1)
                    rj += a; rd += b; cc += c
                r = 10 * np.log10(np.array(rj) / np.array(rd))
                rows.append(dict(dl=dl, L=L, dk=float(dk), draws=draws, rho_joint=float(np.median(rj)),
                                 rho_decoupled=float(np.median(rd)), cond=float(np.median(cc)),
                                 ratio_db_median=float(np.median(r)),
                                 ratio_db_q25=float(np.percentile(r, 25)), ratio_db_q75=float(np.percentile(r, 75))))
    return rows


# ------------------------------------------------------------------ K7 / K8
def k7(frames=400):
    """Average transmitted energy per sample, CP included, of the compared frames
    (coded payload), relative to the nominal 1 per sample."""
    from engine import Payload, build_frame, ofdm_system
    cfg = Config(coded=True)
    S = make_system(cfg)
    So = ofdm_system(cfg)
    rng = np.random.default_rng(7)
    out = {k: [] for k in ("proposed", "per_block", "superimposed_0.1", "superimposed_0.3", "ofdm")}
    B = 16
    for _ in range(frames):
        pay = Payload(True, rng)
        x = build_frame(S, "P" + "D" * (B - 1), pay, "A")
        out["proposed"].append(np.mean(np.abs(S.transmit(x)) ** 2))
        x = build_frame(S, "P" * B, pay, "C")
        out["per_block"].append(np.mean(np.abs(S.transmit(x)) ** 2))
        for eps in (0.1, 0.3):
            d = np.stack([pay.symbols(("S", b), S.N) for b in range(B)])
            xs = np.sqrt(1 - eps) * d
            xs[:, S.m0] += np.sqrt(eps * S.N)
            out[f"superimposed_{eps}"].append(np.mean(np.abs(S.transmit(xs)) ** 2))
        xo = np.stack([So.data_block(rng)] + [pay.symbols(("O", b), S.N) for b in range(1, B)])
        out["ofdm"].append(np.mean(np.abs(So.transmit(xo)) ** 2))
    return {k: dict(mean=float(np.mean(v)), se=float(np.std(v) / np.sqrt(len(v))),
                    p1=float(np.percentile(v, 1)), p99=float(np.percentile(v, 99))) for k, v in out.items()}


def k8():
    """Energy of the unit-norm full-length pilot response captured by the window W,
    integer delays 0..ell_max, Doppler over the searched range."""
    S = MBAFDM()
    e = np.zeros(S.N, complex); e[S.m0] = 1
    s0 = S.idaft(e)
    n = np.arange(S.Ncp, S.Ncp + S.N)
    mn = (2.0, None)
    for l in range(S.ell_max + 1):
        for k in np.linspace(-S.alpha_max - 0.6, S.alpha_max + 0.6, 721):
            a = S.daft(np.roll(s0, l) * np.exp(1j * 2 * np.pi * k * n / S.N))
            c = float(np.sum(np.abs(a[S.W]) ** 2))
            if c < mn[0]:
                mn = (c, (l, float(k)))
    mk = (2.0, None)
    for l in range(S.ell_max + 1):
        for k in np.linspace(-S.alpha_max, S.alpha_max, 601):
            a = S.daft(np.roll(s0, l) * np.exp(1j * 2 * np.pi * k * n / S.N))
            c = float(np.sum(np.abs(a[S.W]) ** 2))
            if c < mk[0]:
                mk = (c, (l, float(k)))
    return dict(min_capture=mn[0], at=mn[1], loss_db=float(10 * np.log10(mn[0])),
                min_capture_kmax=mk[0], at_kmax=mk[1], loss_db_kmax=float(10 * np.log10(mk[0])))


# ------------------------------------------------------------------ K9
def k9():
    """Deterministic sweep of the grating-region peak of the multi-block ambiguity,
    full-length versus pilot-window (restricted, normalized) response: delays
    0..ell_max in quarter samples, reference Dopplers -3, -1, 0.3, 2, Bp = 2, 4, 16,
    Doppler differences in [0.6, 1.4] (plus 1/beta)."""
    S = MBAFDM()
    e = np.zeros(S.N, complex); e[S.m0] = 1
    s0 = S.idaft(e)
    n = np.arange(S.Ncp, S.Ncp + S.N)
    ds = np.union1d(np.linspace(0.6, 1.4, 801), [1 / S.beta])
    ph = np.exp(1j * 2 * np.pi * ds[:, None] * n[None, :] / S.N)
    kk = np.fft.fftfreq(S.N) * S.N
    rows = []
    for l in np.arange(0, S.ell_max + 0.01, 0.25):
        s = np.fft.ifft(np.fft.fft(s0) * np.exp(-1j * 2 * np.pi * kk * l / S.N))
        for k0 in (-3.0, -1.0, 0.3, 2.0):
            base = s * np.exp(1j * 2 * np.pi * k0 * n / S.N)
            a0 = S.daft(base); a1 = S.daft(base[None, :] * ph)
            full = np.abs(a1 @ a0.conj()) / np.sqrt(np.sum(np.abs(a1) ** 2, 1) * np.vdot(a0, a0).real)
            w0, w1 = a0[S.W], a1[:, S.W]
            win = np.abs(w1 @ w0.conj()) / np.sqrt(np.sum(np.abs(w1) ** 2, 1) * np.vdot(w0, w0).real)
            for Bp in (2, 4, 16):
                A = np.abs(np.sin(np.pi * Bp * S.beta * ds) / (Bp * np.sin(np.pi * S.beta * ds) + 1e-300))
                A[np.abs(np.sin(np.pi * S.beta * ds)) < 1e-12] = 1.0
                pf, pw = float(np.max(full * A)), float(np.max(win * A))
                rows.append(dict(ell=float(l), k0=k0, Bp=Bp, full_db=20 * np.log10(pf), win_db=20 * np.log10(pw)))
    out = {}
    for Bp in (2, 4, 16):
        rr = [r for r in rows if r["Bp"] == Bp]
        out[str(Bp)] = dict(full_max_db=max(r["full_db"] for r in rr), win_max_db=max(r["win_db"] for r in rr),
                            max_abs_diff_db=max(abs(r["win_db"] - r["full_db"]) for r in rr))
    out["n_cases"] = len(rows) // 3
    return out


if __name__ == "__main__":
    which = sys.argv[1:] or ["k1", "k2", "k3", "k4", "k5", "k6"]
    res = json.load(open(OUT)) if OUT.exists() else {}
    for w in which:
        res[w] = globals()[w]()
        json.dump(res, open(OUT, "w"), indent=1)
        print(w, json.dumps(res[w])[:600], flush=True)
