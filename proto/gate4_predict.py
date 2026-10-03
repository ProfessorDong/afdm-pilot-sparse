"""Gate 4: pilot-free blocks via Doppler-phase prediction.

Frame of B_tot blocks: the first Bp blocks carry pilot + guard, the remaining
B_tot - Bp blocks carry data on all N chirps (no pilot, no guard).

Receivers
  genie      : true channel in every block
  sb-every   : conventional AFDM, pilot+guard in EVERY block, single-block estimate
  nc-hold    : Bp pilot blocks, non-coherent (free per-block gain) estimate; data
               blocks reuse the last pilot block's channel (cannot extrapolate phase)
  vern-pred  : Bp pilot blocks, coherent vernier estimate; data blocks use the
               predicted channel  h_p exp(j 2pi kappa_p b beta)
  vern-dd    : vern-pred + decision-directed refinement of (kappa, h) using the
               detected data blocks as pseudo-pilots (aperture grows to B_tot)

Spectral efficiency counts only correctly detected data chirps:
  eta = (# correct data symbols) / (B_tot * N).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mbafdm import MBAFDM  # noqa: E402
from mbest import ChirpDomainEstimator  # noqa: E402
from mbdetect import BlockChannel, apply_path  # noqa: E402


def qpsk(rng, n):
    return np.exp(1j * (np.pi / 4 + np.pi / 2 * rng.integers(0, 4, n)))


def detect(C, Hb, yb, known_mask, xknown, idx, sigma2):
    r = yb - Hb[:, known_mask] @ xknown[known_mask]
    Hd = Hb[:, idx]
    z = np.linalg.solve(Hd.conj().T @ Hd + sigma2 * np.eye(idx.size), Hd.conj().T @ r)
    return C.qpsk_hard(z)


class DD:
    """Decision-directed joint refit of (kappa_p, h_p) over all blocks, with the
    integer delays fixed: y_b = sum_p h_p exp(j2pi kappa_p b beta) Phi_b0(l,k) x_b."""

    def __init__(self, S, C):
        self.S, self.C = S, C

    def unit_cols(self, l, k, X):
        # column: block-b response to x_b of a unit-gain path, stacked over b
        return apply_path(self.S, l, k, X).reshape(-1)

    def refit(self, Y, X, paths, iters=2, half=0.02):
        y = Y.reshape(-1)
        paths = list(paths)
        for _ in range(iters):
            for p in range(len(paths)):
                others = [self.unit_cols(l, k, X) for i, (l, k) in enumerate(paths) if i != p]
                l = paths[p][0]

                def obj(k):
                    cols = others + [self.unit_cols(l, k, X)]
                    D = np.stack(cols, 1)
                    g, *_ = np.linalg.lstsq(D, y, rcond=None)
                    return np.sum(np.abs(y - D @ g) ** 2)

                k0 = paths[p][1]
                r = minimize_scalar(obj, bounds=(k0 - half, k0 + half), method="bounded",
                                    options={"xatol": 1e-7})
                paths[p] = (l, r.x)
        D = np.stack([self.unit_cols(l, k, X) for l, k in paths], 1)
        g, *_ = np.linalg.lstsq(D, y, rcond=None)
        return paths, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--P", type=int, default=4)
    ap.add_argument("--snr", type=float, nargs="+", default=[10, 15, 20])
    ap.add_argument("--ep_frac", type=float, default=0.08)
    ap.add_argument("--Bp", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--Btot", type=int, default=16)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--dd", action="store_true")
    ap.add_argument("--out", default="gate4_predict.json")
    a = ap.parse_args()

    S = MBAFDM()
    E = ChirpDomainEstimator(S)
    C = BlockChannel(S)
    dd = DD(S, C)
    rng = np.random.default_rng(31)
    Ep = a.ep_frac * S.N
    N = S.N
    all_idx = np.arange(N)
    pil_known = np.ones(N, bool); pil_known[S.data_idx] = False
    rows = []
    for snr in a.snr:
        sigma2 = 10 ** (-snr / 10)
        for Bp in a.Bp:
            stats = {k: np.zeros(a.Btot) for k in ("genie", "sb-every", "nc-hold", "vern-pred", "vern-dd")}
            ncorrect = {k: 0 for k in stats}
            t0 = time.time()
            for t in range(a.trials):
                ls = rng.choice(S.ell_max + 1, a.P, replace=False)
                ks = rng.uniform(-S.alpha_max, S.alpha_max, a.P)
                hs = (rng.standard_normal(a.P) + 1j * rng.standard_normal(a.P)) / np.sqrt(2 * a.P)
                noise = lambda shape: np.sqrt(sigma2 / 2) * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))

                # ---- frame A: proposed (Bp pilot blocks, then full-data blocks)
                xA = np.zeros((a.Btot, N), complex)
                xA[:Bp] = S.make_frame(Bp, rng, Ep=Ep)
                xA[Bp:] = qpsk(rng, (a.Btot - Bp) * N).reshape(a.Btot - Bp, N)
                YA = S.receive(S.channel(S.transmit(xA), ls, ks, hs) + 0, a.Btot)
                YA = YA + noise(YA.shape)
                # ---- frame B: conventional (pilot + guard every block)
                xB = S.make_frame(a.Btot, rng, Ep=Ep)
                YB = S.receive(S.channel(S.transmit(xB), ls, ks, hs), a.Btot) + noise((a.Btot, N))

                def score(key, b, xh, xtrue, idx):
                    e = np.abs(xh - xtrue[idx]) > 1e-6
                    stats[key][b] += np.mean(e)
                    ncorrect[key] += int(np.sum(~e))

                # genie on frame A
                for b in range(a.Btot):
                    Hb = C.H(b, ls, ks, hs)
                    if b < Bp:
                        score("genie", b, detect(C, Hb, YA[b], pil_known, xA[b], S.data_idx, sigma2), xA[b], S.data_idx)
                    else:
                        score("genie", b, detect(C, Hb, YA[b], np.zeros(N, bool), xA[b], all_idx, sigma2), xA[b], all_idx)
                # conventional: every block single-block estimate
                for b in range(a.Btot):
                    paths, g, _ = E.run(YB[b:b + 1, S.W], a.P, "sb")
                    kk = [p[1] for p in paths]
                    hh = [gp / np.sqrt(Ep) * np.exp(-1j * 2 * np.pi * k * b * S.beta) for gp, k in zip(g, kk)]
                    Hb = C.H(b, [p[0] for p in paths], kk, hh)
                    score("sb-every", b, detect(C, Hb, YB[b], pil_known, xB[b], S.data_idx, sigma2), xB[b], S.data_idx)
                # non-coherent hold
                paths, g, _ = E.run(YA[:Bp, S.W], a.P, "nc")
                P_ = len(paths); G = g.reshape(P_, Bp)
                for b in range(a.Btot):
                    bb = min(b, Bp - 1)
                    hh = [G[p, bb] / np.sqrt(Ep) * np.exp(-1j * 2 * np.pi * paths[p][1] * bb * S.beta) for p in range(P_)]
                    # hold block bb's channel: evaluate at bb's time stamp
                    Hb = C.H(bb, [q[0] for q in paths], [q[1] for q in paths], hh)
                    if b < Bp:
                        score("nc-hold", b, detect(C, Hb, YA[b], pil_known, xA[b], S.data_idx, sigma2), xA[b], S.data_idx)
                    else:
                        score("nc-hold", b, detect(C, Hb, YA[b], np.zeros(N, bool), xA[b], all_idx, sigma2), xA[b], all_idx)
                # vernier + prediction
                paths, g, _ = E.run(YA[:Bp, S.W], a.P, "vern")
                ll = [q[0] for q in paths]; kk = [q[1] for q in paths]; hh = list(g / np.sqrt(Ep))
                Xh = np.array(xA, copy=True)
                for b in range(a.Btot):
                    Hb = C.H(b, ll, kk, hh)
                    if b < Bp:
                        d = detect(C, Hb, YA[b], pil_known, xA[b], S.data_idx, sigma2)
                        score("vern-pred", b, d, xA[b], S.data_idx); Xh[b, S.data_idx] = d
                    else:
                        d = detect(C, Hb, YA[b], np.zeros(N, bool), xA[b], all_idx, sigma2)
                        score("vern-pred", b, d, xA[b], all_idx); Xh[b] = d
                if a.dd:
                    # sequential decision-directed tracking: pilot blocks re-detected with
                    # the vernier estimate, then each pilot-free block is detected with the
                    # current prediction and appended to the aperture before refitting.
                    trk = list(paths); gt = list(g / np.sqrt(Ep))
                    Xk = np.array(xA[:Bp], copy=True)
                    Xk[:, S.data_idx] = Xh[:Bp, S.data_idx]
                    for b in range(Bp):
                        score("vern-dd", b, Xh[b, S.data_idx], xA[b], S.data_idx)
                    for b in range(Bp, a.Btot):
                        Hb = C.H(b, [q[0] for q in trk], [q[1] for q in trk], gt)
                        d = detect(C, Hb, YA[b], np.zeros(N, bool), xA[b], all_idx, sigma2)
                        score("vern-dd", b, d, xA[b], all_idx)
                        Xk = np.vstack([Xk, d[None, :]])
                        trk, g2 = dd.refit(YA[:b + 1], Xk, trk, iters=1)
                        gt = list(g2)
            T = a.trials
            row = {"snr_db": snr, "Bp": Bp, "Btot": a.Btot,
                   "ser_by_block": {k: list(v / T) for k, v in stats.items()},
                   "eta": {k: ncorrect[k] / (T * a.Btot * N) for k in stats}}
            rows.append(row)
            print(f"snr={snr:4.0f} Bp={Bp:2d} eta: " + " ".join(f"{k}={v:.3f}" for k, v in row["eta"].items())
                  + f" ({time.time()-t0:.0f}s)", flush=True)
            for k in ("genie", "sb-every", "nc-hold", "vern-pred") + (("vern-dd",) if a.dd else ()):
                v = row["ser_by_block"][k]
                print(f"   {k:9s} SER first/last-pilot/after: {v[0]:.3f} {v[Bp-1]:.3f} | "
                      + " ".join(f"{x:.3f}" for x in v[Bp:]), flush=True)
    out = Path(__file__).resolve().parents[1] / "runs"
    json.dump({"args": vars(a), "rows": rows}, open(out / a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
