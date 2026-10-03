"""Gate 3: end-to-end SER of the multi-block AFDM frame.

Receivers (all use per-block LMMSE on the estimated DAFT-domain channel):
  genie : true channel
  sb    : single-block chirp-domain estimate, used for every block of its own
          (each block estimated independently -- same frame, same pilots)
  nc    : non-coherent multi-block (shared support, free per-block gains)
  vern  : coherent vernier multi-block
  vern+da : vern with T rounds of data-aided leakage cancellation
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mbafdm import MBAFDM  # noqa: E402
from mbest import ChirpDomainEstimator  # noqa: E402
from mbdetect import BlockChannel  # noqa: E402


def ser_block(C, Hb, yb, xb, sigma2):
    z = C.lmmse(Hb, yb, xb, sigma2)
    xh = C.qpsk_hard(z)
    return np.mean(np.abs(xh - xb[C.S.data_idx]) > 1e-6), xh


def gains_to_h(S, paths, g, B, mode, b):
    sq = np.sqrt(S.Ep_used)
    if mode == "nc":
        P = len(paths)
        gb = g.reshape(P, B)[:, b]
        return [gb[p] / sq * np.exp(-1j * 2 * np.pi * paths[p][1] * b * S.beta) for p in range(P)]
    return [gp / sq for gp in g]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--P", type=int, default=4)
    ap.add_argument("--snr", type=float, nargs="+", default=[0, 5, 10, 15, 20, 25])
    ap.add_argument("--ep_frac", type=float, default=0.08)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--T", type=int, default=2)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--out", default="gate3_ser.json")
    a = ap.parse_args()

    S = MBAFDM()
    E = ChirpDomainEstimator(S)
    C = BlockChannel(S)
    rng = np.random.default_rng(23)
    Ep = a.ep_frac * S.N
    S.Ep_used = Ep
    rows = []
    for snr in a.snr:
        sigma2 = 10 ** (-snr / 10)
        tot = {k: [] for k in ("genie", "sb", "nc", "vern", "vern+da")}
        t0 = time.time()
        for t in range(a.trials):
            ls = rng.choice(S.ell_max + 1, a.P, replace=False)
            ks = rng.uniform(-S.alpha_max, S.alpha_max, a.P)
            hs = (rng.standard_normal(a.P) + 1j * rng.standard_normal(a.P)) / np.sqrt(2 * a.P)
            B = a.B
            x = S.make_frame(B, rng, Ep=Ep)
            r = S.channel(S.transmit(x), ls, ks, hs)
            r = r + np.sqrt(sigma2 / 2) * (rng.standard_normal(r.shape) + 1j * rng.standard_normal(r.shape))
            Y = S.receive(r, B)
            YW = Y[:, S.W]
            # genie
            for b in range(B):
                tot["genie"].append(ser_block(C, C.H(b, ls, ks, hs), Y[b], x[b], sigma2)[0])
            # single-block, each block on its own
            for b in range(B):
                paths, g, _ = E.run(YW[b:b + 1], a.P, "sb")
                kk = [p[1] for p in paths]
                # 'sb' fits block b as if it were block 0: rotate the gain back to absolute time
                hh = [gp / np.sqrt(Ep) * np.exp(-1j * 2 * np.pi * k * b * S.beta) for gp, k in zip(g, kk)]
                tot["sb"].append(ser_block(C, C.H(b, [p[0] for p in paths], kk, hh), Y[b], x[b], sigma2)[0])
            # multi-block
            for mode in ("nc", "vern"):
                paths, g, _ = E.run(YW, a.P, mode)
                ll = [p[0] for p in paths]; kk = [p[1] for p in paths]
                xh = np.array(x, copy=True)
                for b in range(B):
                    Hb = C.H(b, ll, kk, gains_to_h(S, paths, g, B, mode, b))
                    s, d = ser_block(C, Hb, Y[b], x[b], sigma2)
                    tot[mode].append(s)
                    xh[b, S.data_idx] = d
                if mode != "vern":
                    continue
                for it in range(a.T):
                    clean = YW.copy()
                    for b in range(B):
                        Hb = C.H(b, ll, kk, gains_to_h(S, paths, g, B, mode, b))
                        xd = np.zeros(S.N, complex); xd[S.data_idx] = xh[b, S.data_idx]
                        clean[b] -= (Hb @ xd)[S.W]
                    paths, g, _ = E.run(clean, a.P, "vern")
                    ll = [p[0] for p in paths]; kk = [p[1] for p in paths]
                    res = []
                    for b in range(B):
                        Hb = C.H(b, ll, kk, gains_to_h(S, paths, g, B, mode, b))
                        s, d = ser_block(C, Hb, Y[b], x[b], sigma2)
                        res.append(s); xh[b, S.data_idx] = d
                tot["vern+da"].extend(res)
        row = {"snr_db": snr, **{k: float(np.mean(v)) for k, v in tot.items()}}
        rows.append(row)
        print(f"snr={snr:4.0f} " + " ".join(f"{k}={v:.4f}" for k, v in row.items() if k != "snr_db")
              + f"  ({time.time()-t0:.0f}s)", flush=True)
    out = Path(__file__).resolve().parents[1] / "runs"
    json.dump({"args": vars(a), "rows": rows}, open(out / a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
