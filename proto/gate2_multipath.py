"""Gate 2: P-path chirp-domain estimation with data present, vs B and pilot energy.

Per configuration: correct-cell probability of the coarse Doppler, per-path
Doppler RMSE (Hungarian-matched), and pilot-window model NMSE, for the
single-block ('sb'), non-coherent multi-block ('nc') and vernier ('vern')
estimators. The true path count is given (P_max = P) at this gate.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mbafdm import MBAFDM  # noqa: E402
from mbest import ChirpDomainEstimator  # noqa: E402


def draw_channel(S, P, rng):
    ls = rng.choice(S.ell_max + 1, P, replace=False)
    ks = rng.uniform(-S.alpha_max, S.alpha_max, P)
    hs = (rng.standard_normal(P) + 1j * rng.standard_normal(P)) / np.sqrt(2 * P)
    return ls, ks, hs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--P", type=int, default=4)
    ap.add_argument("--snr", type=float, nargs="+", default=[0, 10, 20])
    ap.add_argument("--ep_frac", type=float, nargs="+", default=[0.02, 0.08, 0.34])
    ap.add_argument("--B", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--out", default="gate2_multipath.json")
    a = ap.parse_args()

    S = MBAFDM()
    E = ChirpDomainEstimator(S)
    rng = np.random.default_rng(11)
    rows = []
    Bmax = max(a.B)
    for ep_frac in a.ep_frac:
        Ep = ep_frac * S.N
        for snr in a.snr:
            sigma2 = 10 ** (-snr / 10)
            acc = {(m, B): {"err": [], "cell": [], "nmse": []}
                   for m in ("sb", "nc", "vern") for B in a.B}
            t0 = time.time()
            for t in range(a.trials):
                ls, ks, hs = draw_channel(S, a.P, rng)
                x = S.make_frame(Bmax, rng, Ep=Ep)
                r = S.channel(S.transmit(x), ls, ks, hs)
                r = r + np.sqrt(sigma2 / 2) * (rng.standard_normal(r.shape) + 1j * rng.standard_normal(r.shape))
                Y = S.receive(r, Bmax)[:, S.W]
                # noiseless pilot-only reference for NMSE of the window model
                ref = S.receive(S.channel(S.transmit(S.make_frame(Bmax, rng, Ep=Ep, data=False)),
                                          ls, ks, hs), Bmax)[:, S.W]
                for B in a.B:
                    for m in ("sb", "nc", "vern"):
                        if m != "sb" and B == 1:
                            continue
                        if m == "sb" and B != a.B[0]:
                            continue
                        paths, g, res, coarse = E.run(Y[:B], a.P, m, return_coarse=True)
                        cost = np.array([[abs(pl - tl) * 10 + abs(pk - tk) for tl, tk in zip(ls, ks)]
                                         for pl, pk in paths])
                        ri, ci = linear_sum_assignment(cost)
                        for i, j in zip(ri, ci):
                            ok_l = paths[i][0] == ls[j]
                            acc[(m, B)]["err"].append(paths[i][1] - ks[j] if ok_l else np.nan)
                            acc[(m, B)]["cell"].append(ok_l and abs(paths[i][1] - ks[j]) < 0.5 / S.beta)
                        Bm = 1 if m == "sb" else B
                        D = E.design(paths, Bm, "nc" if m == "nc" else "coh")
                        fit = (D @ g).reshape(Bm, -1)
                        acc[(m, B)]["nmse"].append(np.sum(np.abs(fit - ref[:Bm]) ** 2) / np.sum(np.abs(ref[:Bm]) ** 2))
            for (m, B), v in acc.items():
                if not v["cell"]:
                    continue
                e = np.array(v["err"], float)
                ok = ~np.isnan(e)
                row = {"ep_frac": ep_frac, "snr_db": snr, "B": B, "mode": m, "P": a.P,
                       "p_cell": float(np.mean(v["cell"])),
                       "p_delay_ok": float(np.mean(ok)),
                       "rmse_kappa_delayok": float(np.sqrt(np.mean(e[ok] ** 2))) if ok.any() else None,
                       "median_abs_err": float(np.median(np.abs(e[ok]))) if ok.any() else None,
                       "nmse_db": float(10 * np.log10(np.mean(v["nmse"])))}
                rows.append(row)
                print(f"Ep={ep_frac:.2f}N snr={snr:4.0f} {m:4s} B={B:2d} cell={row['p_cell']:.3f} "
                      f"dly={row['p_delay_ok']:.3f} rmse={row['rmse_kappa_delayok']:.3e} "
                      f"med={row['median_abs_err']:.2e} nmse={row['nmse_db']:6.1f}dB", flush=True)
            print(f"  ({time.time()-t0:.0f}s)", flush=True)
    out = Path(__file__).resolve().parents[1] / "runs"
    out.mkdir(exist_ok=True)
    json.dump({"args": vars(a), "rows": rows}, open(out / a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
