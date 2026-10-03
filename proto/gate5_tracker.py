"""Gate 5: robustness of decision-directed tracking (soft VP-GN vs hard), parallel."""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from mbafdm import MBAFDM  # noqa: E402
from mbest import ChirpDomainEstimator  # noqa: E402
import receivers as R  # noqa: E402

S = MBAFDM()
E = ChirpDomainEstimator(S)


def one(args):
    seed, snr, Bp, Btot, P = args
    rng = np.random.default_rng(seed)
    sigma2 = 10 ** (-snr / 10)
    tau = rng.choice(S.ell_max + 1, P, replace=False).astype(float)
    kap = rng.uniform(-S.alpha_max, S.alpha_max, P)
    h = (rng.standard_normal(P) + 1j * rng.standard_normal(P)) / np.sqrt(2 * P)
    ch = {"tau": tau, "kappa": kap, "h": h}
    kinds = "P" * Bp + "D" * (Btot - Bp)
    x = S.frame(kinds, rng)
    r = S.channel(S.transmit(x), tau, kap, h)
    r = r + np.sqrt(sigma2 / 2) * (rng.standard_normal(r.shape) + 1j * rng.standard_normal(r.shape))
    Y = S.receive(r, Btot)
    xc = S.frame("P" * Btot, rng)
    rc = S.channel(S.transmit(xc), tau, kap, h)
    rc = rc + np.sqrt(sigma2 / 2) * (rng.standard_normal(rc.shape) + 1j * rng.standard_normal(rc.shape))
    Yc = S.receive(rc, Btot)

    def errs(xh_list, xt, kd):
        out = []
        for b, k in enumerate(kd):
            idx = S.data_idx if k == "P" else np.arange(S.N)
            out.append(int(np.sum(np.abs(xh_list[b][idx] - xt[b][idx]) > 1e-6)))
        return out

    res = {}
    res["genie"] = errs(R.genie(S, Y, kinds, sigma2, ch), x, kinds)
    res["conv"] = errs(R.per_block_conventional(S, E, Yc, sigma2, P), xc, "P" * Btot)
    res["pred"] = errs(R.predict_only(S, E, Y, kinds, sigma2, P), x, kinds)
    hard, _ = R.tracked(S, E, Y, kinds, sigma2, P, soft=False, redetect=False)
    res["track-hard"] = errs(hard, x, kinds)
    hard, traj = R.tracked(S, E, Y, kinds, sigma2, P, soft=True, redetect=True)
    res["track-soft"] = errs(hard, x, kinds)
    # Doppler error trajectory (Hungarian-free: match by delay)
    kerr = []
    for b, ks, _ in traj:
        e = [min(abs(k - kt) for t, kt in zip(tau, kap) if True) for k in ks]
        kerr.append(float(np.sqrt(np.mean(np.square(e)))))
    res["kerr"] = kerr
    return snr, Bp, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snr", type=float, nargs="+", default=[5, 10, 15])
    ap.add_argument("--Bp", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--Btot", type=int, default=16)
    ap.add_argument("--P", type=int, default=4)
    ap.add_argument("--trials", type=int, default=16)
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--out", default="gate5_tracker.json")
    a = ap.parse_args()
    jobs = [(1000 * i + 7, snr, Bp, a.Btot, a.P) for snr in a.snr for Bp in a.Bp for i in range(a.trials)]
    t0 = time.time()
    with Pool(a.procs) as pool:
        out = pool.map(one, jobs)
    N = S.N
    rows = []
    for snr in a.snr:
        for Bp in a.Bp:
            rs = [r for s, b, r in out if s == snr and b == Bp]
            kinds = "P" * Bp + "D" * (a.Btot - Bp)
            nsym = sum(S.data_idx.size if k == "P" else N for k in kinds)
            nconv = a.Btot * S.data_idx.size
            row = {"snr": snr, "Bp": Bp}
            for key in ("genie", "conv", "pred", "track-hard", "track-soft"):
                e = np.array([r[key] for r in rs])            # trials x blocks
                tot = nconv if key == "conv" else nsym
                row[f"eta_{key}"] = float((tot - e.sum(1)).mean() / (a.Btot * N))
                per = e.mean(0) / np.array([S.data_idx.size if (key == "conv" or k == "P") else N for k in kinds])
                row[f"ser_{key}"] = per.tolist()
            row["kerr"] = np.mean([r["kerr"] for r in rs], 0).tolist()
            rows.append(row)
            print(f"snr={snr:4.0f} Bp={Bp} " + " ".join(f"{k[4:]}={row[k]:.3f}" for k in row if k.startswith("eta_")), flush=True)
            print("   soft SER/blk: " + " ".join(f"{v:.3f}" for v in row["ser_track-soft"]), flush=True)
            print("   hard SER/blk: " + " ".join(f"{v:.3f}" for v in row["ser_track-hard"]), flush=True)
            print("   kappa rmse  : " + " ".join(f"{v:.1e}" for v in row["kerr"]), flush=True)
    print(f"({time.time()-t0:.0f}s)")
    json.dump({"args": vars(a), "rows": rows}, open(Path(__file__).resolve().parents[1] / "runs" / a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
