"""Frame-level diagnostics and paired uncertainty for the sweeps.

- Path association (initial acquisition and final tracked set) against the true
  channel: a detected path (l, k) matches a true path (tau, kappa) if
  |l - tau| < dl and |k - kappa| < 1/(2 beta), one to one, strongest true path
  first. Reported: fraction of true paths detected, missed channel-energy
  fraction, false (unmatched) paths per frame.
- Prediction-error-to-noise ratio of each predicted block (tracker diagnostics).
- Paired frame bootstrap of throughput ratios and differences: frames (channel
  realizations) are resampled with all receivers kept together.
"""
import json
from collections import defaultdict

import numpy as np

from aggregate import load

BETA = 520 / 512


def associate(det, tau, kap, habs, dl, born=None):
    """det: list of (l, k, ...). Returns (n_true, n_detected_true, missed_energy, n_false)."""
    alive = np.ones(len(tau), bool) if born is None else np.asarray(born) == 0
    order = [p for p in np.argsort(-np.asarray(habs)) if alive[p]]
    used = set()
    hit = 0
    missed = 0.0
    for p in order:
        best = None
        for i, d in enumerate(det):
            if i in used:
                continue
            if abs(d[0] - tau[p]) < dl and abs(d[1] - kap[p]) < 0.5 / BETA:
                best = i
                break
        if best is None:
            missed += habs[p] ** 2
        else:
            used.add(best)
            hit += 1
    tot = sum(habs[p] ** 2 for p in order)
    return len(order), hit, missed / tot if tot > 0 else 0.0, len(det) - len(used)


def frames(name):
    spec, rows = load(name)
    by = defaultdict(list)
    for r in rows:
        if r.get("ok"):
            by[r["point"]].append(r)
    return spec, by


def path_stats(rows, key="acq", dl=1.0):
    nt = nd = 0
    me, nf = [], []
    for r in rows:
        if key == "acq":
            det = r.get("acq")
        else:
            det = r.get(key, {}).get("final")
        if det is None:
            continue
        t = r["true"]
        a, b, m, f = associate(det, t["tau"], t["kappa"], t["h_abs"], dl, t.get("born"))
        nt += a; nd += b; me.append(m); nf.append(f)
    if not me:
        return None
    return dict(p_detect=nd / max(nt, 1), missed_energy=float(np.mean(me)),
                missed_energy_p90=float(np.percentile(me, 90)), false_per_frame=float(np.mean(nf)))


def rho_stats(rows, key="diag-track"):
    R = np.array([r[key]["rho_pred"] for r in rows if key in r])
    if R.size == 0:
        return None
    crc = np.array([r[key]["crc"] for r in rows if key in r])
    return dict(rho_mean_db=(10 * np.log10(R.mean(0))).tolist(),
                rho_median_db=(10 * np.log10(np.median(R, 0))).tolist(),
                crc_fail=(1 - crc.mean(0)).tolist(),
                insert_per_frame=float(np.mean([r[key]["n_insert"] for r in rows if key in r])),
                merge_per_frame=float(np.mean([r[key]["n_merge"] for r in rows if key in r])))


def tp_frame(r, rx, B=16, N=512, Ncp=8):
    return sum(r[rx]["goodbits"]) / (B * (N + Ncp))


def paired(rows, a, b, nboot=4000, seed=0, B=16):
    """Paired frame bootstrap of mean(a)/mean(b) - 1 and mean(a) - mean(b)."""
    rr = [r for r in rows if a in r and b in r]
    x = np.array([tp_frame(r, a, B) for r in rr])
    y = np.array([tp_frame(r, b, B) for r in rr])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (nboot, len(x)))
    xm, ym = x[idx].mean(1), y[idx].mean(1)
    ratio = xm / np.maximum(ym, 1e-12) - 1
    diff = xm - ym
    return dict(n=len(x), ratio=float(x.mean() / max(y.mean(), 1e-12) - 1),
                ratio_lo=float(np.percentile(ratio, 2.5)), ratio_hi=float(np.percentile(ratio, 97.5)),
                diff=float(x.mean() - y.mean()),
                diff_lo=float(np.percentile(diff, 2.5)), diff_hi=float(np.percentile(diff, 97.5)))


if __name__ == "__main__":
    import sys
    spec, by = frames(sys.argv[1])
    for i, g in enumerate(spec["grid"]):
        rows = by.get(i, [])
        if not rows:
            continue
        print(i, {k: v for k, v in g.items() if k != "receivers"}, len(rows))
        print("   acq   ", path_stats(rows, "acq", 1.0))
        for k in ("diag-track", "diag-track-strict"):
            st = rho_stats(rows, k)
            if st:
                print("  ", k, "rho(dB) b=1,2,4,8,15:", [round(st["rho_mean_db"][b], 1) for b in (1, 2, 4, 8, 15)],
                      "crcfail", round(float(np.mean(st["crc_fail"])), 3), "ins", round(st["insert_per_frame"], 2),
                      "merge", round(st["merge_per_frame"], 2))
                print("   final ", path_stats(rows, k, 0.5))
