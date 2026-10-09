"""Number of superimposed pilot chirps M: held-out selection over M in {1, 2, 4}.

Protocol, fixed before any M > 1 result was seen (2026-10-08):
- M = 1 results are the existing tuning sweeps tune_v2-v4; M = 2, 4 are tune_v5 (pilot
  + tracker, K in {1, 2, 4, 8, 16}) and tune_v6 (single-block version, iterations fixed
  at the M = 1 optimum of each SNR). All use the same held-out seeds (seed_by "snr").
- Every configuration is scored on the seeds common to all compared points of its SNR
  (paired), so different M are compared on identical channels and noise.
- Selection per SNR and receiver:
    sp-track     key (throughput, -M, K, eps)       K <= 16
    sp-track-ll  key (throughput, -M, K, eps)       K <= 4
    sp           key (throughput, -M, -iters, -eps)
  i.e. the existing rules, with exact ties resolved toward the smaller M.
- If some SNR selects M > 1, the evaluation sweep m_main is rerun for the superimposed
  receivers with the selected configurations, on the same evaluation channels.
- An optimum on a grid boundary of eps is flagged.
Writes runs/tune_v5_choice.json."""
import json
from collections import defaultdict

import numpy as np

from diagnostics import frames

B, SAMPLES = 16, 16 * 520


def rows_by_config(name):
    """{(snr, config...): {seed: throughput}} for every superimposed receiver of a sweep."""
    spec, by = frames(name)
    out = defaultdict(dict)
    for i, g in enumerate(spec["grid"]):
        for r in by[i]:
            for rx, v in r.items():
                if not (isinstance(v, dict) and "goodbits" in v):
                    continue
                tp = sum(v["goodbits"]) / SAMPLES
                if rx.startswith("sp-track") and rx[8:].isdigit():
                    key = ("sp-track", g["snr_db"], g.get("spt_M", 1), int(rx[8:]), g["spt_eps"])
                elif rx == "sp":
                    key = ("sp", g["snr_db"], g.get("sp_M", 1), g["sp_iters"], g["sp_eps"])
                else:
                    continue
                out[key][r["seed"]] = tp
    return out


def main():
    data = {}
    for nm in ("tune_v2", "tune_v3", "tune_v4", "tune_v5", "tune_v6"):
        try:
            for k, v in rows_by_config(nm).items():
                data.setdefault(k, {}).update(v)
        except FileNotFoundError:
            print("missing", nm)
    choice, paired = {}, {}
    for fam, kmax, label in (("sp-track", 16, "sp_track"), ("sp-track", 4, "sp_track_ll"), ("sp", None, "sp")):
        for snr in (4, 8, 12, 16, 20):
            cand = {k: v for k, v in data.items() if k[0] == fam and k[1] == snr and (kmax is None or k[3] <= kmax)}
            if fam == "sp":                       # M > 1 used the iteration count of the M = 1 optimum
                its = {k[3] for k in cand if k[2] > 1}
                cand = {k: v for k, v in cand.items() if k[3] in its}
            common = set.intersection(*(set(v) for v in cand.values()))
            seeds = sorted(common)
            score = {k: float(np.mean([v[s] for s in seeds])) for k, v in cand.items()}
            if fam == "sp":
                best = max(score, key=lambda k: (score[k], -k[2], -k[3], -k[4]))
            else:
                best = max(score, key=lambda k: (score[k], -k[2], k[3], k[4]))
            per_M = {}
            for M in (1, 2, 4):
                ks = [k for k in score if k[2] == M]
                if ks:
                    kb = max(ks, key=lambda k: (score[k], k[3], k[4]) if fam != "sp" else (score[k], -k[3], -k[4]))
                    per_M[M] = kb
            # paired difference of the best M > 1 configuration to the best M = 1 one
            d = {}
            for M in (2, 4):
                if M in per_M:
                    a = np.array([data[per_M[M]][s] for s in seeds]); b = np.array([data[per_M[1]][s] for s in seeds])
                    rng = np.random.default_rng(snr * 10 + M)
                    I = rng.integers(0, len(seeds), (4000, len(seeds)))
                    bs = a[I].mean(1) / b[I].mean(1) - 1
                    d[M] = {"rel": float(a.mean() / b.mean() - 1), "ci95": [float(x) for x in np.percentile(bs, [2.5, 97.5])],
                            "cfg": list(per_M[M]), "ref": list(per_M[1])}
            choice.setdefault(label, {})[str(snr)] = {"M": best[2], "K_or_iters": best[3], "eps": best[4],
                                                      "tp": score[best], "n_seeds": len(seeds)}
            paired.setdefault(label, {})[str(snr)] = d
    eps_t = sorted({k[4] for k in data if k[0] == "sp-track" and k[2] > 1})
    eps_s = sorted({k[4] for k in data if k[0] == "sp" and k[2] > 1})
    boundary = [(lab, s, c) for lab, cc in choice.items() for s, c in cc.items()
                if c["M"] > 1 and c["eps"] in ((eps_s[0], eps_s[-1]) if lab == "sp" else (eps_t[0], eps_t[-1]))]
    json.dump({"choice": choice, "paired_best_M_vs_M1": paired, "boundary": boundary},
              open("runs/tune_v5_choice.json", "w"), indent=1)
    for lab, cc in choice.items():
        print(lab, {s: (c["M"], c["K_or_iters"], c["eps"], round(c["tp"], 4)) for s, c in cc.items()})
    for lab, pp in paired.items():
        print(lab, {s: {M: (round(100 * v["rel"], 2), [round(100 * x, 2) for x in v["ci95"]]) for M, v in d.items()}
                    for s, d in pp.items()})
    print("boundary:", boundary)


if __name__ == "__main__":
    main()
