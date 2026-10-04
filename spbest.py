"""Superimposed pilot with the proposed tracker, acquired from the best number
K of blocks in {1, 2, 4} at each SNR (e4b_baselines has K=1; e4d_sp_k2/k4
rerun the same channel draws with K=2 and K=4). Picking the best K on the
evaluation draws favors this baseline."""
from aggregate import summarize


def sp_best():
    runs = {1: "e4b_baselines", 2: "e4d_sp_k2", 4: "e4d_sp_k4"}
    best = {}
    for K, name in runs.items():
        try:
            res, _ = summarize(name)
        except FileNotFoundError:
            continue
        for r in res:
            if "tp_sp-track" not in r or r["trials"] < 200:
                continue
            s = r["snr_db"]
            if s not in best or r["tp_sp-track"] > best[s]["tp"]:
                best[s] = {"tp": r["tp_sp-track"], "bler": r["bler_sp-track"], "K": K,
                           "time": r["time_sp-track"], "se": r["tp_sp-track_se"]}
    return best
