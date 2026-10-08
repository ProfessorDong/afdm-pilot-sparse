"""Pick the tuned superimposed-pilot configurations on held-out seeds (tune_v2)
and write them into specs/m_main.json; flags any optimum on a grid boundary.
- superimposed pilot + proposed tracker: (pilot fraction, acquisition blocks K)
- single-block iterative superimposed pilot: (pilot fraction, iterations)"""
import json

from aggregate import summarize

res, fails = summarize("tune_v2")
for nm in ("tune_v3", "tune_v4"):          # extensions of the grid beyond its boundaries
    try:
        r3, f3 = summarize(nm)
        res, fails = res + r3, fails + f3
    except FileNotFoundError:
        pass
grid = [g for nm in ("tune_v2", "tune_v3", "tune_v4") for g in json.load(open(f"specs/{nm}.json"))["grid"]
        if g.get("trials", 1) > 0]
eps_t = sorted({g["spt_eps"] for g in grid if "spt_eps" in g})
eps_s = sorted({g["sp_eps"] for g in grid if "sp_iters" in g})
its = sorted({g["sp_iters"] for g in grid if "sp_iters" in g})
Ks = (1, 2, 4, 8, 16)
best_t, best_s, best_l = {}, {}, {}
for r in res:
    s = r["snr_db"]
    if r.get("spt_eps") is not None and r["trials"] >= 30:
        for K in Ks:
            if f"tp_sp-track{K}" not in r:
                continue
            tp = r[f"tp_sp-track{K}"]
            # ties (e.g. every configuration error-free on the tuning frames) are broken
            # toward more acquisition blocks, then toward the larger pilot fraction
            key = (tp, K, r["spt_eps"])
            if s not in best_t or key > best_t[s]["key"]:
                best_t[s] = {"spt_eps": r["spt_eps"], "K": K, "tp": tp, "key": key}
            if K <= 4 and (s not in best_l or key > best_l[s]["key"]):      # low-latency variant
                best_l[s] = {"spl_eps": r["spt_eps"], "K": K, "tp": tp, "key": key}
    if "tp_sp" in r and r["trials"] >= 30:
        key = (r["tp_sp"], -r["sp_iters"], -r["sp_eps"])       # ties: fewer iterations, smaller fraction
        if s not in best_s or key > best_s[s]["key"]:
            best_s[s] = {"sp_eps": r["sp_eps"], "sp_iters": r["sp_iters"], "tp": r["tp_sp"], "key": key}
warn = []
for s, b in best_t.items():
    # K = 16 = B is the whole frame (structural maximum, not a grid boundary)
    if b["spt_eps"] in (eps_t[0], eps_t[-1]):
        warn.append(("sp-track", s, b))
for s, b in best_l.items():
    if b["spl_eps"] in (eps_t[0], eps_t[-1]):
        warn.append(("sp-track-ll", s, b))
for s, b in best_s.items():
    if b["sp_eps"] in (eps_s[0], eps_s[-1]) or b["sp_iters"] in (its[0], its[-1]):
        warn.append(("sp", s, b))
print("sp-track:", best_t)
print("sp:", best_s)
print("sp-track-ll:", best_l)
print("BOUNDARY:", warn, "fails:", fails)
for d in (best_t, best_s, best_l):
    for v in d.values():
        v.pop("key", None)
json.dump({"sp_track": {str(k): v for k, v in best_t.items()}, "sp": {str(k): v for k, v in best_s.items()},
           "sp_track_ll": {str(k): v for k, v in best_l.items()},
           "boundary": [list(map(str, w)) for w in warn]}, open("runs/tune_v2_choice.json", "w"), indent=1)

# write the choice (nearest tuning SNR) into the main sweep
m = json.load(open("specs/m_main.json"))
for g in m["grid"]:
    if g.get("Bp") == 1 and "sp-track" in g["receivers"]:
        nt = min(best_t, key=lambda s: abs(s - g["snr_db"]))
        ns = min(best_s, key=lambda s: abs(s - g["snr_db"]))
        g["spt_eps"], g["sp_acq_blocks"] = best_t[nt]["spt_eps"], best_t[nt]["K"]
        g["sp_eps"], g["sp_iters"] = best_s[ns]["sp_eps"], best_s[ns]["sp_iters"]
        g["spl_eps"], g["spl_K"] = best_l[nt]["spl_eps"], best_l[nt]["K"]
        if "sp-track-ll" not in g["receivers"]:
            g["receivers"].insert(g["receivers"].index("sp-track") + 1, "sp-track-ll")
json.dump(m, open("specs/m_main.json", "w"), indent=1)
