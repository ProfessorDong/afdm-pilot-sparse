"""Pick the superimposed-pilot configuration (eps, iters) that maximizes coded
throughput at each tuning SNR (held-out seeds), and write it into the main
experiment spec for the nearest tuned SNR. Records the choice for the paper."""
import json
from aggregate import summarize

res, fails = summarize("tune_sp")
res2, fails2 = summarize("tune_sp_ext")
res, fails = res + res2, fails + fails2
best = {}
for r in res:
    s = r["snr_db"]
    if s not in best or r["tp_sp"] > best[s]["tp_sp"]:
        best[s] = r
choice = {s: {"sp_eps": b["sp_eps"], "sp_iters": b["sp_iters"], "tp": b["tp_sp"]} for s, b in best.items()}
print("best per tuning SNR:", choice, "fails:", fails)
for s in sorted(best):
    row = sorted([r for r in res if r["snr_db"] == s], key=lambda r: -r["tp_sp"])
    print(s, [(r["sp_eps"], r["sp_iters"], round(r["tp_sp"], 3)) for r in row])
spec = json.load(open("specs/e4_main.json"))
for g in spec["grid"]:
    if "sp" in g.get("receivers", []):
        near = min(choice, key=lambda s: abs(s - g["snr_db"]))
        g["sp_eps"], g["sp_iters"] = choice[near]["sp_eps"], choice[near]["sp_iters"]
json.dump(spec, open("specs/e4_main.json", "w"), indent=1)
json.dump({str(k): v for k, v in choice.items()}, open("runs/sp_choice.json", "w"), indent=1)
