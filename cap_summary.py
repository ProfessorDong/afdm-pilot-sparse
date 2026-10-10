"""Order-cap sensitivity of the proposed receiver (runs/m_cap*.jsonl), paired against the
evaluated cap P_max = 8 on identical channels and noise (m_main, and m_cap_p8 for P = 8).
Writes runs/cap_sensitivity.json: per (case, cap) the throughput, its paired relative
difference to cap 8 with a whole-frame bootstrap interval, the fraction of frames whose
acquisition stops at the cap, the mean final model order, the median last-block
prediction-error-to-noise ratio and the mean run time."""
import json
from pathlib import Path

import numpy as np

RUNS = Path(__file__).resolve().parent / "runs"
SPECS = Path(__file__).resolve().parent / "specs"
SAMPLES = 16 * 520


def load(name):
    spec = json.load(open(RUNS / f"{name}.spec.json"))
    rows = {}
    for line in open(RUNS / f"{name}.jsonl"):
        r = json.loads(line)
        g = spec["grid"][r["point"]]
        if r.get("ok") and g.get("Bp", 1) == 1:       # one pilot block only (m_main also has Bp > 1)
            key = (g["snr_db"], g.get("P_max", spec["base"].get("P_max")), r["seed"])
            assert key not in rows, key
            rows[key] = r
    return rows


def stats(rows):
    tp = np.array([sum(r["track"]["goodbits"]) / SAMPLES for r in rows])
    return tp


def main():
    out = {}
    base = load("m_main")
    cap = load("m_cap")
    cases = sorted({(k[0], k[1]) for k in cap})
    for snr in sorted({c[0] for c in cases}):
        ref = {k[2]: v for k, v in base.items() if k[0] == snr and k[1] == 8}
        for pm in [8] + sorted(c[1] for c in cases if c[0] == snr):
            cur = ref if pm == 8 else {k[2]: v for k, v in cap.items() if k[0] == snr and k[1] == pm}
            out[f"P4@{snr:g}dB/cap{pm}"] = summarize(cur, ref, pm)
    p8 = load("m_cap_p8")
    snr = sorted({k[0] for k in p8})[0]
    ref = {k[2]: v for k, v in p8.items() if k[1] == 8}
    for pm in sorted({k[1] for k in p8}):
        cur = {k[2]: v for k, v in p8.items() if k[1] == pm}
        out[f"P8@{snr:g}dB/cap{pm}"] = summarize(cur, ref, pm)
    json.dump(out, open(RUNS / "cap_sensitivity.json", "w"), indent=1)
    for k, v in out.items():
        print(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()})


def summarize(cur, ref, pm):
    seeds = sorted(set(cur) & set(ref))
    a = np.array([sum(cur[s]["track"]["goodbits"]) / SAMPLES for s in seeds])
    b = np.array([sum(ref[s]["track"]["goodbits"]) / SAMPLES for s in seeds])
    rng = np.random.default_rng(int(pm) * 7 + len(seeds))
    idx = rng.integers(0, len(seeds), (4000, len(seeds)))
    bs = a[idx].mean(1) / b[idx].mean(1) - 1
    hit = np.mean([len(cur[s]["acq"]) >= pm for s in seeds])
    order = np.mean([len(cur[s]["diag-track"]["final"]) for s in seeds if "diag-track" in cur[s]])
    rho15 = np.median([cur[s]["diag-track"]["rho_pred"][-1] for s in seeds if "diag-track" in cur[s]])
    t = np.mean([cur[s]["timing"]["track"] for s in seeds])
    return {"n": len(seeds), "tp": float(a.mean()), "rel_vs_cap8": float(a.mean() / b.mean() - 1),
            "ci95": [float(x) for x in np.percentile(bs, [2.5, 97.5])], "cap_hit": float(hit),
            "final_order": float(order), "rho_last_db": float(10 * np.log10(rho15)), "time_s": float(t)}


if __name__ == "__main__":
    main()
