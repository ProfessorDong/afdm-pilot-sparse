"""Aggregate a sweep's JSON-lines file into per-point summary statistics."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

RUNS = Path(__file__).resolve().parent / "runs"


def load(name):
    spec = json.load(open(RUNS / f"{name}.spec.json"))
    rows = [json.loads(l) for l in open(RUNS / f"{name}.jsonl")]
    return spec, rows


def summarize(name):
    spec, rows = load(name)
    base = spec.get("base", {})
    by = defaultdict(list)
    fails = 0
    for r in rows:
        if not r.get("ok"):
            fails += 1
            continue
        by[r["point"]].append(r)
    out = []
    for i, g in enumerate(spec["grid"]):
        cfg = {**base, **g}
        rs = by.get(i, [])
        if not rs:
            continue
        B, N, Ncp = cfg.get("B", 16), cfg.get("N", 512), cfg.get("Ncp", 8)
        rec = {"point": i, **g, "trials": len(rs)}
        names = [k for k in rs[0] if isinstance(rs[0][k], dict) and "err" in rs[0][k]]
        for k in names:
            err = np.array([sum(r[k]["err"]) for r in rs], float)
            n = np.array([sum(r[k]["n"]) for r in rs], float)
            gmi = np.array([sum(r[k]["gmi"]) for r in rs], float) / (B * (N + Ncp))
            rec[f"ser_{k}"] = float(err.sum() / n.sum())
            rec[f"gmi_{k}"] = float(gmi.mean())
            rec[f"gmi_{k}_se"] = float(gmi.std(ddof=1) / np.sqrt(len(rs))) if len(rs) > 1 else 0.0
            if "blerr" in rs[0][k]:
                bl = np.array([np.sum(r[k]["blerr"]) for r in rs], float)
                nb = np.array([np.sum(np.array(r[k]["n"]) > 0) for r in rs], float)
                tp = np.array([sum(r[k]["goodbits"]) for r in rs], float) / (B * (N + Ncp))
                rec[f"bler_{k}"] = float(bl.sum() / nb.sum())
                rec[f"tp_{k}"] = float(tp.mean())
                rec[f"tp_{k}_se"] = float(tp.std(ddof=1) / np.sqrt(len(rs))) if len(rs) > 1 else 0.0
            rec[f"time_{k}"] = float(np.mean([r["timing"].get(k, np.nan) for r in rs]))
        out.append(rec)
    return out, fails


if __name__ == "__main__":
    res, fails = summarize(sys.argv[1])
    keys = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    for r in res:
        sel = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
               if keys is None or k in keys or not any(k.startswith(p) for p in ("ser_", "gmi_", "bler_", "tp_", "time_"))}
        print(sel)
    if fails:
        print(f"FAILED trials: {fails}")
