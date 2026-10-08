"""Execution-provenance check: re-simulate recorded frames with the current code.

For every sweep used by the paper, the first completed trial of each grid point is
re-run with `engine.simulate(cfg, seed)` and its per-receiver results (symbol
errors, block errors, decoded payload bits) are compared with the recorded row.
The simulation is deterministic given its seed (one BLAS thread per worker), so
an exact match shows that the released code reproduces the recorded artifacts.
Writes runs/REPLAY.json.
"""
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine import Config, simulate  # noqa: E402

RUNS = Path(__file__).resolve().parent / "runs"
SWEEPS = ["m_main", "m_two", "m_mismatch", "m_robust", "m_frame", "dev_policy", "dev_policy_two",
          "dev_split", "dev_split_two", "dev_c2p_ref", "dev_c2p_irr", "tune_v2", "tune_v3", "tune_v4"]
FIELDS = ("err", "blerr", "goodbits")


def job(args):
    name, point, cfgd, seed, rec = args
    out = simulate(Config(**cfgd), seed)
    bad = []
    for rx, v in rec.items():
        if not isinstance(v, dict) or "err" not in v:
            continue
        for f in FIELDS:
            if f in v and out.get(rx, {}).get(f) != v[f]:
                bad.append(f"{rx}.{f}")
    return name, point, seed, bad


def main(per_point=1, stride=1):
    jobs = []
    for name in SWEEPS:
        sp = RUNS / f"{name}.spec.json"
        if not sp.exists():
            continue
        spec = json.load(open(sp))
        base = spec.get("base", {})
        first = {}
        for line in open(RUNS / f"{name}.jsonl"):
            r = json.loads(line)
            if r.get("ok") and len(first.setdefault(r["point"], [])) < per_point:
                first[r["point"]].append(r)
        for i, g in enumerate(spec["grid"]):
            if i % stride or i not in first:
                continue
            cfgd = {k: (tuple(v) if k == "receivers" else v) for k, v in {**base, **g}.items() if k != "trials"}
            for r in first[i]:
                rec = {k: v for k, v in r.items() if isinstance(v, dict) and "err" in v}
                jobs.append((name, i, cfgd, r["seed"], rec))
    print(f"replaying {len(jobs)} frames", flush=True)
    with Pool(18) as p:
        res = p.map(job, jobs, chunksize=1)
    summary = {}
    for name, point, seed, bad in res:
        s = summary.setdefault(name, {"replayed": 0, "mismatched": []})
        s["replayed"] += 1
        if bad:
            s["mismatched"].append({"point": point, "seed": seed, "fields": bad})
    json.dump(summary, open(RUNS / "REPLAY.json", "w"), indent=1)
    tot = sum(v["replayed"] for v in summary.values())
    mm = sum(len(v["mismatched"]) for v in summary.values())
    print(f"replayed {tot} frames, {mm} mismatches")
    for k, v in summary.items():
        if v["mismatched"]:
            print(k, v["mismatched"][:3])


if __name__ == "__main__":
    main()
