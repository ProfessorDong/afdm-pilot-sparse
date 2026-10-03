"""Parallel sweep runner. Every trial is appended to a JSON-lines file as soon
as it finishes, so an interrupted sweep loses nothing and can be resumed.

Usage: python run_sweep.py SPEC.json
SPEC = {"name": str, "trials": int, "base": {Config overrides},
        "grid": [{Config overrides}, ...], "seed0": int}
"""
import json
import os
import sys
import time
from dataclasses import asdict, replace
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine import Config, simulate  # noqa: E402

RUNS = Path(__file__).resolve().parent / "runs"


def job(args):
    point, cfgd, seed = args
    cfg = Config(**cfgd)
    t0 = time.time()
    try:
        r = simulate(cfg, seed)
        r["ok"] = True
    except Exception as e:  # keep the sweep alive; record the failure
        r = {"ok": False, "error": repr(e)}
    r.update({"point": point, "seed": seed, "wall": time.time() - t0})
    return r


def main():
    spec = json.load(open(sys.argv[1]))
    out = RUNS / f"{spec['name']}.jsonl"
    done = set()
    if out.exists():
        for line in open(out):
            d = json.loads(line)
            done.add((d["point"], d["seed"]))
    base = spec.get("base", {})
    jobs = []
    for i, g in enumerate(spec["grid"]):
        cfgd = {**base, **g}
        for k, v in list(cfgd.items()):
            if isinstance(v, list) and k == "receivers":
                cfgd[k] = tuple(v)
        for t in range(spec["trials"]):
            seed = spec.get("seed0", 0) + 7919 * t + 104729 * i
            if (i, seed) not in done:
                jobs.append((i, cfgd, seed))
    meta = RUNS / f"{spec['name']}.spec.json"
    json.dump(spec, open(meta, "w"), indent=1)
    print(f"{spec['name']}: {len(jobs)} jobs ({len(done)} already done)", flush=True)
    t0 = time.time()
    n = 0
    with Pool(int(spec.get("procs", 18))) as pool, open(out, "a") as f:
        for r in pool.imap_unordered(job, jobs):
            f.write(json.dumps(r) + "\n"); f.flush()
            n += 1
            if n % max(1, len(jobs) // 20) == 0:
                print(f"  {n}/{len(jobs)}  {time.time()-t0:.0f}s", flush=True)
    print(f"done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
