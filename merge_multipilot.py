"""Merge the rerun of the superimposed receivers (runs/m_main_mp.jsonl, held-out choice of
the number M of pilot chirps, pick_v3) into the evaluation sweep runs/m_main.jsonl.

Only the rerun receivers of the affected points are replaced, matched by seed; every other
receiver is untouched. The previous file is kept in runs/superseded/. The evaluation spec
(specs/m_main.json and runs/m_main.spec.json) records the new M, so replay.py re-simulates
the merged rows from scratch."""
import json
import shutil
from pathlib import Path

RUNS = Path(__file__).resolve().parent / "runs"
SPECS = Path(__file__).resolve().parent / "specs"


def main():
    mp_spec = json.load(open(SPECS / "m_main_mp.json"))
    mp = {}
    for line in open(RUNS / "m_main_mp.jsonl"):
        r = json.loads(line)
        assert r.get("ok"), "failed trial in m_main_mp"
        g = mp_spec["grid"][r["point"]]
        mp[(g["snr_db"], r["seed"])] = (g, r)
    arch = RUNS / "superseded" / "m_main_before_multipilot.jsonl"
    if not arch.exists():
        shutil.copy(RUNS / "m_main.jsonl", arch)
    for sp in (SPECS / "m_main.json", RUNS / "m_main.spec.json"):
        spec = json.load(open(sp))
        for g in spec["grid"]:
            for ng in mp_spec["grid"]:
                if g.get("Bp") == 1 and "sp-track" in g["receivers"] and g["snr_db"] == ng["snr_db"]:
                    for k in ("sp_M", "spt_M", "spl_M", "spt_eps", "sp_acq_blocks", "spl_eps", "spl_K",
                              "sp_eps", "sp_iters"):
                        g[k] = ng[k]
        json.dump(spec, open(sp, "w"), indent=1)
    spec = json.load(open(RUNS / "m_main.spec.json"))
    out, n_rep = [], 0
    for line in open(arch):
        r = json.loads(line)
        g = spec["grid"][r["point"]]
        key = (g["snr_db"], r["seed"])
        if r.get("ok") and g.get("Bp") == 1 and key in mp:
            ng, rr = mp[key]
            for rx in ng["receivers"]:
                if not rx.startswith("rng:"):
                    r[rx] = rr[rx]
                    if "timing" in r and rx in rr.get("timing", {}):
                        r["timing"][rx] = rr["timing"][rx]
            n_rep += 1
        out.append(json.dumps(r))
    expected = sum(g["trials"] for g in mp_spec["grid"])
    assert n_rep == expected == len(mp), (n_rep, expected, len(mp))
    # second pass: receivers after the superimposed frames at points where the frame count changed
    lo_path = RUNS / "m_main_lo.jsonl"
    if lo_path.exists():
        lo_spec = json.load(open(SPECS / "m_main_lo.json"))
        lo = {}
        for line in open(lo_path):
            r = json.loads(line)
            assert r.get("ok"), "failed trial in m_main_lo"
            g = lo_spec["grid"][r["point"]]
            lo[(g["snr_db"], r["seed"])] = (g, r)
        n_lo = 0
        for i, line in enumerate(out):
            r = json.loads(line)
            g = spec["grid"][r["point"]]
            key = (g["snr_db"], r["seed"])
            if r.get("ok") and g.get("Bp") == 1 and key in lo:
                ng, rr = lo[key]
                for rx in [x for x in ng["receivers"] if not x.startswith("rng:")] + ["ofdm-track"]:
                    r[rx] = rr[rx]
                    if "timing" in r and rx in rr.get("timing", {}):
                        r["timing"][rx] = rr["timing"][rx]
                out[i] = json.dumps(r)
                n_lo += 1
        assert n_lo == sum(g["trials"] for g in lo_spec["grid"]) == len(lo), (n_lo, len(lo))
        print(f"merged {n_lo} rows from m_main_lo")
    open(RUNS / "m_main.jsonl", "w").write("\n".join(out) + "\n")
    print(f"merged {n_rep} rows; previous file kept as {arch}")


if __name__ == "__main__":
    main()
