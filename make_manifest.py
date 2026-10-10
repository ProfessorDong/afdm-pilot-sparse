"""Release manifest: ties every published number, figure and table to the raw
artifacts that produced it. Writes runs/MANIFEST.json with
- the git commit and the software environment,
- for every sweep used by the paper: SHA-256 of its JSON-lines file, the
  attempted (spec), completed and failed trial counts per grid point,
- SHA-256 of every generated output (numbers, figures, tables) if present.
"""
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
PAPER = ROOT / "paper"
SWEEPS = ["m_main", "m_two", "m_mismatch", "m_robust", "m_frame", "tune_v2", "tune_v3", "tune_v4",
          "dev_c2p_ref", "dev_c2p_irr", "dev_policy", "dev_policy_two", "dev_split", "dev_split_two",
          "tune_v5", "tune_v6", "m_main_mp", "m_main_lo", "m_cap", "m_cap_p8"]
SINGLE = ["theory_check.json", "checks.json", "oracle_guard.json", "numbers.json", "paired.json",
          "tune_v2_choice.json", "tune_v5_choice.json", "physical_mapping.json", "fig_main_meta.json", "fig_diag_meta.json", "REPLAY.json"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    import numpy
    import scipy
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                             cwd=ROOT, text=True).strip())
    except Exception:  # noqa: BLE001
        commit, dirty = None, None
    man = {"commit": commit, "uncommitted_changes": dirty,
           "environment": {"python": sys.version.split()[0], "numpy": numpy.__version__, "scipy": scipy.__version__,
                           "platform": platform.platform(), "processor": platform.processor()},
           "sweeps": {}, "artifacts": {}, "outputs": {}}
    for name in SWEEPS:
        jl, sp = RUNS / f"{name}.jsonl", RUNS / f"{name}.spec.json"
        if not jl.exists():
            continue
        spec = json.load(open(sp))
        rows = [json.loads(l) for l in open(jl)]
        pts = []
        for i, g in enumerate(spec["grid"]):
            rs = [r for r in rows if r["point"] == i]
            pts.append({"point": i, "attempted": g.get("trials", spec["trials"]),
                        "completed": sum(1 for r in rs if r.get("ok")), "failed": sum(1 for r in rs if not r.get("ok"))})
        man["sweeps"][name] = {"sha256": sha(jl), "spec_sha256": sha(sp), "records": len(rows),
                               "failed_total": sum(p["failed"] for p in pts), "points": pts}
    for f in SINGLE:
        if (RUNS / f).exists():
            man["artifacts"][f] = sha(RUNS / f)
    if (RUNS / "REPLAY.json").exists():
        rp = json.load(open(RUNS / "REPLAY.json"))
        man["replay"] = {k: {"replayed": v["replayed"], "mismatched": len(v["mismatched"])} for k, v in rp.items()}
    pdf = PAPER / "AFDM_TVT.pdf"
    if pdf.exists():
        man["manuscript_pdf_sha256"] = sha(pdf)
    if PAPER.exists():
        for p in sorted(PAPER.glob("*.tex")):
            if p.name.startswith(("numbers", "fig_", "tab_")):
                man["outputs"][p.name] = sha(p)
    json.dump(man, open(RUNS / "MANIFEST.json", "w"), indent=1)
    bad = {k: v["failed_total"] for k, v in man["sweeps"].items() if v["failed_total"]}
    short = {k: [p for p in v["points"] if p["completed"] < p["attempted"]] for k, v in man["sweeps"].items()}
    print("commit", commit, "dirty", dirty, "| failed trials:", bad or "none",
          "| incomplete points:", {k: len(v) for k, v in short.items() if v} or "none")


if __name__ == "__main__":
    main()
