"""Generate paper/numbers.tex: every number quoted in the manuscript's prose.

Each macro is computed here from the system design or read from runs/*.
A macro whose source artifact is missing is emitted as a bold '??' so that an
incomplete build is visible in the PDF. Also writes runs/numbers.json, the
audit trail (value + provenance) for every macro.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

from mbafdm import MBAFDM

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
OUT = ROOT / "paper" / "numbers.tex"

NUM = {}


def put(name, value, source, fmt=None):
    NUM[name] = {"tex": value if fmt is None else fmt.format(value), "value": value, "source": source}


def missing(name, source):
    NUM[name] = {"tex": r"\textbf{??}", "value": None, "source": "MISSING: " + source}


# ------------------------------------------------------------------ design
S = MBAFDM()                       # N=512, Ncp=8, alpha_max=3, xi=2, ell_max=7
beta = S.beta
P = 4
Ep = S.Ep
put("NumGuardPct", 100 * len(S.zero_set) / S.N, "|Z|/N, mbafdm.MBAFDM defaults", r"{:.0f}\%")
put("NumGuardGainDb", 10 * np.log10(len(S.zero_set)), "10log10|Z|", "{:.1f}")
eps1 = abs(np.sin(np.pi * S.Ncp / (S.N + S.Ncp))) / (S.N * np.sin(np.pi / (S.N + S.Ncp)))
put("NumEpsOne", eps1, "Theorem 1, eq. (grating), m=1", "{:.4f}")
put("NumEpsOneDb", 20 * np.log10(eps1), "20log10 eps_1", "{:.1f}")
rho1 = P * (3 + 12 * beta ** 2) / (2 * Ep)
put("NumRhoOneDb", 10 * np.log10(rho1), "Corollary 2, eq. (rho1), P=4", "{:.1f}")
put("NumRhoOneLossDb", 10 * np.log10(1 + rho1), "SINR loss 10log10(1+rho1)", "{:.2f}")
rho2 = 5 * P / (2 * (Ep + S.N))
put("NumRhoTwoDb", 10 * np.log10(rho2), "Corollary 2 asymptotic form, b=2", "{:.1f}")
for Bp, nm in ((1, "One"), (2, "Two"), (4, "Four"), (8, "Eight")):
    st2 = (beta ** 2 * (Bp ** 2 - 1) + 1) / 12
    tbar = ((Bp - 1) * beta + 1) / 2
    L = np.arange(Bp, 2000)
    rho = P / (2 * Bp * Ep) * (2 + ((L * beta + 0.5 - tbar) ** 2 + 1 / 12) / st2)
    ok = L[rho <= 0.25]
    put(f"NumHor{nm}", int(ok.max() - Bp + 1) if ok.size else 0, "Corollary 1 exact form, delta=1/4", "{:d}")
# single-block acquisition SNR for a quarter-power path, P_cell >= 1 - 1e-3
# (threshold-aware Proposition 1: global outlier term x local cell term)
from scipy.optimize import brentq
from theory_check import p_global
def _pcell(g):
    loc = 1 - 2 * norm.sf(0.5 / (beta * np.sqrt(3 / (2 * np.pi ** 2 * g))))
    return p_global(g, 1) * loc
g1 = brentq(lambda gdb: _pcell(10 ** (gdb / 10)) - (1 - 1e-3), 0, 30)
put("NumGammaCellDb", g1, "Prop. 1 (threshold-aware), Bp=1, P_cell=1-1e-3", "{:.1f}")
put("NumSnrCellOk", g1 - 10 * np.log10(Ep * 0.25), "Prop. 1 threshold minus 10log10(Ep/4)", "{:.0f}")
# data leakage into the pilot window (deterministic reproduction of check_model [3])
rng = np.random.default_rng(0)
B = 6
leak, pil = [], []
for _ in range(200):
    ls = rng.choice(S.ell_max + 1, 4, replace=False)
    ks = rng.uniform(-S.alpha_max, S.alpha_max, 4)
    hs = (rng.standard_normal(4) + 1j * rng.standard_normal(4)) / np.sqrt(8)
    xd = S.make_frame(B, rng); xd[:, S.m0] = 0
    xp = S.make_frame(B, rng, data=False)
    leak.append(np.sum(np.abs(S.receive(S.channel(S.transmit(xd), ls, ks, hs), B)[:, S.W]) ** 2))
    pil.append(np.sum(np.abs(S.receive(S.channel(S.transmit(xp), ls, ks, hs), B)[:, S.W]) ** 2))
put("NumLeakDb", 10 * np.log10(np.sum(leak) / np.sum(pil)), "data leakage into W, xi=2, 200 draws", "{:.0f}")

# ------------------------------------------------------------------ theory validation
try:
    th = json.load(open(RUNS / "theory_check.json"))
    dev = max(abs(r["p_cell"] - r["p_cell_formula"]) for r in th["e2"])
    put("NumAcqMaxDev", dev, "theory_check e2: max |P_cell sim - Prop.1|", "{:.2f}")
    put("NumAcqTrials", min(r["trials"] for r in th["e2"]), "theory_check e2: trials per point", "{:d}")
    rel = [abs(e / f - 1) for r in th["e3"] for e, f in zip(r["tr_emp"], r["tr_formula"])]
    put("NumTrackMaxDevPct", 100 * max(rel), "theory_check e3: max rel. dev. tracking NMSE vs Thm 2", r"{:.0f}\%")
    rel20 = [abs(e / f - 1) for r in th["e3"] if r["gamma1_db"] == 20 for e, f in zip(r["tr_emp"], r["tr_formula"])]
    put("NumTrackMaxDevPctHi", 100 * max(rel20), "theory_check e3: same, gamma1=20 dB", r"{:.0f}\%")
    put("NumOffDesignDataSnr", 10 - 10 * np.log10(Ep), "data SNR at gamma1=10 dB: 10 - 10log10(Ep)", "{:.1f}")
    put("NumPredTrials", min(r["trials"] for r in th["e3"]), "theory_check e3: in-cell trials per point (min)", "{:d}")
except Exception as e:  # noqa: BLE001
    for n in ("NumAcqMaxDev", "NumAcqTrials", "NumTrackMaxDevPct", "NumTrackMaxDevPctHi", "NumPredTrials"):
        missing(n, f"theory_check ({e!r})")

# ------------------------------------------------------------------ results
try:
    from aggregate import summarize
    e4, _ = summarize("e4_main")
    full = [r for r in e4 if r.get("Bp") == 1 and "tp_conv" in r]
    gains = [(r["snr_db"], r["tp_track"] / r["tp_conv"] - 1, r["tp_track"] / r["tp_genie"]) for r in full]
    hi = [g for g in gains if g[0] >= 10]
    put("NumGainConvPct", 100 * np.mean([g[1] for g in hi]), "e4_main: mean tp_track/tp_conv-1 over SNR>=10 dB", r"{:.0f}\%")
    # lowest SNR from which tracking stays within 2% of genie throughput
    ok = [s for s, _, ratio in sorted(gains) if ratio >= 0.98]
    first = min(s for s in ok if all(rr >= 0.98 for ss, _, rr in gains if ss >= s)) if ok else None
    put("NumSnrGenieMatch", first, "e4_main: lowest SNR with tp_track >= 0.98 tp_genie at and above it", "{:d}")
    put("NumGapGeniePct", 100 * max(1 - g[2] for g in gains if g[0] >= first), "e4_main: max genie gap above match SNR", r"{:.1f}\%")
except Exception as e:  # noqa: BLE001
    for n in ("NumGainConvPct", "NumSnrGenieMatch", "NumGapGeniePct"):
        missing(n, f"e4_main ({e!r})")

# ------------------------------------------------------------------ write
lines = ["% GENERATED by make_numbers.py -- do not edit by hand.\n"]
for k in sorted(NUM):
    lines.append(f"\\newcommand{{\\{k}}}{{{NUM[k]['tex']}}}  % {NUM[k]['source']}\n")
OUT.write_text("".join(lines))
json.dump({k: {"value": v["value"], "tex": v["tex"], "source": v["source"]} for k, v in NUM.items()},
          open(RUNS / "numbers.json", "w"), indent=1, default=float)
print("".join(lines))
