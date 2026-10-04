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
    tex = value if fmt is None else fmt.format(value)
    if isinstance(tex, str) and tex.startswith("-"):
        tex = "\\ensuremath{" + tex + "}"          # true minus sign, not a text hyphen
    NUM[name] = {"tex": tex, "value": value, "source": source}


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

# ------------------------------------------------------------------ grating-region peak sidelobe
_gp = json.load(open(RUNS / "grating_peak.json"))
for _B, _nm in (("2", "Two"), ("16", "Sixteen")):
    put(f"NumGratPeak{_nm}Db", _gp[_B], f"grating_peak.json: max |chi_B| over Delta in [0.6,1.4], B={_B}", "{:.1f}")

# ------------------------------------------------------------------ exact rho (Cor. 2) as simulated
def _rho(blocks_w, bnext, P=4):
    t, w = [], []
    for b, e in blocks_w:
        t += list((b * (S.N + S.Ncp) + S.Ncp + np.arange(S.N)) / S.N); w += [e / S.N] * S.N
    t, w = np.array(t), np.array(w)
    tb = np.sum(w * t) / np.sum(w); st = np.sum(w * (t - tb) ** 2) / np.sum(w)
    tn = (bnext * (S.N + S.Ncp) + S.Ncp + np.arange(S.N)) / S.N
    return P / (2 * np.sum(w)) * (2 + np.mean((tn - tb) ** 2) / st)
put("NumRhoOneDataDb", 10 * np.log10(_rho([(0, S.N)], 1)), "exact rho(1), pilot block with decoded data (energy N)", "{:.1f}")
put("NumRhoTwoExactDb", 10 * np.log10(_rho([(0, S.N), (1, S.N)], 2)), "exact rho(2), two full-energy blocks", "{:.1f}")
put("NumRhoTenExactDb", 10 * np.log10(_rho([(b, S.N) for b in range(10)], 10)), "exact rho(10), ten full-energy blocks", "{:.1f}")
for Bp, nm in ((1, "One"), (2, "Two"), (4, "Four"), (8, "Eight")):
    L = np.arange(Bp, 4000)
    st2 = (beta ** 2 * (Bp ** 2 - 1) + 1) / 12; tbar = ((Bp - 1) * beta + 1) / 2
    rho_d = P / (2 * Bp * S.N) * (2 + ((L * beta + 0.5 - tbar) ** 2 + 1 / 12) / st2)
    ok = L[rho_d <= 0.25]
    put(f"NumHorData{nm}", int(ok.max() - Bp + 1) if ok.size else 0, "Cor. 1 with decoded pilot-block data (energy N per block)", "{:d}")
# guard fraction and overhead ceiling vs N (same channel: l_max=7, alpha_max=3, xi=2)
for Nn, nm in ((1024, "Kilo"), (2048, "TwoKilo")):
    Sn = MBAFDM(N=Nn, Ncp=8, alpha_max=3, xi=2, ell_max=7)
    put(f"NumGuardPct{nm}", 100 * len(Sn.zero_set) / Nn, f"|Z|/N at N={Nn}", r"{:.0f}\%")
put("NumOverheadGainPct", 100 * (1 / (1 - len(S.zero_set) / S.N) - 1), "1/(1-|Z|/N)-1: upper bound of the guard-removal gain", r"{:.0f}\%")

# ------------------------------------------------------------------ receiver constants (from code)
from mbtrack import Tracker
_T = Tracker(S)
put("NumMergeDl", _T.merge_dl, "mbtrack.Tracker.merge_dl", "{:g}")
put("NumMergeDk", _T.merge_dk, "mbtrack.Tracker.merge_dk", "{:g}")
put("NumRetries", ["zero", "one", "two", "three", "four"][_T.retries], "mbtrack.Tracker.retries (word)")

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
    put("NumHiGammaDataSnr", 20 - 10 * np.log10(Ep), "data SNR at gamma1=20 dB: 20 - 10log10(Ep)", "{:.1f}")
    put("NumPredTrials", min(r["trials"] for r in th["e3"]), "theory_check e3: in-cell trials per point (min)", "{:d}")
except Exception as e:  # noqa: BLE001
    for n in ("NumAcqMaxDev", "NumAcqTrials", "NumTrackMaxDevPct", "NumTrackMaxDevPctHi", "NumPredTrials"):
        missing(n, f"theory_check ({e!r})")

# ------------------------------------------------------------------ results (e4_main + e4b + e4c, same channels)
_E4 = ("NumGainConvPct", "NumSnrGenieMatch", "NumGapGeniePct", "NumSpTopSnr", "NumSpFromSnr", "NumSpMaxAdvPct",
       "NumOverSpEightPct", "NumOverConvEightPct", "NumGenieRatioEightPct", "NumGenieRatioSixPct",
       "NumGainConvMaxPct", "NumOverOfdmMinPct", "NumOverOfdmMaxPct", "NumOfdmBetterSnr", "NumOverOpenMinPct",
       "NumOverOpenMaxPct", "NumEvalTrials", "NumTimeSp", "NumTimeTrack", "NumTimeConv", "NumTimeGenie",
       "NumBpTwoBetterTop", "NumOpenBestBpMax", "NumOpenBestBpMin", "NumBlerFloorPct", "NumConvCeilGainPct",
       "NumPlainConvHiPct", "NumSpSingleMaxDiff", "NumOverInterpMinPct", "NumOverInterpMaxPct", "NumOfdmCeilGapPct",
       "NumTimeConvDa", "NumTimeSpTrack", "NumTimeTrackMin", "NumTimeTrackMax")
try:
    from aggregate import summarize
    e4, _ = summarize("e4_main")
    e4b, _ = summarize("e4b_baselines")
    e4c, _ = summarize("e4c_interp")
    by = {r["snr_db"]: r for r in e4 if r.get("Bp") == 1 and "tp_conv" in r}
    bp2 = {r["snr_db"]: r for r in e4 if r.get("Bp") == 2}
    bb = {r["snr_db"]: r for r in e4b}
    cc = {r["snr_db"]: r for r in e4c}
    S_ = sorted(by)
    tr = {s: by[s]["tp_track"] for s in S_}
    ge = {s: by[s]["tp_genie"] for s in S_}
    put("NumEvalTrials", min(r["trials"] for r in e4 + e4b + e4c), "e4*: trials per point (min)", "{:d}")
    # perfect CSI
    ratio = {s: tr[s] / ge[s] for s in S_}
    first = min(s for s in S_ if all(ratio[t] >= 0.98 for t in S_ if t >= s))
    put("NumSnrGenieMatch", first, "lowest SNR with track >= 0.98 genie at and above it", "{:d}")
    put("NumGapGeniePct", 100 * max(1 - ratio[s] for s in S_ if s >= first), "max genie gap above that SNR", r"{:.1f}\%")
    put("NumGenieRatioEightPct", 100 * ratio[8], "track/genie at 8 dB", r"{:.0f}\%")
    put("NumGenieRatioSixPct", 100 * ratio[6], "track/genie at 6 dB", r"{:.0f}\%")
    put("NumBlerFloorPct", 100 * max(by[s]["bler_track"] for s in S_ if s > 12), "max track BLER above 12 dB", r"{:.1f}\%")
    # conventional (data-aided) and its ceiling
    cd = {s: bb[s]["tp_conv-da"] for s in S_}
    put("NumGainConvPct", 100 * np.mean([tr[s] / cd[s] - 1 for s in S_ if s >= 10]), "mean track/conv-da-1, SNR>=10", r"{:.0f}\%")
    put("NumOverConvEightPct", 100 * (tr[8] / cd[8] - 1), "track/conv-da-1 at 8 dB", r"{:.0f}\%")
    put("NumGainConvMaxPct", 100 * max(tr[s] / cd[s] - 1 for s in S_ if s >= 6), "max track/conv-da-1, SNR>=6", r"{:.0f}\%")
    put("NumConvCeilGainPct", 100 * (ge[20] / bb[20]["tp_genie-conv"] - 1), "perfect-CSI ceiling ratio, 20 dB", r"{:.0f}\%")
    put("NumPlainConvHiPct", 100 * max(by[s]["bler_conv"] for s in S_ if s >= 14), "plain conv BLER max >=14 dB", r"{:.1f}\%")
    # superimposed pilot with the same tracker
    sp = {s: bb[s]["tp_sp-track"] for s in S_}
    win = [s for s in S_ if sp[s] > tr[s]]
    put("NumSpFromSnr", min(win), "lowest SNR where sp-track > track", "{:d}")
    put("NumSpTopSnr", max(s for s in S_ if s < min(win)), "highest SNR below that", "{:d}")
    put("NumSpMaxAdvPct", 100 * max(sp[s] / tr[s] - 1 for s in win), "max sp-track advantage", r"{:.1f}\%")
    put("NumOverSpEightPct", 100 * (tr[8] / sp[8] - 1), "track/sp-track-1 at 8 dB", r"{:.0f}\%")
    put("NumSpSingleMaxDiff", max(abs(sp[s] - by[s]["tp_sp"]) for s in S_), "max |sp-track - sp| (bit/s/Hz)", "{:.2f}")
    # periodic pilots, best K
    ip = {s: max(cc[s]["tp_interp4"], cc[s]["tp_interp8"]) for s in S_}
    put("NumOverInterpMinPct", 100 * min(tr[s] / ip[s] - 1 for s in S_ if s >= 6), "min track/best-interp-1, SNR>=6", r"{:.0f}\%")
    put("NumOverInterpMaxPct", 100 * max(tr[s] / ip[s] - 1 for s in S_ if s >= 6), "max track/best-interp-1, SNR>=6", r"{:.0f}\%")
    # OFDM (same tracker)
    of = {s: bb[s]["tp_ofdm-track"] for s in S_}
    put("NumOverOfdmMinPct", 100 * min(tr[s] / of[s] - 1 for s in S_ if s >= 6), "min track/ofdm-1, SNR>=6", r"{:.0f}\%")
    put("NumOverOfdmMaxPct", 100 * max(tr[s] / of[s] - 1 for s in S_ if s >= 6), "max track/ofdm-1, SNR>=6", r"{:.0f}\%")
    put("NumOfdmBetterSnr", max(s for s in S_ if of[s] > tr[s]), "highest SNR where ofdm > track", "{:d}")
    k_ofdm = (S.N * (16 - 1) - 22 * (16 - 1)) / (16 * (S.N + S.Ncp))
    put("NumOfdmCeilGapPct", 100 * (ge[20] / k_ofdm - 1), "perfect-CSI ceiling ratio AFDM pilot frame vs OFDM training frame", r"{:.1f}\%")
    # open loop (best Bp)
    olb = {}
    for r in e4:
        if "tp_openloop" in r:
            if r["snr_db"] not in olb or r["tp_openloop"] > olb[r["snr_db"]][0]:
                olb[r["snr_db"]] = (r["tp_openloop"], r["Bp"])
    put("NumOverOpenMinPct", 100 * min(tr[s] / olb[s][0] - 1 for s in S_ if s >= 6), "min track/best-open-1, SNR>=6", r"{:.0f}\%")
    put("NumOverOpenMaxPct", 100 * max(tr[s] / olb[s][0] - 1 for s in S_ if s >= 6), "max track/best-open-1, SNR>=6", r"{:.0f}\%")
    put("NumOpenBestBpMax", max(v[1] for v in olb.values()), "largest best open-loop Bp", "{:d}")
    put("NumOpenBestBpMin", min(v[1] for v in olb.values()), "smallest best open-loop Bp", "{:d}")
    put("NumBpTwoBetterTop", max(s for s in S_ if bp2[s]["tp_track"] > tr[s]), "highest SNR where Bp=2 beats Bp=1", "{:d}")
    # run time (single thread, mean over SNR, per 16-block frame)
    put("NumTimeTrack", np.mean([by[s]["time_track"] for s in S_]), "time track", "{:.0f}")
    put("NumTimeTrackMin", min(by[s]["time_track"] for s in S_), "time track min over SNR", "{:.0f}")
    put("NumTimeTrackMax", max(by[s]["time_track"] for s in S_), "time track max over SNR", "{:.0f}")
    put("NumTimeConv", np.mean([by[s]["time_conv"] for s in S_]), "time conv", "{:.0f}")
    put("NumTimeConvDa", np.mean([bb[s]["time_conv-da"] for s in S_]), "time conv-da", "{:.0f}")
    put("NumTimeSp", np.mean([by[s]["time_sp"] for s in S_]), "time sp", "{:.0f}")
    put("NumTimeSpTrack", np.mean([bb[s]["time_sp-track"] for s in S_]), "time sp-track", "{:.0f}")
    put("NumTimeGenie", np.mean([by[s]["time_genie"] for s in S_]), "time genie", "{:.0f}")
except Exception as e:  # noqa: BLE001
    for n in _E4:
        missing(n, f"e4 ({e!r})")

# ------------------------------------------------------------------ frame length
try:
    e5, _ = summarize("e5_frame")
    t12 = {r["B"]: r for r in e5 if r["snr_db"] == 12}
    bestB = max(t12, key=lambda b: t12[b]["tp_track"])
    put("NumFrameBestB", bestB, "e5_frame: B maximizing track throughput at 12 dB", "{:d}")
    put("NumFrameGapLongPct", 100 * (1 - t12[max(t12)]["tp_track"] / t12[max(t12)]["tp_genie"]),
        "e5_frame: track genie gap at largest B, 12 dB", r"{:.1f}\%")
    put("NumFrameLongB", max(t12), "e5_frame: largest B", "{:d}")
    put("NumOpenLongTp", t12[max(t12)]["tp_openloop"], "e5_frame: open-loop tp at largest B, 12 dB", "{:.2f}")
    put("NumOpenShortTp", t12[min(t12)]["tp_openloop"], "e5_frame: open-loop tp at smallest B, 12 dB", "{:.2f}")
    put("NumFrameShortB", min(t12), "e5_frame: smallest B", "{:d}")
    put("NumFrameTimeRatio", t12[max(t12)]["time_track"] / t12[16]["time_track"], "e5_frame: track time B_max / B=16", "{:.1f}")
    put("NumFrameTrials", min(r["trials"] for r in e5), "e5_frame: trials per point", "{:d}")
    _rows = [json.loads(l) for l in open(RUNS / "e5_frame.jsonl")]
    _spec = json.load(open(RUNS / "e5_frame.spec.json"))
    _gaps, _nfr = [], []
    for _i, _g in enumerate(_spec["grid"]):
        if _g["snr_db"] != 12:
            continue
        _rs = [r for r in _rows if r["point"] == _i and r["ok"]]
        _den = _g["B"] * (S.N + S.Ncp)
        _d = np.array([(sum(r["genie"]["goodbits"]) - sum(r["track"]["goodbits"])) / _den for r in _rs])
        _gaps.append(100 * _d.mean() / np.mean([sum(r["genie"]["goodbits"]) / _den for r in _rs]))
        _nfr.append(int(np.sum(_d > 1e-9)))
    put("NumFrameGapMinPct", min(_gaps), "e5: min paired genie gap over B at 12 dB", r"{:.1f}\%")
    put("NumFrameGapMaxPct", max(_gaps), "e5: max paired genie gap over B at 12 dB", r"{:.1f}\%")
    put("NumFrameLossMin", min(_nfr), "e5: min #frames (of trials) with track below genie, 12 dB", "{:d}")
    put("NumFrameLossMax", max(_nfr), "e5: max #frames with track below genie, 12 dB", "{:d}")
    put("NumFrameBlerLongPct", 100 * t12[max(t12)]["bler_track"], "e5_frame: track BLER at largest B, 12 dB", r"{:.1f}\%")
    put("NumFrameBlerMidPct", 100 * t12[16]["bler_track"], "e5_frame: track BLER at B=16, 12 dB", r"{:.1f}\%")
except Exception as e:  # noqa: BLE001
    for n in ("NumFrameBestB", "NumFrameGapLongPct", "NumFrameLongB", "NumOpenLongTp", "NumOpenShortTp",
              "NumFrameShortB", "NumFrameTimeRatio", "NumFrameTrials", "NumFrameBlerLongPct", "NumFrameBlerMidPct",
              "NumFrameGapMinPct", "NumFrameGapMaxPct", "NumFrameLossMin", "NumFrameLossMax"):
        missing(n, f"e5_frame ({e!r})")

# ------------------------------------------------------------------ robustness
try:
    e6, _ = summarize("e6_robust")
    VARS = ("rho_max", "window", "model_rho", "pn_var", "born_frac", "reacq_every", "P", "kappa_max", "cfo")

    def f6(**kw):
        for r in e6:
            if all(r.get(k) == v for k, v in kw.items()) and \
               all(k in kw or r.get(k) in (None, False) for k in VARS if k in r):
                return r
        raise KeyError(kw)
    b0 = f6(rho_max=0.0)
    put("NumRobTrials", min(r["trials"] for r in e6), "e6_robust: trials per point", "{:d}")
    put("NumRobBaseRatioPct", 100 * b0["tp_track"] / b0["tp_genie"], "e6: reference track/genie", r"{:.0f}\%")
    rr = [100 * f6(**kw)["tp_track"] / f6(**kw)["tp_genie"] for kw in
          ({"cfo": 0.1}, {"cfo": 0.3}, {"P": 2}, {"P": 6}, {"P": 8}, {"kappa_max": 1.0}, {"kappa_max": 2.0})]
    put("NumRobMinRatioPct", min(rr), "e6: min track/genie over CFO, P, kappa_max variations", r"{:.0f}\%")
    d5 = f6(rho_max=0.005)
    put("NumDriftHsrRatioPct", 100 * d5["tp_track"] / d5["tp_genie"], "e6: track/genie at rho=0.005 (HSR value)", r"{:.0f}\%")
    d2 = f6(rho_max=0.02); d2w = f6(rho_max=0.02, window=6); d2m = f6(rho_max=0.02, model_rho=True)
    put("NumDriftHiDefault", d2["tp_track"], "e6: track at rho=0.02, no window", "{:.2f}")
    put("NumDriftHiWindow", d2w["tp_track"], "e6: track at rho=0.02, window 6", "{:.2f}")
    put("NumDriftHiModel", d2m["tp_track"], "e6: track at rho=0.02, rate model", "{:.2f}")
    put("NumDriftHiGenie", d2["tp_genie"], "e6: genie at rho=0.02", "{:.2f}")
    z0m = f6(rho_max=0.0, model_rho=True); z0w = f6(rho_max=0.0, window=6)
    put("NumDriftZeroModelCostPct", 100 * (1 - z0m["tp_track"] / b0["tp_track"]), "e6: rate-model cost at rho=0", r"{:.1f}\%")
    put("NumDriftZeroWindowCostPct", 100 * (1 - z0w["tp_track"] / b0["tp_track"]), "e6: window cost at rho=0", r"{:.1f}\%")
    pn = sorted([r for r in e6 if r.get("pn_var")], key=lambda r: r["pn_var"])
    put("NumPnHiTrack", pn[-1]["tp_track"], "e6: track at largest phase noise", "{:.2f}")
    put("NumPnHiConv", pn[-1]["tp_conv"], "e6: conv at largest phase noise", "{:.2f}")
    put("NumPnHiGenie", pn[-1]["tp_genie"], "e6: deterministic genie at largest phase noise", "{:.2f}")
    put("NumPnHiLw", "10^{-2}" if abs(pn[-1]["pn_var"] * 512 / (2 * np.pi) - 1e-2) < 1e-9 else "??", "e6: largest linewidth/df")
    bq = f6(born_frac=0.25); bqw = f6(born_frac=0.25, reacq_every=1, window=8); bh = f6(born_frac=0.5)
    put("NumBornQTrack", bq["tp_track"], "e6: track, 25% born", "{:.2f}")
    put("NumBornQTrackW", bqw["tp_track"], "e6: track with per-block re-acq + window, 25% born", "{:.2f}")
    put("NumBornQGenie", bq["tp_genie"], "e6: genie, 25% born", "{:.2f}")
    put("NumBornQConv", bq["tp_conv"], "e6: conv, 25% born", "{:.2f}")
    put("NumBornHTrack", bh["tp_track"], "e6: track, 50% born", "{:.2f}")
    put("NumBornHConv", bh["tp_conv"], "e6: conv, 50% born", "{:.2f}")
except Exception as e:  # noqa: BLE001
    for n in ("NumRobTrials", "NumRobBaseRatioPct", "NumRobMinRatioPct", "NumDriftHsrRatioPct", "NumDriftHiDefault",
              "NumDriftHiWindow", "NumDriftHiModel", "NumDriftHiGenie", "NumDriftZeroModelCostPct",
              "NumDriftZeroWindowCostPct", "NumPnHiTrack", "NumPnHiConv", "NumPnHiGenie", "NumPnHiLw",
              "NumBornQTrack", "NumBornQTrackW", "NumBornQGenie", "NumBornQConv", "NumBornHTrack", "NumBornHConv"):
        missing(n, f"e6_robust ({e!r})")

# ------------------------------------------------------------------ write
lines = ["% GENERATED by make_numbers.py -- do not edit by hand.\n"]
for k in sorted(NUM):
    lines.append(f"\\newcommand{{\\{k}}}{{{NUM[k]['tex']}}}  % {NUM[k]['source']}\n")
OUT.write_text("".join(lines))
json.dump({k: {"value": v["value"], "tex": v["tex"], "source": v["source"]} for k, v in NUM.items()},
          open(RUNS / "numbers.json", "w"), indent=1, default=float)
print("".join(lines))
