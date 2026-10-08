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
vu = (S.N ** 2 - 1) / (12 * S.N ** 2)


def rho_cr(L, E, b):
    """eq. (horizon): decoupled benchmark for block b after L contiguous blocks of energy E."""
    vL = beta ** 2 * (L ** 2 - 1) / 12 + vu
    return P / (2 * L * E) * (2 + (beta ** 2 * (b - (L - 1) / 2) ** 2 + vu) / vL)


rho1 = rho_cr(1, Ep, 1)
assert abs(rho1 - P / (2 * Ep) * (3 + beta ** 2 / vu)) < 1e-12
put("NumRhoOneDb", 10 * np.log10(rho1), "Corollary 2, eq. (rho1) exact, P=4, E=Ep", "{:.1f}")
put("NumRhoOneLossDb", 10 * np.log10(1 + rho1), "SINR loss 10log10(1+rho1)", "{:.2f}")
rho2 = 5 * P / (2 * (Ep + S.N))
put("NumRhoTwoDb", 10 * np.log10(rho2), "Corollary 2 asymptotic form, b=2", "{:.1f}")
for Bp, nm in ((1, "One"), (2, "Two"), (4, "Four"), (8, "Eight")):
    L = np.arange(Bp, 4000)
    ok = L[rho_cr(Bp, Ep, L) <= 0.25]
    put(f"NumHor{nm}", int(ok.max() - Bp + 1) if ok.size else 0, "Corollary 1, eq. (horizon), E=Ep, delta=1/4", "{:d}")
# single-block acquisition SNR for a quarter-power path, P_cell >= 1 - 1e-3
# (threshold-aware Proposition 1: global outlier term x local cell term)
from scipy.optimize import brentq
from theory_check import p_global
def _pcell(g):
    loc = 1 - 2 * norm.sf(0.5 / (beta * np.sqrt(3 / (2 * np.pi ** 2 * g))))
    return p_global(g, 1) * loc
g1 = brentq(lambda gdb: _pcell(10 ** (gdb / 10)) - (1 - 1e-3), 0, 30)
put("NumGammaCellDb", g1, "Prop. 1 (threshold-aware), Bp=1, P_cell=1-1e-3", "{:.1f}")
from theory_check import p_retained, acq_threshold
_T1 = acq_threshold(1)
def _pret(g):
    loc = 1 - 2 * norm.sf(0.5 / (beta * np.sqrt(3 / (2 * np.pi ** 2 * g))))
    return p_retained(g, 1, _T1) * loc
put("NumRetainAtCell", _pret(10 ** (g1 / 10)), "Approx. 1: P_ret at the SNR where P_cell = 1-1e-3", "{:.2f}")
g2 = brentq(lambda gdb: _pret(10 ** (gdb / 10)) - (1 - 1e-3), 0, 30)
put("NumGammaRetainDb", g2, "Approx. 1: SNR for P_ret = 1-1e-3, Bp=1", "{:.1f}")
put("NumSnrRetainOk", g2 - 10 * np.log10(Ep * 0.25), "P_ret threshold minus 10log10(Ep/4): data SNR for a quarter-power path", "{:.0f}")
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
put("NumLeakIntDb", 10 * np.log10(np.sum(leak) / np.sum(pil)), "data leakage into W, integer delays, xi=2, 200 draws", "{:.0f}")

# ------------------------------------------------------------------ grating-region peak sidelobe
_k9 = json.load(open(RUNS / "checks.json"))["k9"]
for _B, _nm in (("2", "Two"), ("16", "Sixteen")):
    put(f"NumGratPeak{_nm}Db", _k9[_B]["full_max_db"], f"checks k9: max full-length grating-region peak over the delay/Doppler sweep, Bp={_B}", "{:.1f}")
put("NumGratWinDiffLowDb", max(_k9[b]["max_abs_diff_db"] for b in ("2", "4")), "checks k9: max |window - full| peak, Bp<=4", "{:.1f}")
put("NumGratWinDiffSixteenDb", _k9["16"]["max_abs_diff_db"], "checks k9: max |window - full| peak, Bp=16", "{:.1f}")
put("NumGratWinSixteenDb", _k9["16"]["win_max_db"], "checks k9: max window grating-region peak, Bp=16", "{:.1f}")
put("NumGratCases", _k9["n_cases"], "checks k9: delay/Doppler cases in the sweep", "{:d}")

# ------------------------------------------------------------------ exact rho (Cor. 2) as simulated
def _rho(blocks_w, bnext, P=4):
    t, w = [], []
    for b, e in blocks_w:
        t += list((b * (S.N + S.Ncp) + S.Ncp + np.arange(S.N)) / S.N); w += [e / S.N] * S.N
    t, w = np.array(t), np.array(w)
    tb = np.sum(w * t) / np.sum(w); st = np.sum(w * (t - tb) ** 2) / np.sum(w)
    tn = (bnext * (S.N + S.Ncp) + S.Ncp + np.arange(S.N)) / S.N
    return P / (2 * np.sum(w)) * (2 + np.mean((tn - tb) ** 2) / st)
def rho_cr_set(idx, E, b):
    """Benchmark for known blocks at absolute indices idx (generalizes eq. (horizon))."""
    idx = np.asarray(idx, float)
    return P / (2 * idx.size * E) * (2 + (beta ** 2 * (b - idx.mean()) ** 2 + vu) / (beta ** 2 * idx.var() + vu))


assert abs(rho_cr_set(range(4), S.N, 9) - rho_cr(4, S.N, 9)) < 1e-15
put("NumSparseBenchDb", 10 * np.log10(rho_cr_set([0, 4, 8, 12], S.N, 15)), "benchmark, known blocks 0,4,8,12, block 15", "{:.1f}")
put("NumContigBenchDb", 10 * np.log10(rho_cr_set([0, 1, 2, 3], S.N, 15)), "benchmark, known blocks 0..3, block 15", "{:.1f}")
put("NumAmbigPeriodFour", 1 / (4 * beta), "slow-time ambiguity period 1/(K beta), K=4", "{:.2f}")
for (L, b), nm in (((1, 1), "OneData"), ((2, 2), "TwoExact"), ((10, 10), "TenExact")):
    v = rho_cr(L, S.N, b)
    assert abs(v - _rho([(q, S.N) for q in range(L)], b)) / v < 1e-6     # eq. (horizon) = direct moments
    put(f"NumRho{nm}Db", 10 * np.log10(v), f"eq. (horizon), {L} full-energy blocks, next block", "{:.1f}")
for Bp, nm in ((1, "One"), (2, "Two"), (4, "Four"), (8, "Eight")):
    L = np.arange(Bp, 4000)
    ok = L[rho_cr(Bp, S.N, L) <= 0.25]
    put(f"NumHorData{nm}", int(ok.max() - Bp + 1) if ok.size else 0, "Cor. 1 with decoded pilot-block data (energy N per block)", "{:d}")
put("NumOverheadGainPct", 100 * (1 / (1 - len(S.zero_set) / S.N) - 1), "1/(1-|Z|/N)-1: upper bound of the guard-removal gain", r"{:.0f}\%")

# ------------------------------------------------------------------ oracle adaptive guard (runs/oracle_guard.json)
_og = json.load(open(RUNS / "oracle_guard.json"))
put("NumOracleGuardPct", 100 * _og["oracle_mean"], "oracle_guard.json: mean guard sized to the realized support", r"{:.1f}\%")
put("NumOracleGainPct", 100 * _og["max_gain_oracle_vs_fixed"], "oracle_guard.json: max gain of an ideal adaptive guard", r"{:.1f}\%")
put("NumPilotBlockSharePct", 100 * _og["pilot_sparse_frame_guard_share"], "|Z|/(16N): guard share of a 16-block pilot-sparse frame", r"{:.1f}\%")
# ------------------------------------------------------------------ numerical checks (runs/checks.json)
_ck = json.load(open(RUNS / "checks.json"))
put("NumKaufFitPct", 100 * _ck["k1"][0]["rel_err"], "checks k1: Kaufman vs exact Jacobian at the fit", r"{:.1f}\%")
put("NumKaufAwayPct", 100 * _ck["k1"][1]["rel_err"], "checks k1: Kaufman vs exact Jacobian away from the fit", r"{:.0f}\%")
put("NumLeakMeanDb", _ck["k2"]["leak_to_pilot_db_mean"], "checks k2: mean data leakage into W / pilot response, evaluation ensemble", "{:.0f}")
put("NumLeakNoiseMedDb", _ck["k2"]["leak_to_noise20_db_median"], "checks k2: median leakage per chirp / noise at 20 dB", "{:.1f}")
put("NumLeakNoiseHiDb", _ck["k2"]["leak_to_noise20_db_p95"], "checks k2: 95th percentile, same", "{:+.1f}")
_rows = {(r["Bp"], r["ell"], r["restricted"]): r["peak_db"] for r in _ck["k3"]["rows"]}
_fr = {r["N"]: r for r in _ck["k4"]["fixed_rate"]}
put("NumGuardPctKiloFs", 100 * _fr[1024]["frac"], "checks k4: |Z|/N at N=1024, fixed sample rate and physical spreads", r"{:.0f}\%")
put("NumGuardPctTwoKiloFs", 100 * _fr[2048]["frac"], "checks k4: |Z|/N at N=2048, same", r"{:.0f}\%")
put("NumTxEnergyDevPct", 100 * max(abs(v["mean"] - 1) for v in _ck["k7"].values()), "checks k7: max |mean energy per sample - 1| over frame types", r"{:.2f}\%")
put("NumWindowCapture", _ck["k8"]["min_capture_kmax"], "checks k8: min energy of the full-length atom captured by W, |kappa|<=kappa_max", "{:.3f}")
put("NumWindowLossDb", _ck["k8"]["loss_db_kmax"], "checks k8: corresponding pilot-SNR loss", "{:.2f}")
put("NumRmsDelayMed", _ck["k5"]["median"], "checks k5: median realized RMS delay spread (samples)", "{:.1f}")
put("NumRmsDelayHi", _ck["k5"]["p90"], "checks k5: 90th percentile realized RMS delay spread (samples)", "{:.1f}")

# paired c2 test (held-out seeds): c2 = sqrt(2)/(4N) vs the default 1/(2N)
def _c2pair():
    from aggregate import load
    out = {}
    (sa, ra), (_, rb) = load("dev_c2p_ref"), load("dev_c2p_irr")
    a = {(r["point"], r["seed"]): r for r in ra if r.get("ok")}
    b = {(r["point"], r["seed"]): r for r in rb if r.get("ok")}
    for i, g in enumerate(sa["grid"]):
        ks = [k for k in a if k[0] == i and k in b]
        for rx in ("genie", "track"):
            x = np.array([sum(a[k][rx]["goodbits"]) for k in ks], float)
            y = np.array([sum(b[k][rx]["goodbits"]) for k in ks], float)
            out[(g["snr_db"], rx)] = (100 * (y.mean() / x.mean() - 1), len(ks))
    return out
_c2 = _c2pair()
put("NumCtwoGenieEightPct", _c2[(8, "genie")][0], "dev_c2p: irrational-c2 change of perfect-CSI throughput at 8 dB", r"{:+.1f}\%")
put("NumCtwoTrackEightPct", _c2[(8, "track")][0], "dev_c2p: irrational-c2 change of proposed throughput at 8 dB", r"{:+.1f}\%")
put("NumCtwoMaxTwelvePct", max(abs(_c2[(12, r)][0]) for r in ("genie", "track")), "dev_c2p: max |change| at 12 dB", r"{:.1f}\%")
put("NumCtwoTrials", _c2[(8, "genie")][1], "dev_c2p: paired channel realizations per point", "{:d}")

# ------------------------------------------------------------------ receiver constants (from code)
from mbtrack import Tracker
_T = Tracker(S)
put("NumMergeDl", _T.merge_dl, "mbtrack.Tracker.merge_dl", "{:g}")
put("NumMergeDk", _T.merge_dk, "mbtrack.Tracker.merge_dk", "{:g}")
put("NumRetries", ["zero", "one", "two", "three", "four"][_T.retries], "mbtrack.Tracker.retries (word)")
put("NumExclRadius", 1 / (_T.reacq_window * beta), "1/(W_r beta): re-acquisition exclusion radius for a full window", "{:.2f}")
put("NumResFrame", 1 / (16 * beta), "1/(B beta): Doppler resolution of a 16-block aperture", "{:.2f}")
put("NumReacqWindow", _T.reacq_window, "mbtrack.Tracker.reacq_window", "{:d}")
put("NumGnIters", ["zero", "one", "two", "three", "four", "five", "six"][_T.gn_iters], "mbtrack.Tracker.gn_iters (word)")

# ------------------------------------------------------------------ theory validation
try:
    th = json.load(open(RUNS / "theory_check.json"))
    dev = max(abs(r["p_cell"] - r["p_cell_formula"]) for r in th["e2"])
    put("NumAcqMaxDev", dev, "theory_check e2: max |P_cell sim - Prop.1|", "{:.2f}")
    put("NumAcqTrials", min(r["trials"] for r in th["e2"]), "theory_check e2: trials per point", "{:d}")
    put("NumRetainMaxDev", max(abs(r["p_retained"] - r["p_retained_formula"]) for r in th["e4"]["rows"]),
        "theory_check e4: max |retention sim - P_ret|", "{:.2f}")
    put("NumRetainCellGap", max(r["p_cell_formula"] - r["p_retained"] for r in th["e4"]["rows"]),
        "theory_check e4: max overestimate of retention by P_cell", "{:.2f}")
    assert th["e5"]["noise"]["p_extra"] == 0 and th["e5"]["one"]["p_extra"] == 0, "false insertions observed: revise text"
    put("NumFalseTrials", th["e5"]["noise"]["trials"], "theory_check e5: trials (no false extra path observed)", "{:d}")
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

# ------------------------------------------------------------------ results (m_main: every receiver on the same channels)
from aggregate import summarize
from diagnostics import frames, path_stats


def _guard(names, src):
    def deco(fn):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            for n in names:
                if n not in NUM:
                    missing(n, f"{src} ({e!r})")
        return fn
    return deco


_MAIN = ("NumGainConvPct", "NumSnrGenieMatch", "NumGapGeniePct", "NumSpTopSnr", "NumSpFromSnr", "NumSpMaxAdvPct",
         "NumOverSpEightPct", "NumOverConvEightPct", "NumGenieRatioEightPct", "NumGenieRatioSixPct",
         "NumOverOfdmMinPct", "NumOverOfdmMaxPct", "NumOfdmBetterSnr", "NumOverOpenMinPct",
         "NumOverOpenMaxPct", "NumEvalTrials", "NumEvalTrialsHi", "NumTimeTrack", "NumTimeGenie",
         "NumBpTwoBetterTop", "NumOpenBestBpMax", "NumOpenBestBpMin", "NumBlerFloorPct", "NumConvCeilGainPct",
         "NumPlainConvHiPct", "NumSpSingleMaxDiff", "NumOverInterpMinPct", "NumOverInterpMaxPct", "NumOfdmCeilGapPct",
         "NumTimeConvDa", "NumTimeSpTrack", "NumTimeTrackMin", "NumTimeTrackMax", "NumSpOverGenieMaxPct",
         "NumOverPtrackMinPct", "NumOverPtrackMaxPct", "NumHybridMaxDiffPct", "NumSptEpsList", "NumSptKList",
         "NumSplEpsList", "NumSplFromSnr", "NumOverSplEightPct", "NumSpItersSatDiff",
         "NumBpTwoSigTop", "NumBpTwoSigLow", "NumBpTwoWorseFrom", "NumBpTwoWorsePct", "NumBlerRatioMax", "NumSpOverGenieFromSnr", "NumHybridMinPct", "NumHybridMaxPct", "NumPtrackBestKList",
         "NumInterpBestKList", "NumUndetected", "NumBlocksAll")


@_guard(_MAIN, "m_main")
def _main():
    mm, _ = summarize("m_main")
    by = {r["snr_db"]: r for r in mm if r.get("Bp") == 1 and "tp_track" in r}
    bp2 = {r["snr_db"]: r for r in mm if r.get("Bp") == 2}
    S_ = sorted(by)
    tr = {s: by[s]["tp_track"] for s in S_}
    ge = {s: by[s]["tp_genie"] for s in S_}
    put("NumEvalTrials", min(by[s]["trials"] for s in S_), "m_main: channel realizations per SNR (min)", "{:d}")
    put("NumEvalTrialsHi", max(by[s]["trials"] for s in S_), "m_main: channel realizations at the key SNRs", "{:d}")
    ratio = {s: tr[s] / ge[s] for s in S_}
    first = min(s for s in S_ if all(ratio[t] >= 0.98 for t in S_ if t >= s))
    put("NumSnrGenieMatch", first, "lowest SNR with track >= 0.98 genie at and above it", "{:d}")
    put("NumGapGeniePct", 100 * max(1 - ratio[s] for s in S_ if s >= first), "max genie gap above that SNR", r"{:.1f}\%")
    put("NumGenieRatioEightPct", 100 * ratio[8], "track/genie at 8 dB", r"{:.0f}\%")
    put("NumGenieRatioSixPct", 100 * ratio[6], "track/genie at 6 dB", r"{:.0f}\%")
    put("NumBlerFloorPct", 100 * max(by[s]["bler_track"] for s in S_ if s > 12), "max track BLER above 12 dB", r"{:.1f}\%")
    cd = {s: by[s]["tp_conv-da"] for s in S_}
    put("NumGainConvPct", 100 * np.mean([tr[s] / cd[s] - 1 for s in S_ if s >= 10]),
        "mean over the simulated SNR points >= 10 dB of track/conv-da-1", r"{:.0f}\%")
    put("NumOverConvEightPct", 100 * (tr[8] / cd[8] - 1), "track/conv-da-1 at 8 dB", r"{:.0f}\%")
    put("NumConvCeilGainPct", 100 * (ge[20] / by[20]["tp_genie-conv"] - 1), "perfect-CSI ceiling ratio, 20 dB", r"{:.0f}\%")
    put("NumPlainConvHiPct", 100 * max(by[s]["bler_conv"] for s in S_ if s >= 14), "plain conv BLER max >=14 dB", r"{:.1f}\%")
    sp = {s: by[s]["tp_sp-track"] for s in S_}
    win = [s for s in S_ if sp[s] > tr[s]]
    put("NumSpFromSnr", min(win), "lowest SNR where sp-track > track", "{:d}")
    put("NumSpTopSnr", max(s for s in S_ if s < min(win)), "highest SNR below that", "{:d}")
    put("NumSpMaxAdvPct", 100 * max(sp[s] / tr[s] - 1 for s in win), "max sp-track advantage", r"{:.1f}\%")
    put("NumOverSpEightPct", 100 * (tr[8] / sp[8] - 1), "track/sp-track-1 at 8 dB", r"{:.0f}\%")
    put("NumSpOverGenieMaxPct", 100 * max(sp[s] / ge[s] - 1 for s in S_), "max sp-track over perfect CSI of the pilot-sparse frame", r"{:.1f}\%")
    put("NumSpSingleMaxDiff", max(abs(sp[s] - by[s]["tp_sp"]) for s in S_), "max |sp-track - sp| (bit/s/Hz)", "{:.2f}")
    ch = json.load(open(RUNS / "tune_v2_choice.json"))["sp_track"]
    put("NumSptEpsList", ", ".join(f"{ch[k]['spt_eps']:g}" for k in sorted(ch, key=float)), "tune_v2: tuned pilot fraction per tuning SNR")
    put("NumSptKList", ", ".join(f"{ch[k]['K']}" for k in sorted(ch, key=float)), "tune_v2: tuned acquisition blocks per tuning SNR")
    # single-block superimposed pilot: gain of the last iteration step at the chosen fraction
    _tv = [r for nm in ("tune_v2", "tune_v3", "tune_v4") for r in summarize(nm)[0] if r.get("sp_iters") and r["trials"] >= 30]
    _cs = json.load(open(RUNS / "tune_v2_choice.json"))["sp"]
    _sat = []
    for k, v in _cs.items():
        if v["sp_iters"] == 20:
            t = {r["sp_iters"]: r["tp_sp"] for r in _tv if r["snr_db"] == float(k) and r["sp_eps"] == v["sp_eps"]}
            _sat.append(t[20] - t[16])
    put("NumSpItersSatDiff", max(_sat) if _sat else 0.0, "tune: throughput gain from 16 to 20 iterations (single-block SP)", "{:.3f}")
    cl = json.load(open(RUNS / "tune_v2_choice.json"))["sp_track_ll"]
    put("NumSplEpsList", ", ".join(f"{cl[k]['spl_eps']:g}" for k in sorted(cl, key=float)), "tune: low-latency pilot fraction per tuning SNR")
    spl = {s: by[s]["tp_sp-track-ll"] for s in S_}
    wl = [s for s in S_ if spl[s] > tr[s]]
    put("NumSplFromSnr", min(wl) if wl else "--", "lowest SNR where low-latency sp-track > track", "{}")
    put("NumOverSplEightPct", 100 * (tr[8] / spl[8] - 1), "track/sp-track-ll-1 at 8 dB", r"{:.0f}\%")
    ip = {s: max(by[s]["tp_interp4"], by[s]["tp_interp8"]) for s in S_}
    put("NumOverInterpMinPct", 100 * min(tr[s] / ip[s] - 1 for s in S_ if s >= 6), "min track/best-interp-1, SNR>=6", r"{:.0f}\%")
    put("NumOverInterpMaxPct", 100 * max(tr[s] / ip[s] - 1 for s in S_ if s >= 6), "max track/best-interp-1, SNR>=6", r"{:.0f}\%")
    pt = {s: max(by[s]["tp_ptrack4"], by[s]["tp_ptrack8"]) for s in S_}
    put("NumOverPtrackMinPct", 100 * min(tr[s] / pt[s] - 1 for s in S_ if s >= 6), "min track/best-ptrack-1, SNR>=6", r"{:.0f}\%")
    put("NumOverPtrackMaxPct", 100 * max(tr[s] / pt[s] - 1 for s in S_ if s >= 6), "max track/best-ptrack-1, SNR>=6", r"{:.0f}\%")
    put("NumHybridMaxDiffPct", 100 * max(abs(by[s]["tp_track-hybrid"] / tr[s] - 1) for s in S_ if s >= 4),
        "max |hybrid/strict-1|, SNR>=4", r"{:.1f}\%")
    of = {s: by[s]["tp_ofdm-track"] for s in S_}
    put("NumOverOfdmMinPct", 100 * min(tr[s] / of[s] - 1 for s in S_ if s >= 6), "min track/ofdm-1, SNR>=6", r"{:.0f}\%")
    put("NumOverOfdmMaxPct", 100 * max(tr[s] / of[s] - 1 for s in S_ if s >= 6), "max track/ofdm-1, SNR>=6", r"{:.0f}\%")
    lo = [s for s in S_ if of[s] > tr[s]]
    put("NumOfdmBetterSnr", max(lo) if lo else "--", "highest SNR where ofdm > track", "{}")
    k_ofdm = (S.N * (16 - 1) - 22 * (16 - 1)) / (16 * (S.N + S.Ncp))
    put("NumOfdmCeilGapPct", 100 * (ge[20] / k_ofdm - 1), "perfect-CSI ceiling ratio AFDM pilot frame vs OFDM training frame", r"{:.1f}\%")
    # paired comparisons across grid points on the common channel draws (same seeds per SNR)
    from diagnostics import paired
    spec_m, fb_m = frames("m_main")
    gi = {(g["snr_db"], g.get("Bp")): i for i, g in enumerate(spec_m["grid"])}

    def pair_points(s, bp, rx_b):
        ra = {r["seed"]: r for r in fb_m[gi[(s, 1)]]}
        rb = {r["seed"]: r for r in fb_m[gi[(s, bp)]]}
        return [{**ra[k], "_b": rb[k][rx_b]} for k in ra if k in rb]
    olb = {}
    for s in S_:
        best = None
        for bp in (1, 2, 4, 8):
            rows = pair_points(s, bp, "openloop")
            d = paired(rows, "track", "_b", seed=s)
            if best is None or d["ratio"] < best[0]["ratio"]:
                best = (d, bp)
        olb[s] = best
    put("NumOverOpenMinPct", 100 * min(olb[s][0]["ratio"] for s in S_ if s >= 6), "paired: min track/best-open-1, SNR>=6", r"{:.0f}\%")
    put("NumOverOpenMaxPct", 100 * max(olb[s][0]["ratio"] for s in S_ if s >= 6), "paired: max track/best-open-1, SNR>=6", r"{:.0f}\%")
    put("NumOpenBestBpMax", max(v[1] for v in olb.values()), "largest best open-loop Bp", "{:d}")
    put("NumOpenBestBpMin", min(v[1] for v in olb.values()), "smallest best open-loop Bp", "{:d}")
    sig2 = []
    for s in S_:
        d = paired(pair_points(s, 2, "track"), "_b", "track", seed=s)     # Bp=2 relative to Bp=1
        if d["ratio_lo"] > 0:
            sig2.append(s)
    put("NumBpTwoSigTop", max(sig2) if sig2 else "--", "paired: highest SNR where Bp=2 beats Bp=1 significantly", "{}")
    b2 = [s for s in S_ if paired(pair_points(s, 2, "track"), "_b", "track", seed=s)["ratio"] > 0]
    put("NumBpTwoBetterTop", max(b2) if b2 else "--", "paired: highest SNR where Bp=2 beats Bp=1 in the mean", "{}")
    put("NumBpTwoSigLow", min(sig2) if sig2 else "--", "paired: lowest SNR where Bp=2 beats Bp=1 significantly", "{}")
    worse = {}
    for s in S_:
        d = paired(pair_points(s, 2, "track"), "_b", "track", seed=s)
        if d["ratio_hi"] < 0:
            worse[s] = d["ratio"]
    put("NumBpTwoWorseFrom", min(worse) if worse else "--", "paired: lowest SNR where Bp=2 is significantly worse", "{}")
    put("NumBpTwoWorsePct", -100 * min(worse.values()) if worse else 0.0, "paired: largest significant Bp=2 deficit", r"{:.1f}\%")
    bl = {s: by[s]["bler_track"] / max(by[s]["bler_genie"], 1e-9) for s in S_ if 6 <= s <= 12}
    put("NumBlerRatioMax", max(bl.values()), "max BLER ratio track/genie, 6-12 dB", "{:.0f}")
    over = [s for s in S_ if by[s]["tp_sp-track"] > ge[s]]
    put("NumSpOverGenieFromSnr", min(over), "lowest SNR where sp-track > genie of the pilot-sparse frame", "{:d}")
    hy = {s: by[s]["tp_track-hybrid"] / tr[s] - 1 for s in S_ if s >= 4}
    put("NumHybridMinPct", 100 * min(hy.values()), "min hybrid/strict-1, SNR>=4", r"{:+.1f}\%")
    put("NumHybridMaxPct", 100 * max(hy.values()), "max hybrid/strict-1, SNR>=4", r"{:+.1f}\%")
    pk = {s: (4 if by[s]["tp_ptrack4"] >= by[s]["tp_ptrack8"] else 8) for s in S_}
    put("NumPtrackBestKList", ", ".join(str(pk[s]) for s in S_), "best periodic+tracker spacing per SNR")
    ik = {s: (4 if by[s]["tp_interp4"] >= by[s]["tp_interp8"] else 8) for s in S_}
    put("NumInterpBestKList", ", ".join(str(ik[s]) for s in S_), "best periodic interpolation spacing per SNR")
    # undetected errors: CRC passed, payload wrong (proposed)
    nb = nu = 0
    for i, g in enumerate(spec_m["grid"]):
        if g.get("Bp") == 1:
            for r in fb_m[i]:
                for c, e in zip(r["diag-track"]["crc"], r["track"]["blerr"]):
                    nb += 1; nu += int(c == 1 and e == 1)
    assert nu == 0, "undetected errors observed: revise text"
    put("NumUndetected", nu, "m_main: blocks with CRC passed but payload wrong (proposed)", "{:d}")
    put("NumBlocksAll", nb, "m_main: blocks of the proposed receiver, Bp=1 points", "{:d}")
    put("NumTimeTrack", np.mean([by[s]["time_track"] for s in S_]), "time track", "{:.0f}")
    put("NumTimeTrackMin", min(by[s]["time_track"] for s in S_), "time track min over SNR", "{:.0f}")
    put("NumTimeTrackMax", max(by[s]["time_track"] for s in S_), "time track max over SNR", "{:.0f}")
    put("NumTimeConvDa", np.mean([by[s]["time_conv-da"] for s in S_]), "time conv-da", "{:.0f}")
    put("NumTimeSpTrack", np.mean([by[s]["time_sp-track"] for s in S_]), "time sp-track", "{:.0f}")
    put("NumTimeGenie", np.mean([by[s]["time_genie"] for s in S_]), "time genie", "{:.0f}")


_DIAG = ("NumCapHitPct", "NumCapHitMaxPct", "NumRhoMeanLastTen", "NumRhoMeanLastSixteen", "NumAcqDetSix", "NumAcqDetHi", "NumAcqMissSix", "NumAcqMissHi", "NumFinalDetHi", "NumFalseAcq", "NumFalseFinal",
         "NumInsertPerFrame", "NumMergePerFrame", "NumRhoMedFirstTen", "NumRhoMedLastTen", "NumRhoMedFirstSixteen",
         "NumRhoMedLastSixteen", "NumRhoCircSixteen", "NumIbiDb", "NumCircMisDb", "NumMisGenieDiffPct",
         "NumMisTrackDiffPct", "NumLockLossFrames", "NumHiFrames", "NumBurstGenieOk", "NumBurstRhoHigh", "NumRhoFloorRiseDb", "NumLockErrSharePct", "NumBenchFifteenDb",
         "NumCircBenchGapDb", "NumMisFloorTenDb")


@_guard(_DIAG, "m_main diagnostics / m_mismatch")
def _diag():
    spec, fb = frames("m_main")
    pidx = {g["snr_db"]: i for i, g in enumerate(spec["grid"]) if g.get("Bp") == 1}
    a6 = path_stats(fb[pidx[6]], "acq", 1.0)
    hi = [s for s in pidx if s >= 12]
    ahi = [path_stats(fb[pidx[s]], "acq", 1.0) for s in hi]
    fhi = [path_stats(fb[pidx[s]], "diag-track", 0.5) for s in hi]
    put("NumAcqDetSix", 100 * a6["p_detect"], "m_main acq: fraction of true paths found at 6 dB", r"{:.0f}\%")
    put("NumAcqMissSix", 100 * a6["missed_energy"], "m_main acq: missed channel energy at 6 dB", r"{:.1f}\%")
    put("NumAcqDetHi", 100 * min(a["p_detect"] for a in ahi), "m_main acq: min fraction found, SNR>=12", r"{:.0f}\%")
    put("NumAcqMissHi", 100 * max(a["missed_energy"] for a in ahi), "m_main acq: max missed energy, SNR>=12", r"{:.2f}\%")
    put("NumFinalDetHi", 100 * min(f["p_detect"] for f in fhi), "m_main final: min fraction tracked, SNR>=12", r"{:.0f}\%")
    pooled_a = path_stats([r for s in hi for r in fb[pidx[s]]], "acq", 1.0)
    pooled_f = path_stats([r for s in hi for r in fb[pidx[s]]], "diag-track", 0.5)
    put("NumFalseAcq", pooled_a["false_per_frame"], "m_main acq: unmatched components per frame, pooled over frames, SNR>=12", "{:.1f}")
    put("NumFalseFinal", pooled_f["false_per_frame"], "m_main final: unmatched components per frame, pooled, SNR>=12", "{:.1f}")
    hif = [r for s in hi for r in fb[pidx[s]]]
    capf = np.mean([len(r["acq"]) >= 8 for r in hif])
    put("NumCapHitPct", 100 * capf, "m_main: frames whose acquisition reaches the cap P_max=8, SNR>=12, pooled", r"{:.0f}\%")
    caps = [np.mean([len(r["acq"]) >= 8 for r in fb[pidx[s]]]) for s in pidx]
    put("NumCapHitMaxPct", 100 * max(caps), "m_main: max over SNR of frames reaching the cap", r"{:.0f}\%")
    hir = [r for s in hi for r in fb[pidx[s]] if "diag-track" in r]          # same population as above
    put("NumInsertPerFrame", np.mean([r["diag-track"]["n_insert"] for r in hir]), "m_main: re-acquisition insertions per frame, SNR>=12", "{:.1f}")
    put("NumMergePerFrame", np.mean([r["diag-track"]["n_merge"] for r in hir]), "m_main: merges per frame, SNR>=12", "{:.1f}")
    if "n_split" in hir[0]["diag-track"]:
        put("NumSplitPerFrame", np.mean([r["diag-track"]["n_split"] for r in hir]), "m_main: accepted splits per frame, SNR>=12", "{:.2f}")
    for s, nm in ((10, "Ten"), (16, "Sixteen")):
        R = np.array([r["diag-track"]["rho_pred"] for r in fb[pidx[s]]])
        med = 10 * np.log10(np.median(R, 0))
        put(f"NumRhoMedFirst{nm}", med[1], f"m_main diag: median rho(1), {s} dB", "{:.1f}")
        put(f"NumRhoMedLast{nm}", med[-1], f"m_main diag: median rho(15), {s} dB", "{:.1f}")
        NUM.setdefault("_last", {})[s] = med[-1]
        put(f"NumRhoMeanLast{nm}", 10 * np.log10(np.mean(R[:, -1])), f"m_main diag: mean rho(15), {s} dB", "{:.1f}")
    put("NumRhoFloorRiseDb", NUM["_last"][16] - NUM["_last"][10], "median rho(15) at 16 dB minus at 10 dB", "{:.1f}")
    del NUM["_last"]
    nl = sum(1 for s in hi for r in fb[pidx[s]] if sum(r["track"]["blerr"]) >= 3)
    put("NumLockLossFrames", nl, "m_main: frames with >=3 block errors (proposed), SNR>=12", "{:d}")
    put("NumHiFrames", sum(len(fb[pidx[s]]) for s in hi), "m_main: frames at SNR>=12", "{:d}")
    burst = [r for s in hi for r in fb[pidx[s]] if sum(r["track"]["blerr"]) >= 3]
    put("NumBurstGenieOk", sum(1 for r in burst if sum(r["genie"]["blerr"]) <= 1), "m_main: of those, frames with <=1 block error under perfect CSI", "{:d}")
    put("NumBurstRhoHigh", sum(1 for r in burst if np.median(r["diag-track"]["rho_pred"][1:]) > 1.0),
        "m_main: of those, frames whose median prediction-error-to-noise ratio exceeds 0 dB", "{:d}")
    _e = [sum(r["track"]["blerr"]) for s in hi for r in fb[pidx[s]]]
    put("NumLockErrSharePct", 100 * sum(e for e in _e if e >= 3) / max(1, sum(_e)), "m_main: share of block errors (SNR>=12) in those frames", r"{:.0f}\%")
    sm, bm = frames("m_mismatch")
    ibi, cm, rc = [], [], None
    gdiff, tdiff = [], []
    by_snr = {}
    for i, g in enumerate(sm["grid"]):
        by_snr.setdefault(g["snr_db"], {})["circ" if g.get("circular") else f"t{g.get('taps', 24)}"] = bm[i]
        if not g.get("circular") and g.get("taps", 24) == 24:
            ibi += [r["diag-track"]["ibi"] for r in bm[i]]
            cm += [r["diag-track"]["circ_mismatch"] for r in bm[i]]
        if g.get("circular") and g["snr_db"] == 16:
            R = np.array([r["diag-track"]["rho_pred"] for r in bm[i]])
            rc = 10 * np.log10(np.median(R, 0))[-1]
    put("NumIbiDb", 10 * np.log10(np.median(ibi)), "m_mismatch: median IBI / received power (49-tap filter)", "{:.0f}")
    put("NumCircMisDb", 10 * np.log10(np.median(cm)), "m_mismatch: median ||H_bb - H_circ||^2/||H_bb||^2", "{:.0f}")
    put("NumRhoCircSixteen", rc, "m_mismatch: median rho(15), matched model, 16 dB", "{:.1f}")
    _vu = (S.N ** 2 - 1) / (12 * S.N ** 2)
    _bench = 10 * np.log10(rho_cr(15, S.N, 15))
    put("NumBenchFifteenDb", _bench, "eq. (horizon): benchmark after 15 full-energy blocks", "{:.1f}")
    put("NumCircBenchGapDb", rc - _bench, "m_mismatch: matched-model median rho(15) minus benchmark", "{:.1f}")
    put("NumMisFloorTenDb", 10 * np.log10(np.median(cm)) + 10, "mismatch level relative to noise at 10 dB", "{:.0f}")
    tp = lambda rows, rx: np.mean([sum(r[rx]["goodbits"]) for r in rows])
    for s, d in by_snr.items():
        gdiff.append(100 * abs(tp(d["circ"], "genie") / tp(d["t24"], "genie") - 1))
        tdiff.append(100 * abs(tp(d["circ"], "track") / tp(d["t24"], "track") - 1))
    put("NumMisGenieDiffPct", max(gdiff), "m_mismatch: max |circular/FIR - 1| perfect CSI", r"{:.1f}\%")
    put("NumMisTrackDiffPct", max(tdiff), "m_mismatch: max |circular/FIR - 1| proposed", r"{:.1f}\%")


_TWO = ("NumJointDelayMaxDb", "NumTwoTrials", "NumTwoMinRatioPct", "NumTwoMinDk", "NumTwoCloseRatioPct", "NumTwoFarRatioPct", "NumTwoDelayRatioPct", "NumJointPenaltyOneDb",
        "NumJointPenaltyFourDb")


@_guard(_TWO, "m_two / checks k6")
def _two():
    res, _ = summarize("m_two")
    put("NumTwoTrials", min(r["trials"] for r in res), "m_two: trials per point", "{:d}")
    f = {(r["tp_dl"], r["tp_dk"]): r for r in res}
    dks = sorted({k[1] for k in f})
    put("NumTwoCloseRatioPct", 100 * f[(0.0, dks[0])]["tp_track"] / f[(0.0, dks[0])]["tp_genie"],
        f"m_two: track/genie, equal delay, dk={dks[0]}", r"{:.0f}\%")
    put("NumTwoFarRatioPct", 100 * f[(0.0, dks[-1])]["tp_track"] / f[(0.0, dks[-1])]["tp_genie"],
        f"m_two: track/genie, equal delay, dk={dks[-1]}", r"{:.0f}\%")
    rat = {d: f[(0.0, d)]["tp_track"] / f[(0.0, d)]["tp_genie"] for d in dks}
    mn = min(rat.values())
    near = sorted(d for d in dks if rat[d] - mn < 0.005)
    put("NumTwoMinRatioPct", 100 * mn, "m_two: min track/genie, equal delay", r"{:.0f}\%")
    put("NumTwoMinDk", f"{near[0]:g}" if len(near) == 1 else f"{near[0]:g}--{near[-1]:g}", "m_two: separation(s) at the minimum")
    put("NumTwoDelayRatioPct", 100 * min(f[(1.0, d)]["tp_track"] / f[(1.0, d)]["tp_genie"] for d in dks),
        "m_two: min track/genie, delays one sample apart", r"{:.1f}\%")
    ck = json.load(open(RUNS / "checks.json"))["k6"]
    for L, nm in ((1, "One"), (4, "Four")):
        rr = sorted([r for r in ck if r["dl"] == 0 and r["L"] == L], key=lambda r: r["dk"])
        put(f"NumJointPenalty{nm}Db", rr[0]["ratio_db_median"], f"checks k6: median rho_J/rho_CR at the smallest dk, equal delay, L={L}", "{:.0f}")
    _jd = max(abs(r["ratio_db_median"]) for r in ck if r["dl"] == 1)
    assert _jd < 0.05, "joint and decoupled benchmarks differ for delay-separated paths: revise text"
    put("NumJointDelayMaxDb", _jd, "checks k6: max |rho_J/rho_CR| (dB), delays one sample apart", "{:.2f}")


_FRAME = ("NumFrameBurstSharePct", "NumFrameBestB", "NumFrameGapLongPct", "NumFrameLongB", "NumOpenLongTp", "NumOpenShortTp", "NumFrameShortB",
          "NumFrameTimeRatio", "NumFrameTrials", "NumFrameBlerLongPct", "NumFrameBlerMidPct", "NumFrameGapMinPct",
          "NumFrameGapMaxPct", "NumFrameLossMin", "NumFrameLossMax")


@_guard(_FRAME, "m_frame")
def _frame():
    e5, _ = summarize("m_frame")
    t12 = {r["B"]: r for r in e5 if r["snr_db"] == 12}
    bestB = max(t12, key=lambda b: t12[b]["tp_track"])
    put("NumFrameBestB", bestB, "m_frame: B maximizing track throughput at 12 dB", "{:d}")
    put("NumFrameGapLongPct", 100 * (1 - t12[max(t12)]["tp_track"] / t12[max(t12)]["tp_genie"]),
        "m_frame: track genie gap at largest B, 12 dB", r"{:.1f}\%")
    put("NumFrameLongB", max(t12), "m_frame: largest B", "{:d}")
    put("NumOpenLongTp", t12[max(t12)]["tp_openloop"], "m_frame: open-loop tp at largest B, 12 dB", "{:.2f}")
    put("NumOpenShortTp", t12[min(t12)]["tp_openloop"], "m_frame: open-loop tp at smallest B, 12 dB", "{:.2f}")
    put("NumFrameShortB", min(t12), "m_frame: smallest B", "{:d}")
    put("NumFrameTimeRatio", t12[max(t12)]["time_track"] / t12[16]["time_track"], "m_frame: track time B_max / B=16", "{:.1f}")
    put("NumFrameTrials", min(r["trials"] for r in e5), "m_frame: trials per point", "{:d}")
    spec, fb = frames("m_frame")
    gaps, nfr, shares = [], [], []
    for i, g in enumerate(spec["grid"]):
        if g["snr_db"] != 12:
            continue
        rs = fb[i]
        den = g["B"] * (S.N + S.Ncp)
        d = np.array([(sum(r["genie"]["goodbits"]) - sum(r["track"]["goodbits"])) / den for r in rs])
        gaps.append(100 * d.mean() / np.mean([sum(r["genie"]["goodbits"]) / den for r in rs]))
        nfr.append(int(sum(1 for r in rs if sum(r["track"]["blerr"]) >= 3)))
        if g["B"] >= 8:
            dd = np.array([sum(r["genie"]["goodbits"]) - sum(r["track"]["goodbits"]) for r in rs], float)
            bur = np.array([sum(r["track"]["blerr"]) >= 3 for r in rs])
            shares.append(dd[bur].sum() / dd.sum() if dd.sum() > 0 else 1.0)
    put("NumFrameGapMinPct", min(gaps), "m_frame: min paired genie gap over B at 12 dB", r"{:.1f}\%")
    put("NumFrameGapMaxPct", max(gaps), "m_frame: max paired genie gap over B at 12 dB", r"{:.1f}\%")
    put("NumFrameBurstSharePct", 100 * min(shares), "m_frame: min share of the genie gap in frames with >=3 block errors, B>=8, 12 dB", r"{:.0f}\%")
    put("NumFrameLossMin", min(nfr), "m_frame: min #frames with >=3 block errors (proposed), 12 dB", "{:d}")
    put("NumFrameLossMax", max(nfr), "m_frame: max #frames with >=3 block errors (proposed), 12 dB", "{:d}")
    put("NumFrameBlerLongPct", 100 * t12[max(t12)]["bler_track"], "m_frame: track BLER at largest B, 12 dB", r"{:.1f}\%")
    put("NumFrameBlerMidPct", 100 * t12[16]["bler_track"], "m_frame: track BLER at B=16, 12 dB", r"{:.1f}\%")


_ROB = ("NumRobTrials", "NumRobBaseRatioPct", "NumRobMinRatioPct", "NumDriftHsrRatioPct", "NumDriftHiDefault",
        "NumDriftHiWindow", "NumDriftHiModel", "NumDriftHiGenie", "NumDriftHiPtrack", "NumDriftZeroModelCostPct",
        "NumDriftZeroWindowCostPct", "NumPnHiTrack", "NumPnHiConv", "NumPnHiGenie", "NumPnHiGenieDet", "NumPnHiLw",
        "NumBornQTrack", "NumBornQTrackW", "NumBornQGenie", "NumBornQConv", "NumBornQPtrack", "NumBornHTrack",
        "NumBornHConv", "NumBornHPtrack", "NumBornHTrackW", "NumDriftGainTwoPct", "NumDriftGainFivePct",
        "NumDriftPtrackTenTp", "NumDriftTrackTenTp")


@_guard(_ROB, "m_robust")
def _rob():
    e6, _ = summarize("m_robust")
    VARS = ("rho_max", "window", "model_rho", "pn_var", "born_frac", "reacq_every", "P", "kappa_max", "cfo")

    def f6(**kw):
        for r in e6:
            if all(r.get(k) == v for k, v in kw.items()) and \
               all(k in kw or r.get(k) in (None, False) for k in VARS if k in r):
                return r
        raise KeyError(kw)
    b0 = f6(rho_max=0.0)
    put("NumRobTrials", min(r["trials"] for r in e6), "m_robust: trials per point", "{:d}")
    put("NumRobBaseRatioPct", 100 * b0["tp_track"] / b0["tp_genie"], "m_robust: reference track/genie", r"{:.0f}\%")
    rr = [100 * f6(**kw)["tp_track"] / f6(**kw)["tp_genie"] for kw in
          ({"cfo": 0.1}, {"cfo": 0.3}, {"P": 2}, {"P": 6}, {"P": 8}, {"kappa_max": 1.0}, {"kappa_max": 2.0})]
    put("NumRobMinRatioPct", min(rr), "m_robust: min track/genie over CFO, P, kappa_max variations", r"{:.0f}\%")
    d5 = f6(rho_max=0.005)
    put("NumDriftHsrRatioPct", 100 * d5["tp_track"] / d5["tp_genie"], "m_robust: track/genie at rho=0.005", r"{:.0f}\%")
    d2 = f6(rho_max=0.02); d2w = f6(rho_max=0.02, window=6); d2m = f6(rho_max=0.02, model_rho=True)
    put("NumDriftHiDefault", d2["tp_track"], "m_robust: track at rho=0.02", "{:.2f}")
    put("NumDriftHiWindow", d2w["tp_track"], "m_robust: track at rho=0.02, window 6", "{:.2f}")
    put("NumDriftHiModel", d2m["tp_track"], "m_robust: track at rho=0.02, rate model", "{:.2f}")
    put("NumDriftHiGenie", d2["tp_genie"], "m_robust: genie at rho=0.02", "{:.2f}")
    put("NumDriftHiPtrack", d2["tp_ptrack4"], "m_robust: periodic pilots (K=4) + tracker at rho=0.02", "{:.2f}")
    z0m = f6(rho_max=0.0, model_rho=True); z0w = f6(rho_max=0.0, window=6)
    put("NumDriftZeroModelCostPct", 100 * (1 - z0m["tp_track"] / b0["tp_track"]), "m_robust: rate-model cost at rho=0", r"{:.1f}\%")
    put("NumDriftZeroWindowCostPct", 100 * (1 - z0w["tp_track"] / b0["tp_track"]), "m_robust: window cost at rho=0", r"{:.1f}\%")
    pn = sorted([r for r in e6 if r.get("pn_var")], key=lambda r: r["pn_var"])
    put("NumPnHiTrack", pn[-1]["tp_track"], "m_robust: track at largest phase noise", "{:.2f}")
    put("NumPnHiConv", pn[-1]["tp_conv-da"], "m_robust: conv-da at largest phase noise", "{:.2f}")
    put("NumPnHiGenie", pn[-1]["tp_genie"], "m_robust: phase-noise-aware genie at largest phase noise", "{:.2f}")
    put("NumPnHiGenieDet", pn[-1]["tp_genie-det"], "m_robust: deterministic genie at largest phase noise", "{:.2f}")
    put("NumPnHiLw", "10^{-2}" if abs(pn[-1]["pn_var"] * 512 / (2 * np.pi) - 1e-2) < 1e-9 else "??", "m_robust: largest linewidth/df")
    bq = f6(born_frac=0.25); bqw = f6(born_frac=0.25, reacq_every=1, window=8); bh = f6(born_frac=0.5)
    put("NumBornQTrack", bq["tp_track"], "m_robust: track, 25% born", "{:.2f}")
    put("NumBornQTrackW", bqw["tp_track"], "m_robust: track with per-block re-acq + window, 25% born", "{:.2f}")
    put("NumBornQGenie", bq["tp_genie"], "m_robust: genie, 25% born", "{:.2f}")
    put("NumBornQConv", bq["tp_conv-da"], "m_robust: conv-da, 25% born", "{:.2f}")
    put("NumBornQPtrack", bq["tp_ptrack4"], "m_robust: ptrack4, 25% born", "{:.2f}")
    put("NumBornHTrack", bh["tp_track"], "m_robust: track, 50% born", "{:.2f}")
    put("NumBornHConv", bh["tp_conv-da"], "m_robust: conv-da, 50% born", "{:.2f}")
    put("NumBornHPtrack", bh["tp_ptrack4"], "m_robust: ptrack4, 50% born", "{:.2f}")
    put("NumBornHTrackW", f6(born_frac=0.5, reacq_every=1, window=8)["tp_track"], "m_robust: per-block re-acq + window, 50% born", "{:.2f}")
    g2 = 100 * max(f6(rho_max=0.002, model_rho=True)["tp_track"], f6(rho_max=0.002, window=6)["tp_track"]) / f6(rho_max=0.002)["tp_track"] - 100
    g5 = 100 * max(f6(rho_max=0.005, model_rho=True)["tp_track"], f6(rho_max=0.005, window=6)["tp_track"]) / f6(rho_max=0.005)["tp_track"] - 100
    put("NumDriftGainTwoPct", g2, "m_robust: best remedy gain at drift 0.002", r"{:.1f}\%")
    put("NumDriftGainFivePct", g5, "m_robust: best remedy gain at drift 0.005", r"{:.1f}\%")
    # paired comparisons of the drift remedies (seed_by snr: every point uses the same channels)
    from diagnostics import paired
    spec_r, fb_r = frames("m_robust")
    gi = {}
    for i, g in enumerate(spec_r["grid"]):
        key = (g.get("rho_max", 0.0), g.get("window"), bool(g.get("model_rho", False)), g.get("pn_var"),
               g.get("born_frac"), g.get("P"), g.get("kappa_max"), g.get("cfo"), g.get("reacq_every"))
        gi[key] = i

    def pr(rho, var):
        base = (rho, None, False, None, None, None, None, None, None)
        alt = (rho, 6, False, None, None, None, None, None, None) if var == "window" else \
            (rho, None, True, None, None, None, None, None, None)
        ra = {r["seed"]: r for r in fb_r[gi[base]]}
        rb = {r["seed"]: r for r in fb_r[gi[alt]]}
        rows = [{**ra[k], "_b": rb[k]["track"]} for k in ra if k in rb]
        assert len(rows) == len(ra), "drift variants not paired"
        return paired(rows, "_b", "track", seed=int(1000 * rho) + 1)
    for rho, nm in ((0.0, "Zero"), (0.002, "Two"), (0.005, "Five"), (0.02, "Twenty")):
        for var, vn in (("window", "Win"), ("rate", "Rate")):
            d = pr(rho, var)
            put(f"NumDrift{vn}{nm}Pct", 100 * d["ratio"], f"m_robust paired: {var}/default-1 at drift {rho}", r"{:+.1f}\%")
            put(f"NumDrift{vn}{nm}Ci", f"[{100 * d['ratio_lo']:+.1f}, {100 * d['ratio_hi']:+.1f}]",
                f"m_robust paired: 95% bootstrap interval, {var}, drift {rho}")
    put("NumDriftPtrackTenTp", f6(rho_max=0.01)["tp_ptrack4"], "m_robust: ptrack4 at drift 0.01", "{:.2f}")
    put("NumDriftTrackTenTp", f6(rho_max=0.01)["tp_track"], "m_robust: track at drift 0.01", "{:.2f}")


# ------------------------------------------------------------------ receiver design choices (held-out seeds)
@_guard(("NumRollbackTwoPct", "NumRollbackEightPct", "NumKeepMinPct", "NumKeepMaxPct", "NumValidateMinPct",
         "NumValidateMaxPct", "NumSplitMinPct", "NumSplitMaxPct", "NumDevTrials"), "dev_policy")
def _design():
    from diagnostics import paired
    def pr(name, a, b):
        spec, fb = frames(name)
        out = {}
        for i, g in enumerate(spec["grid"]):
            d = paired(fb[i], a, b, seed=1)
            out[(g.get("snr_db"), g.get("tp_dk"))] = d["ratio"]
        return out
    rb2 = pr("dev_policy_two", "track-rollback", "track-keep")
    rb = pr("dev_policy", "track-rollback", "track-keep")
    put("NumRollbackTwoPct", -100 * min(rb2.values()), "dev_policy_two: largest loss of reverting to the committed path set vs keep", r"{:.0f}\%")
    put("NumRollbackEightPct", -100 * rb[(8, None)], "dev_policy: loss of reverting vs keep at 8 dB", r"{:.1f}\%")
    kp = list(pr("dev_policy", "track-keep", "track-legacy").values()) + list(pr("dev_policy_two", "track-keep", "track-legacy").values())
    put("NumKeepMinPct", 100 * min(kp), "dev_policy: min keep/legacy-1", r"{:+.1f}\%")
    put("NumKeepMaxPct", 100 * max(kp), "dev_policy: max keep/legacy-1", r"{:+.1f}\%")
    vk = list(pr("dev_policy", "track-validate", "track-keep").values()) + list(pr("dev_policy_two", "track-validate", "track-keep").values())
    put("NumValidateMinPct", 100 * min(vk), "dev_policy: min validate/keep-1", r"{:+.1f}\%")
    put("NumValidateMaxPct", 100 * max(vk), "dev_policy: max validate/keep-1", r"{:+.1f}\%")
    sv = list(pr("dev_split", "track-keepsplit", "track-keep").values()) + list(pr("dev_split_two", "track-keepsplit", "track-keep").values())
    put("NumSplitMinPct", 100 * min(sv), "dev_split: min keep+split/keep-1", r"{:+.1f}\%")
    put("NumSplitMaxPct", 100 * max(sv), "dev_split: max keep+split/keep-1", r"{:+.1f}\%")
    put("NumDevTrials", min(len(v) for nm in ("dev_policy", "dev_policy_two") for v in frames(nm)[1].values()), "dev_policy: frames per point (min)", "{:d}")


# ------------------------------------------------------------------ write
lines = ["% GENERATED by make_numbers.py -- do not edit by hand.\n"]
for k in sorted(NUM):
    lines.append(f"\\newcommand{{\\{k}}}{{{NUM[k]['tex']}}}  % {NUM[k]['source']}\n")
OUT.parent.mkdir(exist_ok=True)
OUT.write_text("".join(lines))
json.dump({k: {"value": v["value"], "tex": v["tex"], "source": v["source"]} for k, v in NUM.items()},
          open(RUNS / "numbers.json", "w"), indent=1, default=float)
print("".join(lines))
