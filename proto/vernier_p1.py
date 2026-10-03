"""Gate 1: single-path Doppler estimation from the chirp-domain pilot window.

Compares, versus B and pilot SNR:
  SB   : single block (b=0 only), grid + local refine
  NC   : non-coherent multi-block (sum_b |c_b|^2)
  VERN : two-stage vernier: cell from NC, fine kappa from the coherent
         slow-time score restricted to kappa_c +- 1/(2 beta)
  ML   : coherent multi-block ML over the full range (envelope x array factor)
against the nuisance-eliminated CRB (unknown complex gain, known integer delay).

Reports RMSE, CRB, and the correct-cell probability Pr(|k_c - k| < 1/(2 beta)).
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mbafdm import MBAFDM  # noqa: E402

S = MBAFDM()
L = S.ell_max + 1
KG = np.arange(-S.alpha_max - 0.6, S.alpha_max + 0.6 + 1e-9, 0.02)
A = S.atoms(range(L), KG)                       # (L, K, W)
A /= np.linalg.norm(A, axis=-1, keepdims=True)


def corr(y, l, k):
    a = S.atoms([l], [k])[0, 0]
    return (y @ np.conj(a)) / np.linalg.norm(a)   # (B,)


def steer(k, B):
    return np.exp(-1j * 2 * np.pi * k * np.arange(B) * S.beta)


def coh(y, l, k):
    return np.abs(steer(k, len(y)) @ corr(y, l, k)) ** 2


def nc(y, l, k):
    return np.sum(np.abs(corr(y, l, k)) ** 2)


def refine(f, k0, half):
    r = minimize_scalar(lambda k: -f(k), bounds=(k0 - half, k0 + half),
                        method="bounded", options={"xatol": 1e-6})
    return r.x


def estimate(y):
    B = y.shape[0]
    C = np.einsum("lkw,bw->lkb", np.conj(A), y)   # (L, K, B)
    out = {}
    # SB
    s = np.abs(C[:, :, 0]) ** 2
    l, i = np.unravel_index(np.argmax(s), s.shape)
    out["SB"] = refine(lambda k: nc(y[:1], l, k), KG[i], 0.02)
    # NC
    s = np.sum(np.abs(C) ** 2, axis=-1)
    l, i = np.unravel_index(np.argmax(s), s.shape)
    kc = refine(lambda k: nc(y, l, k), KG[i], 0.02)
    out["NC"] = kc
    # VERN: coherent search within the cell around the NC estimate
    half = 0.5 / S.beta
    kk = np.arange(kc - half, kc + half, 0.002 / max(B, 1))
    sc = [coh(y, l, k) for k in kk]
    out["VERN"] = refine(lambda k: coh(y, l, k), kk[int(np.argmax(sc))], 0.002 / max(B, 1))
    out["kc"] = kc
    # ML over full range: grid then refine
    st = np.exp(-1j * 2 * np.pi * np.outer(KG, np.arange(B)) * S.beta)   # (K, B)
    s = np.abs(np.einsum("lkb,kb->lk", C, st)) ** 2
    l2, i2 = np.unravel_index(np.argmax(s), s.shape)
    # the 0.02 grid under-samples the 1/(B beta) main lobe for large B: densify locally
    kk = np.arange(KG[i2] - 0.6, KG[i2] + 0.6, 0.002 / max(B, 1))
    sc = [coh(y, l2, k) for k in kk]
    out["ML"] = refine(lambda k: coh(y, l2, k), kk[int(np.argmax(sc))], 0.002 / max(B, 1))
    return out


def crb(l, k, B, sigma2, h=1.0, d=1e-5):
    a = S.atoms([l], [k - d, k, k + d])[0]
    ph = lambda kk: np.exp(1j * 2 * np.pi * kk * np.arange(B) * S.beta)
    mu = lambda kk, ai: h * np.outer(ph(kk), ai).reshape(-1)
    dk = (mu(k + d, a[2]) - mu(k - d, a[0])) / (2 * d)
    base = np.outer(ph(k), a[1]).reshape(-1)
    D = np.stack([dk, base, 1j * base], 1)
    J = 2 / sigma2 * np.real(D.conj().T @ D)
    return 1.0 / (J[0, 0] - J[0, 1:] @ np.linalg.solve(J[1:, 1:], J[1:, 0]))


def run(Bs, pilot_snrs_db, trials, seed=7):
    rng = np.random.default_rng(seed)
    res = []
    for psnr in pilot_snrs_db:
        sigma2 = 10 ** (-psnr / 10)       # unit-energy pilot, unit-modulus gain
        for B in Bs:
            err = {k: [] for k in ("SB", "NC", "VERN", "ML")}
            cell, cr = [], []
            for t in range(trials):
                l = int(rng.integers(0, L))
                k = rng.uniform(-S.alpha_max, S.alpha_max)
                h = np.exp(1j * rng.uniform(0, 2 * np.pi))
                a = S.atoms([l], [k])[0, 0]
                y = h * np.exp(1j * 2 * np.pi * k * np.arange(B)[:, None] * S.beta) * a[None, :]
                y = y + np.sqrt(sigma2 / 2) * (rng.standard_normal(y.shape) + 1j * rng.standard_normal(y.shape))
                o = estimate(y)
                for key in err:
                    err[key].append(o[key] - k)
                cell.append(abs(o["kc"] - k) < 0.5 / S.beta)
                cr.append(crb(l, k, B, sigma2))
            row = {"pilot_snr_db": psnr, "B": B,
                   **{f"rmse_{key}": float(np.sqrt(np.mean(np.square(v)))) for key, v in err.items()},
                   **{f"medae_{key}": float(np.median(np.abs(v))) for key, v in err.items()},
                   "p_cell": float(np.mean(cell)), "sqrt_crb": float(np.sqrt(np.mean(cr)))}
            res.append(row)
            print(f"psnr={psnr:5.1f} B={B:2d} cell={row['p_cell']:.3f} "
                  f"SB={row['rmse_SB']:.2e} NC={row['rmse_NC']:.2e} "
                  f"VERN={row['rmse_VERN']:.2e} ML={row['rmse_ML']:.2e} "
                  f"sqrtCRB={row['sqrt_crb']:.2e}", flush=True)
    return res


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "runs"
    out.mkdir(exist_ok=True)
    res = run(Bs=[1, 2, 4, 8, 16], pilot_snrs_db=[-5, 0, 5, 10, 20], trials=300)
    json.dump({"system": {"N": S.N, "Ncp": S.Ncp, "Q": S.Q, "ell_max": S.ell_max,
                          "alpha_max": S.alpha_max, "xi": S.xi, "beta": S.beta},
               "results": res}, open(out / "gate1_vernier_p1.json", "w"), indent=1)
