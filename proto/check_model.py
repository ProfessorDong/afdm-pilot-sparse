"""Gate 0: verify the multi-block AFDM structure the whole design rests on.

1. Per-block pilot response through one path equals exp(j2 pi kappa b beta) * atom.
2. The atom's energy is concentrated at the chirp-domain index m0 + alpha + Q ell.
3. With data present, leakage of data into the pilot window is bounded (xi guard).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mbafdm import MBAFDM  # noqa: E402

rng = np.random.default_rng(0)
sys_ = MBAFDM()
print(f"N={sys_.N} Ncp={sys_.Ncp} Q={sys_.Q} ell_max={sys_.ell_max} "
      f"separable={sys_.separable} overhead={sys_.overhead:.3f} beta={sys_.beta:.5f}")

B = 6
x = sys_.make_frame(B, rng, data=False)
worst = 0.0
for trial in range(20):
    l = int(rng.integers(0, sys_.ell_max + 1))
    k = rng.uniform(-sys_.alpha_max, sys_.alpha_max)
    y = sys_.receive(sys_.channel(sys_.transmit(x), [l], [k], [1.0]), B)[:, sys_.W]
    a = sys_.atoms([l], [k])[0, 0] * np.sqrt(sys_.N * 0 + len(sys_.zero_set))
    for b in range(B):
        pred = np.exp(1j * 2 * np.pi * k * b * sys_.beta) * a
        worst = max(worst, np.linalg.norm(y[b] - pred) / np.linalg.norm(pred))
print(f"[1] max rel. error of exp(j2pi kappa b beta)*atom model: {worst:.2e}")

# [2] where is the energy?
l, k = 3, 1.37
a = sys_.atoms([l], [k])[0, 0]
peak = sys_.W[np.argmax(np.abs(a))]
print(f"[2] ell={l} kappa={k}: peak at DAFT index {peak}, predicted "
      f"m0+round(kappa)-Q*ell = {sys_.index_of(l, k)}; "
      f"energy in window = {np.sum(np.abs(a)**2):.4f}")

# [3] data leakage into W (no pilot), as fraction of pilot-response energy
xd = sys_.make_frame(B, rng, data=True)
xd[:, sys_.m0] = 0
leak, pil = [], []
for trial in range(200):
    P = 4
    ls = rng.choice(sys_.ell_max + 1, P, replace=False)
    ks = rng.uniform(-sys_.alpha_max, sys_.alpha_max, P)
    hs = (rng.standard_normal(P) + 1j * rng.standard_normal(P)) / np.sqrt(2 * P)
    yd = sys_.receive(sys_.channel(sys_.transmit(xd), ls, ks, hs), B)[:, sys_.W]
    yp = sys_.receive(sys_.channel(sys_.transmit(sys_.make_frame(B, rng, data=False)), ls, ks, hs), B)[:, sys_.W]
    leak.append(np.sum(np.abs(yd) ** 2)); pil.append(np.sum(np.abs(yp) ** 2))
r = np.sum(leak) / np.sum(pil)
print(f"[3] data-leakage / pilot energy in window: {r:.2e}  ({10*np.log10(r):.1f} dB)")
