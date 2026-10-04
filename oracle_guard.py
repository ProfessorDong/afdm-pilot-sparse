"""Upper bound on what an ideal adaptive guard could save (runs/oracle_guard.json).

For each channel draw of the main configuration, size the guard to the realized
support (largest delay rounded up, since fractional delays spill to the next
integer delay; largest integer Doppler), keeping c1 fixed, and compare the mean
guard with the fixed design."""
import json

import numpy as np

from engine import Config, draw_channel, make_system


def main(draws=20000, seed=2026):
    cfg = Config(frac_delay=True)
    S = make_system(cfg)
    rng = np.random.default_rng(seed)
    fr = []
    for _ in range(draws):
        ch = draw_channel(cfg, rng)
        l_eff = int(np.ceil(ch["tau"].max()))
        a_eff = int(np.max(np.abs(np.round(ch["kappa"]))))
        g = S.Q * l_eff + 2 * (a_eff + S.xi)
        fr.append((2 * g + 1) / S.N)
    fr = np.array(fr)
    fixed = len(S.zero_set) / S.N
    res = {"fixed_guard": fixed, "oracle_mean": float(fr.mean()),
           "max_gain_oracle_vs_fixed": float((1 - fr.mean()) / (1 - fixed) - 1),
           "pilot_sparse_frame_guard_share": len(S.zero_set) / (16 * S.N)}
    json.dump(res, open("runs/oracle_guard.json", "w"), indent=1)
    print(res)


if __name__ == "__main__":
    main()
