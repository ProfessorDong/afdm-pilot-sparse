"""Unit tests for the receiver primitives (run: python -m pytest tests -q)."""
import numpy as np

from engine import Config, simulate, _block_matrix_exact, make_system, draw_channel
from mbtrack import Tracker, apply_path, channel_matrix


def test_reacquire_sees_past_an_excluded_maximum():
    """A residual peak at a tracked path (stronger) must not hide a new path."""
    S = make_system(Config())
    rng = np.random.default_rng(1)
    B = 4
    X = np.exp(1j * np.pi / 4 * (2 * rng.integers(0, 4, (B, S.N)) + 1))
    blocks = np.arange(B)
    strong = (2.0, 1.3, 0.0)
    new = (5.0, -0.7, 0.0)
    Y = 1.0 * apply_path(S, *strong[:2], X, blocks) + 0.3 * apply_path(S, *new[:2], X, blocks)
    Y += 0.01 * (rng.standard_normal(Y.shape) + 1j * rng.standard_normal(Y.shape))
    T = Tracker(S, P_cap=8)
    # tracked path with half its gain: its residual peak (0.5) exceeds the new path (0.3)
    out = T.reacquire(Y, X, blocks, [strong], np.array([0.5 + 0j]), 1e-4)
    assert len(out) == 2
    assert abs(out[1][0] - new[0]) < 1 and abs(out[1][1] - new[1]) < 0.05


def test_circular_mode_is_exact():
    cfg = Config(frac_delay=True, circular=True)
    S = make_system(cfg)
    ch = draw_channel(cfg, np.random.default_rng(3))
    X = np.exp(1j * np.pi / 4 * (2 * np.random.default_rng(4).integers(0, 4, (3, S.N)) + 1))
    Y = sum(ch["h"][p] * apply_path(S, ch["tau"][p], ch["kappa"][p], X, np.arange(3)) for p in range(len(ch["tau"])))
    for b in range(3):
        H = channel_matrix(S, b, ch["tau"], ch["kappa"], ch["h"])
        assert np.allclose(H @ X[b], Y[b], atol=1e-10)


def test_phase_noise_aware_block_matrix():
    """Integer delays (no IBI): the PN-aware block matrix reproduces the channel."""
    cfg = Config(frac_delay=False)
    S = make_system(cfg)
    ch = draw_channel(cfg, np.random.default_rng(5))
    B, Tb = 3, S.N + S.Ncp
    pn = np.cumsum(np.sqrt(1e-4) * np.random.default_rng(6).standard_normal(B * Tb))
    X = np.exp(1j * np.pi / 4 * (2 * np.random.default_rng(7).integers(0, 4, (B, S.N)) + 1))
    Y = S.receive(S.channel(S.transmit(X), ch["tau"], ch["kappa"], ch["h"], pn_traj=pn), B)
    for b in range(B):
        H = _block_matrix_exact(S, ch, b, pn=pn[b * Tb + S.Ncp:(b + 1) * Tb])
        assert np.allclose(H @ X[b], Y[b], atol=1e-9)


def test_all_receivers_run():
    cfg = Config(frac_delay=True, coded=True, decision="decoded", P_max=8, reacq_every=4, B=6, snr_db=12,
                 diag=True, ofdm=True,
                 receivers=("genie", "conv", "conv-da", "genie-conv", "openloop", "track", "track-hybrid",
                            "sp-track1", "sp-track2", "sp-track-ll", "interp2", "ptrack2"))
    r = simulate(cfg, 11)
    for k in ("genie", "openloop", "track", "track-hybrid", "sp-track1", "sp-track2", "interp2", "ptrack2",
              "ofdm-track", "conv-da"):
        assert k in r, k
    assert len(r["diag-track"]["rho_pred"]) == 6
