"""Superimposed pilot on M equally spaced chirps (engine.sp_pilot, MultiPilotAcquirer)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import Acquirer, Config, MultiPilotAcquirer, make_system, simulate, sp_pilot  # noqa: E402


def test_pilot_energy_and_positions():
    S = make_system(Config())
    for M in (1, 2, 4):
        x = sp_pilot(S, 0.2, M)
        assert np.isclose(np.vdot(x, x).real, 0.2 * S.N)
        assert np.count_nonzero(x) == M and x[S.m0] != 0


def test_multipilot_atoms_reduce_to_single_pilot():
    """The multi-pilot atom formula, evaluated for one pilot, equals S.atoms."""
    S = make_system(Config())
    m = MultiPilotAcquirer.__new__(MultiPilotAcquirer)
    m.S, m.M = S, 1
    m.p = np.zeros(S.N, complex); m.p[S.m0] = 1.0
    m.Wsel = S.W
    ref = S.atoms([0, 3, 7], [-2.3, 0.0, 1.7])
    assert np.allclose(m._atoms([0, 3, 7], [-2.3, 0.0, 1.7]), ref, atol=1e-12)


def test_multipilot_acquisition_recovers_paths_noiseless():
    """Pilot only (no data, no noise), integer delays: the paths are recovered."""
    S = make_system(Config())
    paths = [(1, -1.37), (4, 0.62), (6, 2.21)]
    gains = np.array([1.0, 0.6 * np.exp(1j * 0.9), 0.4 * np.exp(-1j * 2.0)])
    for M in (2, 4):
        acq = MultiPilotAcquirer(S, M)
        x = sp_pilot(S, 0.2, M)[None, :]
        Y = S.receive(S.channel(S.transmit(x), [p[0] for p in paths], [p[1] for p in paths], gains), 1)
        Ep0, S.Ep = S.Ep, 0.2 * S.N
        est, g = acq.run(Y, 8, len(paths), sigma2=1e-6)
        S.Ep = Ep0
        est = sorted(est)
        assert [e[0] for e in est] == [p[0] for p in paths]
        assert np.allclose([e[1] for e in est], [p[1] for p in paths], atol=2e-3)


def test_multipilot_sp_track_decodes_at_high_snr():
    """End to end: the superimposed pilot + tracker with M = 2 and 4 decodes at 20 dB."""
    for M in (2, 4):
        cfg = Config(snr_db=20.0, coded=True, decision="decoded", P_max=8, frac_delay=True, reacq_every=4,
                     receivers=("sp-track",), spt_eps=0.3, sp_acq_blocks=16, spt_M=M)
        out = simulate(cfg, 12345)
        assert sum(out["sp-track"]["blerr"]) <= 1, out["sp-track"]["blerr"]


def test_multipilot_results_do_not_depend_on_process_history():
    """The cached multi-pilot acquirer must use the current trial's system object
    (its pilot energy scales the initial gains): a trial gives the same result
    whether or not other trials ran before it in the same process."""
    import engine
    cfg = Config(snr_db=8.0, coded=True, decision="decoded", P_max=8, frac_delay=True, reacq_every=4,
                 receivers=("sp-track",), spt_eps=0.1, sp_acq_blocks=4, spt_M=4)
    engine._MPA.clear()
    fresh = simulate(cfg, 777)
    engine._MPA.clear()
    simulate(cfg, 778)                      # populate the cache with another trial
    again = simulate(cfg, 777)
    for f in ("err", "blerr", "goodbits"):
        assert fresh["sp-track"][f] == again["sp-track"][f]
