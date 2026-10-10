# Pilot-Sparse Multi-Block AFDM

Reference implementation and artifacts for

> **Pilot-Sparse AFDM with Decision-Directed Doppler Tracking for High-Mobility Links**
> L. Dong, submitted to IEEE Transactions on Vehicular Technology.

An embedded-pilot AFDM block spends a third of its chirps on the pilot and the
null region around it. Consecutive blocks see each path through the same delay
and Doppler, so a path's coefficient advances by a deterministic phase
`exp(j 2 pi kappa b beta)`, `beta = (N + Ncp)/N`, from block to block. One
pilot block initializes the detectable paths; the receiver then predicts,
detects and decodes each block carrying data on all chirps, appends it to a
trusted aperture if its CRC passes, refits the path parameters, and re-acquires
missed paths from the decoded data.

## Status

The manuscript's LaTeX sources are not included here; the generators write
their outputs (numbers.tex, fig_*.tex, tab_*.tex) to `paper/`, which they create
if needed. `python make_manifest.py` writes `runs/MANIFEST.json` with the commit,
environment, SHA-256 and completed/failed trial counts of every sweep, and the
hashes of the generated outputs. Every simulation
artifact that the paper's numbers, figures and tables are computed from is in
`runs/`. The scripts that
write the paper's numbers, figures and tables (`make_*.py`) target a `paper/`
directory that is not part of this repository.

## Layout

```
mbafdm.py        sample-exact multi-block AFDM transmitter and physical channel
                 (fractional delays, Doppler rate, path births, CFO, shared phase noise)
engine.py        one simulated frame through every compared receiver on the same channel
mbtrack.py       path operator, LMMSE, variable-projection Gauss-Newton tracker with a
                 trusted (CRC-verified) aperture, data-aided re-acquisition
coding.py        rate-1/2 K=7 convolutional code, CRC-16, soft Viterbi
run_sweep.py     parallel, resumable sweeps (JSON lines, one line per trial);
                 "seed_by": "snr" pairs every grid point of an SNR on the same channels
aggregate.py     per-point statistics
diagnostics.py   path association, prediction-error diagnostics, paired frame bootstrap
theory_check.py  validation of Approximation 1 (correct cell, retention), Theorem 1
checks.py        numerical checks: Kaufman Jacobian, leakage, ambiguity, guard scaling,
                 RMS delay spread, joint two-path Cramer-Rao benchmark
oracle_guard.py  bound on the saving of an ideal adaptive guard
pick_v2.py       tuned superimposed-pilot configurations (held-out seeds) -> specs/m_main.json
pick_v3.py       held-out choice of the number M of superimposed pilot chirps (rule fixed in
                 its docstring before the M > 1 results were seen)
merge_multipilot.py  merges the rerun of the superimposed receivers into runs/m_main.jsonl
replay.py        re-simulates the first trial of every grid point and compares it with the
                 recorded row (runs/REPLAY.json); make_manifest.py writes runs/MANIFEST.json
make_numbers.py  every number quoted in the paper's text -> paper/numbers.tex
make_figures.py  every figure and the paired-interval table -> paper/fig_*.tex, tab_paired.tex
make_tables.py   parameter table and impairment table
tests/           unit tests (python -m pytest tests -q)
specs/           sweep definitions
runs/            artifacts
proto/           development gates (not used by the paper)
```

## Reproducing the paper

```bash
pip install -r requirements.txt
python -m pytest tests -q                    # unit tests (~30 s)
python run_sweep.py specs/tune_v2.json       # superimposed-pilot tuning, held-out seeds
python run_sweep.py specs/tune_v3.json       #   grid extension beyond the first boundaries
python run_sweep.py specs/tune_v4.json       #   grid extension (single-block iterations)
python pick_v2.py                            # writes the tuned settings into specs/m_main.json
python run_sweep.py specs/m_main.json        # Figs. 3-5, Table III: all receivers, same channels
python run_sweep.py specs/tune_v5.json       # superimposed pilot on M = 2, 4 chirps + tracker, held-out
python run_sweep.py specs/tune_v6.json       #   single-block version, held-out
python pick_v3.py                            # held-out choice of M (runs/tune_v5_choice.json)
python run_sweep.py specs/m_main_mp.json     # superimposed receivers with the chosen M, evaluation seeds
python run_sweep.py specs/m_main_lo.json     # receivers whose random stream shifted (0-6 dB)
python merge_multipilot.py                   # merges both into runs/m_main.jsonl
python run_sweep.py specs/m_cap.json        # order-cap sensitivity (P_max 6, 12), m_main channels
python run_sweep.py specs/m_cap_p8.json     #   P = 8 paths, cap 8 vs 12, m_robust channels
python run_sweep.py specs/m_two.json         # Fig. 6(b): two closely spaced paths
python run_sweep.py specs/m_mismatch.json    # matched-model control (Fig. 5)
python run_sweep.py specs/m_robust.json      # Table IV
python run_sweep.py specs/m_frame.json       # Fig. 7
python run_sweep.py specs/dev_c2p_ref.json   # paired c2 check (held-out seeds)
python run_sweep.py specs/dev_c2p_irr.json
python theory_check.py && python checks.py && python oracle_guard.py   # all checks (e1-e5, k1-k9) by default
python make_numbers.py && python make_figures.py && python make_tables.py && python make_tables.py robust
python replay.py && python make_manifest.py  # provenance: re-simulation check and manifest
```

The 16 and 20 dB points of `m_main` were recomputed after the tie-breaking rule of
`pick_v2.py` was fixed (ties toward more acquisition blocks); the earlier rows are
kept in `runs/superseded/m_main_points8-9_tiebreak.jsonl`. The superimposed-pilot rows of
`m_main` come from `m_main_mp` and `m_main_lo` (merged by `merge_multipilot.py` into
`runs/superseded/m_main_before_multipilot.jsonl`); the reruns use draw-only receiver tokens
(`rng:conv`, `rng:sp`, `rng:conv-da`) that reproduce the random stream of a full run, and
`replay.py` confirms that a full run of `specs/m_main.json` reproduces the merged rows. Seeds are fixed in the specs; sweeps resume from their JSON-lines files after an
interruption. CPU only; one BLAS thread per worker. The full set of sweeps takes
about three days on 18 cores.

## Verified invariants

- Per-block pilot response equals `exp(j2 pi kappa b beta)` times the block-0
  response to 1e-14 (Lemma 1); the dense and FFT block matrices agree to 1e-15.
- A path lands at chirp index `m0 + alpha - Q*ell`.
- The grating-lobe level of Proposition 1 matches the measured atom correlation to
  four digits for N in {256, 512, 1024} and Ncp in {4,...,64}.

## License

MIT -- see `LICENSE`.
