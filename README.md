# Pilot-Sparse Multi-Block AFDM

Reference implementation and artifacts for

> **Pilot-Sparse AFDM with Decision-Directed Doppler Tracking for High-Mobility Links**
> L. Dong, submitted to IEEE Transactions on Vehicular Technology.

An embedded-pilot AFDM block spends a third of its chirps on the null guard
around the pilot. Consecutive blocks see each path through the same delay and
Doppler, so a path's coefficient advances by a deterministic phase
`exp(j 2 pi kappa b beta)`, `beta = (N + Ncp)/N`, from block to block. One
pilot block acquires every path; the receiver then predicts, detects, decodes
and appends each guard-free data block and refits the path parameters.

## Status

The manuscript's LaTeX sources are not included here. Every simulation
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
python -m pytest tests -q                    # smoke test (~10 s)
python run_sweep.py specs/tune_v2.json       # superimposed-pilot tuning, held-out seeds
python run_sweep.py specs/tune_v3.json       #   grid extension beyond the first boundaries
python run_sweep.py specs/tune_v4.json       #   grid extension (single-block iterations)
python pick_v2.py                            # writes the tuned settings into specs/m_main.json
python run_sweep.py specs/m_main.json        # Figs. 3-5, Table III: all receivers, same channels
python run_sweep.py specs/m_two.json         # Fig. 6(b): two closely spaced paths
python run_sweep.py specs/m_mismatch.json    # matched-model control (Fig. 5)
python run_sweep.py specs/m_robust.json      # Table IV
python run_sweep.py specs/m_frame.json       # Fig. 7
python run_sweep.py specs/dev_c2p_ref.json   # paired c2 check (held-out seeds)
python run_sweep.py specs/dev_c2p_irr.json
python theory_check.py && python checks.py && python oracle_guard.py
python make_numbers.py && python make_figures.py && python make_tables.py && python make_tables.py robust
```

Seeds are fixed in the specs; sweeps resume from their JSON-lines files after an
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
