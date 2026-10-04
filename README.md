# Pilot-Sparse Multi-Block AFDM

Reference implementation and artifacts for

> **Pilot-Sparse Multi-Block AFDM with Single-Block Acquisition and
> Decision-Directed Doppler Tracking for High-Mobility Links**
> L. Dong, submitted to IEEE Transactions on Vehicular Technology.

An embedded-pilot AFDM block spends a third of its chirps on the null guard
around the pilot. Consecutive blocks see each path through the same delay and
Doppler, so a path's coefficient advances by a deterministic phase
`exp(j 2 pi kappa b beta)`, `beta = (N + Ncp)/N`, from block to block. One
pilot block acquires every path; the receiver then predicts, detects, decodes
and appends each guard-free data block and refits the path parameters.

## Status

The manuscript is being submitted to IEEE TVT; its LaTeX sources are not
included here. Simulation artifacts for the main results (`e4_main`,
`e5_frame`, `e6_robust`) are added when the sweeps complete. The scripts that
write the paper's numbers, figures and tables (`make_*.py`) target a `paper/`
directory that is not part of this repository.

## Layout

```
mbafdm.py        sample-exact multi-block AFDM transmitter and physical channel
                 (fractional delays, Doppler rate, path births, CFO, phase noise)
engine.py        one simulated frame through every compared receiver; metrics
mbtrack.py       path operator, LMMSE, variable-projection Gauss-Newton tracker,
                 data-aided re-acquisition
coding.py        rate-1/2 K=7 convolutional code, CRC-16, soft Viterbi
run_sweep.py     parallel, resumable sweeps (JSON lines, one line per trial)
aggregate.py     per-point statistics
theory_check.py  numerical validation of Theorem 1, Proposition 1, Theorem 2
make_numbers.py  every number quoted in the paper's text -> paper/numbers.tex
make_figures.py  every figure -> paper/fig_*.tex
make_tables.py   parameter table -> paper/tab_params.tex
specs/           sweep definitions
runs/            artifacts
proto/           development gates (not used by the paper)
pick_sp.py       picks the tuned superimposed-pilot configuration per SNR
audit_acronyms.py  checks that every acronym is defined at first use
queue.sh, queue2.sh  run the experiment chain
```

## Reproducing the paper

```bash
python run_sweep.py specs/tune_sp.json      # superimposed-pilot tuning (held-out seeds)
python run_sweep.py specs/tune_sp_ext.json  # extended grid: no chosen configuration on the boundary
python pick_sp.py                           # writes the tuned (eps, iters) into specs/e4_main.json
python theory_check.py                      # Fig. 2 data
python run_sweep.py specs/tune_sptrack.json # superimposed pilot + tracker tuning (held-out seeds)
python run_sweep.py specs/e4_main.json      # Figs. 3-4: proposed, open loop, single-block SP
python run_sweep.py specs/e4b_baselines.json # Figs. 3-4: data-aided per-block pilot, its ceiling,
                                            #   SP + same tracker, OFDM same tracker (same channels)
python run_sweep.py specs/e4c_interp.json   # Fig. 3: periodic pilot blocks (same channels)
python run_sweep.py specs/e5_frame.json     # Fig. 5
python run_sweep.py specs/e6_robust.json    # robustness table
python make_numbers.py && python make_figures.py && python make_tables.py && python make_tables.py robust
cd paper && pdflatex AFDM_TVT && bibtex AFDM_TVT && pdflatex AFDM_TVT && pdflatex AFDM_TVT
```

`queue.sh` runs the whole chain. Seeds are fixed in the specs; sweeps resume
from their JSON-lines files after an interruption. CPU only; the code sets one
BLAS thread per worker.

## Verified invariants

- Per-block pilot response equals `exp(j2 pi kappa b beta)` times the block-0
  response to 1e-14 (Lemma 1); the dense and FFT block matrices agree to 1e-15.
- A path lands at chirp index `m0 + alpha - Q*ell`.
- The grating-lobe level of Theorem 1 matches the measured atom correlation to
  four digits for N in {256, 512, 1024} and Ncp in {4,...,64}.

## License

MIT -- see `LICENSE`.
