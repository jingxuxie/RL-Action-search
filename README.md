# Search or Refresh?
## Candidate-Exposure Certificates for Offline Policy Extraction

**Status: complete first research draft with self-contained proofs and executed CPU experiments; not yet submission-ready or independently reviewed.**

The project studies how a fixed-proposal best-of-K policy exposes uncertain critic rankings, when budget-matched Bellman updates are needed, and when a fixed offline dataset cannot justify more search. The expected-max Bellman operator itself is EMaQ's, not a new contribution.

- **Manuscript:** [paper/main.pdf](paper/main.pdf), with [LaTeX source](paper/main.tex) and full proofs.
- **Recorded findings:** [results/summary.json](results/summary.json), [research status](notes/research_status.md), and [claim audit](notes/proof_audit.md).
- **Implementation:** [src/search_refresh](src/search_refresh); [experiments](experiments); [tests](tests).
- **Prior-work boundary:** [notes/literature.md](notes/literature.md).

### Main mathematical results

1. An exact delayed fork isolates continuation-policy mismatch even with an exact reference-policy critic. The consequential ranking changes after exactly d synchronous local sweeps. This is **not** a lower bound against arbitrary planning; backward dynamic programming is faster in a known layered model.
2. A closed-form, O(A^2 log A) candidate-exposure envelope Gamma upper-bounds worst-case ranking regret over a rectangular value box. If R is the worst regret of **one fixed value vector**, then `Gamma/4 <= R <= Gamma`, with equality `R=Gamma` for two actions. Gamma is **not** generally the exact fixed-vector worst case, nor a Bellman-consistent MDP minimax certificate.
3. Loss propagation and simultaneous model confidence sets produce certificates valid across adaptive refresh iterations and search budgets on the same offline dataset. A separate conservative gate certifies replacement of the current policy.
4. A two-environment lower bound identifies a search-exposed statistical limit of order `(1 - 2^(1-K))/sqrt(n)` that unlimited computation cannot remove.

### Completed validation and important negative results

| Check or experiment | Recorded result |
|---|---|
| Unit/property tests | 303 passing locally |
| Interval geometries | 400; 5,840 enumerated menus; 6,000 checked box vertices |
| Maximum menu-formula discrepancy | 1.78e-15 |
| Delayed-fork grid | 2,952 configurations |
| Random exact graph/budget pairs | 234; 5.2% fewer backups, but **4.134x** diagnostic-inclusive operation proxy |
| Targeted rare-choice pairs | 180; 52.7% fewer backups, 48.0% smaller proxy |
| Main offline data | 480 datasets, 1,440 proposals, 907 accepted, 445 near-optimality certificates |
| Main observed failures | One model-confidence-set miss; zero false acceptances or false near-optimality certificates |
| Estimation-only control | 500 datasets, 1,500 proposals; 625 harmful ungated proposals; 301 accepted, zero harmful acceptances |
| Fitted linear critics | 180 task/data/budget configurations; five refresh sweeps help 56.1%, hurt 43.9%; mean return gain 0.00726 |

The positive rare-choice result is a **controlled mechanism test**, not a typical-benchmark speedup. The operation proxy is documented, not a hardware-independent FLOP or wall-clock claim. Exact tabular backward planning is included and is cheaper than the synchronous stopping procedures. Confidence-based selection can reject substantial genuine improvement. Three proposals per dataset and multiple configurations per task are dependent; aggregate counts are not independent trials.

### Reproduce

Python 3.10 or later is supported by the source. The locally tested environment is Python 3.13.5; exact scientific dependency versions are in `requirements-tested.txt`.

```bash
python -m pip install -r requirements-tested.txt
python -m pip install -e .
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
python -m pytest -q
python experiments/run_all.py --suite all
python experiments/stress_controls.py
python experiments/run_all.py --suite summary
python experiments/plot_results.py
python scripts/verify_results.py
bash scripts/build_paper.sh
```

The last command requires a LaTeX installation with `pdflatex`, `bibtex` or `bibtex8`, and the packages listed in `paper/main.tex`. `make reproduce` executes the same scientific pipeline; `make paper` builds the PDF. No GPU, pretrained models, external datasets, or paid API are required.

The data generator stores binomial/multinomial sufficient statistics from the stated stratified independent one-step sampling scheme. It is not a behavior-trajectory benchmark. Actual sample totals and zero-count rows are recorded. `results/manifest.json` gives hashes of the scientific CSVs, excluding machine-dependent runtimes. Cross-platform floating-point or library differences can change byte-level hashes even when numerical results agree.

### What is not yet established

This is not a globally optimal joint search/refresh scheduler, not a general neural offline-RL algorithm, and not an empirical benchmark paper. The dominant practical issue is certificate cost and conservatism. Independent novelty/proof review, a cheaper diagnostic implementation, and a convincing setting where the criterion beats inexpensive planning/refresh baselines are required before calling it submission-ready.

The initial package was generated and self-checked with AI assistance. Numerical tests are not formal verification, and mathematical claims should be reviewed independently.
