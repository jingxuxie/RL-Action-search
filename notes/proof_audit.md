# Claim and proof audit

This is a self-review, not independent peer review, formal verification, or a novelty certificate.

| Claim | Assumptions | Status / independent numerical check |
|---|---|---|
| Expected-max Bellman operator and matched optimum | Fixed known proposal; independent menus | Established EMaQ construction, explicitly credited; finite-horizon derivation supplied |
| Delayed root switch after exactly d sweeps | Q-beta initialization, synchronous full sweeps, forced intermediate states, strict root gaps | Complete direct induction; fork tests and 2,952 grid configurations |
| Exposure envelope integral | Finite actions, deterministic score tie priority, rectangular value bounds | Complete inclusion-exclusion/tail-integral proof; independent menu enumeration |
| Gamma/4 <= R <= Gamma | One fixed value vector in the same local box; no Bellman realizability restriction | Complete random-endpoint argument; corner enumeration. Factor four not claimed sharp |
| Two-action equality | At most two supported alternatives | Complete co-occurrence proof; analytic tests |
| Local-to-return bound | Matched-K benchmark and common-menu coupling | Complete Bellman identity/unrolling proof; direct exact-value equality tests |
| Uniform adaptive stopping | Stratified independent samples, fixed outcome-independent counts, Bernoulli rewards, known layer supports and proposal | Full binomial/TV confidence event plus deterministic residual induction. Not proved for arbitrary trajectory data |
| Safe replacement | Robust bounds on old-policy values; all-state advantage test or disjoint root value bounds | Complete box minimization plus performance-difference proof; TV LP independently checked with scipy.linprog |
| Statistical lower bound | Two specified environments, fixed K, n risky samples | Complete KL/testing proof; exact binomial Bayes risk grid |
| Practical speedup | Explicit operation proxy, not actual model training cost | Conditional, not universal: random graph diagnostic cost is 4.134x generic |

## Errors and overclaims explicitly avoided

- `E_menu sup_v` and `sup_v E_menu` are different. Three uniform actions with K=3 and [0,1] intervals give 8/9 versus 2/3.
- Exact Q-beta is not exact Q-K. Neither should be confused with unrestricted Q-star.
- Larger K can screen low-ranked inversions, so ranking loss need not be monotone in K.
- The d-sweep result is not a lower bound against backward dynamic programming, asynchronous schedules, or arbitrary solvers.
- A final stopping diagnostic is a paid backup: m updates cost m+1 diagnostic/backup calls.
- Finite-data iterations share the dataset. Validity follows from a uniform model event, not fictional independent errors per iteration.
- Failure to certify is not proof of harm, not proof of information-theoretic impossibility, and not a negative experimental result by itself.
- The same small dataset may produce multiple proposals. Counts of proposals are not counts of independent experiments.
- Zero observed false decisions does not imply zero failure probability. One model-confidence-set miss is recorded, retained, and reported.
- Positive results on a targeted family were not preregistered, and are not asserted to describe typical benchmark prevalence.
- The certificate's factor-four approximation refers to local interval uncertainty, not an optimal MDP-robust guarantee.
- A local value function stored as a table permits compiling the extraction policy. Repeated K-score inference is not unavoidable.

## Remaining scientific questions

1. Can a cheap screening implementation retain useful exposure sensitivity without O(A^2 log A) overhead?
2. Is the factor four tight, or can the fixed-vector relaxation be approximated more sharply?
3. Can occupancy-aware, coupled confidence sets reduce conservatism without recreating full planning cost?
4. Can refresh scheduling beat straightforward planning/evaluation in a realistic small function-approximation setting?
5. Is the exposure theorem sufficiently distinct from existing robust ranking and random-assortment optimization work? This requires an independent specialist literature audit.
