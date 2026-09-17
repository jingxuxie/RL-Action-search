# Research status — September 2026

## Delivered

An 18-page compiled first research draft with complete derivations and proofs, modular NumPy/SciPy implementation, 303 locally passing tests, twelve scientific CSV files, nine separately generated figures, and reproducibility commands. The project began with an empty repository.

The scope evolved from a proposed general search-versus-refresh compute allocator to a narrower, proved candidate-exposure certificate and safe fixed-dataset refresh framework. The paper is explicit about this change. A global optimal allocation theorem has **not** been obtained.

## What the results support

The exact example isolates a sequential source of nonmonotonic action-search performance even when the reference-policy critic is perfect. The candidate-exposure theorem measures when uncertain relative action values can actually affect the extracted policy. It gives a deterministic local box guarantee and an adaptive high-probability finite-data application under a specified sampling model.

Fewer backups are genuinely possible in a targeted family (52.7% average reduction), but not uniformly valuable: on random graphs the reduction is 5.2% and the complete operation proxy is 4.134 times the generic baseline. An ordered tabular planner is cheaper still. These findings argue against claiming a broadly faster RL algorithm at present.

The low-coverage one-step control distinguishes estimation-only degradation from continuation mismatch: it contains 625 harmful ungated proposals among 1,500. The conservative gate accepts 301 proposals with no observed harmful acceptances, but can also reject improvements for a long time. Among 180 fitted-linear configurations, five refresh sweeps help 56.1% and hurt 43.9%. No neural performance extrapolation is supported.

## Readiness assessment

This is a substantive complete research draft, not a submission-ready paper. The outstanding issues are practical diagnostic cost, statistical conservatism, specificity/novelty of the core robust-ranking theorem, and lack of an independently reviewed proof or convincing function-approximation advantage. Large compute is not the immediate bottleneck.

The highest-priority next research step is a cheaper exposure diagnostic (or a clear impossibility/tradeoff result), evaluated against the already included backward-DP and full-refresh controls. Broad benchmark expansion before resolving this issue would risk adding results without strengthening the central claim.

## Reproducibility and provenance

All reported numbers come from executed deterministic-seed CPU experiments. Exact expected policy returns replace noisy rollout estimates. Offline samples are generated as exact binomial/multinomial sufficient statistics of the declared one-step sampling model. Family selection and additional mechanism controls were not preregistered. The initial manuscript, code, proofs, and audit were AI-assisted; self-checks are not independent validation.

The repository's materialization workflow reproduces the source-derived results and builds the manuscript on a fresh runner. Local runtime records and runner records need not agree. Cross-platform numeric equality is tested with tolerances; CSV manifests identify each produced set of scientific outputs.
