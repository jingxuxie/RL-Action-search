# Data and validation protocol

- Independent one-step reward and transition draws, stratified by state and action. Counts are fixed from the known proposal before outcomes; no dependence on stopping outcomes.
- Bernoulli rewards and multinomial successor counts are stored as sufficient statistics. This is distributionally the same experiment as retaining the raw independent records, not an approximation by fractional samples.
- Known layer/terminal support is a modeling assumption, not knowledge of transition probabilities. Reward confidence bounds do not reveal zero intermediate rewards.
- Best-of-K expected values and marginal policies are integrated analytically. Independent brute-force enumeration checks the implementation for tiny cases.
- True models evaluate final policies and audit confidence containment, but never decide the stopping or acceptance rules. Named hindsight oracles are evaluation references only.
- Baselines include generic residual stopping, full synchronous propagation, and ordered backward dynamic programming. Baseline critic work is shared. Final diagnostics and rejected-candidate gates are paid.
- Each dataset generates up to three sequential budget proposals, and multiple budgets share each task. Within-condition independent seeds, not aggregate rows, determine independent replication.
- Pointwise standard errors in plotted mean-return curves use independent seeds within the relevant condition. No family-wide confidence claim or p-value is inferred from those error bars.
- Scientific CSV hashes exclude elapsed timings and build metadata. Floating-point/library/platform differences may alter last digits; summary verification tolerates those differences but does not allow changed discrete scientific counts.
- Controls and parameter grids were developed during research, not preregistered. No hidden benchmark test set is claimed.
