# Prior-work boundary

Primary-source review, September 2026. This is a targeted check, not an exhaustive novelty certification.

1. **Ghasemipour, Schuurmans, and Gu (2021), EMaQ: Expected-Max Q-Learning Operator for Simple Yet Effective Offline and Online RL.** ICML, PMLR 139:3682–3691. https://proceedings.mlr.press/v139/ghasemipour21a.html
   The expected-max operator, matched-budget critic, fixed-point interpretation, and monotonicity of the matched optimum are prior work. EMaQ also discusses empirical budget sensitivity. None of those facts is claimed new.
2. **Scherrer et al. (2015), Approximate Modified Policy Iteration and its Application to the Game of Tetris.** JMLR 16(49):1629–1676. https://jmlr.org/papers/v16/scherrer15a.html
   Evaluation/improvement tradeoffs and generic approximate-policy-iteration error propagation are established. Our standard contraction appendix is not a new theorem.
3. **Farahmand (2011), Action-Gap Phenomenon in Reinforcement Learning.** NeurIPS 24:172–180. https://www.sologen.net/papers/ActionGapInRL.pdf
   Decision gaps can make policy error smaller than norm-based critic error suggests. The proposed distinction is the exact random-candidate exposure envelope for rectangular uncertainty.
4. **Laroche, Trichelair, and Tachet des Combes (2019), Safe Policy Improvement with Baseline Bootstrapping.** ICML, PMLR 97:3652–3661. https://proceedings.mlr.press/v97/laroche19a.html
   Safe offline improvement is established. The present replacement gate is a conservative application of robust policy bounds, not an origin claim.
5. **Park et al. (2024), Is Value Learning Really the Main Bottleneck in Offline RL?** NeurIPS. https://arxiv.org/abs/2406.09329
   Policy extraction and generalization can be important independently of value fitting. This provides motivation, not novelty for our phenomenon by itself.
6. **Huang et al. (2025), Is Best-of-N the Best of Them? Coverage, Scaling, and Optimality in Inference-Time Alignment.** https://arxiv.org/abs/2503.21878
   Best-of-N can expose reward-model error; coverage and pessimistic extraction are already studied. Our exact reference-policy fork isolates continuation mismatch instead.
7. **Lin et al. (2026), Decoupling Policy Extraction for Offline Reinforcement Learning.** arXiv:2608.20909, August 2026. https://arxiv.org/abs/2608.20909
   Frozen proposals, independent value scoring, and nonmonotonic candidate-budget behavior are relevant existing work. Declining return versus K alone is not a sufficient contribution.

## Most plausible differentiator

The closed-form menu-wise rectangular envelope, its factor-four comparison with the one-fixed-vector robust loss, and its adaptive fixed-data stopping application form the proposed contribution. The constant-factor result is not a tightness claim. Its relationship to robust ranking, assortment choice, and interval regret deserves a broader independent audit before submission.
