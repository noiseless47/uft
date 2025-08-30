# Week 4 — Sep 14 (Sun) → Sep 20 (Sat)

**Objectives:** integrate Multi‑Armed Bandit for dynamic feature selection; deliver simulator.

**Tasks:**

1. **MAB module (Sep 14–15):**

   * Define arms = feature subsets/transform recipes; reward = validation objective.
   * Implement **Thompson Sampling** (Bernoulli/Logistic reward transform) + **UCB1** baseline; rolling update per fold.
2. **HPO with Optuna (Sep 16):** nested with temporal CV; prune via median pruner; budgeted trials (e.g., 100–200).
3. **Monte Carlo (Sep 17–18):**

   * Poisson/Skellam‑based goals model parameterized by team & XI‑adjusted xG; simulate **N=5k–10k** match outcomes.
   * Output: expected points, win/draw/loss probs, goal diff distribution.
4. **Ablations (Sep 19):** remove compatibility, remove MAB, remove transfer pre‑train; quantify deltas.
5. **G4 Sim + MAB + HPO Complete (Sep 20).**

**Outputs:** MAB report, HPO study artifact, simulator module, ablation tables & plots.
