# Week 1 — Aug 24 (Sun) → Aug 30 (Sat)

**Objectives:** lock scope, audit data, finalize metrics, remove leakage, set reproducibility basics.

**Tasks:**

1. **Project scope & risks workshop (Aug 24):**

   * Define exact prediction targets: (a) player selection probability into preliminary 20–25, (b) XI + 7 subs under formation constraints.
   * Finalize evaluation protocol: **rolling‑origin temporal CV** (train on t₀…t, validate on t+Δ) + nested CV for HPO.
   * Choose main formation sets (e.g., 4‑3‑3, 4‑2‑3‑1, 3‑5‑2) and constraint templates.
   * Risk register (see §6), define mitigations.

2. **Data audit (Aug 25–26):**

   * Schema: matches, events, players, squads, injuries/fitness, minutes, positions, opponent descriptors, xG (per shot & team‑level rolling xG), cards/suspensions.
   * Integrity: ID consistency, missingness, outliers (e.g., impossible minutes), unit normalization.
   * **Leakage checks:** no post‑selection info in features; watch for post‑match stats bleeding into selection.
   * **Deliverable:** Data Profiling Report + **Data Freeze list** (tables, columns, and extraction queries).

3. **Feature catalog (+ owner) (Aug 26):**

   * Time‑series windows: last 3/5/10 matches rolling means for xG, xA, presses, duels, distance, fitness proxy.
   * Opponent‑specific: performance vs opponent archetypes (clustered by style) and venue.
   * Availability: injury status, travel, fatigue (fixture congestion), yellow‑card risk.
   * Role encoding: position one‑hots + learned role embeddings.

4. **Repo & Docker setup (Aug 27):**

   * `python==3.11`, `scikit‑learn`, `lightgbm`, `xgboost`, `pandas`, `numpy`, `scipy`, `optuna`, `pulp` or `ortools`, `matplotlib`, `mlflow`.
   * **Makefile** targets: `make data`, `make train_stage1`, `make train_stage2`, `make sim`, `make figures`, `make paper`.
   * **MLflow** experiment tracking + fixed random seeds; set GPU if available for XGBoost/LGBM.

5. **Baselines & sanity (Aug 28–29):**

   * Simple heuristics: minutes played, coach’s last match XI, Elo‑like player score.
   * Quick RF baseline on small slice to validate labels & metrics.

6. **G1 Data Freeze (Aug 30):**

   * Commit `DATA_VERSION` tag; document extraction scripts; lock feature list v1.

**Outputs:** Data Profiling Report, Feature Catalog v1, Dockerfile, `ENV.md`, Baseline sanity plots.
