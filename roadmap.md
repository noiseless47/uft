# A→Z Roadmap to Publish “Data‑Driven Football Squad Selection” by **October 9, 2025**

**Context (from abstract):** Two‑tier ML framework for football squad selection: Stage‑1 ensemble triage (RF/XGBoost/LightGBM) → Stage‑2 gradient boosting + matrix factorization for XI + 7 subs, with compatibility matrices, time‑series features, transfer learning (pre‑train on >10k matches, fine‑tune team‑specific), MAB feature selection, xG as core predictive feature, K‑fold + Monte Carlo validation, Dockerized (sklearn + LightGBM).

---

## 0) North‑Star Goals & Success Criteria

* **Submission target (by Oct 9, 2025):** Full paper + reproducibility package (code, Docker, fixed seed runs) + arXiv preprint + supplemental.
* **Primary scientific claims to demonstrate:**

  1. Two‑tier pipeline outperforms single‑stage baselines for squad selection accuracy & downstream match outcome proxies.
  2. Matrix‑factorization‑derived compatibility improves XI quality beyond independent player ranking.
  3. Transfer learning from 10k+ matches boosts generalization to unfamiliar squads/opponents.
  4. MAB‑driven feature selection yields stable, compact feature sets with no significant performance loss.
* **Hard acceptance criteria:**

  * ≥ **3–5%** relative improvement vs. strongest single‑stage baseline on main metric.
  * Robustness to **temporal shift** (rolling‑origin CV) + **unfamiliar team** configs.
  * **Full reproducibility** from a clean machine using Docker; single script to reproduce tables/figures.

**Main metrics (report together):** AUC/PR‑AUC for selection likelihood, Brier score & calibration (ECE), lineup‑level expected points/goal difference via Monte Carlo, constraint satisfaction rate, and ablation deltas for compatibility & MAB modules.

---

## 1) Timeline at a Glance (Asia/Kolkata)

* **Week 1 (Aug 24–Aug 30):** Scope lock, data audit, environment, repo/Docker skeleton, baseline sanity.
* **Week 2 (Aug 31–Sep 6):** Stage‑1 pipeline (RF/XGB/LGBM) + temporal CV + calibration.
* **Week 3 (Sep 7–Sep 13):** Stage‑2 (GBM + Matrix Factorization) + lineup optimization + compatibility.
* **Week 4 (Sep 14–Sep 20):** MAB feature selection + HPO + Monte Carlo simulator.
* **Week 5 (Sep 21–Sep 27):** Transfer learning pre‑train → fine‑tune + robustness & shift tests.
* **Week 6 (Sep 28–Oct 4):** Error analysis, fairness, final experiments freeze, full writing draft.
* **Week 7 (Oct 5–Oct 9):** Paper polish, reproducibility checks, arXiv & submission package.

**Milestone gates:**

* **G1 Data Freeze** (Aug 30)
* **G2 Stage‑1 Ready** (Sep 6)
* **G3 Stage‑2 Ready** (Sep 13)
* **G4 Sim + MAB + HPO Complete** (Sep 20)
* **G5 Transfer + Robustness Complete** (Sep 27)
* **G6 Results Freeze** (Oct 2)
* **G7 Camera‑ready & Repro Pack** (Oct 8)

---

## 2) Detailed Day‑by‑Day / Week‑by‑Week Plan

### Week 1 — **Aug 24 (Sun) → Aug 30 (Sat)** — Foundation, Data & Repro

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

---

### Week 2 — **Aug 31 (Sun) → Sep 6 (Sat)** — Stage‑1 Ensemble Triage

**Objectives:** deliver Stage‑1 model suite with calibrated probabilities & temporal CV.

**Tasks:**

1. **Pipelines (Aug 31–Sep 1):** unified sklearn pipelines for RF, XGBoost, LightGBM with imputation, scaling where needed, and categorical handling.
2. **Temporal CV & calibration (Sep 2):** rolling folds + probability calibration (Platt/Isotonic) → Brier/ECE.
3. **Feature importance & stability (Sep 3):** permutation importance, SHAP on holdout; check stability across folds.
4. **Model selection (Sep 4):** pick champion per team/season via validation objective (PR‑AUC + ECE).
5. **Stage‑1 output API (Sep 5):** standardized outputs: per‑player selection probability + uncertainty.
6. **G2 Stage‑1 Ready (Sep 6):** frozen config, logged in MLflow, reproducible run ID.

**Outputs:** Stage‑1 report, calibrated curves, importance plots, `stage1_model.pkl`, schema for downstream.

---

### Week 3 — **Sep 7 (Sun) → Sep 13 (Sat)** — Stage‑2 + Compatibility + XI Optimization

**Objectives:** build XI + 7 subs decision layer with compatibility matrices & MF.

**Tasks:**

1. **Compatibility matrix (Sep 7–8):**

   * Construct **co‑play matrices** (minutes together), **synergy scores** (on‑pitch goal difference/xG delta while co‑playing), role adjacency weights.
   * **Matrix Factorization:** implicit‑feedback ALS or BPR to derive player latent vectors; cosine similarity = compatibility prior.
2. **Stage‑2 model (Sep 9):** gradient boosting using inputs: Stage‑1 posteriors, compatibility features per candidate XI, opponent context.
3. **XI & bench optimizer (Sep 10–11):**

   * Mixed‑integer programming (PuLP/OR‑Tools):

     * Constraints: formation role counts, fitness thresholds, min GK/DEF/MID/FWD on bench, minutes caps, injury/ban, home/away.
     * Objective: maximize **composite lineup score** predicted by Stage‑2 + synergy regularizer.
4. **Validation (Sep 12):**

   * Back‑test against historical selections & out‑of‑sample matches; measure lineup‑level metrics.
5. **G3 Stage‑2 Ready (Sep 13):** API returns XI+7 with rationale (top features/compatibility edges).

**Outputs:** Compatibility notebook, `compat_matrix.npz`, `stage2_model.pkl`, optimizer module, validation report.

---

### Week 4 — **Sep 14 (Sun) → Sep 20 (Sat)** — MAB Feature Selection + Monte Carlo Simulator

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

---

### Week 5 — **Sep 21 (Sun) → Sep 27 (Sat)** — Transfer Learning & Robustness

**Objectives:** pre‑train on 10k+ matches; fine‑tune; stress‑test generalization.

**Tasks:**

1. **Pre‑training corpus prep (Sep 21):** normalization to common schema; domain labels.
2. **Pre‑train (Sep 22):**

   * Train Stage‑1/2 initializers on large multi‑league corpus; save weights/trees/feature stats.
3. **Fine‑tune (Sep 23):** team‑specific re‑weighting; check catastrophic forgetting.
4. **Robustness (Sep 24–25):** unfamiliar team config tests, opponent style shift, injury shocks; sensitivity to window sizes.
5. **Calibration refresh (Sep 26):** re‑calibrate post fine‑tune; check Brier/ECE.
6. **G5 Transfer + Robustness Complete (Sep 27).**

**Outputs:** Transfer report, robustness dashboard, final model artifacts v1.0.

---

### Week 6 — **Sep 28 (Sun) → Oct 4 (Sat)** — Error Analysis, Ethics, Results Freeze, Writing

**Objectives:** finalize results, perform deep analysis, write full draft.

**Tasks:**

1. **Error/uncertainty analysis (Sep 28–29):**

   * Where Stage‑2 fails (formation mismatches, rare combos); prediction intervals via bootstrap; calibration by segment (home/away, top/bottom teams).
2. **Fairness & bias checks (Sep 30):**

   * Audit by age, position, tenure; ensure no protected‑attribute leakage; document exclusions.
3. **Composite score finalization (Oct 1):** weighting between predicted outcome, compatibility, consistency; justify with Pareto front.
4. **Results Freeze (Oct 2):** lock random seeds, export CSVs for all tables/figures; tag MLflow run IDs.
5. **Full writing draft (Oct 3–4):** all sections complete (see §4), figures & tables integrated.

**Outputs:** Error analysis appendix, Bias/Fairness statement, `results_freeze_2025-10-02/`, Draft v1.

---

### Week 7 — **Oct 5 (Sun) → Oct 9 (Thu)** — Polish, Repro Pack, Submission

**Objectives:** camera‑ready polish, reproducibility, packaging, submission by Oct 9.

**Tasks:**

1. **Internal peer review (Oct 5):** two reviewers; checklist in §7; log actions.
2. **Reproducibility pack (Oct 6):** Docker image push; `run_all.sh` reproduces paper in < X hours; artifact README.
3. **Polish writing (Oct 7):** tighten abstract, contributions, limitations; language edit.
4. **Final checks (Oct 8):** figure numbering, references, license, ethics & competing interests.
5. **Submission & arXiv (Oct 9):** upload manuscript + supplement + code archive DOI (e.g., Zenodo/OSF) + release tag.

**Outputs:** Camera‑ready PDF, Supplement, arXiv, Code release, Submission confirmation.

---

## 3) Technical Design Details (A→Z)

### 3.1 Data & Splits

* **Entities:** matches, events, players, squads, injuries/fitness, positions, opponent/team features, formations.
* **xG:** either provided or computed; rollups per player (last 3/5/10), team xG for simulator.
* **Temporal splitting:** rolling‑origin CV with gaps to avoid leakage; nested HPO; leave‑one‑season‑out as robustness.
* **Labels:**

  * Stage‑1: player selected to 20–25 squad (binary, per match context).
  * Stage‑2: lineup viability/strength score for XI (+7), learned as outcome proxy using historical choices & match results.

### 3.2 Features (examples)

* **Performance:** xG/xA, key passes, pressures, duels, interceptions, progressive actions, defensive actions, set‑piece contributions.
* **Form & fatigue:** minutes last N matches, days since last match, travel, congestion index.
* **Opponent context:** opponent style cluster, venue, weather proxy, referee leniency proxy (cards).
* **Compatibility features:** co‑play minutes, plus MF latent similarity, role adjacency constraints, side balance (left/right).
* **Stability:** exponentially‑weighted moving averages to capture recency.

### 3.3 Models

* **Stage‑1:** RF / XGBoost / LightGBM (calibrated). Objective tuned for PR‑AUC & Brier.
* **Stage‑2:** Gradient boosting using aggregate candidate‑XI features + compatibility features.
* **Matrix Factorization:** implicit ALS/BPR for player‑player synergy; dimensions 16–64; regularization via λ search.
* **MAB Feature Selection:** arms = curated feature subsets (e.g., with/without fatigue, with/without opponent‑cluster); Thompson Sampling primary.

### 3.4 Optimization for XI + 7

* **Decision variables:** binary for player‑in‑XI, player‑on‑bench.
* **Constraints:** formation role counts; min GK; bench coverage; injury/ban; minutes/fatigue caps; max non‑homegrown (if relevant); chemistry penalties.
* **Objective:** maximize Stage‑2 predicted score + λ·compatibility − γ·risk (injury/fatigue/discipline).

### 3.5 Monte Carlo Simulator

* **Team goal rates:** base on team & XI‑adjusted xG, opponent defense, venue.
* **Sim:** 5k–10k draws → expected points, distribution of goal difference; sensitivity by formation.

### 3.6 Validation & Statistics

* **Report:** PR‑AUC, ROC‑AUC, Brier, ECE; lineup‑level expected points; reliability diagrams; Diebold‑Mariano for forecast comparisons; McNemar/DeLong where applicable.
* **Ablations:** −compatibility, −MAB, −transfer; **frozen seeds**.
* **Uncertainty:** bootstrap CIs; calibration across segments.

---

## 4) Writing Plan (Section‑by‑Section)

1. **Abstract:** 150–200 words, quantitative claims.
2. **Introduction:** problem, contributions (bulleted), summary of results.
3. **Related Work:** squad selection, xG & team strength, compatibility/synergy, transfer learning in sports, feature selection (MAB).
4. **Method:** Stage‑1, Stage‑2, MF compatibility, optimizer, simulator, MAB; diagrams.
5. **Data:** sources, preprocessing, leakage controls, ethics & licenses.
6. **Experiments:** splits, baselines, metrics, ablations, robustness, calibration.
7. **Results:** main table, calibration, simulator outcomes, case studies.
8. **Discussion:** interpretation, limitations, failure modes.
9. **Reproducibility:** code, Docker, seeds, data availability.
10. **Conclusion:** takeaways + future work.
11. **Appendix:** extra tables, hyperparams, feature catalog, risk/bias audit, algorithmic details.

**Figure list:** pipeline diagram; compatibility graph; reliability curves; ablation bar chart; Pareto front of composite score weights; simulator distributions.

**Table list:** dataset summary; Stage‑1 metrics; Stage‑2 lineup metrics; ablation deltas; robustness tests; hyperparameter grid.

---

## 5) Engineering & Reproducibility

* **Repo layout:**

  * `data/` (scripts only, no raw proprietary data), `src/` (stage1, stage2, mf, optimizer, sim, mab), `configs/`, `experiments/`, `notebooks/`, `docker/`, `paper/`.
* **Determinism:** global seed; document non‑deterministic ops; log hashes of datasets.
* **One‑click run:** `run_all.sh` builds Docker, runs experiments, exports figures/tables.
* **Tracking:** MLflow tags per figure/table; link to commit SHA.

---

## 6) Risk Register & Mitigations

* **Data sparsity for co‑play:** back‑off to role/cluster averages; MF regularization; Bayesian shrinkage.
* **Temporal leakage:** strict date filters; feature windows ending **before** selection time.
* **Overfitting to a team/coach:** transfer pre‑train + time‑aware validation.
* **Compute/time overrun:** limit HPO trials; early stopping; prioritize ablations.
* **Reproducibility drift:** Docker only; pin library versions; lock random seeds; store artifacts.

---

## 7) Checklists

**Experiment Freeze (Oct 2):**

* [ ] All tables regenerated from `results_freeze_2025-10-02/`
* [ ] Seeds, configs, data versions logged
* [ ] Ablations complete with CIs
* [ ] Bias/fairness audit done

**Submission Pack (Oct 8–9):**

* [ ] PDF compiles from `paper/`
* [ ] Figures vectorized, <10MB total
* [ ] Supplement (methods, extra tables)
* [ ] Code + Docker + README + license
* [ ] Data access notes (how to obtain)
* [ ] ArXiv source + bbl files

---

## 8) Resource Plan

* **Compute:** 1× mid‑range GPU helpful (for XGB/LGBM speed), else CPU ok; budget \~10–20 GPU‑hours or \~2–3 CPU‑days.
* **People:** 1 lead author, 1 engineer (shared), 1 reviewer.
* **Storage:** <20 GB artifacts; clean up raw caches.

---

## 9) Deliverables by Date (high‑level)

* **Aug 30:** Data Freeze, repo/Docker ready.
* **Sep 6:** Stage‑1 calibrated.
* **Sep 13:** Stage‑2 + optimizer ready.
* **Sep 20:** MAB + simulator + ablations.
* **Sep 27:** Transfer & robustness complete.
* **Oct 2:** Results Freeze.
* **Oct 4:** Full draft.
* **Oct 9:** Submission & arXiv.

---

## 10) Appendix: Implementation Notes

* Prefer **Isotonic** calibration when data volume allows; else Platt.
* Use **grouped k‑fold by match** when needed to prevent leakage across same match context.
* For MF, start with **ALS (implicit)**, k=32, α=40, λ tuned; back‑off to cosine over role embeddings if data thin.
* Optimizer: warm‑start with greedy heuristic to cut MIP solve time; set 30–60s time limit per solve.
* Use **Optuna** + MLflow callback to log trials; prune aggressively.
* Simulator validation: compare simulated win probs vs. betting‑market proxies (if available) on holdout.

---

**You now have a concrete, date‑specific execution plan from A→Z that lands a polished, reproducible paper by October 9, 2025.**
