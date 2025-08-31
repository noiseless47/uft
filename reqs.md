⚽ Project Requirements Analysis for Data-Driven Football Squad Selection
1. Data Requirements

Your models depend heavily on both breadth (10k+ pro matches for pre-training) and depth (team-specific detail). You’ll need three layers of data:

A) Core Match/Event Data

Sources (public/commercial):

Open-source: StatsBomb Open Data (~6k matches), FiveThirtyEight SPI, FBref (via statsbombpy / soccerdata), football-data.co.uk.

Commercial (if budget): Opta, Wyscout, InStat (gives fitness, player load, co-play minutes).

Fields required:

Match ID, competition, season, team IDs, opponent IDs, formations, venue, date.

Event streams: shots, passes, duels, tackles, presses, dribbles, fouls, cards.

Advanced metrics: xG, xA (or derivable).

B) Player Metadata & Fitness

Unique player IDs across seasons.

Age, position(s), minutes played.

Injury reports, suspensions, recovery dates (public feeds often partial).

Travel/fatigue proxies (fixture congestion, days since last match).

C) Contextual/Compatibility Data

Lineups (XI + bench) per match.

Substitutions and minutes together (to derive co-play matrix).

Team style descriptors (possession vs. counter, pressing intensity, etc. — cluster from event data if not provided).

Referee assignment and card tendency (optional).

Data Volume Targets:

≥10,000 matches across leagues (2015–2025) for pre-training.

≥2–3 seasons for your focus team(s) for fine-tuning.

Enough co-play instances per pair (or fallback to role-based priors).

2. Feature Requirements

Time-series windows: last 3/5/10 matches rolling means for xG, passes, duels, distance.

Opponent-specific: past vs opponent archetypes, home/away split.

Fitness: minutes, injuries, suspensions, yellow-card accumulation.

Compatibility: co-play synergy, MF latent embeddings, adjacency (GK–CB, CB–FB, MID–FWD).

Formation encoding: categorical or learned embeddings.

Stability: exponentially weighted moving averages.

3. Modeling Requirements
Stage-1: Player → Squad (20–25)

Models: Random Forest, XGBoost, LightGBM.

Outputs: calibrated probability of selection.

Calibration: Isotonic or Platt scaling.

Stage-2: Squad → XI + 7

Inputs: Stage-1 posteriors, compatibility features, team/opponent context.

Model: Gradient boosting (LightGBM) + compatibility MF vectors.

Decision: Mixed-Integer Programming (MIP) for formation-constrained XI/bench.

Add-Ons

Transfer Learning: Pre-train on multi-league, fine-tune team-specific.

MAB Feature Selection: Thompson Sampling on subsets.

Simulator: Poisson/Skellam Monte Carlo (5k–10k runs).

4. Validation Requirements

Splits: rolling-origin temporal CV; leave-one-season-out robustness.

Metrics: PR-AUC, Brier score, Expected Calibration Error (ECE), lineup-level expected points.

Ablations: w/o compatibility, w/o MAB, w/o transfer.

Uncertainty: bootstrapped confidence intervals.

5. Compute & Infrastructure

Hardware:

CPU baseline sufficient (for scikit-learn + LGBM).

GPU optional (faster XGBoost/LGBM on large pre-training set).

Storage: ~20 GB for raw + features + artifacts.

Software stack:

Python 3.11, scikit-learn, lightgbm, xgboost, pandas, numpy, optuna, mlflow, matplotlib, pulp/ortools.

Docker (cross-platform reproducibility).

Git + DVC or similar for data versioning.

6. Experiment Tracking & Reproducibility

Tracking: MLflow runs tagged with commit SHAs.

Reproducibility:

run_all.sh → builds Docker, executes experiments, exports figures/tables.

Seed locking + pinned library versions.

Artifacts: all models (stage1_model.pkl, stage2_model.pkl), compatibility matrices, simulator outputs, ablation tables.

7. Writing & Publication Requirements

Paper sections (drafted in Week 6): Abstract, Intro, Related Work, Method, Data, Experiments, Results, Discussion, Reproducibility, Conclusion.

Figures: pipeline diagram, reliability curves, compatibility graph, ablation bars, simulator distributions.

Tables: dataset summary, Stage-1 metrics, Stage-2 lineup results, ablation deltas, robustness tests.

Target venues:

Sports Analytics: MIT Sloan Sports Analytics Conf.

ML/AI Workshops: KDD Sports Analytics, NeurIPS ML4Sports.

General Preprint: arXiv (for early dissemination).

8. Human/Process Requirements

Team roles:

Lead researcher (end-to-end).

Engineer (pipeline, Docker, optimization).

Reviewer (writing polish, sanity checks).

Checkpoints: G1–G7 milestones from roadmap.

Ethics: document dataset licensing; fairness analysis (age, position, tenure bias).

✅ Summary:
You’ll need 10k+ match dataset (event + squad + fitness), curated feature engineering (performance, fatigue, compatibility), two-tier modeling (Stage-1 ensemble + Stage-2 GBM+MF), transfer + MAB modules, Monte Carlo simulation, full temporal CV validation, Dockerized reproducibility pipeline, and a paper package (tables/figures + code release) by Oct 9, 2025.