Complete project artifact & requirements checklist (no code)
1) Top-level repo & minimal structure (recommended tree)

(You don’t need to follow this byte-for-byte, but every named file/folder below must exist and contain the described content.)

/project-root
├─ README.md
├─ CONTRIBUTING.md
├─ LICENSE
├─ DATA_ACCESS.md
├─ run_all.sh
├─ Makefile
├─ docker/Dockerfile
├─ env/requirements.txt
├─ env/environment.yml
├─ configs/config.yaml
├─ src/
│  ├─ pipelines/   (descriptions + orchestration scripts)
│  ├─ models/      (model metadata + saved artifacts)
│  ├─ utils/       (utilities descriptions)
│  └─ cli_spec.md
├─ data/
│  ├─ raw/
│  ├─ processed/
│  ├─ features/
│  └─ manifests/
├─ experiments/
│  ├─ mlflow/ or tracking.sqlite
│  ├─ optuna_studies/
│  ├─ runs/
│  └─ experiments_index.csv
├─ artifacts/
│  ├─ stage1_model_v1.meta
│  ├─ stage1_model_v1.bin
│  ├─ stage2_model_v1.meta
│  └─ compat_matrix_v1.npz
├─ notebooks/
├─ paper/
│  ├─ draft.tex / draft.md
│  ├─ figures/
│  ├─ tables/
│  └─ supplement/
├─ tests/
│  ├─ test_data_schema.md
│  └─ unit_tests_descriptions.md
├─ docs/
│  ├─ pipeline_diagram.svg
│  └─ model_card.md
└─ reviewer_quickstart.md


Each file/folder above is explained in the sections below.

2) Required documents & README files

README.md — top-level: project purpose, quickstart steps (high-level), expected runtime of run_all.sh (approx), main contact. (If you claim “reproducible in X hours”, state measured time on a common machine.)

CONTRIBUTING.md — coding conventions, branch workflow, how to add experiments.

LICENSE — project license (MIT/Apache) suited to code + instructions on dataset licensing.

DATA_ACCESS.md — exact instructions to obtain each dataset used (download links, queries, credentials required, sample SQL/API queries, expected filenames after download, and instructions to place them under data/raw/). For proprietary data, include the contact or purchase instructions.

reviewer_quickstart.md — 1–2 page guide for a reviewer to reproduce main table/figure in ≤2 hours (commands to run, which MLflow run ID to use, which small sample dataset).

model_card.md — who/what the model is for, intended uses, limitations, training data summary, performance metrics, bias statements, ethical considerations.

pipeline_diagram.svg — architecture diagram showing Stage-1 → Stage-2 → optimizer → simulator.

3) Data artifacts (what must be present and its schema)

A. data/raw/ — raw, unchanged files

Store only what you are allowed to publish. If any raw data is proprietary, keep extraction scripts and record checksums rather than raw files.

B. data/processed/ — cleaned, merged, dated

Include a manifest.csv listing: filename, checksum (SHA256), row count, created_date, description.

C. data/features/ — computed feature tables

Provide at least these CSVs (with these mandatory columns and data types):

players.csv

player_id (string/int), player_name, dob (YYYY-MM-DD), primary_position, secondary_positions (list), country, height_cm, weight_kg (optional).

matches.csv

match_id, season, date (YYYY-MM-DD), competition, home_team_id, away_team_id, venue, referee_id, weather (optional).

lineups.csv (one row per player per match)

match_id, team_id, player_id, is_starting (0/1), position_label (e.g., 'LB', 'CB'), minutes_played, sub_in_minute, sub_out_minute.

events_aggregated_per_player_windowed.csv (one row per player per match context)

match_id, player_id, window_last_n_matches (3/5/10), xG_sum, xA_sum, key_passes, progressive_passes, pressures, tackles, duels_won, distance_covered (if available), minutes_total.

fitness_and_injury.csv

player_id, date, injury_status (enum: healthy/injured/unknown), injury_type, days_since_last_match, yellow_card_count_rolling, sickness_flag.

co_play_minutes.csv

player_id_a, player_id_b, match_id, minutes_together, on_field_events_diff (optional).

xg_shot_level.csv (if computed)

match_id, player_id, shot_id, xG, xG_model_version, shot_result.

D. Data dictionary (data/data_dictionary.csv or MD)

For every column above: name, dtype, allowed values, units, description, source, missingness rate, transformation notes.

E. Synthetic sample dataset (data/sample_small/)

Small, synthetic but realistic sample dataset to let reviewers run the pipeline without licensed data.

4) Feature catalog & metadata

features_catalog.csv with: feature_name, description, source (raw field), window, transformation, type (numerical/categorical), expected_range, missing_treatment.

A human-readable feature_catalog.md that explains reasoning behind each feature (why keep/drop, expected signal).

5) Configuration & environment files

configs/config.yaml — central config file capturing: random seeds, dataset version tags, model hyperparams defaults, WIP toggles (simulate_only: true/false).

env/requirements.txt — pinned package versions (no caret ranges). Example: scikit-learn==1.2.2, lightgbm==3.3.5. (Pin exact versions that worked.)

env/environment.yml — optional Conda environment file for reproducibility.

.env.example — placeholders for any API keys (never commit real keys).

6) Docker & reproducibility

docker/Dockerfile — builds an environment that runs run_all.sh. Document base image and exact steps in README.md.

run_all.sh — single entrypoint (what it does: build image → run data preprocessing → train stage1 → train stage2 → run simulator → export figures and tables). Include --dry-run and --fast-demo modes.

reproducibility_checklist.md — exact steps a reproducer should follow and the expected artifacts/files produced (with checksums).

7) Models & model metadata (artifacts)

For each model (Stage-1 & Stage-2): provide

Saved model binary file: e.g., artifacts/stage1_lightgbm_v1.bin.

stage1_model_v1.meta containing: model type, training run ID, training data version, hyperparameters, date trained, random seed, MLflow run link (if available), feature order, feature normalizers/scalers (mean/std), preprocessing steps applied and their parameters.

feature_transformer_meta.json — dict describing any encoders (one-hot / ordinal), thresholds, imputation strategy + values used.

A short README in artifacts/ describing how to load and evaluate the model.

8) Experiment tracking & HPO artifacts

experiments/experiments_index.csv — for every experiment: exp_id, description, commit_sha, date, dataset_version, seed, main_metrics (PR-AUC, Brier), artifact_paths.

Optuna / HPO study files: optuna_studies/study_stage1_v1.db or export JSON of study.

MLflow exports (or an alternative): logs for runs, metric plots, parameter lists, model artifacts.

HPO summary tables and the "winning" configs exported as CSV & saved in paper/tables/.

9) Evaluation outputs (what must exist)

Main numeric table(s) (paper/tables/stage1_metrics.csv, stage2_lineup_table.csv) with:

row per model/variant

PR-AUC, ROC-AUC, Brier score, ECE, calibration slope/intercept, mean predicted probability, baseline heuristics.

Ablation table(s) with deltas and bootstrap CIs: e.g., ablation_results.csv columns: variant, metric, delta, 95%_CI_lower, 95%_CI_upper, p_value.

Statistical test outputs stored as paper/tables/stat_tests.json (Diebold-Mariano results, McNemar counts, DeLong p-values).

Calibration artifacts: reliability plot data (bins with mean_pred, empirical_freq, count) as CSV.

Lineup evaluation: lineup_sim_results.csv per lineup simulated (lineup_id, expected_points, win_prob, draw_prob, loss_prob, goal_diff_mean, goal_diff_95CI).

10) Monte-Carlo simulator & its outputs

simulator/config.json — parameters: number_of_sims (default 5000–10000), goal_rate_model_version, random_seed, home_advantage_factor.

simulator/results/ — for every simulated match: CSVs including sim_id, lineup_id, goals_team, goals_opponent, outcome, timestamp.

Summary files: simulator/summary_by_lineup.csv with expected values and distributions.

11) Compatibility / matrix factorization artifacts

artifacts/compat_matrix_v1.npz — serialized compatibility matrix (player × player), plus compat_meta.json describing MF algorithm used, latent dimensions, regularization, training data version.

docs/compatibility_method.md — explanation of how compatibility was computed and fallback strategies if co-play sparsity is high.

12) Optimizer / decision module artifacts

artifacts/optimizer_logs/ — MIP solver logs and solution certificates (time limits, optimality gap).

artifacts/optimizer_inputs/ — the exact input file used by the optimizer for each tested match (list of candidate players, their stage1 probs, compatibility submatrix, constraints).

docs/optimizer_spec.md — objective function, constraint list, hyperparameters (λ for synergy, γ for fatigue penalty), and solver choice + settings.

13) Writing & paper assets (what to include in paper/)

paper/draft.md or .tex — full manuscript draft (title, abstract, methods, results, discussion).

paper/figures/ — each figure as both *.svg (vector) and *.png (raster), with a short caption file figure_x_caption.txt.

Pipeline diagram, compatibility heatmap, example lineup graph, reliability diagram, ablation bar charts, simulator distributions.

paper/tables/ — CSVs for all tables used in the paper (not just final PDFs).

paper/supplement/ — extra methods, hyperparameter grids (hyperparams.csv), full ablation matrices, extended results by season/competition.

paper/references.bib — bibliography file.

paper/cover_letter.md — if you’re submitting to a conference or journal, include your planned cover letter.

14) Tests & continuous integration (descriptions only)

tests/ folder with descriptions of tests (no code necessary, but tests must be implemented):

Data schema tests: ensure all required columns exist and types are correct in data/processed/.

Sanity tests: non-negative minutes, xG in [0,1] per shot, sum of starting XI minutes ≤ 11*90.

Determinism tests: same seed + data → identical model artifact checksum.

Solver feasibility tests: given a constrained input, the optimizer returns a feasible solution.

Optional CI config (GitHub Actions) description: run unit tests on PRs; run light smoke test on run_all.sh --fast-demo.

15) Documentation for replicators & reviewers

REPRODUCE.md — step-by-step reproduction guide covering:

Setup (Docker/Conda)

How to obtain/prepare data

Commands to run Stage-1 training (with run ID), Stage-2, and simulator

How to regenerate figures/tables used in paper

Where to find outputs in artifacts/

LIMITATIONS.md — realistic limits, what the model cannot do, known failure modes.

16) Ethics, licensing & legal artifacts

data/licenses/ — copy of dataset license text (StatsBomb, FBref, commercial vendor EULAs) or a clear statement if not redistributable.

ETHICS.md — privacy considerations, sensitive attribute handling, bias/fairness audit summary.

DATA_PRIVACY_CHECKLIST.md — what was anonymized/removed, how personal data was handled.

17) Supplementary presentation/demo materials

demo/ — short demo plan:

demo/slides.pdf — 8–12 slides summarising the pipeline & key results.

demo/video.mp4 — optional short screen recording (2–4 minutes) showing outputs.

demo/example_output_lineup.csv — sample of final predicted XI + bench with fields (match_id, player_id, predicted_prob, composite_score, compatibility_score, minutes_projection).

poster/ — printable poster PDF for conferences (optional).

18) Administrative & submission artifacts

SUBMISSION/ — copy of paper, supplement, cover letter, suggested reviewers list, conflicts of interest, and any required forms (e.g., data usage forms).

zenodo_metadata.json — if you plan to archive code/data on Zenodo: metadata for DOI minting.

release_notes.md — for major releases / tags.

19) Logging, monitoring & performance records

perf/ — timing logs for training and inference (train_time_hours, memory_peak, GPU_utilization) for each run.

cost_estimates.md — estimated compute cost (CPU/GPU hours) for the full pre-train + fine-tune pipeline.

20) Team & process artifacts

ROLES.md — who is responsible for data, modeling, optimization, writing, experiments, paper submission.

MILESTONES.md — checkpoints & acceptance criteria (aligns with roadmap).

RISK_REGISTER.md — list of risks and mitigation steps (e.g., data gaps → fallback to synthesized compatibility).

21) Appendices you must include in paper / supplement

Full feature catalog (CSV) with exact formulas for engineered features.

Full hyperparameter grid and the final hyperparameters chosen.

Random seed list and explanation of nondeterministic steps.

Extended ablation results (per-season and per-competition).

Implementation details for matrix factorization (algorithm, k, α, λ), optimizer setup (solver, time limits).

22) Naming conventions & metadata hygiene (must have)

All artifacts must include: projectname_component_version_date_commitsha in their filenames when possible.

e.g., stage1_lightgbm_v1_2025-09-20_abc123.bin

manifests/ must list SHA256 checksums for critical files.

Every experiment in experiments_index.csv must map to a commit SHA and Docker image tag.

23) Short list of recommended non-code deliverables to impress reviewers

Reproducible 1-click demo: docker run that produces the main Figure 2.

Small synthetic dataset + script so reviewers can run everything without licensed data.

Model card + datasheet for each dataset used.

Zenodo DOI linking to code and non-restricted artifacts (cite in paper).

Reviewer quickstart that runs in ≤2 hours.

24) Final QA checklist before submission

(Place this in paper/submission_checklist.md and tick off before upload.)

Paper text complete + references.

Figures vectorized and match figure captions.

All tables reproduced from results_freeze artifacts.

Docker image built and tested by a clean checkout.

reproducibility_checklist.md verified by a second person.

Data license statements included in the Methods.

Model card + limitation statements included.

Zenodo/OSF link (if releasing code) ready.

Closing notes (practical priorities)

First deliverables to produce now: README, DATA_ACCESS.md, data manifest, sample synthetic dataset, and run_all.sh --fast-demo. That lets reviewers sanity-check your workflow immediately.

Make the synthetic dataset realistic: same schema + plausible numbers; it avoids legal/licensing pain and enables rapid review.

Be obsessive about metadata: every model must say which data version and commit produced it — otherwise reproducibility collapses.