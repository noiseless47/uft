#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import yaml
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.config import FIG_DIR, ARTIFACTS_DIR, MLFLOW_TRACKING_URI
from src.utils.logging import get_logger
from src.utils.seed import set_global_seed
from src.data.loaders import load_config_data_paths, load_tables
from src.features.build_features import build_week1_features
from src.models.baseline_stage1 import run_temporal_cv_calibrated

logger = get_logger("baseline")

def main(exp_cfg_path: str):
    with open(exp_cfg_path, "r") as f:
        exp_cfg = yaml.safe_load(f)

    set_global_seed(exp_cfg.get("random_seed", 42))
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(exp_cfg["experiment_name"])

    data_cfg_path = exp_cfg["data_config"]
    paths = load_config_data_paths(data_cfg_path)
    with open(data_cfg_path, "r") as f:
        data_cfg = yaml.safe_load(f)

    tables = load_tables(paths)

    # Build features
    windows = data_cfg.get("feature_windows", [3,5,10])
    feat_df = build_week1_features(
        matches=tables["matches"],
        events=tables["events"],
        squads=tables["squads"],
        players=tables["players"],
        windows=windows,
    )

    # Sanity: drop rows with missing label
    label_col = exp_cfg["label"]["column"]
    feat_df = feat_df.dropna(subset=[label_col])

    # Sort by time
    feat_df = feat_df.sort_values("match_date").reset_index(drop=True)

    # Fit baseline RF + calibration with temporal CV
    cv = exp_cfg["cv"]
    calib = exp_cfg["calibration"]["method"]
    figs_dir = Path(exp_cfg["figures_dir"]) if "figures_dir" in exp_cfg else FIG_DIR
    figs_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = Path(exp_cfg["artifacts_dir"]) if "artifacts_dir" in exp_cfg else ARTIFACTS_DIR
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    rf_params = exp_cfg["model"]["params"]

    with mlflow.start_run(run_name=exp_cfg.get("run_name", "run")):
        mlflow.log_params({
            "model": exp_cfg["model"]["type"],
            **{f"rf__{k}": v for k, v in rf_params.items()},
            "cv_n_folds": cv["n_folds"],
            "cv_gap_days": cv["gap_days"],
            "feature_windows": ",".join(map(str, windows)),
            "calibration": calib
        })

        # Attach model params to pipeline inside run
        # (We pass params by overriding in the model file)
        # Quick trick: monkey-patch for now by injecting into kwargs
        # We'll set in-place within the training procedure by recreating classifier each fold.

        res = run_temporal_cv_calibrated(
            df=feat_df.copy(),
            label_col=label_col,
            n_folds=cv["n_folds"],
            gap_days=cv.get("gap_days", 0),
            calibration=calib,
            artifacts_dir=artifacts_dir,
            figures_dir=figs_dir,
            random_seed=exp_cfg.get("random_seed", 42),
        )

        # Aggregate metrics
        import numpy as np
        pr_mean = float(np.mean([m["pr_auc"] for m in res.fold_metrics]))
        br_mean = float(np.mean([m["brier"] for m in res.fold_metrics]))
        ece_mean = float(np.mean([m["ece"] for m in res.fold_metrics]))

        mlflow.log_metrics({"pr_auc_mean": pr_mean, "brier_mean": br_mean, "ece_mean": ece_mean})

        # Save per-fold metrics CSV
        import pandas as pd
        pd.DataFrame(res.fold_metrics).to_csv(artifacts_dir / "week1_baseline_fold_metrics.csv", index=False)
        mlflow.log_artifact(str(artifacts_dir / "week1_baseline_fold_metrics.csv"))

        # Log figures
        for fig in (figs_dir.glob("reliability_*.png")):
            mlflow.log_artifact(str(fig))

    logger.info("Week-1 baseline complete.")
    logger.info(f"Figures saved to: {figs_dir}")
    logger.info(f"Artifacts saved to: {artifacts_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    args = ap.parse_args()
    main(args.exp)
