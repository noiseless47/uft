from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from ..eval.metrics import pr_auc, brier, expected_calibration_error
from ..viz.plots import reliability_plot

@dataclass
class TemporalCVResult:
    fold_metrics: List[Dict[str, float]]
    y_true_all: np.ndarray
    y_prob_all: np.ndarray

def _split_temporal(df: pd.DataFrame, n_folds: int, gap_days: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    # Assumes df sorted by match_date
    dates = df["match_date"].sort_values().unique()
    folds = []
    # Simple rolling split by unique dates
    chunk = len(dates) // (n_folds + 1)
    for i in range(1, n_folds + 1):
        val_start = i * chunk
        val_end = (i + 1) * chunk if i < n_folds else len(dates)
        val_dates = dates[val_start:val_end]
        train_dates = dates[:max(0, val_start)]
        if len(train_dates) == 0 or len(val_dates) == 0:
            continue
        # Apply temporal gap by trimming last 'gap_days' from training
        train_max_date = val_dates.min()
        train_mask = df["match_date"] < (train_max_date - np.timedelta64(gap_days, "D"))
        val_mask = df["match_date"].isin(val_dates)
        folds.append((train_mask.values, val_mask.values))
    return folds

def _build_pipeline(df: pd.DataFrame, model_params: Dict[str, Any]) -> Tuple[Pipeline, List[str], List[str]]:
    cat_cols = df.select_dtypes(include=["category", "object"]).columns.tolist()
    num_cols = [c for c in df.columns if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    clf = RandomForestClassifier(**model_params)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    return pipe, num_cols, cat_cols

def run_temporal_cv_calibrated(
    df: pd.DataFrame,
    label_col: str,
    n_folds: int,
    gap_days: int,
    calibration: str = "auto",
    artifacts_dir=None,
    figures_dir=None,
    random_seed: int = 42,
) -> TemporalCVResult:
    np.random.seed(random_seed)

    y = df[label_col].astype(int).values
    X = df.drop(columns=[label_col, "match_id", "team_id", "player_id"])

    pipe, _, _ = _build_pipeline(X, model_params={})
    # we will clone with params per run; RF params are set by caller before fit

    folds = _split_temporal(df, n_folds=n_folds, gap_days=gap_days)
    all_probs, all_true = [], []
    fold_metrics = []

    for fi, (tr_mask, va_mask) in enumerate(folds, start=1):
        X_train, y_train = X[tr_mask], y[tr_mask]
        X_val, y_val = X[va_mask], y[va_mask]

        # Rebuild with params each fold to avoid state carryover
        pipe, _, _ = _build_pipeline(X, model_params={})
        base_clf = pipe

        base_clf.fit(X_train, y_train)

        # Choose calibration method
        method = "isotonic"
        pos_rate = y_train.mean()
        # If positives too few, use sigmoid
        if calibration == "sigmoid" or (calibration == "auto" and (len(y_train) * pos_rate < 200)):
            method = "sigmoid"

        calibrated = CalibratedClassifierCV(base_clf, method=method, cv="prefit")
        calibrated.fit(X_val, y_val)
        probs = calibrated.predict_proba(X_val)[:, 1]

        pr = pr_auc(y_val, probs)
        br = brier(y_val, probs)
        ece = expected_calibration_error(y_val, probs, n_bins=15)

        fold_metrics.append({"fold": fi, "pr_auc": pr, "brier": br, "ece": ece})
        all_probs.append(probs)
        all_true.append(y_val)

        # Reliability plot per fold
        if figures_dir is not None:
            reliability_plot(y_val, probs,
                             outpath=(figures_dir / f"reliability_fold{fi}.png"),
                             n_bins=15,
                             title=f"Reliability Fold {fi} ({method})")

    y_prob_all = np.concatenate(all_probs)
    y_true_all = np.concatenate(all_true)

    # Aggregate reliability
    if figures_dir is not None:
        reliability_plot(y_true_all, y_prob_all,
                         outpath=(figures_dir / "reliability_all.png"),
                         n_bins=15,
                         title="Reliability (All Folds)")

    return TemporalCVResult(fold_metrics=fold_metrics, y_true_all=y_true_all, y_prob_all=y_prob_all)
