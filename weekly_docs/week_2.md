# Week 2 — Aug 31 (Sun) → Sep 6 (Sat)

**Objectives:** deliver Stage‑1 model suite with calibrated probabilities & temporal CV.

**Tasks:**

1. **Pipelines (Aug 31–Sep 1):** unified sklearn pipelines for RF, XGBoost, LightGBM with imputation, scaling where needed, and categorical handling.
2. **Temporal CV & calibration (Sep 2):** rolling folds + probability calibration (Platt/Isotonic) → Brier/ECE.
3. **Feature importance & stability (Sep 3):** permutation importance, SHAP on holdout; check stability across folds.
4. **Model selection (Sep 4):** pick champion per team/season via validation objective (PR‑AUC + ECE).
5. **Stage‑1 output API (Sep 5):** standardized outputs: per‑player selection probability + uncertainty.
6. **G2 Stage‑1 Ready (Sep 6):** frozen config, logged in MLflow, reproducible run ID.

**Outputs:** Stage‑1 report, calibrated curves, importance plots, `stage1_model.pkl`, schema for downstream.
