from __future__ import annotations
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss

def expected_calibration_error(y_true, y_prob, n_bins=15):
    """ECE with equal-width bins in [0,1]."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if not np.any(mask):
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (np.sum(mask) / n) * abs(acc - conf)
    return float(ece)

def pr_auc(y_true, y_prob):
    return float(average_precision_score(y_true, y_prob))

def brier(y_true, y_prob):
    return float(brier_score_loss(y_true, y_prob))
