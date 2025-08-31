"""
Model utility functions for the football squad selection pipeline.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Determine if sample is in bin m (between bin lower & upper)
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def plot_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, 
                          model_name: str = "Model", n_bins: int = 10,
                          save_path: Optional[str] = None) -> plt.Figure:
    """Plot calibration curve (reliability diagram)."""
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins
    )
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot calibration curve
    ax.plot(mean_predicted_value, fraction_of_positives, "s-", 
            label=f"{model_name} (ECE={calculate_ece(y_true, y_prob):.3f})")
    
    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(f"Calibration Plot - {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Calibration plot saved: {save_path}")
    
    return fig


def calculate_brier_decomposition(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Calculate Brier score decomposition (reliability, resolution, uncertainty)."""
    # Overall Brier score
    brier_score = brier_score_loss(y_true, y_prob)
    
    # Base rate (uncertainty)
    base_rate = y_true.mean()
    uncertainty = base_rate * (1 - base_rate)
    
    # Bin predictions for reliability calculation
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    reliability = 0
    resolution = 0
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        if i == n_bins - 1:  # Include upper boundary for last bin
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        
        if in_bin.sum() > 0:
            bin_prob = y_prob[in_bin].mean()
            bin_actual = y_true[in_bin].mean()
            bin_count = in_bin.sum()
            bin_weight = bin_count / len(y_true)
            
            reliability += bin_weight * (bin_prob - bin_actual) ** 2
            resolution += bin_weight * (bin_actual - base_rate) ** 2
    
    return {
        'brier_score': brier_score,
        'reliability': reliability,
        'resolution': resolution,
        'uncertainty': uncertainty,
        'brier_skill_score': (resolution - reliability) / uncertainty
    }


def bootstrap_metric(y_true: np.ndarray, y_pred: np.ndarray, 
                    metric_func, n_bootstrap: int = 1000, 
                    confidence_level: float = 0.95) -> Tuple[float, float, float]:
    """Calculate bootstrap confidence intervals for a metric."""
    np.random.seed(42)
    
    bootstrap_scores = []
    n_samples = len(y_true)
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        # Calculate metric
        score = metric_func(y_true_boot, y_pred_boot)
        bootstrap_scores.append(score)
    
    bootstrap_scores = np.array(bootstrap_scores)
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(bootstrap_scores, lower_percentile)
    ci_upper = np.percentile(bootstrap_scores, upper_percentile)
    mean_score = np.mean(bootstrap_scores)
    
    return mean_score, ci_lower, ci_upper


def feature_importance_analysis(models: dict, feature_names: list, 
                               top_k: int = 20) -> pd.DataFrame:
    """Analyze feature importance across multiple models."""
    importance_data = []
    
    for model_name, model_data in models.items():
        if 'feature_importance' in model_data:
            for feature, importance in model_data['feature_importance'].items():
                importance_data.append({
                    'model': model_name,
                    'feature': feature,
                    'importance': importance
                })
    
    importance_df = pd.DataFrame(importance_data)
    
    # Calculate average importance across models
    avg_importance = importance_df.groupby('feature')['importance'].agg([
        'mean', 'std', 'count'
    ]).reset_index()
    
    avg_importance.columns = ['feature', 'importance_mean', 'importance_std', 'n_models']
    avg_importance = avg_importance.sort_values('importance_mean', ascending=False)
    
    return avg_importance.head(top_k)


def plot_feature_importance(importance_df: pd.DataFrame, 
                           save_path: Optional[str] = None) -> plt.Figure:
    """Plot feature importance with error bars."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot horizontal bar chart
    y_pos = np.arange(len(importance_df))
    ax.barh(y_pos, importance_df['importance_mean'], 
            xerr=importance_df['importance_std'], capsize=3)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(importance_df['feature'])
    ax.set_xlabel('Feature Importance')
    ax.set_title('Average Feature Importance Across Models')
    ax.grid(True, alpha=0.3)
    
    # Invert y-axis to show most important features at top
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved: {save_path}")
    
    return fig


def model_stability_analysis(cv_results: list) -> dict:
    """Analyze model stability across CV folds."""
    metrics = ['pr_auc', 'brier_score', 'ece'] if 'ece' in cv_results[0] else ['pr_auc', 'brier_score']
    
    stability_stats = {}
    
    for metric in metrics:
        values = [fold[metric] for fold in cv_results if metric in fold]
        
        if values:
            stability_stats[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'cv': np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
            }
    
    return stability_stats
