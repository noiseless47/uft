#!/usr/bin/env python3
"""
Stage-1 ensemble training for player selection probability.
Implements Random Forest, XGBoost, and LightGBM with temporal CV and calibration.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import yaml
import logging
import joblib
import mlflow
import mlflow.sklearn
from datetime import datetime

# ML libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_curve, roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import optuna

sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.logging_config import setup_logging
from utils.data_utils import create_temporal_splits
from utils.model_utils import calculate_ece, plot_calibration_curve

logger = logging.getLogger(__name__)


class Stage1Trainer:
    """Stage-1 ensemble trainer with temporal CV and calibration."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.stage1_config = self.config['stage1']
        self.random_seed = self.config['random_seeds']['model_training']
        self.mlflow_config = self.config['mlflow']
        
        # Setup MLflow
        mlflow.set_tracking_uri(self.mlflow_config['tracking_uri'])
        mlflow.set_experiment(self.mlflow_config['experiment_name'])
        
    def load_features_and_targets(self, data_dir: str = "data/features") -> Tuple[pd.DataFrame, pd.Series]:
        """Load features and target variables."""
        logger.info(f"Loading features from {data_dir}")
        
        # Load main features
        features_path = os.path.join(data_dir, "stage1_features.csv")
        features_df = pd.read_csv(features_path)
        
        # Load targets
        targets_path = os.path.join(data_dir, "squad_selection_features.csv")
        targets_df = pd.read_csv(targets_path)
        
        # Merge features with targets
        data = features_df.merge(targets_df[['match_id', 'player_id', 'selected_to_squad']], 
                                on=['match_id', 'player_id'], how='inner')
        
        # Separate features and targets
        feature_cols = [col for col in data.columns 
                       if col not in ['match_id', 'player_id', 'selected_to_squad']]
        
        X = data[feature_cols]
        y = data['selected_to_squad']
        
        logger.info(f"Loaded features: {X.shape}, targets: {y.shape}")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y, data[['match_id', 'player_id']]
    
    def create_temporal_cv_splits(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create temporal cross-validation splits."""
        logger.info("Creating temporal CV splits...")
        
        # Load match dates for temporal ordering
        matches_df = pd.read_csv("data/processed/matches.csv")
        data_with_dates = data.merge(matches_df[['match_id', 'date']], on='match_id')
        
        splits = create_temporal_splits(
            data_with_dates, 
            date_column='date',
            n_splits=self.config['data']['temporal_cv']['n_splits'],
            gap_days=self.config['data']['temporal_cv']['gap_days'],
            test_size_days=self.config['data']['temporal_cv']['test_size_days']
        )
        
        logger.info(f"Created {len(splits)} temporal CV splits")
        return splits
    
    def train_random_forest(self, X: pd.DataFrame, y: pd.Series, 
                           cv_splits: List[Dict]) -> Dict[str, Any]:
        """Train Random Forest with temporal CV."""
        logger.info("Training Random Forest...")
        
        rf_config = self.stage1_config['models']['random_forest']
        
        model = RandomForestClassifier(
            random_state=self.random_seed,
            **rf_config
        )
        
        # Temporal cross-validation
        cv_scores = []
        feature_importances = []
        
        for fold_idx, split in enumerate(cv_splits):
            X_train = X.iloc[split['train_indices']]
            X_test = X.iloc[split['test_indices']]
            y_train = y.iloc[split['train_indices']]
            y_test = y.iloc[split['test_indices']]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            pr_auc = self._calculate_pr_auc(y_test, y_pred_proba)
            brier = brier_score_loss(y_test, y_pred_proba)
            
            cv_scores.append({
                'fold': fold_idx,
                'pr_auc': pr_auc,
                'brier_score': brier,
                'n_train': len(X_train),
                'n_test': len(X_test)
            })
            
            feature_importances.append(model.feature_importances_)
            
            logger.info(f"RF Fold {fold_idx}: PR-AUC={pr_auc:.3f}, Brier={brier:.3f}")
        
        # Train final model on all data
        model.fit(X, y)
        
        # Calculate average feature importance
        avg_feature_importance = np.mean(feature_importances, axis=0)
        
        results = {
            'model': model,
            'cv_scores': cv_scores,
            'feature_importance': dict(zip(X.columns, avg_feature_importance)),
            'config': rf_config
        }
        
        logger.info(f"RF training complete. Avg PR-AUC: {np.mean([s['pr_auc'] for s in cv_scores]):.3f}")
        return results
    
    def train_xgboost(self, X: pd.DataFrame, y: pd.Series, 
                     cv_splits: List[Dict]) -> Dict[str, Any]:
        """Train XGBoost with temporal CV."""
        logger.info("Training XGBoost...")
        
        xgb_config = self.stage1_config['models']['xgboost']
        
        model = xgb.XGBClassifier(
            random_state=self.random_seed,
            eval_metric='logloss',
            **xgb_config
        )
        
        # Temporal cross-validation
        cv_scores = []
        feature_importances = []
        
        for fold_idx, split in enumerate(cv_splits):
            X_train = X.iloc[split['train_indices']]
            X_test = X.iloc[split['test_indices']]
            y_train = y.iloc[split['train_indices']]
            y_test = y.iloc[split['test_indices']]
            
            # Train with early stopping
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                early_stopping_rounds=50,
                verbose=False
            )
            
            # Evaluate
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            pr_auc = self._calculate_pr_auc(y_test, y_pred_proba)
            brier = brier_score_loss(y_test, y_pred_proba)
            
            cv_scores.append({
                'fold': fold_idx,
                'pr_auc': pr_auc,
                'brier_score': brier,
                'n_train': len(X_train),
                'n_test': len(X_test)
            })
            
            feature_importances.append(model.feature_importances_)
            
            logger.info(f"XGB Fold {fold_idx}: PR-AUC={pr_auc:.3f}, Brier={brier:.3f}")
        
        # Train final model
        model.fit(X, y)
        
        # Calculate average feature importance
        avg_feature_importance = np.mean(feature_importances, axis=0)
        
        results = {
            'model': model,
            'cv_scores': cv_scores,
            'feature_importance': dict(zip(X.columns, avg_feature_importance)),
            'config': xgb_config
        }
        
        logger.info(f"XGB training complete. Avg PR-AUC: {np.mean([s['pr_auc'] for s in cv_scores]):.3f}")
        return results
    
    def train_lightgbm(self, X: pd.DataFrame, y: pd.Series, 
                      cv_splits: List[Dict]) -> Dict[str, Any]:
        """Train LightGBM with temporal CV."""
        logger.info("Training LightGBM...")
        
        lgb_config = self.stage1_config['models']['lightgbm']
        
        model = lgb.LGBMClassifier(
            random_state=self.random_seed,
            objective='binary',
            metric='binary_logloss',
            **lgb_config
        )
        
        # Temporal cross-validation
        cv_scores = []
        feature_importances = []
        
        for fold_idx, split in enumerate(cv_splits):
            X_train = X.iloc[split['train_indices']]
            X_test = X.iloc[split['test_indices']]
            y_train = y.iloc[split['train_indices']]
            y_test = y.iloc[split['test_indices']]
            
            # Train with early stopping
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            
            # Evaluate
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            pr_auc = self._calculate_pr_auc(y_test, y_pred_proba)
            brier = brier_score_loss(y_test, y_pred_proba)
            
            cv_scores.append({
                'fold': fold_idx,
                'pr_auc': pr_auc,
                'brier_score': brier,
                'n_train': len(X_train),
                'n_test': len(X_test)
            })
            
            feature_importances.append(model.feature_importances_)
            
            logger.info(f"LGB Fold {fold_idx}: PR-AUC={pr_auc:.3f}, Brier={brier:.3f}")
        
        # Train final model
        model.fit(X, y)
        
        # Calculate average feature importance
        avg_feature_importance = np.mean(feature_importances, axis=0)
        
        results = {
            'model': model,
            'cv_scores': cv_scores,
            'feature_importance': dict(zip(X.columns, avg_feature_importance)),
            'config': lgb_config
        }
        
        logger.info(f"LGB training complete. Avg PR-AUC: {np.mean([s['pr_auc'] for s in cv_scores]):.3f}")
        return results
    
    def calibrate_models(self, models: Dict[str, Any], X: pd.DataFrame, y: pd.Series,
                        cv_splits: List[Dict]) -> Dict[str, Any]:
        """Calibrate model probabilities using temporal CV."""
        logger.info("Calibrating model probabilities...")
        
        calibrated_models = {}
        calibration_method = self.stage1_config['calibration']['method']
        
        for model_name, model_results in models.items():
            logger.info(f"Calibrating {model_name}...")
            
            model = model_results['model']
            
            # Use temporal splits for calibration
            calibrated_scores = []
            
            for fold_idx, split in enumerate(cv_splits):
                X_train = X.iloc[split['train_indices']]
                X_cal = X.iloc[split['test_indices']]
                y_train = y.iloc[split['train_indices']]
                y_cal = y.iloc[split['test_indices']]
                
                # Train base model
                if hasattr(model, 'fit'):
                    base_model = type(model)(**model_results['config'])
                    base_model.set_params(random_state=self.random_seed)
                    base_model.fit(X_train, y_train)
                    
                    # Calibrate
                    calibrated_model = CalibratedClassifierCV(
                        base_model, method=calibration_method, cv='prefit'
                    )
                    calibrated_model.fit(X_cal, y_cal)
                    
                    # Evaluate calibration
                    y_pred_cal = calibrated_model.predict_proba(X_cal)[:, 1]
                    ece = calculate_ece(y_cal, y_pred_cal)
                    brier = brier_score_loss(y_cal, y_pred_cal)
                    
                    calibrated_scores.append({
                        'fold': fold_idx,
                        'ece': ece,
                        'brier_score': brier
                    })
            
            # Train final calibrated model
            calibrated_final = CalibratedClassifierCV(
                model, method=calibration_method, 
                cv=self.stage1_config['calibration']['cv_folds']
            )
            calibrated_final.fit(X, y)
            
            calibrated_models[model_name] = {
                'model': calibrated_final,
                'calibration_scores': calibrated_scores,
                'base_results': model_results
            }
            
            avg_ece = np.mean([s['ece'] for s in calibrated_scores])
            logger.info(f"{model_name} calibration complete. Avg ECE: {avg_ece:.3f}")
        
        return calibrated_models
    
    def evaluate_ensemble(self, models: Dict[str, Any], X: pd.DataFrame, y: pd.Series,
                         cv_splits: List[Dict]) -> Dict[str, Any]:
        """Evaluate ensemble performance."""
        logger.info("Evaluating ensemble performance...")
        
        ensemble_scores = []
        
        for fold_idx, split in enumerate(cv_splits):
            X_test = X.iloc[split['test_indices']]
            y_test = y.iloc[split['test_indices']]
            
            # Get predictions from all models
            predictions = {}
            for model_name, model_data in models.items():
                model = model_data['model']
                pred_proba = model.predict_proba(X_test)[:, 1]
                predictions[model_name] = pred_proba
            
            # Ensemble prediction (simple average)
            ensemble_pred = np.mean(list(predictions.values()), axis=0)
            
            # Evaluate ensemble
            pr_auc = self._calculate_pr_auc(y_test, ensemble_pred)
            roc_auc = roc_auc_score(y_test, ensemble_pred)
            brier = brier_score_loss(y_test, ensemble_pred)
            ece = calculate_ece(y_test, ensemble_pred)
            
            ensemble_scores.append({
                'fold': fold_idx,
                'pr_auc': pr_auc,
                'roc_auc': roc_auc,
                'brier_score': brier,
                'ece': ece
            })
            
            logger.info(f"Ensemble Fold {fold_idx}: PR-AUC={pr_auc:.3f}, ECE={ece:.3f}")
        
        # Calculate average metrics
        avg_metrics = {
            'pr_auc_mean': np.mean([s['pr_auc'] for s in ensemble_scores]),
            'pr_auc_std': np.std([s['pr_auc'] for s in ensemble_scores]),
            'roc_auc_mean': np.mean([s['roc_auc'] for s in ensemble_scores]),
            'brier_score_mean': np.mean([s['brier_score'] for s in ensemble_scores]),
            'ece_mean': np.mean([s['ece'] for s in ensemble_scores])
        }
        
        return {
            'fold_scores': ensemble_scores,
            'average_metrics': avg_metrics
        }
    
    def save_models(self, models: Dict[str, Any], output_dir: str = "artifacts"):
        """Save trained models and metadata."""
        logger.info(f"Saving models to {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        for model_name, model_data in models.items():
            # Save model
            model_path = os.path.join(output_dir, f"stage1_{model_name}.pkl")
            joblib.dump(model_data['model'], model_path)
            
            # Save metadata
            metadata = {
                'model_type': model_name,
                'training_date': datetime.now().isoformat(),
                'config': model_data['base_results']['config'],
                'cv_scores': model_data['base_results']['cv_scores'],
                'calibration_scores': model_data['calibration_scores'],
                'feature_importance': model_data['base_results']['feature_importance'],
                'random_seed': self.random_seed
            }
            
            metadata_path = os.path.join(output_dir, f"stage1_{model_name}_metadata.yaml")
            with open(metadata_path, 'w') as f:
                yaml.dump(metadata, f)
            
            logger.info(f"Saved {model_name}: {model_path}")
    
    def _calculate_pr_auc(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Calculate Precision-Recall AUC."""
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        return np.trapz(precision, recall)
    
    def log_to_mlflow(self, models: Dict[str, Any], ensemble_results: Dict[str, Any]):
        """Log results to MLflow."""
        with mlflow.start_run(run_name="stage1_ensemble_training"):
            # Log parameters
            mlflow.log_params(self.stage1_config)
            mlflow.log_param("random_seed", self.random_seed)
            mlflow.log_param("n_models", len(models))
            
            # Log ensemble metrics
            for metric, value in ensemble_results['average_metrics'].items():
                mlflow.log_metric(f"ensemble_{metric}", value)
            
            # Log individual model metrics
            for model_name, model_data in models.items():
                cv_scores = model_data['base_results']['cv_scores']
                avg_pr_auc = np.mean([s['pr_auc'] for s in cv_scores])
                avg_brier = np.mean([s['brier_score'] for s in cv_scores])
                
                mlflow.log_metric(f"{model_name}_pr_auc", avg_pr_auc)
                mlflow.log_metric(f"{model_name}_brier_score", avg_brier)
            
            # Log artifacts
            mlflow.log_artifacts("artifacts", artifact_path="stage1_models")
            
            logger.info("Results logged to MLflow")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Train Stage-1 ensemble models")
    parser.add_argument('--data', default='data/features',
                       help='Features directory')
    parser.add_argument('--output', default='artifacts',
                       help='Output directory for models')
    parser.add_argument('--config', default='configs/config.yaml',
                       help='Configuration file')
    parser.add_argument('--models', default='rf,xgb,lgb',
                       help='Models to train (comma-separated)')
    parser.add_argument('--fast', action='store_true',
                       help='Fast training mode')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger.info("Starting Stage-1 ensemble training")
    
    try:
        # Initialize trainer
        trainer = Stage1Trainer(args.config)
        
        # Load data
        X, y, metadata = trainer.load_features_and_targets(args.data)
        
        # Create temporal CV splits
        full_data = pd.concat([metadata, X, y], axis=1)
        cv_splits = trainer.create_temporal_cv_splits(full_data)
        
        # Train models
        models_to_train = args.models.split(',')
        trained_models = {}
        
        if 'rf' in models_to_train:
            trained_models['random_forest'] = trainer.train_random_forest(X, y, cv_splits)
        
        if 'xgb' in models_to_train:
            trained_models['xgboost'] = trainer.train_xgboost(X, y, cv_splits)
        
        if 'lgb' in models_to_train:
            trained_models['lightgbm'] = trainer.train_lightgbm(X, y, cv_splits)
        
        # Calibrate models
        calibrated_models = trainer.calibrate_models(trained_models, X, y, cv_splits)
        
        # Evaluate ensemble
        ensemble_results = trainer.evaluate_ensemble(calibrated_models, X, y, cv_splits)
        
        # Save models
        trainer.save_models(calibrated_models, args.output)
        
        # Log to MLflow
        trainer.log_to_mlflow(calibrated_models, ensemble_results)
        
        logger.info("Stage-1 training completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Stage-1 training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
