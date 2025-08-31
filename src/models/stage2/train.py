#!/usr/bin/env python3
"""
Stage-2 Gradient Boosting Model Training.
Uses Stage-1 outputs + compatibility matrix for final squad selection.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
import logging
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
import xgboost as xgb
import mlflow
import mlflow.sklearn

sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.logging_config import setup_logging
from utils.model_utils import calculate_ece, plot_calibration_curve
from stage1.api import Stage1API

logger = logging.getLogger(__name__)


class Stage2ModelTrainer:
    """Train Stage-2 gradient boosting model for final squad selection."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.stage2_config = self.config['stage2']
        self.random_seed = self.config['random_seeds']['global']
        np.random.seed(self.random_seed)
        
        # Initialize Stage-1 API
        self.stage1_api = Stage1API()
        
    def load_training_data(self, data_dir: str = "data/processed") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load training data and historical selections."""
        logger.info("Loading Stage-2 training data...")
        
        # Load lineups (actual selections)
        lineups_df = pd.read_csv(os.path.join(data_dir, "lineups.csv"))
        
        # Load players data
        players_df = pd.read_csv(os.path.join(data_dir, "players.csv"))
        
        # Load features
        features_df = pd.read_csv(os.path.join(data_dir, "features.csv"))
        
        logger.info(f"Loaded {len(lineups_df)} lineup records")
        logger.info(f"Loaded {len(players_df)} players")
        logger.info(f"Loaded {len(features_df)} feature records")
        
        return lineups_df, players_df, features_df
    
    def load_compatibility_matrix(self, artifacts_dir: str = "artifacts") -> Tuple[np.ndarray, Dict]:
        """Load pre-computed compatibility matrix."""
        logger.info("Loading compatibility matrix...")
        
        matrix_path = os.path.join(artifacts_dir, "compat_matrix_v1.npz")
        matrix_data = np.load(matrix_path, allow_pickle=True)
        
        compat_matrix = matrix_data['compatibility_matrix']
        player_to_idx = matrix_data['player_to_idx'].item()
        idx_to_player = matrix_data['idx_to_player'].item()
        
        logger.info(f"Loaded compatibility matrix: {compat_matrix.shape}")
        return compat_matrix, {'player_to_idx': player_to_idx, 'idx_to_player': idx_to_player}
    
    def create_stage2_features(self, lineups_df: pd.DataFrame, 
                              features_df: pd.DataFrame,
                              compat_matrix: np.ndarray,
                              player_mappings: Dict) -> pd.DataFrame:
        """Create Stage-2 features combining Stage-1 outputs with compatibility."""
        logger.info("Creating Stage-2 features...")
        
        stage2_features = []
        
        for _, lineup in lineups_df.iterrows():
            match_date = lineup['match_date']
            team_id = lineup['team_id']
            
            # Get all eligible players for this match
            eligible_players = features_df[
                (features_df['match_date'] == match_date) & 
                (features_df['team_id'] == team_id)
            ].copy()
            
            if len(eligible_players) == 0:
                continue
            
            # Get Stage-1 predictions for all eligible players
            try:
                stage1_probs = self.stage1_api.predict_batch(
                    eligible_players, 
                    return_uncertainty=True
                )
                eligible_players['stage1_prob'] = stage1_probs['probabilities']
                eligible_players['stage1_uncertainty'] = stage1_probs['uncertainty']
            except Exception as e:
                logger.warning(f"Stage-1 prediction failed for {match_date}: {e}")
                continue
            
            # Calculate compatibility features for each player
            for _, player_row in eligible_players.iterrows():
                player_id = player_row['player_id']
                
                # Get player index for compatibility matrix
                if player_id not in player_mappings['player_to_idx']:
                    continue
                
                player_idx = player_mappings['player_to_idx'][player_id]
                
                # Calculate compatibility with recently selected teammates
                recent_teammates = self._get_recent_teammates(
                    player_id, team_id, match_date, lineups_df, days_back=30
                )
                
                teammate_compatibility = self._calculate_teammate_compatibility(
                    player_idx, recent_teammates, compat_matrix, player_mappings
                )
                
                # Create Stage-2 feature vector
                stage2_feature = {
                    'player_id': player_id,
                    'match_date': match_date,
                    'team_id': team_id,
                    
                    # Stage-1 outputs
                    'stage1_prob': player_row['stage1_prob'],
                    'stage1_uncertainty': player_row['stage1_uncertainty'],
                    
                    # Original features (subset)
                    'performance_score': player_row.get('performance_score', 0),
                    'fitness_score': player_row.get('fitness_score', 1),
                    'position_versatility': player_row.get('position_versatility', 1),
                    'recent_form': player_row.get('recent_form_3_matches', 0),
                    
                    # Compatibility features
                    'avg_teammate_compatibility': teammate_compatibility['avg_compatibility'],
                    'max_teammate_compatibility': teammate_compatibility['max_compatibility'],
                    'min_teammate_compatibility': teammate_compatibility['min_compatibility'],
                    'n_compatible_teammates': teammate_compatibility['n_compatible'],
                    
                    # Squad context features
                    'squad_avg_stage1_prob': eligible_players['stage1_prob'].mean(),
                    'squad_std_stage1_prob': eligible_players['stage1_prob'].std(),
                    'player_rank_in_squad': (eligible_players['stage1_prob'] > player_row['stage1_prob']).sum(),
                    
                    # Target: was actually selected
                    'selected': int(player_id in lineup.get('selected_players', '').split(','))
                }
                
                stage2_features.append(stage2_feature)
        
        stage2_df = pd.DataFrame(stage2_features)
        logger.info(f"Created {len(stage2_df)} Stage-2 training examples")
        
        return stage2_df
    
    def _get_recent_teammates(self, player_id: str, team_id: str, 
                            match_date: str, lineups_df: pd.DataFrame,
                            days_back: int = 30) -> List[str]:
        """Get recently played teammates."""
        match_date_dt = pd.to_datetime(match_date)
        cutoff_date = match_date_dt - pd.Timedelta(days=days_back)
        
        recent_lineups = lineups_df[
            (lineups_df['team_id'] == team_id) &
            (pd.to_datetime(lineups_df['match_date']) >= cutoff_date) &
            (pd.to_datetime(lineups_df['match_date']) < match_date_dt)
        ]
        
        teammates = set()
        for _, lineup in recent_lineups.iterrows():
            selected_players = lineup.get('selected_players', '').split(',')
            if player_id in selected_players:
                teammates.update([p for p in selected_players if p != player_id])
        
        return list(teammates)
    
    def _calculate_teammate_compatibility(self, player_idx: int, teammates: List[str],
                                        compat_matrix: np.ndarray, 
                                        player_mappings: Dict) -> Dict:
        """Calculate compatibility statistics with teammates."""
        if not teammates:
            return {
                'avg_compatibility': 0.5,
                'max_compatibility': 0.5,
                'min_compatibility': 0.5,
                'n_compatible': 0
            }
        
        compatibilities = []
        for teammate_id in teammates:
            if teammate_id in player_mappings['player_to_idx']:
                teammate_idx = player_mappings['player_to_idx'][teammate_id]
                compat_score = compat_matrix[player_idx, teammate_idx]
                compatibilities.append(compat_score)
        
        if not compatibilities:
            return {
                'avg_compatibility': 0.5,
                'max_compatibility': 0.5,
                'min_compatibility': 0.5,
                'n_compatible': 0
            }
        
        compatibilities = np.array(compatibilities)
        return {
            'avg_compatibility': compatibilities.mean(),
            'max_compatibility': compatibilities.max(),
            'min_compatibility': compatibilities.min(),
            'n_compatible': (compatibilities > 0.6).sum()  # Threshold for "compatible"
        }
    
    def train_stage2_models(self, stage2_df: pd.DataFrame) -> Dict:
        """Train Stage-2 gradient boosting models."""
        logger.info("Training Stage-2 models...")
        
        # Prepare features and target
        feature_cols = [col for col in stage2_df.columns if col not in 
                       ['player_id', 'match_date', 'team_id', 'selected']]
        
        X = stage2_df[feature_cols]
        y = stage2_df['selected']
        
        logger.info(f"Training features: {feature_cols}")
        logger.info(f"Training samples: {len(X)}, Positive rate: {y.mean():.3f}")
        
        # Temporal cross-validation
        tscv = TimeSeriesSplit(n_splits=self.stage2_config['cv_folds'])
        
        models = {}
        cv_scores = {}
        
        # Train multiple algorithms
        algorithms = {
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=self.stage2_config['gb_params']['n_estimators'],
                learning_rate=self.stage2_config['gb_params']['learning_rate'],
                max_depth=self.stage2_config['gb_params']['max_depth'],
                subsample=self.stage2_config['gb_params']['subsample'],
                random_state=self.random_seed
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=self.stage2_config['lgb_params']['n_estimators'],
                learning_rate=self.stage2_config['lgb_params']['learning_rate'],
                max_depth=self.stage2_config['lgb_params']['max_depth'],
                num_leaves=self.stage2_config['lgb_params']['num_leaves'],
                subsample=self.stage2_config['lgb_params']['subsample'],
                random_state=self.random_seed,
                verbose=-1
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=self.stage2_config['xgb_params']['n_estimators'],
                learning_rate=self.stage2_config['xgb_params']['learning_rate'],
                max_depth=self.stage2_config['xgb_params']['max_depth'],
                subsample=self.stage2_config['xgb_params']['subsample'],
                random_state=self.random_seed,
                eval_metric='logloss'
            )
        }
        
        for algo_name, model in algorithms.items():
            logger.info(f"Training {algo_name}...")
            
            fold_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Train model
                model.fit(X_train, y_train)
                
                # Validate
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                
                fold_score = {
                    'roc_auc': roc_auc_score(y_val, y_pred_proba),
                    'pr_auc': average_precision_score(y_val, y_pred_proba),
                    'brier': brier_score_loss(y_val, y_pred_proba),
                    'ece': calculate_ece(y_val, y_pred_proba)
                }
                
                fold_scores.append(fold_score)
                logger.info(f"{algo_name} Fold {fold}: ROC-AUC={fold_score['roc_auc']:.3f}")
            
            # Calculate mean CV scores
            cv_score = {metric: np.mean([fold[metric] for fold in fold_scores]) 
                       for metric in fold_scores[0].keys()}
            cv_scores[algo_name] = cv_score
            
            # Train final model on all data
            final_model = model.__class__(**model.get_params())
            final_model.fit(X, y)
            
            # Calibrate probabilities
            calibrated_model = CalibratedClassifierCV(
                final_model, method='isotonic', cv=3
            )
            calibrated_model.fit(X, y)
            
            models[algo_name] = {
                'model': final_model,
                'calibrated_model': calibrated_model,
                'cv_scores': cv_score,
                'feature_importance': self._get_feature_importance(final_model, feature_cols)
            }
            
            logger.info(f"{algo_name} training completed - ROC-AUC: {cv_score['roc_auc']:.3f}")
        
        return models, cv_scores
    
    def _get_feature_importance(self, model, feature_cols: List[str]) -> Dict:
        """Extract feature importance from trained model."""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            importance = np.ones(len(feature_cols))  # Fallback
        
        return dict(zip(feature_cols, importance))
    
    def select_best_model(self, models: Dict, cv_scores: Dict) -> str:
        """Select best Stage-2 model based on validation performance."""
        logger.info("Selecting best Stage-2 model...")
        
        # Primary metric: ROC-AUC, secondary: PR-AUC
        best_model = None
        best_score = -1
        
        for algo_name, scores in cv_scores.items():
            # Weighted score: 60% ROC-AUC + 40% PR-AUC
            combined_score = 0.6 * scores['roc_auc'] + 0.4 * scores['pr_auc']
            
            logger.info(f"{algo_name}: Combined score = {combined_score:.3f}")
            
            if combined_score > best_score:
                best_score = combined_score
                best_model = algo_name
        
        logger.info(f"Selected best model: {best_model} (score: {best_score:.3f})")
        return best_model
    
    def create_ensemble_model(self, models: Dict, cv_scores: Dict) -> Dict:
        """Create ensemble of Stage-2 models with weighted averaging."""
        logger.info("Creating Stage-2 ensemble...")
        
        # Calculate weights based on ROC-AUC performance
        weights = {}
        total_score = sum(scores['roc_auc'] for scores in cv_scores.values())
        
        for algo_name, scores in cv_scores.items():
            weights[algo_name] = scores['roc_auc'] / total_score
        
        ensemble_info = {
            'models': models,
            'weights': weights,
            'individual_scores': cv_scores
        }
        
        logger.info(f"Ensemble weights: {weights}")
        return ensemble_info
    
    def validate_stage2_performance(self, ensemble_info: Dict, 
                                  stage2_df: pd.DataFrame) -> Dict:
        """Validate Stage-2 ensemble performance."""
        logger.info("Validating Stage-2 performance...")
        
        feature_cols = [col for col in stage2_df.columns if col not in 
                       ['player_id', 'match_date', 'team_id', 'selected']]
        
        X = stage2_df[feature_cols]
        y = stage2_df['selected']
        
        # Get ensemble predictions
        ensemble_probs = np.zeros(len(X))
        
        for algo_name, weight in ensemble_info['weights'].items():
            model = ensemble_info['models'][algo_name]['calibrated_model']
            probs = model.predict_proba(X)[:, 1]
            ensemble_probs += weight * probs
        
        # Calculate metrics
        validation_metrics = {
            'roc_auc': roc_auc_score(y, ensemble_probs),
            'pr_auc': average_precision_score(y, ensemble_probs),
            'brier_score': brier_score_loss(y, ensemble_probs),
            'ece': calculate_ece(y, ensemble_probs),
            'n_samples': len(y),
            'positive_rate': y.mean()
        }
        
        logger.info("Stage-2 Ensemble Performance:")
        for metric, value in validation_metrics.items():
            logger.info(f"  {metric}: {value:.3f}")
        
        return validation_metrics
    
    def save_stage2_models(self, ensemble_info: Dict, validation_metrics: Dict,
                          output_dir: str = "artifacts"):
        """Save Stage-2 models and metadata."""
        logger.info(f"Saving Stage-2 models to {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save ensemble
        ensemble_path = os.path.join(output_dir, "stage2_ensemble.pkl")
        joblib.dump(ensemble_info, ensemble_path)
        
        # Save individual models
        for algo_name, model_info in ensemble_info['models'].items():
            model_path = os.path.join(output_dir, f"stage2_{algo_name}.pkl")
            joblib.dump(model_info, model_path)
        
        # Save metadata
        metadata = {
            'stage2_config': self.stage2_config,
            'validation_metrics': validation_metrics,
            'ensemble_weights': ensemble_info['weights'],
            'individual_cv_scores': ensemble_info['individual_scores'],
            'training_date': pd.Timestamp.now().isoformat(),
            'random_seed': self.random_seed
        }
        
        metadata_path = os.path.join(output_dir, "stage2_metadata.yaml")
        with open(metadata_path, 'w') as f:
            yaml.dump(metadata, f)
        
        logger.info("Stage-2 models saved successfully")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Train Stage-2 models")
    parser.add_argument('--data', default='data/processed',
                       help='Processed data directory')
    parser.add_argument('--artifacts', default='artifacts',
                       help='Artifacts directory')
    parser.add_argument('--config', default='configs/config.yaml',
                       help='Configuration file')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger.info("Starting Stage-2 model training")
    
    try:
        # Initialize trainer
        trainer = Stage2ModelTrainer(args.config)
        
        # Load data
        lineups_df, players_df, features_df = trainer.load_training_data(args.data)
        
        # Load compatibility matrix
        compat_matrix, player_mappings = trainer.load_compatibility_matrix(args.artifacts)
        
        # Create Stage-2 features
        stage2_df = trainer.create_stage2_features(
            lineups_df, features_df, compat_matrix, player_mappings
        )
        
        # Train models
        models, cv_scores = trainer.train_stage2_models(stage2_df)
        
        # Create ensemble
        ensemble_info = trainer.create_ensemble_model(models, cv_scores)
        
        # Validate performance
        validation_metrics = trainer.validate_stage2_performance(ensemble_info, stage2_df)
        
        # Save models
        trainer.save_stage2_models(ensemble_info, validation_metrics, args.artifacts)
        
        # Log to MLflow
        with mlflow.start_run(run_name="stage2_training"):
            mlflow.log_params(trainer.stage2_config)
            mlflow.log_metrics(validation_metrics)
            mlflow.log_metrics({f"weight_{k}": v for k, v in ensemble_info['weights'].items()})
            mlflow.log_artifacts(args.artifacts, artifact_path="stage2")
        
        logger.info("Stage-2 training completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Stage-2 training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
