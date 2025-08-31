#!/usr/bin/env python3
"""
Stage-1 API for player selection probability prediction.
Provides standardized interface for Stage-2 and external usage.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import yaml
import logging
import joblib
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


class Stage1Predictor:
    """Stage-1 ensemble predictor with standardized API."""
    
    def __init__(self, models_dir: str = "artifacts", config_path: str = "configs/config.yaml"):
        """Initialize Stage-1 predictor with trained models."""
        self.models_dir = models_dir
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.models = {}
        self.feature_names = None
        self.is_loaded = False
        
    def load_models(self) -> bool:
        """Load all trained Stage-1 models."""
        logger.info("Loading Stage-1 models...")
        
        model_files = {
            'random_forest': 'stage1_random_forest.pkl',
            'xgboost': 'stage1_xgboost.pkl',
            'lightgbm': 'stage1_lightgbm.pkl'
        }
        
        loaded_count = 0
        
        for model_name, filename in model_files.items():
            model_path = os.path.join(self.models_dir, filename)
            
            if os.path.exists(model_path):
                try:
                    self.models[model_name] = joblib.load(model_path)
                    loaded_count += 1
                    logger.info(f"Loaded {model_name} model")
                except Exception as e:
                    logger.warning(f"Failed to load {model_name}: {e}")
            else:
                logger.warning(f"Model file not found: {model_path}")
        
        if loaded_count == 0:
            logger.error("No models could be loaded")
            return False
        
        # Load feature names from first available model metadata
        self._load_feature_metadata()
        
        self.is_loaded = True
        logger.info(f"Successfully loaded {loaded_count} models")
        return True
    
    def _load_feature_metadata(self):
        """Load feature names and metadata."""
        for model_name in self.models.keys():
            metadata_path = os.path.join(self.models_dir, f"stage1_{model_name}_metadata.yaml")
            
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = yaml.safe_load(f)
                    
                    if 'feature_importance' in metadata:
                        self.feature_names = list(metadata['feature_importance'].keys())
                        logger.info(f"Loaded feature names from {model_name} metadata")
                        break
                        
                except Exception as e:
                    logger.warning(f"Failed to load metadata for {model_name}: {e}")
    
    def predict_selection_probability(self, features: Union[pd.DataFrame, Dict, List[Dict]],
                                    return_individual: bool = False,
                                    return_uncertainty: bool = False) -> Union[np.ndarray, Dict]:
        """
        Predict player selection probabilities.
        
        Args:
            features: Player features as DataFrame, dict, or list of dicts
            return_individual: Return individual model predictions
            return_uncertainty: Return prediction uncertainty estimates
            
        Returns:
            Array of probabilities or dict with additional info
        """
        if not self.is_loaded:
            if not self.load_models():
                raise RuntimeError("No models available for prediction")
        
        # Convert input to DataFrame
        if isinstance(features, dict):
            features_df = pd.DataFrame([features])
        elif isinstance(features, list):
            features_df = pd.DataFrame(features)
        else:
            features_df = features.copy()
        
        # Validate features
        self._validate_features(features_df)
        
        # Get predictions from all models
        individual_predictions = {}
        
        for model_name, model in self.models.items():
            try:
                pred_proba = model.predict_proba(features_df)[:, 1]
                individual_predictions[model_name] = pred_proba
            except Exception as e:
                logger.warning(f"Prediction failed for {model_name}: {e}")
        
        if not individual_predictions:
            raise RuntimeError("All model predictions failed")
        
        # Ensemble prediction (simple average)
        ensemble_pred = np.mean(list(individual_predictions.values()), axis=0)
        
        # Calculate uncertainty (standard deviation across models)
        if len(individual_predictions) > 1:
            uncertainty = np.std(list(individual_predictions.values()), axis=0)
        else:
            uncertainty = np.zeros_like(ensemble_pred)
        
        # Prepare return value
        if return_individual or return_uncertainty:
            result = {
                'ensemble_probability': ensemble_pred,
                'model_count': len(individual_predictions)
            }
            
            if return_individual:
                result['individual_predictions'] = individual_predictions
            
            if return_uncertainty:
                result['uncertainty'] = uncertainty
                result['confidence'] = 1 - uncertainty  # Simple confidence measure
            
            return result
        else:
            return ensemble_pred
    
    def predict_squad_selection(self, features: pd.DataFrame, 
                               squad_size: int = 23,
                               formation_constraints: Optional[Dict] = None) -> Dict:
        """
        Predict squad selection with constraints.
        
        Args:
            features: Player features DataFrame with player_id column
            squad_size: Target squad size
            formation_constraints: Position constraints (optional)
            
        Returns:
            Dict with selected players and metadata
        """
        if 'player_id' not in features.columns:
            raise ValueError("Features must include 'player_id' column")
        
        # Get selection probabilities
        feature_cols = [col for col in features.columns if col != 'player_id']
        probabilities = self.predict_selection_probability(features[feature_cols])
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'player_id': features['player_id'],
            'selection_probability': probabilities
        })
        
        # Apply formation constraints if provided
        if formation_constraints:
            results_df = self._apply_formation_constraints(results_df, features, formation_constraints)
        
        # Select top players
        selected_players = results_df.nlargest(squad_size, 'selection_probability')
        
        return {
            'selected_players': selected_players['player_id'].tolist(),
            'selection_probabilities': selected_players.set_index('player_id')['selection_probability'].to_dict(),
            'squad_size': len(selected_players),
            'avg_probability': selected_players['selection_probability'].mean(),
            'min_probability': selected_players['selection_probability'].min(),
            'timestamp': datetime.now().isoformat()
        }
    
    def _validate_features(self, features_df: pd.DataFrame):
        """Validate input features."""
        if features_df.empty:
            raise ValueError("Features DataFrame is empty")
        
        # Check for required features (if we have feature names)
        if self.feature_names:
            missing_features = set(self.feature_names) - set(features_df.columns)
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                
                # Fill missing features with zeros
                for feature in missing_features:
                    features_df[feature] = 0
        
        # Check for NaN values
        if features_df.isnull().any().any():
            logger.warning("NaN values detected in features, filling with median")
            features_df.fillna(features_df.median(), inplace=True)
    
    def _apply_formation_constraints(self, results_df: pd.DataFrame, 
                                   features: pd.DataFrame,
                                   constraints: Dict) -> pd.DataFrame:
        """Apply formation constraints to selection."""
        # This is a simplified version - full implementation would be in Stage-2
        logger.info("Applying basic formation constraints")
        
        # For now, just ensure we have minimum positions if position info available
        if 'primary_position' in features.columns:
            results_with_pos = results_df.merge(
                features[['player_id', 'primary_position']], 
                on='player_id'
            )
            
            # Ensure minimum GK, DEF, MID, FWD if specified
            position_groups = {
                'GK': ['GK'],
                'DEF': ['CB', 'LB', 'RB'],
                'MID': ['DM', 'CM', 'AM'],
                'FWD': ['LW', 'RW', 'CF']
            }
            
            adjusted_results = []
            
            for group, positions in position_groups.items():
                min_required = constraints.get(f'min_{group.lower()}', 0)
                
                if min_required > 0:
                    group_players = results_with_pos[
                        results_with_pos['primary_position'].isin(positions)
                    ].nlargest(min_required, 'selection_probability')
                    
                    adjusted_results.append(group_players)
            
            if adjusted_results:
                # Combine constrained selections
                constrained_df = pd.concat(adjusted_results).drop_duplicates('player_id')
                
                # Add remaining players to fill squad
                remaining_players = results_with_pos[
                    ~results_with_pos['player_id'].isin(constrained_df['player_id'])
                ]
                
                final_results = pd.concat([constrained_df, remaining_players])
                return final_results[['player_id', 'selection_probability']]
        
        return results_df
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models."""
        if not self.is_loaded:
            return {'status': 'not_loaded', 'models': []}
        
        model_info = []
        
        for model_name in self.models.keys():
            metadata_path = os.path.join(self.models_dir, f"stage1_{model_name}_metadata.yaml")
            
            info = {'name': model_name, 'loaded': True}
            
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = yaml.safe_load(f)
                    
                    info.update({
                        'training_date': metadata.get('training_date'),
                        'cv_performance': metadata.get('cv_scores', []),
                        'config': metadata.get('config', {})
                    })
                except Exception as e:
                    info['metadata_error'] = str(e)
            
            model_info.append(info)
        
        return {
            'status': 'loaded',
            'models': model_info,
            'feature_count': len(self.feature_names) if self.feature_names else 'unknown',
            'ensemble_method': 'simple_average'
        }
    
    def batch_predict(self, features_list: List[pd.DataFrame],
                     batch_size: int = 1000) -> List[np.ndarray]:
        """Predict probabilities for multiple feature sets efficiently."""
        logger.info(f"Batch prediction for {len(features_list)} feature sets")
        
        results = []
        
        for i in range(0, len(features_list), batch_size):
            batch = features_list[i:i+batch_size]
            batch_results = []
            
            for features in batch:
                pred = self.predict_selection_probability(features)
                batch_results.append(pred)
            
            results.extend(batch_results)
            
            if i % (batch_size * 10) == 0:
                logger.info(f"Processed {i + len(batch)} / {len(features_list)} batches")
        
        return results


# Convenience functions for common usage patterns
def predict_player_selection(player_features: Dict, models_dir: str = "artifacts") -> float:
    """Quick prediction for a single player."""
    predictor = Stage1Predictor(models_dir)
    probability = predictor.predict_selection_probability(player_features)
    return float(probability[0])


def predict_squad_from_csv(csv_path: str, models_dir: str = "artifacts", 
                          squad_size: int = 23) -> Dict:
    """Predict squad selection from CSV file."""
    features_df = pd.read_csv(csv_path)
    
    predictor = Stage1Predictor(models_dir)
    result = predictor.predict_squad_selection(features_df, squad_size)
    
    return result


if __name__ == "__main__":
    # Example usage
    setup_logging()
    
    # Initialize predictor
    predictor = Stage1Predictor()
    
    if predictor.load_models():
        # Get model info
        info = predictor.get_model_info()
        print("Model Info:", info)
        
        # Example prediction (would need real features)
        # features = pd.DataFrame({...})  # Your features here
        # probabilities = predictor.predict_selection_probability(features)
        # print("Selection probabilities:", probabilities)
    else:
        print("Failed to load models")
