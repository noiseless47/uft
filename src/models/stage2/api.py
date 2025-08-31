#!/usr/bin/env python3
"""
Stage-2 API for final squad selection predictions.
Combines Stage-1 outputs with compatibility matrix for optimized selections.
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

sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.logging_config import setup_logging
from stage1.api import Stage1API

logger = logging.getLogger(__name__)


class Stage2API:
    """API for Stage-2 squad selection predictions."""
    
    def __init__(self, artifacts_dir: str = "artifacts", 
                 config_path: str = "configs/config.yaml"):
        """Initialize Stage-2 API with pre-trained models."""
        self.artifacts_dir = artifacts_dir
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load models and artifacts
        self._load_models()
        self._load_compatibility_matrix()
        
        # Initialize Stage-1 API
        self.stage1_api = Stage1API()
        
        logger.info("Stage-2 API initialized successfully")
    
    def _load_models(self):
        """Load pre-trained Stage-2 models."""
        logger.info("Loading Stage-2 models...")
        
        ensemble_path = os.path.join(self.artifacts_dir, "stage2_ensemble.pkl")
        self.ensemble_info = joblib.load(ensemble_path)
        
        metadata_path = os.path.join(self.artifacts_dir, "stage2_metadata.yaml")
        with open(metadata_path, 'r') as f:
            self.metadata = yaml.safe_load(f)
        
        logger.info(f"Loaded ensemble with {len(self.ensemble_info['models'])} models")
    
    def _load_compatibility_matrix(self):
        """Load compatibility matrix and player mappings."""
        logger.info("Loading compatibility matrix...")
        
        matrix_path = os.path.join(self.artifacts_dir, "compat_matrix_v1.npz")
        matrix_data = np.load(matrix_path, allow_pickle=True)
        
        self.compat_matrix = matrix_data['compatibility_matrix']
        self.player_to_idx = matrix_data['player_to_idx'].item()
        self.idx_to_player = matrix_data['idx_to_player'].item()
        
        logger.info(f"Loaded compatibility matrix: {self.compat_matrix.shape}")
    
    def predict_squad_selection(self, eligible_players: pd.DataFrame,
                              formation: str = "4-3-3",
                              return_probabilities: bool = False) -> Dict:
        """
        Predict final squad selection using Stage-2 model.
        
        Args:
            eligible_players: DataFrame with player features
            formation: Target formation (e.g., "4-3-3")
            return_probabilities: Whether to return selection probabilities
            
        Returns:
            Dictionary with selected squad and metadata
        """
        logger.info(f"Predicting squad selection for {len(eligible_players)} players")
        
        # Get Stage-1 predictions
        stage1_results = self.stage1_api.predict_batch(
            eligible_players, return_uncertainty=True
        )
        
        # Create Stage-2 features
        stage2_features = self._create_stage2_features(
            eligible_players, stage1_results
        )
        
        # Get Stage-2 predictions
        stage2_probs = self._predict_stage2_probabilities(stage2_features)
        
        # Combine predictions
        final_probs = self._combine_stage_predictions(
            stage1_results['probabilities'], stage2_probs
        )
        
        # Apply formation constraints and optimize
        selected_squad = self._optimize_squad_selection(
            eligible_players, final_probs, formation
        )
        
        result = {
            'selected_players': selected_squad['players'],
            'formation': formation,
            'total_score': selected_squad['total_score'],
            'position_assignments': selected_squad['positions']
        }
        
        if return_probabilities:
            result['stage1_probabilities'] = stage1_results['probabilities']
            result['stage2_probabilities'] = stage2_probs
            result['final_probabilities'] = final_probs
            result['uncertainty'] = stage1_results['uncertainty']
        
        logger.info(f"Selected {len(selected_squad['players'])} players")
        return result
    
    def _create_stage2_features(self, eligible_players: pd.DataFrame,
                              stage1_results: Dict) -> pd.DataFrame:
        """Create Stage-2 features for prediction."""
        stage2_features = eligible_players.copy()
        
        # Add Stage-1 outputs
        stage2_features['stage1_prob'] = stage1_results['probabilities']
        stage2_features['stage1_uncertainty'] = stage1_results['uncertainty']
        
        # Add compatibility features
        for idx, player_row in stage2_features.iterrows():
            player_id = player_row['player_id']
            
            if player_id in self.player_to_idx:
                player_idx = self.player_to_idx[player_id]
                
                # Calculate average compatibility with other eligible players
                other_player_indices = [
                    self.player_to_idx[pid] for pid in stage2_features['player_id']
                    if pid in self.player_to_idx and pid != player_id
                ]
                
                if other_player_indices:
                    compatibilities = [self.compat_matrix[player_idx, other_idx] 
                                     for other_idx in other_player_indices]
                    
                    stage2_features.loc[idx, 'avg_teammate_compatibility'] = np.mean(compatibilities)
                    stage2_features.loc[idx, 'max_teammate_compatibility'] = np.max(compatibilities)
                    stage2_features.loc[idx, 'min_teammate_compatibility'] = np.min(compatibilities)
                    stage2_features.loc[idx, 'n_compatible_teammates'] = sum(c > 0.6 for c in compatibilities)
                else:
                    stage2_features.loc[idx, 'avg_teammate_compatibility'] = 0.5
                    stage2_features.loc[idx, 'max_teammate_compatibility'] = 0.5
                    stage2_features.loc[idx, 'min_teammate_compatibility'] = 0.5
                    stage2_features.loc[idx, 'n_compatible_teammates'] = 0
            else:
                # Default values for unknown players
                stage2_features.loc[idx, 'avg_teammate_compatibility'] = 0.5
                stage2_features.loc[idx, 'max_teammate_compatibility'] = 0.5
                stage2_features.loc[idx, 'min_teammate_compatibility'] = 0.5
                stage2_features.loc[idx, 'n_compatible_teammates'] = 0
        
        # Add squad context features
        stage2_features['squad_avg_stage1_prob'] = stage1_results['probabilities'].mean()
        stage2_features['squad_std_stage1_prob'] = stage1_results['probabilities'].std()
        
        for idx, prob in enumerate(stage1_results['probabilities']):
            rank = (stage1_results['probabilities'] > prob).sum()
            stage2_features.loc[idx, 'player_rank_in_squad'] = rank
        
        return stage2_features
    
    def _predict_stage2_probabilities(self, stage2_features: pd.DataFrame) -> np.ndarray:
        """Get Stage-2 ensemble predictions."""
        feature_cols = [
            'stage1_prob', 'stage1_uncertainty', 'performance_score', 
            'fitness_score', 'position_versatility', 'recent_form',
            'avg_teammate_compatibility', 'max_teammate_compatibility',
            'min_teammate_compatibility', 'n_compatible_teammates',
            'squad_avg_stage1_prob', 'squad_std_stage1_prob', 'player_rank_in_squad'
        ]
        
        # Filter available features
        available_features = [col for col in feature_cols if col in stage2_features.columns]
        X = stage2_features[available_features]
        
        # Get ensemble predictions
        ensemble_probs = np.zeros(len(X))
        
        for algo_name, weight in self.ensemble_info['weights'].items():
            model = self.ensemble_info['models'][algo_name]['calibrated_model']
            probs = model.predict_proba(X)[:, 1]
            ensemble_probs += weight * probs
        
        return ensemble_probs
    
    def _combine_stage_predictions(self, stage1_probs: np.ndarray, 
                                 stage2_probs: np.ndarray) -> np.ndarray:
        """Combine Stage-1 and Stage-2 predictions."""
        # Weighted combination: 40% Stage-1 + 60% Stage-2
        alpha = 0.4
        return alpha * stage1_probs + (1 - alpha) * stage2_probs
    
    def _optimize_squad_selection(self, eligible_players: pd.DataFrame,
                                final_probs: np.ndarray, formation: str) -> Dict:
        """Optimize squad selection with formation constraints."""
        logger.info(f"Optimizing squad for formation: {formation}")
        
        # Parse formation
        formation_positions = self._parse_formation(formation)
        
        # Simple greedy selection with position constraints
        selected_players = []
        selected_positions = []
        total_score = 0
        
        # Sort players by probability
        player_prob_pairs = list(zip(eligible_players.index, final_probs))
        player_prob_pairs.sort(key=lambda x: x[1], reverse=True)
        
        position_counts = {pos: 0 for pos in formation_positions}
        
        for player_idx, prob in player_prob_pairs:
            player_row = eligible_players.iloc[player_idx]
            player_id = player_row['player_id']
            primary_pos = player_row.get('primary_position', 'CM')
            
            # Check if we can assign this player to a position
            assigned_position = None
            
            # Try primary position first
            if primary_pos in position_counts and position_counts[primary_pos] < formation_positions[primary_pos]:
                assigned_position = primary_pos
            else:
                # Try compatible positions
                compatible_positions = self._get_compatible_positions(primary_pos)
                for pos in compatible_positions:
                    if pos in position_counts and position_counts[pos] < formation_positions[pos]:
                        assigned_position = pos
                        break
            
            if assigned_position:
                selected_players.append(player_id)
                selected_positions.append(assigned_position)
                position_counts[assigned_position] += 1
                total_score += prob
                
                # Stop when formation is complete
                if sum(position_counts.values()) >= sum(formation_positions.values()):
                    break
        
        return {
            'players': selected_players,
            'positions': dict(zip(selected_players, selected_positions)),
            'total_score': total_score
        }
    
    def _parse_formation(self, formation: str) -> Dict[str, int]:
        """Parse formation string into position requirements."""
        formation_map = {
            "4-3-3": {'GK': 1, 'CB': 2, 'LB': 1, 'RB': 1, 'CM': 3, 'LW': 1, 'RW': 1, 'CF': 1},
            "4-4-2": {'GK': 1, 'CB': 2, 'LB': 1, 'RB': 1, 'CM': 2, 'LW': 1, 'RW': 1, 'CF': 2},
            "3-5-2": {'GK': 1, 'CB': 3, 'CM': 3, 'LW': 1, 'RW': 1, 'CF': 2},
            "4-2-3-1": {'GK': 1, 'CB': 2, 'LB': 1, 'RB': 1, 'DM': 2, 'AM': 3, 'CF': 1}
        }
        
        return formation_map.get(formation, formation_map["4-3-3"])
    
    def _get_compatible_positions(self, primary_position: str) -> List[str]:
        """Get positions compatible with primary position."""
        compatibility_map = {
            'GK': [],
            'CB': ['DM'],
            'LB': ['LW', 'CM'],
            'RB': ['RW', 'CM'],
            'DM': ['CB', 'CM'],
            'CM': ['DM', 'AM', 'LW', 'RW'],
            'AM': ['CM', 'LW', 'RW', 'CF'],
            'LW': ['LB', 'AM', 'CF'],
            'RW': ['RB', 'AM', 'CF'],
            'CF': ['AM', 'LW', 'RW']
        }
        
        return compatibility_map.get(primary_position, [])
    
    def get_model_info(self) -> Dict:
        """Get Stage-2 model information."""
        return {
            'stage2_metadata': self.metadata,
            'ensemble_weights': self.ensemble_info['weights'],
            'validation_metrics': self.metadata['validation_metrics'],
            'compatibility_matrix_shape': self.compat_matrix.shape,
            'n_players_in_matrix': len(self.player_to_idx)
        }
    
    def predict_player_compatibility(self, player_id_a: str, 
                                   player_id_b: str) -> Optional[float]:
        """Get compatibility score between two specific players."""
        if (player_id_a not in self.player_to_idx or 
            player_id_b not in self.player_to_idx):
            return None
        
        idx_a = self.player_to_idx[player_id_a]
        idx_b = self.player_to_idx[player_id_b]
        
        return float(self.compat_matrix[idx_a, idx_b])
    
    def get_most_compatible_players(self, player_id: str, 
                                  n_players: int = 10) -> List[Tuple[str, float]]:
        """Get most compatible players for a given player."""
        if player_id not in self.player_to_idx:
            return []
        
        player_idx = self.player_to_idx[player_id]
        compatibilities = self.compat_matrix[player_idx, :]
        
        # Get top compatible players (excluding self)
        top_indices = np.argsort(compatibilities)[::-1]
        
        result = []
        for idx in top_indices:
            if len(result) >= n_players:
                break
            
            other_player_id = self.idx_to_player[idx]
            if other_player_id != player_id:
                result.append((other_player_id, float(compatibilities[idx])))
        
        return result


def main():
    """Demo usage of Stage-2 API."""
    setup_logging()
    
    try:
        # Initialize API
        api = Stage2API()
        
        # Load sample data
        players_df = pd.read_csv("data/processed/players.csv")
        features_df = pd.read_csv("data/processed/features.csv")
        
        # Get recent match data
        recent_features = features_df.head(25)  # Sample 25 players
        
        # Predict squad selection
        result = api.predict_squad_selection(
            recent_features, 
            formation="4-3-3",
            return_probabilities=True
        )
        
        print("\n=== Stage-2 Squad Selection Results ===")
        print(f"Formation: {result['formation']}")
        print(f"Total Score: {result['total_score']:.3f}")
        print(f"Selected Players: {len(result['selected_players'])}")
        
        print("\nPosition Assignments:")
        for player_id, position in result['position_assignments'].items():
            print(f"  {player_id}: {position}")
        
        # Show model info
        model_info = api.get_model_info()
        print(f"\nModel Performance:")
        print(f"  ROC-AUC: {model_info['validation_metrics']['roc_auc']:.3f}")
        print(f"  PR-AUC: {model_info['validation_metrics']['pr_auc']:.3f}")
        
        # Demo compatibility query
        if len(result['selected_players']) >= 2:
            player_a = result['selected_players'][0]
            player_b = result['selected_players'][1]
            compat_score = api.predict_player_compatibility(player_a, player_b)
            print(f"\nCompatibility between {player_a} and {player_b}: {compat_score:.3f}")
        
        logger.info("Stage-2 API demo completed successfully")
        
    except Exception as e:
        logger.error(f"Stage-2 API demo failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
