#!/usr/bin/env python3
"""
Compatibility matrix generation using Matrix Factorization.
Implements ALS (Alternating Least Squares) for player-player synergy modeling.
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
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
import mlflow

sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


class CompatibilityMatrixBuilder:
    """Build player compatibility matrix using Matrix Factorization."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.compat_config = self.config['stage2']['compatibility']
        self.random_seed = self.config['random_seeds']['global']
        np.random.seed(self.random_seed)
        
    def load_coplay_data(self, data_dir: str = "data/processed") -> pd.DataFrame:
        """Load co-play minutes data."""
        logger.info("Loading co-play data...")
        
        coplay_path = os.path.join(data_dir, "co_play_minutes.csv")
        coplay_df = pd.read_csv(coplay_path)
        
        # Load lineups for additional context
        lineups_path = os.path.join(data_dir, "lineups.csv")
        lineups_df = pd.read_csv(lineups_path)
        
        # Load events for synergy calculation
        events_path = os.path.join(data_dir, "events_aggregated.csv")
        events_df = pd.read_csv(events_path)
        
        logger.info(f"Loaded co-play data: {len(coplay_df)} records")
        return coplay_df, lineups_df, events_df
    
    def create_coplay_matrix(self, coplay_df: pd.DataFrame) -> Tuple[csr_matrix, Dict, Dict]:
        """Create sparse co-play matrix for Matrix Factorization."""
        logger.info("Creating co-play matrix...")
        
        # Get unique players
        all_players = set(coplay_df['player_id_a'].unique()) | set(coplay_df['player_id_b'].unique())
        player_to_idx = {player_id: idx for idx, player_id in enumerate(sorted(all_players))}
        idx_to_player = {idx: player_id for player_id, idx in player_to_idx.items()}
        
        n_players = len(all_players)
        logger.info(f"Matrix size: {n_players} x {n_players} players")
        
        # Create sparse matrix
        rows, cols, data = [], [], []
        
        for _, row in coplay_df.iterrows():
            player_a_idx = player_to_idx[row['player_id_a']]
            player_b_idx = player_to_idx[row['player_id_b']]
            
            # Use minutes together as interaction strength
            # Add small synergy bonus if available
            interaction_strength = row['minutes_together']
            if 'on_field_events_diff' in row and pd.notna(row['on_field_events_diff']):
                # Positive synergy increases interaction strength
                synergy_bonus = max(0, row['on_field_events_diff']) * 10
                interaction_strength += synergy_bonus
            
            # Add both directions (symmetric matrix)
            rows.extend([player_a_idx, player_b_idx])
            cols.extend([player_b_idx, player_a_idx])
            data.extend([interaction_strength, interaction_strength])
        
        # Create sparse matrix
        coplay_matrix = csr_matrix((data, (rows, cols)), shape=(n_players, n_players))
        
        logger.info(f"Created sparse matrix: {coplay_matrix.nnz} non-zero entries")
        logger.info(f"Sparsity: {(1 - coplay_matrix.nnz / (n_players ** 2)):.3f}")
        
        return coplay_matrix, player_to_idx, idx_to_player
    
    def train_matrix_factorization(self, coplay_matrix: csr_matrix) -> AlternatingLeastSquares:
        """Train Matrix Factorization model using ALS."""
        logger.info("Training Matrix Factorization model...")
        
        # Configure ALS model
        model = AlternatingLeastSquares(
            factors=self.compat_config['factors'],
            regularization=self.compat_config['regularization'],
            iterations=self.compat_config['iterations'],
            random_state=self.random_seed,
            use_gpu=False  # Set to True if GPU available
        )
        
        # Train model
        logger.info("Starting ALS training...")
        model.fit(coplay_matrix)
        
        logger.info("Matrix Factorization training completed")
        return model
    
    def compute_compatibility_scores(self, mf_model: AlternatingLeastSquares,
                                   player_to_idx: Dict, idx_to_player: Dict) -> pd.DataFrame:
        """Compute pairwise compatibility scores from trained MF model."""
        logger.info("Computing compatibility scores...")
        
        # Get player embeddings
        player_embeddings = mf_model.user_factors  # Players as "users"
        
        # Compute cosine similarity matrix
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(player_embeddings)
        
        # Convert to DataFrame format
        compatibility_data = []
        n_players = len(player_to_idx)
        
        for i in range(n_players):
            for j in range(i + 1, n_players):  # Upper triangle only
                player_a = idx_to_player[i]
                player_b = idx_to_player[j]
                compatibility_score = similarity_matrix[i, j]
                
                compatibility_data.append({
                    'player_id_a': player_a,
                    'player_id_b': player_b,
                    'compatibility_score': compatibility_score,
                    'embedding_distance': np.linalg.norm(
                        player_embeddings[i] - player_embeddings[j]
                    )
                })
        
        compatibility_df = pd.DataFrame(compatibility_data)
        
        # Add statistics
        logger.info(f"Compatibility scores - Mean: {compatibility_df['compatibility_score'].mean():.3f}")
        logger.info(f"Compatibility scores - Std: {compatibility_df['compatibility_score'].std():.3f}")
        
        return compatibility_df
    
    def enhance_with_positional_compatibility(self, compatibility_df: pd.DataFrame,
                                            players_df: pd.DataFrame) -> pd.DataFrame:
        """Enhance compatibility with positional synergy."""
        logger.info("Adding positional compatibility...")
        
        # Define position adjacency (which positions work well together)
        position_synergy = {
            'GK': {'CB': 0.8, 'LB': 0.3, 'RB': 0.3},
            'CB': {'GK': 0.8, 'CB': 0.9, 'LB': 0.7, 'RB': 0.7, 'DM': 0.8},
            'LB': {'CB': 0.7, 'LW': 0.8, 'CM': 0.6, 'DM': 0.5},
            'RB': {'CB': 0.7, 'RW': 0.8, 'CM': 0.6, 'DM': 0.5},
            'DM': {'CB': 0.8, 'CM': 0.9, 'AM': 0.7},
            'CM': {'DM': 0.9, 'CM': 0.8, 'AM': 0.8, 'LW': 0.5, 'RW': 0.5},
            'AM': {'CM': 0.8, 'CF': 0.8, 'LW': 0.7, 'RW': 0.7},
            'LW': {'LB': 0.8, 'CM': 0.5, 'AM': 0.7, 'CF': 0.6},
            'RW': {'RB': 0.8, 'CM': 0.5, 'AM': 0.7, 'CF': 0.6},
            'CF': {'AM': 0.8, 'LW': 0.6, 'RW': 0.6, 'CF': 0.4}
        }
        
        # Merge with player positions
        enhanced_compat = compatibility_df.merge(
            players_df[['player_id', 'primary_position']].rename(columns={'player_id': 'player_id_a', 'primary_position': 'position_a'}),
            on='player_id_a'
        ).merge(
            players_df[['player_id', 'primary_position']].rename(columns={'player_id': 'player_id_b', 'primary_position': 'position_b'}),
            on='player_id_b'
        )
        
        # Calculate positional synergy
        def get_positional_synergy(pos_a, pos_b):
            return position_synergy.get(pos_a, {}).get(pos_b, 0.1)  # Default low synergy
        
        enhanced_compat['positional_synergy'] = enhanced_compat.apply(
            lambda row: get_positional_synergy(row['position_a'], row['position_b']), axis=1
        )
        
        # Combine MF compatibility with positional synergy
        alpha = 0.7  # Weight for MF compatibility
        enhanced_compat['final_compatibility'] = (
            alpha * enhanced_compat['compatibility_score'] + 
            (1 - alpha) * enhanced_compat['positional_synergy']
        )
        
        logger.info("Enhanced compatibility with positional synergy")
        return enhanced_compat
    
    def create_full_compatibility_matrix(self, enhanced_compat: pd.DataFrame,
                                       player_to_idx: Dict) -> np.ndarray:
        """Create full symmetric compatibility matrix."""
        logger.info("Creating full compatibility matrix...")
        
        n_players = len(player_to_idx)
        compat_matrix = np.eye(n_players)  # Identity for self-compatibility
        
        # Fill matrix with compatibility scores
        for _, row in enhanced_compat.iterrows():
            idx_a = player_to_idx[row['player_id_a']]
            idx_b = player_to_idx[row['player_id_b']]
            score = row['final_compatibility']
            
            compat_matrix[idx_a, idx_b] = score
            compat_matrix[idx_b, idx_a] = score  # Symmetric
        
        # Fill missing entries with position-based defaults
        avg_compat_by_pos = enhanced_compat.groupby(['position_a', 'position_b'])['final_compatibility'].mean()
        
        for i in range(n_players):
            for j in range(i + 1, n_players):
                if compat_matrix[i, j] == 0:  # Missing entry
                    # Use average compatibility for position pair
                    compat_matrix[i, j] = 0.3  # Default moderate compatibility
                    compat_matrix[j, i] = 0.3
        
        logger.info(f"Full compatibility matrix created: {compat_matrix.shape}")
        return compat_matrix
    
    def save_compatibility_artifacts(self, compat_matrix: np.ndarray, 
                                   enhanced_compat: pd.DataFrame,
                                   mf_model: AlternatingLeastSquares,
                                   player_mappings: Dict,
                                   output_dir: str = "artifacts"):
        """Save all compatibility artifacts."""
        logger.info(f"Saving compatibility artifacts to {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save compatibility matrix
        matrix_path = os.path.join(output_dir, "compat_matrix_v1.npz")
        np.savez_compressed(matrix_path, 
                           compatibility_matrix=compat_matrix,
                           player_to_idx=player_mappings['player_to_idx'],
                           idx_to_player=player_mappings['idx_to_player'])
        
        # Save detailed compatibility scores
        compat_scores_path = os.path.join(output_dir, "compatibility_scores.csv")
        enhanced_compat.to_csv(compat_scores_path, index=False)
        
        # Save MF model
        mf_model_path = os.path.join(output_dir, "mf_model.pkl")
        joblib.dump(mf_model, mf_model_path)
        
        # Save metadata
        metadata = {
            'algorithm': self.compat_config['algorithm'],
            'factors': self.compat_config['factors'],
            'regularization': self.compat_config['regularization'],
            'iterations': self.compat_config['iterations'],
            'matrix_shape': compat_matrix.shape,
            'n_players': len(player_mappings['player_to_idx']),
            'sparsity': 1 - np.count_nonzero(compat_matrix) / compat_matrix.size,
            'training_date': pd.Timestamp.now().isoformat(),
            'random_seed': self.random_seed
        }
        
        metadata_path = os.path.join(output_dir, "compat_matrix_metadata.yaml")
        with open(metadata_path, 'w') as f:
            yaml.dump(metadata, f)
        
        logger.info("Compatibility artifacts saved successfully")
    
    def validate_compatibility_matrix(self, compat_matrix: np.ndarray,
                                    enhanced_compat: pd.DataFrame) -> Dict:
        """Validate compatibility matrix quality."""
        logger.info("Validating compatibility matrix...")
        
        validation_results = {
            'matrix_symmetric': np.allclose(compat_matrix, compat_matrix.T),
            'diagonal_ones': np.allclose(np.diag(compat_matrix), 1.0),
            'values_in_range': (compat_matrix >= 0).all() and (compat_matrix <= 1).all(),
            'mean_compatibility': compat_matrix.mean(),
            'std_compatibility': compat_matrix.std(),
            'sparsity': 1 - np.count_nonzero(compat_matrix) / compat_matrix.size
        }
        
        # Position-based validation
        position_stats = enhanced_compat.groupby(['position_a', 'position_b']).agg({
            'final_compatibility': ['mean', 'std', 'count']
        }).round(3)
        
        validation_results['position_compatibility_stats'] = position_stats
        
        # Log validation results
        for key, value in validation_results.items():
            if key != 'position_compatibility_stats':
                logger.info(f"Validation - {key}: {value}")
        
        return validation_results


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Build compatibility matrix")
    parser.add_argument('--data', default='data/processed',
                       help='Processed data directory')
    parser.add_argument('--output', default='artifacts',
                       help='Output directory')
    parser.add_argument('--config', default='configs/config.yaml',
                       help='Configuration file')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger.info("Starting compatibility matrix generation")
    
    try:
        # Initialize builder
        builder = CompatibilityMatrixBuilder(args.config)
        
        # Load data
        coplay_df, lineups_df, events_df = builder.load_coplay_data(args.data)
        
        # Load players data for position information
        players_df = pd.read_csv(os.path.join(args.data, "players.csv"))
        
        # Create co-play matrix
        coplay_matrix, player_to_idx, idx_to_player = builder.create_coplay_matrix(coplay_df)
        
        # Train Matrix Factorization
        mf_model = builder.train_matrix_factorization(coplay_matrix)
        
        # Compute compatibility scores
        compatibility_df = builder.compute_compatibility_scores(
            mf_model, player_to_idx, idx_to_player
        )
        
        # Enhance with positional compatibility
        enhanced_compat = builder.enhance_with_positional_compatibility(
            compatibility_df, players_df
        )
        
        # Create full matrix
        compat_matrix = builder.create_full_compatibility_matrix(
            enhanced_compat, player_to_idx
        )
        
        # Validate matrix
        validation_results = builder.validate_compatibility_matrix(
            compat_matrix, enhanced_compat
        )
        
        # Save artifacts
        player_mappings = {
            'player_to_idx': player_to_idx,
            'idx_to_player': idx_to_player
        }
        
        builder.save_compatibility_artifacts(
            compat_matrix, enhanced_compat, mf_model, player_mappings, args.output
        )
        
        # Log to MLflow
        with mlflow.start_run(run_name="compatibility_matrix_generation"):
            mlflow.log_params(builder.compat_config)
            mlflow.log_metrics({
                'n_players': len(player_to_idx),
                'matrix_sparsity': validation_results['sparsity'],
                'mean_compatibility': validation_results['mean_compatibility'],
                'std_compatibility': validation_results['std_compatibility']
            })
            mlflow.log_artifacts(args.output, artifact_path="compatibility")
        
        logger.info("Compatibility matrix generation completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Compatibility matrix generation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
