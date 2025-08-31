#!/usr/bin/env python3
"""
Feature engineering pipeline for football squad selection.
Creates time-series features, compatibility features, and opponent context.
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
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans

sys.path.append(str(Path(__file__).parent.parent))
from utils.logging_config import setup_logging
from utils.data_utils import create_feature_windows, impute_missing_values

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering pipeline for squad selection."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.feature_windows = self.config['data']['feature_windows']
        self.random_seed = self.config['random_seeds']['global']
        np.random.seed(self.random_seed)
        
    def load_processed_data(self, data_dir: str = "data/processed") -> Dict[str, pd.DataFrame]:
        """Load processed data files."""
        logger.info(f"Loading processed data from {data_dir}")
        
        required_files = [
            'players.csv', 'matches.csv', 'lineups.csv',
            'events_aggregated.csv', 'fitness_and_injury.csv', 'co_play_minutes.csv'
        ]
        
        data = {}
        for filename in required_files:
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                data[filename.replace('.csv', '')] = pd.read_csv(filepath)
                logger.info(f"Loaded {filename}: {len(data[filename.replace('.csv', '')])} rows")
            else:
                logger.warning(f"File not found: {filepath}")
        
        return data
    
    def create_performance_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create player performance features with time windows."""
        logger.info("Creating performance features...")
        
        events_df = data['events_aggregated']
        lineups_df = data['lineups']
        matches_df = data['matches']
        
        # Merge with match dates
        events_with_dates = events_df.merge(
            matches_df[['match_id', 'date']], on='match_id'
        )
        events_with_dates['date'] = pd.to_datetime(events_with_dates['date'])
        
        # Performance metrics to engineer
        perf_metrics = [
            'xG_sum', 'xA_sum', 'key_passes', 'progressive_passes',
            'pressures', 'tackles', 'duels_won', 'distance_covered'
        ]
        
        # Create windowed features
        features_list = []
        
        for window in [3, 5, 10]:
            window_data = events_with_dates[events_with_dates['window_last_n_matches'] == window]
            
            for metric in perf_metrics:
                if metric in window_data.columns:
                    # Basic statistics
                    features_list.append(
                        window_data.groupby('player_id')[metric].agg([
                            'mean', 'std', 'max', 'min'
                        ]).add_prefix(f"{metric}_w{window}_")
                    )
                    
                    # Rate features (per 90 minutes)
                    if 'minutes_total' in window_data.columns:
                        rate_feature = (window_data[metric] / window_data['minutes_total'] * 90)
                        features_list.append(
                            rate_feature.groupby(window_data['player_id']).agg([
                                'mean', 'std'
                            ]).add_prefix(f"{metric}_per90_w{window}_")
                        )
        
        # Combine all features
        performance_features = pd.concat(features_list, axis=1).fillna(0)
        performance_features.index.name = 'player_id'
        
        logger.info(f"Created performance features: {performance_features.shape}")
        return performance_features.reset_index()
    
    def create_fitness_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create fitness and availability features."""
        logger.info("Creating fitness features...")
        
        fitness_df = data['fitness_and_injury'].copy()
        fitness_df['date'] = pd.to_datetime(fitness_df['date'])
        
        # Encode injury status
        injury_encoder = LabelEncoder()
        fitness_df['injury_status_encoded'] = injury_encoder.fit_transform(
            fitness_df['injury_status'].fillna('unknown')
        )
        
        # Create fitness features
        fitness_features = fitness_df.groupby('player_id').agg({
            'days_since_last_match': ['mean', 'std', 'min', 'max'],
            'yellow_card_count_rolling': ['mean', 'max', 'sum'],
            'injury_status_encoded': 'mean',
            'sickness_flag': 'sum'
        }).round(3)
        
        # Flatten column names
        fitness_features.columns = [
            f"fitness_{col[0]}_{col[1]}" for col in fitness_features.columns
        ]
        
        # Add injury frequency
        injury_counts = fitness_df.groupby('player_id')['injury_status'].apply(
            lambda x: (x == 'injured').sum()
        )
        fitness_features['injury_frequency'] = injury_counts
        
        logger.info(f"Created fitness features: {fitness_features.shape}")
        return fitness_features.reset_index()
    
    def create_opponent_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create opponent and context features."""
        logger.info("Creating opponent features...")
        
        matches_df = data['matches'].copy()
        lineups_df = data['lineups'].copy()
        
        # Create team style clusters (simplified)
        # In practice, this would use detailed tactical data
        np.random.seed(self.random_seed)
        unique_teams = pd.concat([
            matches_df['home_team_id'], 
            matches_df['away_team_id']
        ]).unique()
        
        # Assign random style clusters for demo
        team_styles = pd.DataFrame({
            'team_id': unique_teams,
            'style_cluster': np.random.randint(0, 4, len(unique_teams)),
            'avg_possession': np.random.uniform(0.4, 0.7, len(unique_teams)),
            'avg_pressing_intensity': np.random.uniform(0.3, 0.8, len(unique_teams))
        })
        
        # Create opponent features for each match-player combination
        opponent_features = []
        
        for _, match in matches_df.iterrows():
            match_lineups = lineups_df[lineups_df['match_id'] == match['match_id']]
            
            for _, lineup in match_lineups.iterrows():
                # Determine opponent
                if lineup['team_id'] == match['home_team_id']:
                    opponent_id = match['away_team_id']
                    is_home = 1
                else:
                    opponent_id = match['home_team_id']
                    is_home = 0
                
                # Get opponent style
                opponent_style = team_styles[team_styles['team_id'] == opponent_id]
                
                if len(opponent_style) > 0:
                    opponent_features.append({
                        'match_id': match['match_id'],
                        'player_id': lineup['player_id'],
                        'opponent_team_id': opponent_id,
                        'is_home': is_home,
                        'opponent_style_cluster': opponent_style['style_cluster'].iloc[0],
                        'opponent_possession': opponent_style['avg_possession'].iloc[0],
                        'opponent_pressing': opponent_style['avg_pressing_intensity'].iloc[0],
                        'venue_familiar': 1 if is_home else 0
                    })
        
        opponent_df = pd.DataFrame(opponent_features)
        logger.info(f"Created opponent features: {opponent_df.shape}")
        return opponent_df
    
    def create_position_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create position and role features."""
        logger.info("Creating position features...")
        
        players_df = data['players'].copy()
        lineups_df = data['lineups'].copy()
        
        # Position encoding
        all_positions = ['GK', 'CB', 'LB', 'RB', 'DM', 'CM', 'AM', 'LW', 'RW', 'CF']
        
        # One-hot encode primary position
        for pos in all_positions:
            players_df[f"pos_primary_{pos}"] = (players_df['primary_position'] == pos).astype(int)
        
        # Position versatility (number of secondary positions)
        players_df['position_versatility'] = players_df['secondary_positions'].apply(
            lambda x: len(eval(x)) if pd.notna(x) and x != '[]' else 0
        )
        
        # Playing time by position
        position_minutes = lineups_df.groupby(['player_id', 'position_label'])['minutes_played'].sum()
        position_experience = position_minutes.unstack(fill_value=0)
        position_experience.columns = [f"exp_{col}_minutes" for col in position_experience.columns]
        
        # Merge position features
        position_features = players_df.merge(
            position_experience.reset_index(), on='player_id', how='left'
        ).fillna(0)
        
        # Select relevant columns
        feature_cols = ['player_id'] + [col for col in position_features.columns 
                                       if col.startswith(('pos_', 'exp_')) or col == 'position_versatility']
        
        logger.info(f"Created position features: {len(feature_cols)-1} features")
        return position_features[feature_cols]
    
    def create_compatibility_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create basic compatibility features (full matrix in Stage-2)."""
        logger.info("Creating compatibility features...")
        
        co_play_df = data['co_play_minutes'].copy()
        
        # Aggregate co-play statistics per player
        player_compatibility = []
        
        for player_id in co_play_df['player_id_a'].unique():
            # Get all co-play records for this player
            player_coplay_a = co_play_df[co_play_df['player_id_a'] == player_id]
            player_coplay_b = co_play_df[co_play_df['player_id_b'] == player_id]
            
            # Combine both directions
            total_coplay_minutes = (
                player_coplay_a['minutes_together'].sum() + 
                player_coplay_b['minutes_together'].sum()
            )
            
            unique_partners = len(set(
                list(player_coplay_a['player_id_b']) + 
                list(player_coplay_b['player_id_a'])
            ))
            
            avg_synergy = np.concatenate([
                player_coplay_a['on_field_events_diff'].fillna(0),
                player_coplay_b['on_field_events_diff'].fillna(0)
            ]).mean()
            
            player_compatibility.append({
                'player_id': player_id,
                'total_coplay_minutes': total_coplay_minutes,
                'unique_partners': unique_partners,
                'avg_synergy_score': avg_synergy,
                'coplay_experience': total_coplay_minutes / 90  # matches equivalent
            })
        
        compatibility_df = pd.DataFrame(player_compatibility)
        logger.info(f"Created compatibility features: {compatibility_df.shape}")
        return compatibility_df
    
    def create_target_variables(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create target variables for Stage-1 and Stage-2."""
        logger.info("Creating target variables...")
        
        lineups_df = data['lineups'].copy()
        matches_df = data['matches'].copy()
        
        # Stage-1 target: selected to squad (starting or bench)
        squad_selection = lineups_df.groupby(['match_id', 'player_id']).agg({
            'is_starting': 'max',  # 1 if starting
            'minutes_played': 'sum'
        }).reset_index()
        
        squad_selection['selected_to_squad'] = (squad_selection['minutes_played'] > 0).astype(int)
        
        # Stage-2 target: lineup quality score (simplified)
        # In practice, this would be based on match outcomes and performance
        lineup_quality = []
        
        for match_id in lineups_df['match_id'].unique():
            match_lineups = lineups_df[lineups_df['match_id'] == match_id]
            
            for team_id in match_lineups['team_id'].unique():
                team_lineup = match_lineups[match_lineups['team_id'] == team_id]
                starting_xi = team_lineup[team_lineup['is_starting'] == 1]
                
                if len(starting_xi) == 11:  # Valid lineup
                    # Synthetic quality score based on player performance
                    quality_score = np.random.uniform(0.4, 0.9)  # Placeholder
                    
                    lineup_quality.append({
                        'match_id': match_id,
                        'team_id': team_id,
                        'lineup_quality_score': quality_score,
                        'formation_valid': 1
                    })
        
        lineup_quality_df = pd.DataFrame(lineup_quality)
        
        targets = {
            'squad_selection': squad_selection,
            'lineup_quality': lineup_quality_df
        }
        
        logger.info(f"Created target variables: {len(targets)} target types")
        return targets
    
    def combine_all_features(self, feature_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine all feature sets into final feature matrix."""
        logger.info("Combining all features...")
        
        # Start with performance features as base
        combined_features = feature_dfs['performance'].copy()
        
        # Merge other feature sets
        for feature_name, feature_df in feature_dfs.items():
            if feature_name != 'performance' and 'player_id' in feature_df.columns:
                combined_features = combined_features.merge(
                    feature_df, on='player_id', how='left'
                )
                logger.info(f"Merged {feature_name} features")
        
        # Fill missing values
        combined_features = impute_missing_values(
            combined_features, 
            strategy='median',
            group_by='player_id'
        )
        
        logger.info(f"Final feature matrix: {combined_features.shape}")
        return combined_features
    
    def save_features(self, features: Dict[str, pd.DataFrame], 
                     output_dir: str = "data/features"):
        """Save engineered features."""
        logger.info(f"Saving features to {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        for feature_name, feature_df in features.items():
            filename = f"{feature_name}_features.csv"
            filepath = os.path.join(output_dir, filename)
            feature_df.to_csv(filepath, index=False)
            logger.info(f"Saved {filename}: {feature_df.shape}")
        
        # Save feature catalog
        self._create_feature_catalog(features, output_dir)
    
    def _create_feature_catalog(self, features: Dict[str, pd.DataFrame], output_dir: str):
        """Create feature catalog documenting all features."""
        catalog_data = []
        
        for feature_set, df in features.items():
            for col in df.columns:
                if col != 'player_id':
                    catalog_data.append({
                        'feature_name': col,
                        'feature_set': feature_set,
                        'description': self._get_feature_description(col),
                        'data_type': str(df[col].dtype),
                        'missing_rate': df[col].isnull().mean(),
                        'min_value': df[col].min() if df[col].dtype in ['int64', 'float64'] else None,
                        'max_value': df[col].max() if df[col].dtype in ['int64', 'float64'] else None,
                        'unique_values': df[col].nunique()
                    })
        
        catalog_df = pd.DataFrame(catalog_data)
        catalog_path = os.path.join(output_dir, "features_catalog.csv")
        catalog_df.to_csv(catalog_path, index=False)
        logger.info(f"Saved feature catalog: {catalog_path}")
    
    def _get_feature_description(self, feature_name: str) -> str:
        """Get human-readable description for feature."""
        descriptions = {
            'xG_sum': 'Expected goals in time window',
            'xA_sum': 'Expected assists in time window',
            'key_passes': 'Key passes leading to shots',
            'progressive_passes': 'Forward progressive passes',
            'pressures': 'Defensive pressures applied',
            'tackles': 'Successful tackles',
            'duels_won': 'Duels won (aerial and ground)',
            'distance_covered': 'Distance covered per match (km)',
            'minutes_played': 'Total minutes played',
            'is_home': 'Playing at home venue (1) or away (0)',
            'opponent_style_cluster': 'Opponent tactical style cluster (0-3)',
            'position_versatility': 'Number of positions player can play',
            'injury_frequency': 'Historical injury frequency',
            'coplay_experience': 'Experience playing with teammates'
        }
        
        # Check for pattern matches
        for pattern, desc in descriptions.items():
            if pattern in feature_name:
                return desc
        
        # Default description
        if '_mean_' in feature_name:
            return f"Mean {feature_name.split('_')[0]} over time window"
        elif '_std_' in feature_name:
            return f"Standard deviation of {feature_name.split('_')[0]}"
        elif '_per90_' in feature_name:
            return f"Per-90-minute rate of {feature_name.split('_')[0]}"
        elif 'pos_primary_' in feature_name:
            return f"Primary position is {feature_name.split('_')[-1]}"
        
        return f"Feature: {feature_name}"


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Engineer features for squad selection")
    parser.add_argument('--input', default='data/processed',
                       help='Input directory with processed data')
    parser.add_argument('--output', default='data/features',
                       help='Output directory for features')
    parser.add_argument('--config', default='configs/config.yaml',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger.info("Starting feature engineering pipeline")
    
    try:
        # Initialize feature engineer
        engineer = FeatureEngineer(args.config)
        
        # Load processed data
        data = engineer.load_processed_data(args.input)
        
        if not data:
            logger.error("No processed data found")
            return 1
        
        # Create different feature sets
        feature_sets = {}
        
        # Performance features
        feature_sets['performance'] = engineer.create_performance_features(data)
        
        # Fitness features
        feature_sets['fitness'] = engineer.create_fitness_features(data)
        
        # Opponent features
        feature_sets['opponent'] = engineer.create_opponent_features(data)
        
        # Position features
        feature_sets['position'] = engineer.create_position_features(data)
        
        # Compatibility features
        feature_sets['compatibility'] = engineer.create_compatibility_features(data)
        
        # Create targets
        targets = engineer.create_target_variables(data)
        feature_sets.update(targets)
        
        # Combine main features for Stage-1
        main_features = engineer.combine_all_features({
            k: v for k, v in feature_sets.items() 
            if k in ['performance', 'fitness', 'opponent', 'position', 'compatibility']
        })
        feature_sets['stage1_features'] = main_features
        
        # Save all features
        engineer.save_features(feature_sets, args.output)
        
        logger.info("Feature engineering completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
