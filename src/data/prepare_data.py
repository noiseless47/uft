#!/usr/bin/env python3
"""
Data preparation pipeline for football squad selection.
Handles data collection, cleaning, and initial processing.
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
import hashlib

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.logging_config import setup_logging
from utils.data_utils import validate_schema, compute_checksum

logger = logging.getLogger(__name__)


class DataPreparer:
    """Main data preparation pipeline."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_config = self.config['data']
        self.random_seed = self.config['random_seeds']['global']
        np.random.seed(self.random_seed)
        
    def collect_data(self, sources: List[str], synthetic: bool = False) -> Dict[str, pd.DataFrame]:
        """Collect data from specified sources."""
        logger.info(f"Collecting data from sources: {sources}")
        
        if synthetic:
            return self._generate_synthetic_data()
        
        data = {}
        
        if 'statsbomb' in sources:
            data.update(self._collect_statsbomb_data())
        
        if 'fbref' in sources:
            data.update(self._collect_fbref_data())
            
        if 'fivethirtyeight' in sources:
            data.update(self._collect_fivethirtyeight_data())
            
        return data
    
    def _generate_synthetic_data(self, n_matches: int = 1000, n_players: int = 500) -> Dict[str, pd.DataFrame]:
        """Generate synthetic data for testing and demos."""
        logger.info(f"Generating synthetic data: {n_matches} matches, {n_players} players")
        
        # Generate players
        positions = ['GK', 'CB', 'LB', 'RB', 'DM', 'CM', 'AM', 'LW', 'RW', 'CF']
        countries = ['England', 'Spain', 'Germany', 'France', 'Italy', 'Brazil', 'Argentina']
        
        players = pd.DataFrame({
            'player_id': range(1, n_players + 1),
            'player_name': [f"Player_{i}" for i in range(1, n_players + 1)],
            'dob': pd.date_range('1990-01-01', '2005-12-31', periods=n_players),
            'primary_position': np.random.choice(positions, n_players),
            'secondary_positions': [
                str(np.random.choice(positions, np.random.randint(0, 3)).tolist()) 
                for _ in range(n_players)
            ],
            'country': np.random.choice(countries, n_players),
            'height_cm': np.random.normal(180, 8, n_players).clip(160, 200),
            'weight_kg': np.random.normal(75, 8, n_players).clip(60, 95)
        })
        
        # Generate teams
        n_teams = 20
        teams = pd.DataFrame({
            'team_id': range(1, n_teams + 1),
            'team_name': [f"Team_{i}" for i in range(1, n_teams + 1)]
        })
        
        # Generate matches
        start_date = datetime(2023, 8, 1)
        matches = []
        
        for i in range(n_matches):
            home_team = np.random.randint(1, n_teams + 1)
            away_team = np.random.choice([t for t in range(1, n_teams + 1) if t != home_team])
            
            matches.append({
                'match_id': i + 1,
                'season': '2023-24',
                'date': (start_date + timedelta(days=i // 10)).strftime('%Y-%m-%d'),
                'competition': 'Premier League',
                'home_team_id': home_team,
                'away_team_id': away_team,
                'venue': f"Stadium_{home_team}",
                'referee_id': f"REF_{np.random.randint(1, 50)}",
                'weather': np.random.choice(['Clear', 'Rain', 'Cloudy', None])
            })
        
        matches_df = pd.DataFrame(matches)
        
        # Generate lineups
        lineups = []
        for match_id in range(1, n_matches + 1):
            match = matches_df[matches_df['match_id'] == match_id].iloc[0]
            
            for team_id in [match['home_team_id'], match['away_team_id']]:
                # Select 18 players for squad
                available_players = np.random.choice(n_players, 18, replace=False) + 1
                
                # Starting XI
                starting_xi = available_players[:11]
                bench = available_players[11:]
                
                # Add starting XI
                for i, player_id in enumerate(starting_xi):
                    position = np.random.choice(positions[1:])  # No GK for now
                    if i == 0:  # First player is GK
                        position = 'GK'
                    
                    lineups.append({
                        'match_id': match_id,
                        'team_id': team_id,
                        'player_id': int(player_id),
                        'is_starting': 1,
                        'position_label': position,
                        'minutes_played': np.random.randint(60, 91),
                        'sub_in_minute': None,
                        'sub_out_minute': np.random.choice([None, np.random.randint(60, 90)], p=[0.7, 0.3])
                    })
                
                # Add bench
                for player_id in bench:
                    sub_in = np.random.choice([None, np.random.randint(45, 85)], p=[0.6, 0.4])
                    minutes = np.random.randint(0, 45) if sub_in else 0
                    
                    lineups.append({
                        'match_id': match_id,
                        'team_id': team_id,
                        'player_id': int(player_id),
                        'is_starting': 0,
                        'position_label': np.random.choice(positions[1:]),
                        'minutes_played': minutes,
                        'sub_in_minute': sub_in,
                        'sub_out_minute': None
                    })
        
        lineups_df = pd.DataFrame(lineups)
        
        # Generate aggregated events
        events_agg = []
        for _, lineup in lineups_df.iterrows():
            if lineup['minutes_played'] > 0:
                for window in [3, 5, 10]:
                    events_agg.append({
                        'match_id': lineup['match_id'],
                        'player_id': lineup['player_id'],
                        'window_last_n_matches': window,
                        'xG_sum': np.random.exponential(0.5),
                        'xA_sum': np.random.exponential(0.3),
                        'key_passes': np.random.poisson(2),
                        'progressive_passes': np.random.poisson(5),
                        'pressures': np.random.poisson(8),
                        'tackles': np.random.poisson(3),
                        'duels_won': np.random.poisson(4),
                        'distance_covered': np.random.normal(10, 2).clip(5, 15),
                        'minutes_total': window * np.random.randint(60, 91)
                    })
        
        events_df = pd.DataFrame(events_agg)
        
        # Generate fitness data
        fitness_data = []
        for player_id in range(1, n_players + 1):
            for match_id in range(1, min(n_matches + 1, 100)):  # Limit for synthetic
                fitness_data.append({
                    'player_id': player_id,
                    'date': matches_df[matches_df['match_id'] == match_id]['date'].iloc[0],
                    'injury_status': np.random.choice(['healthy', 'injured', 'unknown'], p=[0.85, 0.1, 0.05]),
                    'injury_type': np.random.choice([None, 'muscle', 'joint', 'other'], p=[0.9, 0.05, 0.03, 0.02]),
                    'days_since_last_match': np.random.randint(0, 14),
                    'yellow_card_count_rolling': np.random.poisson(0.5),
                    'sickness_flag': np.random.choice([0, 1], p=[0.95, 0.05])
                })
        
        fitness_df = pd.DataFrame(fitness_data)
        
        # Generate co-play data
        co_play_data = []
        for match_id in range(1, min(n_matches + 1, 100)):
            match_lineups = lineups_df[lineups_df['match_id'] == match_id]
            starting_players = match_lineups[match_lineups['is_starting'] == 1]['player_id'].values
            
            # Generate pairs of starting players
            for i, player_a in enumerate(starting_players):
                for player_b in starting_players[i+1:]:
                    minutes_together = np.random.randint(45, 91)
                    co_play_data.append({
                        'player_id_a': player_a,
                        'player_id_b': player_b,
                        'match_id': match_id,
                        'minutes_together': minutes_together,
                        'on_field_events_diff': np.random.normal(0, 1)
                    })
        
        co_play_df = pd.DataFrame(co_play_data)
        
        return {
            'players': players,
            'matches': matches_df,
            'lineups': lineups_df,
            'events_aggregated': events_df,
            'fitness_and_injury': fitness_df,
            'co_play_minutes': co_play_df
        }
    
    def _collect_statsbomb_data(self) -> Dict[str, pd.DataFrame]:
        """Collect data from StatsBomb Open Data."""
        try:
            from statsbombpy import sb
            logger.info("Collecting StatsBomb data...")
            
            # Get competitions
            competitions = sb.competitions()
            
            # Focus on major leagues
            major_comps = competitions[
                competitions['competition_name'].isin([
                    'Premier League', 'La Liga', 'Serie A', 'Bundesliga', 'Ligue 1'
                ])
            ]
            
            all_matches = []
            all_lineups = []
            
            for _, comp in major_comps.head(2).iterrows():  # Limit for demo
                try:
                    matches = sb.matches(
                        competition_id=comp['competition_id'],
                        season_id=comp['season_id']
                    )
                    all_matches.append(matches)
                    
                    # Get lineups for sample of matches
                    for match_id in matches['match_id'].head(10):
                        try:
                            lineup = sb.lineups(match_id=match_id)
                            # Process lineup data
                            for team_id, team_lineup in lineup.items():
                                team_lineup['match_id'] = match_id
                                team_lineup['team_id'] = team_id
                                all_lineups.append(team_lineup)
                        except Exception as e:
                            logger.warning(f"Failed to get lineup for match {match_id}: {e}")
                            
                except Exception as e:
                    logger.warning(f"Failed to get data for competition {comp['competition_name']}: {e}")
            
            matches_df = pd.concat(all_matches, ignore_index=True) if all_matches else pd.DataFrame()
            lineups_df = pd.concat(all_lineups, ignore_index=True) if all_lineups else pd.DataFrame()
            
            return {
                'matches_statsbomb': matches_df,
                'lineups_statsbomb': lineups_df
            }
            
        except ImportError:
            logger.warning("statsbombpy not available, skipping StatsBomb data")
            return {}
        except Exception as e:
            logger.error(f"Error collecting StatsBomb data: {e}")
            return {}
    
    def _collect_fbref_data(self) -> Dict[str, pd.DataFrame]:
        """Collect data from FBref."""
        try:
            import soccerdata as sd
            logger.info("Collecting FBref data...")
            
            fbref = sd.FBref()
            
            # Get recent season data
            leagues = ['ENG-Premier League', 'ESP-La Liga']
            seasons = ['2022-23', '2023-24']
            
            all_matches = []
            for league in leagues:
                for season in seasons:
                    try:
                        matches = fbref.read_matches(league, season)
                        matches['competition'] = league
                        matches['season'] = season
                        all_matches.append(matches)
                    except Exception as e:
                        logger.warning(f"Failed to get {league} {season}: {e}")
            
            matches_df = pd.concat(all_matches, ignore_index=True) if all_matches else pd.DataFrame()
            
            return {'matches_fbref': matches_df}
            
        except ImportError:
            logger.warning("soccerdata not available, skipping FBref data")
            return {}
        except Exception as e:
            logger.error(f"Error collecting FBref data: {e}")
            return {}
    
    def _collect_fivethirtyeight_data(self) -> Dict[str, pd.DataFrame]:
        """Collect FiveThirtyEight SPI data."""
        try:
            import requests
            logger.info("Collecting FiveThirtyEight data...")
            
            url = "https://projects.fivethirtyeight.com/soccer-api/club/spi_matches.csv"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            from io import StringIO
            spi_data = pd.read_csv(StringIO(response.text))
            
            return {'spi_matches': spi_data}
            
        except Exception as e:
            logger.error(f"Error collecting FiveThirtyEight data: {e}")
            return {}
    
    def clean_and_merge_data(self, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Clean and merge data from different sources."""
        logger.info("Cleaning and merging data...")
        
        # For synthetic data, return as-is with minor cleaning
        if 'players' in raw_data:
            return self._clean_synthetic_data(raw_data)
        
        # For real data, implement proper merging logic
        return self._clean_real_data(raw_data)
    
    def _clean_synthetic_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Clean synthetic data."""
        cleaned = {}
        
        for table_name, df in data.items():
            # Basic cleaning
            df_clean = df.copy()
            
            # Handle missing values
            if 'height_cm' in df_clean.columns:
                df_clean['height_cm'] = df_clean['height_cm'].fillna(df_clean['height_cm'].median())
            
            if 'weight_kg' in df_clean.columns:
                df_clean['weight_kg'] = df_clean['weight_kg'].fillna(df_clean['weight_kg'].median())
            
            # Ensure data types
            if 'date' in df_clean.columns:
                df_clean['date'] = pd.to_datetime(df_clean['date'])
            
            cleaned[table_name] = df_clean
            logger.info(f"Cleaned {table_name}: {len(df_clean)} rows")
        
        return cleaned
    
    def _clean_real_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Clean and merge real data sources."""
        # Implementation for real data merging
        # This would include ID mapping, deduplication, etc.
        logger.info("Real data cleaning not implemented yet - using synthetic data")
        return {}
    
    def save_processed_data(self, data: Dict[str, pd.DataFrame], output_dir: str = "data/processed"):
        """Save processed data with manifest."""
        logger.info(f"Saving processed data to {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        manifest_data = []
        
        for table_name, df in data.items():
            filename = f"{table_name}.csv"
            filepath = os.path.join(output_dir, filename)
            
            # Save CSV
            df.to_csv(filepath, index=False)
            
            # Compute metadata
            checksum = compute_checksum(filepath)
            file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
            
            manifest_data.append({
                'filename': filename,
                'checksum': f"sha256:{checksum}",
                'row_count': len(df),
                'created_date': datetime.now().strftime('%Y-%m-%d'),
                'description': f"Processed {table_name} data",
                'source': 'synthetic' if 'players' in data else 'mixed',
                'file_size_mb': round(file_size, 2)
            })
            
            logger.info(f"Saved {filename}: {len(df)} rows, {file_size:.1f}MB")
        
        # Save manifest
        manifest_df = pd.DataFrame(manifest_data)
        manifest_path = os.path.join("data/manifests", "processed_data_manifest.csv")
        os.makedirs("data/manifests", exist_ok=True)
        manifest_df.to_csv(manifest_path, index=False)
        
        logger.info(f"Saved data manifest: {manifest_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Prepare football data for modeling")
    parser.add_argument('--sources', default='statsbomb,fbref', 
                       help='Data sources (comma-separated)')
    parser.add_argument('--synthetic', action='store_true',
                       help='Generate synthetic data instead')
    parser.add_argument('--n_matches', type=int, default=1000,
                       help='Number of synthetic matches')
    parser.add_argument('--n_players', type=int, default=500,
                       help='Number of synthetic players')
    parser.add_argument('--output', default='data/processed',
                       help='Output directory')
    parser.add_argument('--config', default='configs/config.yaml',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger.info("Starting data preparation pipeline")
    
    try:
        # Initialize preparer
        preparer = DataPreparer(args.config)
        
        # Collect data
        sources = args.sources.split(',') if not args.synthetic else []
        raw_data = preparer.collect_data(
            sources=sources, 
            synthetic=args.synthetic
        )
        
        if not raw_data:
            logger.error("No data collected")
            return 1
        
        # Clean and merge
        processed_data = preparer.clean_and_merge_data(raw_data)
        
        # Save processed data
        preparer.save_processed_data(processed_data, args.output)
        
        logger.info("Data preparation completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Data preparation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
