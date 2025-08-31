# Data Schema Test Specifications

## Overview
Comprehensive tests to validate data integrity, schema compliance, and temporal consistency across all data tables.

## Core Schema Tests

### 1. Players Table Tests
**File**: `data/processed/players.csv`

**Required Columns**:
- `player_id` (string/int, unique, not null)
- `player_name` (string, not null)
- `dob` (date, YYYY-MM-DD format)
- `primary_position` (string, valid position codes)
- `secondary_positions` (list/string, valid position codes)
- `country` (string, ISO country codes)
- `height_cm` (float, range 150-220)
- `weight_kg` (float, range 50-120, optional)

**Validation Rules**:
- No duplicate player_ids
- DOB must be realistic (1980-2010 range)
- Position codes must be valid: GK, CB, LB, RB, DM, CM, AM, LW, RW, CF
- Height/weight within human ranges
- Country codes must be valid ISO 3166-1

### 2. Matches Table Tests
**File**: `data/processed/matches.csv`

**Required Columns**:
- `match_id` (string/int, unique, not null)
- `season` (string, format: YYYY-YY)
- `date` (date, YYYY-MM-DD)
- `competition` (string, not null)
- `home_team_id` (string/int, not null)
- `away_team_id` (string/int, not null)
- `venue` (string, optional)
- `referee_id` (string/int, optional)
- `weather` (string, optional)

**Validation Rules**:
- No duplicate match_ids
- Date must be within reasonable range (2015-2025)
- home_team_id ≠ away_team_id
- Season format must match date year
- Competition names must be consistent

### 3. Lineups Table Tests
**File**: `data/processed/lineups.csv`

**Required Columns**:
- `match_id` (foreign key to matches)
- `team_id` (string/int, not null)
- `player_id` (foreign key to players)
- `is_starting` (binary, 0/1)
- `position_label` (string, valid positions)
- `minutes_played` (int, 0-120 range)
- `sub_in_minute` (int, 0-120, optional)
- `sub_out_minute` (int, 0-120, optional)

**Validation Rules**:
- Foreign key integrity (match_id, player_id exist)
- Exactly 11 starting players per team per match
- Minutes played ≤ 90 for most players
- sub_in_minute < sub_out_minute when both present
- Position labels must be valid
- No player in multiple positions simultaneously

### 4. Events Aggregated Table Tests
**File**: `data/processed/events_aggregated_per_player_windowed.csv`

**Required Columns**:
- `match_id` (foreign key)
- `player_id` (foreign key)
- `window_last_n_matches` (int, 3/5/10)
- `xG_sum` (float, ≥0)
- `xA_sum` (float, ≥0)
- `key_passes` (int, ≥0)
- `progressive_passes` (int, ≥0)
- `pressures` (int, ≥0)
- `tackles` (int, ≥0)
- `duels_won` (int, ≥0)
- `distance_covered` (float, ≥0, optional)
- `minutes_total` (int, ≥0)

**Validation Rules**:
- All performance metrics ≥ 0
- xG_sum reasonable range (0-10 per window)
- minutes_total ≤ window_size * 90
- Foreign key integrity maintained

### 5. Fitness and Injury Table Tests
**File**: `data/processed/fitness_and_injury.csv`

**Required Columns**:
- `player_id` (foreign key)
- `date` (date, YYYY-MM-DD)
- `injury_status` (enum: healthy/injured/unknown)
- `injury_type` (string, optional)
- `days_since_last_match` (int, ≥0)
- `yellow_card_count_rolling` (int, ≥0)
- `sickness_flag` (binary, 0/1)

**Validation Rules**:
- Date must be valid and within data range
- injury_status must be valid enum value
- days_since_last_match reasonable (0-365)
- yellow_card_count_rolling ≤ 10 (reasonable limit)

## Temporal Consistency Tests

### 1. Chronological Order
- Match dates in ascending order within seasons
- Player features computed only from past matches
- No future information leakage in windowed features

### 2. Data Availability
- Player must exist before match participation
- Injury status must be recorded before/on match date
- Feature windows must have sufficient historical data

### 3. Lineup Consistency
- Starting XI must sum to exactly 11 players per team
- Substitutions must respect timing constraints
- Minutes played must be consistent with substitution data

## Data Quality Tests

### 1. Completeness
- Missing data rate < 30% for critical columns
- Complete lineups for all matches
- Player biographical data completeness > 90%

### 2. Accuracy
- Performance metrics within expected ranges
- No impossible values (negative minutes, etc.)
- Statistical outliers flagged and reviewed

### 3. Consistency
- Player names consistent across tables
- Team IDs consistent across seasons
- Position labels standardized

## Leakage Detection Tests

### 1. Temporal Leakage
- Feature computation cutoff before match date
- No post-match statistics in pre-match features
- Proper temporal splits in cross-validation

### 2. Data Leakage
- No target variable information in features
- No selection outcome in Stage-1 features
- No lineup information in player-level features

## Performance Tests

### 1. Data Loading
- CSV files load without errors
- Memory usage within limits
- Loading time < 5 minutes for full dataset

### 2. Feature Engineering
- Feature computation completes successfully
- No infinite or NaN values in features
- Feature distributions within expected ranges

## Test Implementation

### Automated Tests
```python
# Example test structure
def test_players_schema():
    """Test players.csv schema compliance."""
    df = pd.read_csv('data/processed/players.csv')
    assert 'player_id' in df.columns
    assert df['player_id'].is_unique
    assert df['height_cm'].between(150, 220).all()

def test_temporal_consistency():
    """Test no future information leakage."""
    # Implementation details...
    pass

def test_lineup_constraints():
    """Test lineup formation constraints."""
    # Implementation details...
    pass
```

### Manual Validation
- Visual inspection of sample records
- Statistical summaries and distributions
- Cross-table join verification

## Error Handling

### Common Issues
- **Missing Files**: Graceful handling with clear error messages
- **Schema Violations**: Detailed reporting of failed validations
- **Data Type Errors**: Automatic type inference with warnings
- **Encoding Issues**: UTF-8 handling for international player names

### Recovery Strategies
- **Imputation**: Position-based medians for missing performance data
- **Fallbacks**: Default values for optional fields
- **Filtering**: Remove records that fail critical validations
- **Logging**: Comprehensive logging of all data issues

## Reporting

### Test Results Format
```
Data Schema Validation Report
Generated: 2025-08-30 23:30:10

PASSED: 45/50 tests
FAILED: 5/50 tests
WARNINGS: 12

Critical Issues:
- players.csv: 3 duplicate player_ids found
- lineups.csv: 2 matches with incorrect starting XI count

Warnings:
- fitness_and_injury.csv: 15% missing injury_type values
- events_aggregated.csv: 5 outlier xG values detected

Recommendations:
1. Clean duplicate player records
2. Verify lineup data for affected matches
3. Review outlier xG values for data quality
```

### Success Criteria
- All critical tests pass (100%)
- Warning rate < 20%
- No data leakage detected
- Temporal consistency verified
- Foreign key integrity maintained
