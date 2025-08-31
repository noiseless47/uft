# Data Access Instructions

## Overview

This project requires three layers of football data for optimal performance:
1. **Core Match/Event Data** (10k+ matches for pre-training)
2. **Player Metadata & Fitness** (biographical and availability data)
3. **Contextual/Compatibility Data** (lineups, co-play matrices)

## Data Sources & Access

### 1. Open Source Data (Recommended for Reproduction)

#### StatsBomb Open Data
- **Access**: Free registration at https://github.com/statsbomb/open-data
- **Coverage**: ~6,000 matches across major competitions
- **Installation**:
  ```bash
  pip install statsbombpy
  ```
- **Usage**:
  ```python
  from statsbombpy import sb
  matches = sb.matches(competition_id=11, season_id=1)  # La Liga 2019/20
  events = sb.events(match_id=15946)
  ```
- **Expected Files**: Place in `data/raw/statsbomb/`
- **License**: CC BY-NC-SA 4.0 (Non-commercial use only)

#### FBref Data
- **Access**: https://fbref.com/ (web scraping with rate limits)
- **Installation**:
  ```bash
  pip install soccerdata
  ```
- **Usage**:
  ```python
  import soccerdata as sd
  fbref = sd.FBref()
  matches = fbref.read_matches("ENG-Premier League", "2023-24")
  ```
- **Expected Files**: `data/raw/fbref/matches_*.csv`, `data/raw/fbref/players_*.csv`
- **License**: Subject to Sports Reference Terms of Use

#### FiveThirtyEight SPI
- **Access**: https://github.com/fivethirtyeight/data/tree/master/soccer-spi
- **Direct Download**:
  ```bash
  wget https://projects.fivethirtyeight.com/soccer-api/club/spi_matches.csv
  ```
- **Expected Files**: `data/raw/fivethirtyeight/spi_matches.csv`
- **License**: CC BY 4.0

### 2. Commercial Data Sources (Optional, Enhanced Performance)

#### Opta Sports Data
- **Contact**: sales@optasports.com
- **Coverage**: Comprehensive event data, player tracking
- **Cost**: €10,000-50,000+ per season/league
- **Delivery**: API access or data files
- **Expected Schema**: Opta F24 XML format or JSON API
- **Placement**: `data/raw/opta/`

#### Wyscout
- **Contact**: info@wyscout.com
- **Coverage**: Player performance metrics, video analysis
- **Cost**: Academic discounts available
- **API Documentation**: https://apidocs.wyscout.com/
- **Expected Files**: `data/raw/wyscout/matches/`, `data/raw/wyscout/players/`

#### InStat
- **Contact**: info@instatfootball.com
- **Specialty**: Fitness data, player load, injury risk
- **Coverage**: Physical performance metrics
- **Expected Files**: `data/raw/instat/fitness_*.csv`

### 3. Data Acquisition Scripts

#### Automated Download (Open Sources)
```bash
# Run data collection pipeline
python src/data/collect_data.py --sources statsbomb,fbref,fivethirtyeight
```

#### Manual Steps for Commercial Data
1. **Opta**: Request API credentials, configure in `.env`
2. **Wyscout**: Download match/player CSVs, place in `data/raw/wyscout/`
3. **InStat**: Export fitness reports as CSV, place in `data/raw/instat/`

## Required Data Schema

### Core Tables (Must Exist)

#### `players.csv`
```
player_id,player_name,dob,primary_position,secondary_positions,country,height_cm,weight_kg
12345,"Lionel Messi","1987-06-24","RW","[CF,AM]","Argentina",170,72
```

#### `matches.csv`
```
match_id,season,date,competition,home_team_id,away_team_id,venue,referee_id,weather
987654,"2023-24","2024-03-15","Premier League",1,2,"Old Trafford","REF001","Clear"
```

#### `lineups.csv`
```
match_id,team_id,player_id,is_starting,position_label,minutes_played,sub_in_minute,sub_out_minute
987654,1,12345,1,"RW",90,NULL,NULL
987654,1,67890,0,"CF",25,65,NULL
```

#### `events_aggregated_per_player_windowed.csv`
```
match_id,player_id,window_last_n_matches,xG_sum,xA_sum,key_passes,progressive_passes,pressures,tackles,duels_won,distance_covered,minutes_total
987654,12345,5,2.3,1.8,12,45,23,8,15,10.2,450
```

#### `fitness_and_injury.csv`
```
player_id,date,injury_status,injury_type,days_since_last_match,yellow_card_count_rolling,sickness_flag
12345,"2024-03-15","healthy",NULL,3,1,0
```

#### `co_play_minutes.csv`
```
player_id_a,player_id_b,match_id,minutes_together,on_field_events_diff
12345,67890,987654,65,2.1
```

### Data Validation

After data acquisition, run validation:
```bash
python src/data/validate_schema.py
```

Expected output:
- Schema compliance report
- Missing data summary
- Data quality metrics
- Checksum verification

## Synthetic Sample Dataset

For reviewers without data access, we provide synthetic datasets:

### Location
`data/sample_small/` - Contains realistic but synthetic data matching the schema above

### Generation
```bash
python src/data/generate_synthetic.py --n_matches 100 --n_players 500
```

### Coverage
- 100 matches across 2 seasons
- 20 teams, 500 players
- Realistic statistical distributions
- Maintains temporal relationships

## Data Privacy & Ethics

### Personal Data Handling
- Player names anonymized in public releases
- Date of birth rounded to year only
- No sensitive personal information stored
- GDPR compliance for EU data subjects

### Commercial Data Usage
- Respect licensing terms strictly
- No redistribution of proprietary data
- Academic use only where specified
- Proper attribution in publications

## Troubleshooting

### Common Issues

#### StatsBomb API Rate Limits
```python
import time
from statsbombpy import sb

# Add delays between requests
time.sleep(1)
matches = sb.matches(competition_id=11, season_id=1)
```

#### FBref Scraping Blocks
- Use VPN if blocked by region
- Implement exponential backoff
- Respect robots.txt

#### Missing Data Fields
- Check `data/manifests/missing_data_report.csv`
- Use fallback values from `configs/data_defaults.yaml`
- Implement imputation strategies

### Data Quality Checks

Run comprehensive validation:
```bash
make validate-data
```

Checks include:
- Schema compliance
- Temporal consistency
- Statistical outliers
- Cross-table referential integrity

## Contact for Data Issues

- **Technical Issues**: [tech-email]
- **Licensing Questions**: [legal-email]
- **Academic Partnerships**: [academic-email]

## Data Update Schedule

- **Open Data**: Monthly refresh
- **Commercial Data**: As per license agreement
- **Synthetic Data**: Updated with each release

Last Updated: 2025-08-30
