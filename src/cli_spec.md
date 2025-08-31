# Command Line Interface Specification

## Overview

The Football Squad Selection pipeline provides a comprehensive CLI for data processing, model training, and evaluation.

## Main Entry Points

### 1. Data Pipeline
```bash
# Data collection and preprocessing
python src/data/collect_data.py [OPTIONS]
python src/data/prepare_data.py [OPTIONS]
python src/features/engineer_features.py [OPTIONS]
```

### 2. Model Training
```bash
# Stage-1 ensemble training
python src/models/stage1/train.py [OPTIONS]

# Stage-2 training with compatibility
python src/models/stage2/train.py [OPTIONS]
```

### 3. Evaluation and Simulation
```bash
# Model evaluation
python src/evaluation/evaluate_models.py [OPTIONS]

# Monte Carlo simulation
python src/simulation/monte_carlo.py [OPTIONS]
```

## Command Specifications

### Data Collection
```bash
python src/data/collect_data.py \
    --sources statsbomb,fbref \
    --competitions "Premier League,La Liga" \
    --seasons 2020-21,2021-22,2022-23 \
    --output data/raw/ \
    --parallel 4
```

**Options:**
- `--sources`: Data sources (statsbomb, fbref, fivethirtyeight)
- `--competitions`: Competition names or IDs
- `--seasons`: Season identifiers
- `--output`: Output directory
- `--parallel`: Number of parallel workers
- `--rate-limit`: Requests per second limit

### Feature Engineering
```bash
python src/features/engineer_features.py \
    --input data/processed/ \
    --output data/features/ \
    --windows 3,5,10 \
    --config configs/features.yaml
```

**Options:**
- `--input`: Processed data directory
- `--output`: Feature output directory
- `--windows`: Rolling window sizes
- `--config`: Feature configuration file
- `--parallel`: Parallel processing

### Stage-1 Training
```bash
python src/models/stage1/train.py \
    --data data/features/ \
    --models rf,xgb,lgb \
    --cv temporal \
    --calibration isotonic \
    --output artifacts/stage1/
```

**Options:**
- `--data`: Feature data directory
- `--models`: Models to train (rf, xgb, lgb, all)
- `--cv`: Cross-validation strategy (temporal, kfold)
- `--calibration`: Calibration method (isotonic, platt)
- `--output`: Model output directory
- `--hpo`: Enable hyperparameter optimization
- `--n-trials`: Number of HPO trials

### Stage-2 Training
```bash
python src/models/stage2/train.py \
    --stage1-models artifacts/stage1/ \
    --compatibility-matrix artifacts/compat_matrix.npz \
    --output artifacts/stage2/ \
    --formations "4-3-3,4-2-3-1"
```

**Options:**
- `--stage1-models`: Stage-1 model directory
- `--compatibility-matrix`: Compatibility matrix file
- `--output`: Output directory
- `--formations`: Supported formations
- `--optimizer`: MIP solver (pulp, ortools)

### Monte Carlo Simulation
```bash
python src/simulation/monte_carlo.py \
    --models artifacts/ \
    --n-sims 5000 \
    --output experiments/simulation/ \
    --parallel 8
```

**Options:**
- `--models`: Trained models directory
- `--n-sims`: Number of simulations
- `--output`: Simulation results directory
- `--parallel`: Parallel workers
- `--goal-model`: Goal scoring model (poisson, negative_binomial)

### Evaluation
```bash
python src/evaluation/evaluate_models.py \
    --models artifacts/ \
    --data data/features/ \
    --output paper/tables/ \
    --metrics pr_auc,brier_score,ece
```

**Options:**
- `--models`: Models directory
- `--data`: Test data directory
- `--output`: Results output directory
- `--metrics`: Evaluation metrics
- `--bootstrap`: Bootstrap confidence intervals
- `--n-bootstrap`: Number of bootstrap samples

## Utility Commands

### Data Validation
```bash
python src/data/validate_schema.py --data data/processed/
python src/data/check_leakage.py --features data/features/
```

### Model Inspection
```bash
python src/utils/model_inspector.py \
    --model artifacts/stage1_rf.pkl \
    --feature-names data/features/feature_names.txt
```

### Hyperparameter Optimization
```bash
python src/hpo/optimize.py \
    --model stage1 \
    --study-name stage1_hpo \
    --n-trials 100 \
    --timeout 3600
```

## Configuration Files

### Main Config
`configs/config.yaml` - Central configuration

### Model Configs
- `configs/stage1.yaml` - Stage-1 model parameters
- `configs/stage2.yaml` - Stage-2 model parameters
- `configs/features.yaml` - Feature engineering settings

### Experiment Configs
`configs/experiments/` - Individual experiment configurations

## Environment Variables

```bash
# MLflow tracking
export MLFLOW_TRACKING_URI="file:./experiments/mlflow"

# Data paths
export DATA_PATH="data/"
export ARTIFACTS_PATH="artifacts/"

# Compute settings
export N_JOBS=8
export OMP_NUM_THREADS=4

# API keys (if using commercial data)
export OPTA_API_KEY="your_key_here"
export WYSCOUT_API_KEY="your_key_here"
```

## Logging and Debugging

### Log Levels
```bash
# Debug mode
python src/models/stage1/train.py --log-level DEBUG

# Quiet mode
python src/models/stage1/train.py --log-level ERROR
```

### Performance Profiling
```bash
# Memory profiling
python -m memory_profiler src/models/stage1/train.py

# Time profiling
python -m cProfile -o profile.stats src/models/stage1/train.py
```

## Examples

### Quick Start Example
```bash
# Generate synthetic data and run fast demo
python src/data/generate_synthetic.py --n_matches 100
./run_all.sh --fast-demo
```

### Full Pipeline Example
```bash
# Download real data
python src/data/collect_data.py --sources statsbomb,fbref

# Run complete pipeline
./run_all.sh

# Generate paper outputs
make figures
make paper
```

### Development Example
```bash
# Start development environment
make docker-dev

# Or run locally
jupyter lab
```

## Error Handling

### Common Exit Codes
- `0`: Success
- `1`: General error
- `2`: Data validation error
- `3`: Model training error
- `4`: Simulation error
- `5`: Configuration error

### Debugging Tips
1. Check logs in `logs/` directory
2. Validate data schema first
3. Ensure sufficient disk space (20GB+)
4. Verify random seeds for reproducibility
5. Check MLflow for experiment tracking
