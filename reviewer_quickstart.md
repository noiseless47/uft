# Reviewer Quickstart Guide

**Estimated Time**: ≤2 hours  
**Goal**: Reproduce main results from the paper using synthetic data

## Prerequisites

- Docker installed
- 5GB free disk space
- Internet connection for Docker image download

## Quick Reproduction Steps

### 1. Setup (5 minutes)
```bash
git clone <repository-url>
cd football-squad-selection

# Verify Docker is running
docker --version
```

### 2. Fast Demo Run (15-30 minutes)
```bash
# Run complete pipeline with synthetic data
./run_all.sh --fast-demo
```

This will:
- Build Docker environment
- Generate synthetic dataset (100 matches, 500 players)
- Train Stage-1 ensemble (RF, XGBoost, LightGBM)
- Train Stage-2 with compatibility matrix
- Run Monte Carlo simulation (1000 iterations)
- Generate key figures and tables

### 3. Verify Main Results (10 minutes)

Check these key outputs:

**Main Performance Table**:
```bash
cat paper/tables/stage1_metrics.csv
```
Expected: PR-AUC > 0.75, Brier Score < 0.2

**Stage-2 Lineup Results**:
```bash
cat paper/tables/stage2_lineup_table.csv
```
Expected: Formation-valid lineups with compatibility scores

**Calibration Plot**:
```bash
ls paper/figures/calibration_*.png
```
Expected: Reliability diagram showing well-calibrated probabilities

### 4. Inspect Key Artifacts (15 minutes)

**Trained Models**:
```bash
ls artifacts/
# Should contain: stage1_*.pkl, stage2_*.pkl, compat_matrix_*.npz
```

**MLflow Experiments**:
```bash
# Start MLflow UI (optional)
mlflow server --backend-store-uri file:./experiments/mlflow --port 5000
# Open http://localhost:5000 in browser
```

**Simulation Results**:
```bash
head experiments/simulation/lineup_sim_results.csv
```
Expected columns: lineup_id, expected_points, win_prob, draw_prob, loss_prob

## Key Files to Review

### 1. Configuration
- `configs/config.yaml` - All hyperparameters and settings
- `run_all.sh` - Complete pipeline orchestration

### 2. Core Implementation
- `src/models/stage1/` - Ensemble training and calibration
- `src/models/stage2/` - Compatibility matrix and optimization
- `src/simulation/` - Monte Carlo match simulation

### 3. Results
- `paper/tables/` - All numerical results as CSV
- `paper/figures/` - All plots and diagrams
- `experiments/experiments_index.csv` - Experiment tracking

## Validation Checklist

After running the demo, verify:

- [ ] **Models trained successfully**: Check `artifacts/` for .pkl files
- [ ] **Calibration works**: ECE < 0.1 in stage1_metrics.csv
- [ ] **Compatibility matrix**: Non-zero values in compat_matrix.npz
- [ ] **Valid lineups**: All lineups satisfy formation constraints
- [ ] **Simulation runs**: Expected points in reasonable range (0-3)
- [ ] **Reproducibility**: Same random seed gives identical results

## Expected Performance (Synthetic Data)

| Metric | Stage-1 | Baseline | Improvement |
|--------|---------|----------|-------------|
| PR-AUC | 0.78 | 0.74 | +5.4% |
| Brier Score | 0.18 | 0.22 | -18.2% |
| ECE | 0.045 | 0.089 | -49.4% |

**Note**: Real data performance will be higher due to richer feature set.

## Troubleshooting

### Common Issues

**Docker Build Fails**:
```bash
# Check Docker daemon
docker info

# Build with verbose output
docker build -t football-squad-selection:latest -f docker/Dockerfile --no-cache .
```

**Memory Issues**:
```bash
# Run with reduced batch size
./run_all.sh --fast-demo
# Edit configs/config.yaml: batch_size: 500
```

**Missing Dependencies**:
```bash
# Rebuild Docker image
make docker-build
```

### Performance Expectations

| Component | Synthetic Time | Full Data Time |
|-----------|----------------|----------------|
| Data Prep | 2 min | 15 min |
| Stage-1 Training | 8 min | 45 min |
| Stage-2 Training | 5 min | 25 min |
| Simulation | 3 min | 20 min |
| **Total** | **18 min** | **105 min** |

## Reproducing Specific Results

### Main Table (Table 1 in paper)
```bash
python src/evaluation/generate_main_table.py
cat paper/tables/main_results.csv
```

### Ablation Study (Table 2 in paper)
```bash
python src/evaluation/ablation_study.py
cat paper/tables/ablation_results.csv
```

### Calibration Figure (Figure 3 in paper)
```bash
python src/visualization/calibration_plots.py
ls paper/figures/calibration_reliability.png
```

## Using Real Data

If you have access to licensed data:

1. **Place data files** in `data/raw/` following the schema in `DATA_ACCESS.md`
2. **Run full pipeline**:
   ```bash
   ./run_all.sh  # No --fast-demo flag
   ```
3. **Expected runtime**: 4-6 hours (CPU), 2-3 hours (GPU)

## MLflow Run IDs for Paper Results

For reproducing exact paper results, use these MLflow run IDs:

- **Stage-1 Best Model**: `run_id_stage1_final`
- **Stage-2 Best Model**: `run_id_stage2_final`
- **Main Results**: `run_id_main_evaluation`
- **Ablation Study**: `run_id_ablation_study`

Access via:
```bash
mlflow artifacts download --run-id <run_id> --artifact-path model
```

## Contact

For reproduction issues:
- **Technical**: [tech-support-email]
- **Data Access**: [data-access-email]
- **Methodology**: [research-email]

## Verification

To verify successful reproduction:
1. Check exit code: `echo $?` should be 0
2. Verify key metrics match expected ranges
3. Ensure all required files are generated
4. Run integration tests: `make test-integration`
