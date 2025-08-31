# Unit Tests Descriptions

## Overview
Comprehensive unit test suite covering all components of the football squad selection pipeline.

## Data Processing Tests

### `test_data_collection.py`
- **StatsBomb API integration**: Valid API responses, rate limiting
- **FBref scraping**: HTML parsing, data extraction accuracy
- **Data merging**: Consistent player/match IDs across sources
- **Error handling**: Network failures, API rate limits

### `test_feature_engineering.py`
- **Rolling windows**: Correct computation of 3/5/10 match averages
- **Performance metrics**: xG, xA, passing accuracy calculations
- **Fitness features**: Days since last match, fixture congestion
- **Opponent features**: Style clusters, venue effects
- **Edge cases**: New players, missing data, season boundaries

### `test_data_validation.py`
- **Schema compliance**: Column types, required fields
- **Range validation**: Realistic values for all metrics
- **Temporal consistency**: No future information leakage
- **Foreign key integrity**: Valid references between tables

## Model Tests

### `test_stage1_models.py`
- **Model training**: RF, XGBoost, LightGBM convergence
- **Calibration**: Isotonic/Platt scaling effectiveness
- **Feature importance**: Stability across CV folds
- **Prediction format**: Probability outputs in [0,1] range
- **Serialization**: Model save/load consistency

### `test_stage2_models.py`
- **Compatibility matrix**: MF algorithm convergence
- **Lineup optimization**: MIP solver feasibility
- **Formation constraints**: Valid XI + 7 configurations
- **Integration**: Stage-1 outputs → Stage-2 inputs
- **Edge cases**: Injured players, formation switches

### `test_mab_feature_selection.py`
- **Thompson Sampling**: Arm selection logic
- **UCB1 algorithm**: Upper confidence bound calculations
- **Reward computation**: Feature subset performance tracking
- **Convergence**: Optimal arm identification over time
- **Stability**: Consistent selections across runs

## Pipeline Integration Tests

### `test_temporal_cv.py`
- **Split generation**: Proper temporal ordering
- **Gap handling**: No data leakage across splits
- **Fold consistency**: Same splits across different runs
- **Edge cases**: Short seasons, irregular fixtures

### `test_calibration.py`
- **Reliability diagrams**: Calibration curve computation
- **ECE calculation**: Expected Calibration Error
- **Brier score**: Proper decomposition (reliability + resolution)
- **Bootstrap CIs**: Confidence interval generation

### `test_compatibility_matrix.py`
- **Co-play extraction**: Minutes together calculation
- **Matrix factorization**: ALS algorithm implementation
- **Similarity computation**: Cosine similarity between embeddings
- **Sparsity handling**: Fallback for rare player pairs

## Simulation Tests

### `test_monte_carlo.py`
- **Goal generation**: Poisson/Skellam model accuracy
- **Match simulation**: Realistic score distributions
- **Lineup evaluation**: Expected points calculation
- **Parallel execution**: Consistent results across workers
- **Random seed**: Reproducible simulation outcomes

### `test_optimizer.py`
- **Constraint satisfaction**: Formation, fitness, availability
- **Objective function**: Composite score maximization
- **Solver integration**: PuLP/OR-Tools compatibility
- **Solution quality**: Optimal vs heuristic comparison
- **Performance**: Solve time within limits

## Validation and Evaluation Tests

### `test_evaluation_metrics.py`
- **PR-AUC calculation**: Precision-recall curve accuracy
- **ROC-AUC computation**: Receiver operating characteristic
- **Brier score**: Proper scoring rule implementation
- **Statistical tests**: Diebold-Mariano, McNemar, DeLong
- **Bootstrap sampling**: Confidence interval generation

### `test_ablation_studies.py`
- **Component removal**: Compatibility, MAB, transfer learning
- **Performance deltas**: Accurate improvement measurement
- **Statistical significance**: Proper hypothesis testing
- **Result consistency**: Reproducible ablation outcomes

## Reproducibility Tests

### `test_determinism.py`
- **Random seed control**: Identical outputs with same seed
- **Model artifacts**: Consistent checksums across runs
- **Data processing**: Deterministic feature engineering
- **Pipeline execution**: End-to-end reproducibility

### `test_environment.py`
- **Package versions**: Exact dependency matching
- **Docker consistency**: Same results in container vs local
- **Cross-platform**: Windows/Linux/Mac compatibility
- **Resource usage**: Memory and CPU within limits

## Performance Tests

### `test_scalability.py`
- **Large datasets**: Performance with 50k+ matches
- **Memory usage**: Peak memory within 8GB limit
- **Training time**: Reasonable training duration
- **Inference speed**: <1 second per lineup prediction

### `test_robustness.py`
- **Missing data**: Graceful handling of incomplete records
- **Outliers**: Robust to extreme performance values
- **New teams**: Generalization to unseen teams
- **Formation changes**: Adaptation to tactical shifts

## Error Handling Tests

### `test_error_recovery.py`
- **API failures**: Graceful degradation with data source issues
- **Model failures**: Fallback to simpler models
- **Optimization failures**: Heuristic solutions when MIP fails
- **Data corruption**: Detection and recovery strategies

### `test_input_validation.py`
- **Invalid inputs**: Proper error messages for bad data
- **Type checking**: Correct handling of wrong data types
- **Range validation**: Bounds checking for all parameters
- **Configuration errors**: Clear messages for config issues

## Integration Tests

### `test_end_to_end.py`
- **Full pipeline**: Data → Stage-1 → Stage-2 → Simulation
- **Output validation**: All expected files generated
- **Performance benchmarks**: Meets minimum accuracy thresholds
- **Resource consumption**: Within specified limits

### `test_mlflow_integration.py`
- **Experiment logging**: Proper parameter and metric tracking
- **Artifact storage**: Model and result persistence
- **Run comparison**: Consistent experiment comparison
- **Metadata tracking**: Complete provenance information

## Test Data

### Synthetic Test Data
- **Location**: `tests/data/synthetic/`
- **Coverage**: All required tables with realistic distributions
- **Size**: Small enough for fast testing (100 matches, 200 players)
- **Consistency**: Maintains referential integrity

### Edge Case Data
- **Missing values**: Various missingness patterns
- **Outliers**: Extreme but valid performance values
- **Boundary conditions**: Season transitions, new players
- **Error conditions**: Invalid data for error handling tests

## Test Execution

### Local Testing
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_data_schema.py -v
python -m pytest tests/test_stage1_models.py -v

# Run with coverage
python -m pytest tests/ --cov=src/ --cov-report=html
```

### CI/CD Testing
```bash
# Fast test suite (for PRs)
python -m pytest tests/ -m "not slow" --maxfail=5

# Full test suite (for releases)
python -m pytest tests/ --cov=src/ --cov-fail-under=80
```

### Performance Testing
```bash
# Memory profiling
python -m pytest tests/test_scalability.py --profile-memory

# Timing tests
python -m pytest tests/test_performance.py --benchmark-only
```

## Test Configuration

### pytest.ini
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
    data: marks tests as data-related
    model: marks tests as model-related
```

### Test Fixtures
- **Sample data**: Consistent test datasets
- **Mock models**: Pre-trained models for testing
- **Configuration**: Test-specific config files
- **Temporary directories**: Isolated test environments

## Success Criteria

### Coverage Requirements
- **Code coverage**: ≥80% for all modules
- **Branch coverage**: ≥70% for critical paths
- **Function coverage**: 100% for public APIs

### Performance Requirements
- **Test execution**: Complete suite in <10 minutes
- **Memory usage**: <2GB peak during testing
- **Determinism**: 100% reproducible test outcomes

### Quality Gates
- All tests must pass before merge
- No critical security vulnerabilities
- Documentation updated for new features
- Performance regression checks pass
