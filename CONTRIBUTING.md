# Contributing Guidelines

## Development Workflow

### Branch Strategy
- `main`: Production-ready code
- `develop`: Integration branch for features
- `feature/*`: Individual feature branches
- `experiment/*`: Experimental model variants

### Code Standards

#### Python Style
- Follow PEP 8
- Use type hints for function signatures
- Maximum line length: 88 characters (Black formatter)
- Docstrings: Google style

#### Naming Conventions
- Files: `snake_case.py`
- Classes: `PascalCase`
- Functions/variables: `snake_case`
- Constants: `UPPER_CASE`

#### Example:
```python
from typing import Tuple, Optional
import pandas as pd

class Stage1Ensemble:
    """Stage-1 ensemble for player selection probability."""
    
    def predict_proba(self, X: pd.DataFrame) -> Tuple[np.ndarray, Optional[dict]]:
        """Predict selection probabilities with uncertainty estimates.
        
        Args:
            X: Feature matrix with shape (n_samples, n_features)
            
        Returns:
            Tuple of (probabilities, metadata)
        """
        pass
```

### Experiment Management

#### Adding New Experiments
1. Create experiment config in `configs/experiments/`
2. Add entry to `experiments/experiments_index.csv`
3. Use MLflow for tracking:
```python
import mlflow

with mlflow.start_run(experiment_id="stage1_ensemble"):
    mlflow.log_params(config)
    mlflow.log_metrics(results)
    mlflow.log_artifacts("artifacts/")
```

#### Reproducibility Requirements
- Set random seeds: `np.random.seed(42)`, `random.seed(42)`
- Log exact package versions
- Save model artifacts with metadata
- Document data preprocessing steps

### Data Handling

#### Feature Engineering
- Add new features to `src/features/`
- Update `features_catalog.csv`
- Include feature validation tests

#### Data Validation
- Schema checks in `tests/test_data_schema.md`
- Range validation for numerical features
- Consistency checks across time periods

### Model Development

#### Stage-1 Models
- Location: `src/models/stage1/`
- Must implement calibration
- Include feature importance analysis
- Temporal CV validation required

#### Stage-2 Models
- Location: `src/models/stage2/`
- Integration with compatibility matrices
- MIP optimization constraints
- Lineup-level validation

### Testing

#### Required Tests
- Data schema validation
- Model determinism (same seed → same output)
- Pipeline integration tests
- Constraint satisfaction for optimizer

#### Running Tests
```bash
make test                    # All tests
make test-data              # Data validation only
make test-models            # Model tests only
make test-integration       # End-to-end pipeline
```

### Documentation

#### Code Documentation
- Docstrings for all public functions
- Inline comments for complex logic
- README files in each major directory

#### Experiment Documentation
- Clear description in experiment config
- Results interpretation in notebooks
- Failure analysis for unsuccessful experiments

### Pull Request Process

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/mab-feature-selection
   ```

2. **Development**
   - Write code following standards
   - Add tests for new functionality
   - Update documentation

3. **Pre-commit Checks**
   ```bash
   make lint                # Code formatting
   make test                # Run test suite
   make validate-data       # Data consistency
   ```

4. **Submit PR**
   - Clear title and description
   - Link to relevant issues
   - Include performance metrics if applicable

5. **Review Process**
   - Code review by team member
   - Automated CI checks
   - Integration test on clean environment

### Performance Guidelines

#### Computational Efficiency
- Profile code for bottlenecks
- Use vectorized operations (pandas/numpy)
- Consider memory usage for large datasets
- GPU utilization where beneficial

#### Model Training
- Early stopping for iterative algorithms
- Hyperparameter optimization with Optuna
- Parallel processing where possible
- Progress logging for long-running jobs

### Release Process

#### Version Tagging
- Semantic versioning: `v1.2.3`
- Tag format: `YYYY-MM-DD_component_version`
- Example: `2025-09-06_stage1_v1.0`

#### Artifact Management
- Model binaries with metadata
- Compatibility matrices versioned
- Configuration snapshots
- Performance benchmarks

### Environment Management

#### Dependencies
- Pin exact versions in `requirements.txt`
- Test with fresh virtual environments
- Document system dependencies
- Maintain Docker compatibility

#### Configuration
- Environment-specific configs
- Secrets management (never commit keys)
- Path handling for cross-platform compatibility

### Troubleshooting

#### Common Issues
- **Memory errors**: Reduce batch sizes, use data generators
- **Convergence issues**: Adjust learning rates, increase iterations
- **Reproducibility**: Check random seeds, library versions
- **Performance**: Profile code, optimize bottlenecks

#### Getting Help
- Check existing issues and documentation
- Include error logs and environment details
- Provide minimal reproducible example
- Tag relevant team members
