# Football Squad Selection - Makefile
# Provides convenient targets for common tasks

.PHONY: help install build data train_stage1 train_stage2 sim figures paper test clean docker-build docker-run

# Default target
help:
	@echo "Football Squad Selection Pipeline"
	@echo ""
	@echo "Available targets:"
	@echo "  install      - Install dependencies"
	@echo "  build        - Build Docker image"
	@echo "  data         - Prepare and validate data"
	@echo "  train_stage1 - Train Stage-1 ensemble models"
	@echo "  train_stage2 - Train Stage-2 models with compatibility"
	@echo "  sim          - Run Monte Carlo simulation"
	@echo "  figures      - Generate all figures and tables"
	@echo "  paper        - Compile paper and supplement"
	@echo "  test         - Run all tests"
	@echo "  clean        - Clean artifacts and logs"
	@echo "  all          - Run complete pipeline"
	@echo ""
	@echo "Docker targets:"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run pipeline in Docker"
	@echo "  docker-dev   - Start development environment"

# Installation
install:
	pip install -r env/requirements.txt

# Docker operations
docker-build:
	docker build -t $(PROJECT_NAME):latest -f docker/Dockerfile --target prod .
	docker build -t $(PROJECT_NAME):dev -f docker/Dockerfile --target dev .

docker-run:
	docker run --rm -v $$(pwd):/app $(PROJECT_NAME):latest

docker-dev:
	docker run --rm -p 8888:8888 -v $$(pwd):/app $(PROJECT_NAME):dev

# Data pipeline
data:
	python src/data/prepare_data.py
	python src/data/validate_schema.py
	python src/features/engineer_features.py

# Model training
train_stage1:
	python src/models/stage1/train.py
	python src/models/stage1/calibrate.py
	python src/models/stage1/evaluate.py

train_stage2:
	python src/models/stage2/compatibility_matrix.py
	python src/models/stage2/train.py
	python src/models/stage2/optimize_lineup.py

# Simulation
sim:
	python src/simulation/monte_carlo.py
	python src/simulation/analyze_results.py

# Visualization and reporting
figures:
	python src/visualization/generate_figures.py
	python src/visualization/generate_tables.py

paper:
	cd paper && pdflatex draft.tex

# Testing
test:
	python -m pytest tests/ -v

test-data:
	python src/data/validate_schema.py --test-mode

test-models:
	python -m pytest tests/test_models.py -v

test-integration:
	python -m pytest tests/test_integration.py -v

# Validation
validate-data:
	python src/data/validate_schema.py
	python src/data/check_leakage.py

lint:
	black src/ tests/
	flake8 src/ tests/

# Cleanup
clean:
	rm -rf artifacts/*.pkl
	rm -rf experiments/runs/*
	rm -rf logs/*.log
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	find . -name "*.pyc" -delete

clean-all: clean
	rm -rf data/processed/*
	rm -rf data/features/*
	rm -rf paper/figures/*
	rm -rf paper/tables/*

# Complete pipeline
all: data train_stage1 train_stage2 sim figures test

# Fast demo
demo:
	./run_all.sh --fast-demo

# Development setup
dev-setup: install
	pre-commit install
	jupyter lab --generate-config

# Performance profiling
profile:
	python -m cProfile -o profile.stats src/models/stage1/train.py
	python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"

# MLflow server
mlflow-server:
	mlflow server --backend-store-uri file:./experiments/mlflow --default-artifact-root ./artifacts --host 0.0.0.0 --port 5000
