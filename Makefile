SHELL := /bin/bash

.PHONY: help venv install data profile baseline figures clean docker-build docker-run

help:
	@echo "Targets:"
	@echo "  install        - pip install requirements"
	@echo "  data           - (placeholder) ensure data/ dirs exist"
	@echo "  profile        - run data profiling & schema check"
	@echo "  baseline       - train Stage-1 baseline (temporal CV + calibration)"
	@echo "  figures        - regenerate reliability plots"
	@echo "  docker-build   - build Docker image"
	@echo "  docker-run     - run Docker with mounted repo"
	@echo "  clean          - remove caches and intermediate artifacts"

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

data:
	mkdir -p data/raw data/interim data/processed outputs/reports outputs/figures outputs/artifacts mlruns

profile: data
	python scripts/01_profile_data.py --config configs/data.yaml

baseline:
	python scripts/02_train_baseline_stage1.py --exp configs/experiment_baseline.yaml

figures:
	@echo "Figures are generated during baseline training."

docker-build:
	docker build -t fss:dev .

docker-run:
	docker run --rm -it -p 5000:5000 -v $$(pwd):/workspace fss:dev /bin/bash

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	find . -type f -name "*.pyc" -delete
