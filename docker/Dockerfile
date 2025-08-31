# Multi-stage Docker build for Football Squad Selection Pipeline
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY env/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/features data/manifests \
    experiments/mlflow experiments/optuna_studies experiments/runs \
    artifacts notebooks paper/figures paper/tables \
    tests docs logs

# Set permissions for run script
RUN chmod +x run_all.sh

# Expose MLflow port
EXPOSE 5000

# Default command
CMD ["./run_all.sh"]

# Development stage with additional tools
FROM base as dev

RUN pip install --no-cache-dir \
    jupyter \
    ipykernel \
    black \
    flake8 \
    pytest \
    pytest-cov

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Production stage - minimal image
FROM base as prod

# Remove development dependencies and clean up
RUN apt-get autoremove -y build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Run as non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

CMD ["./run_all.sh"]
