FROM python:3.11-slim

# System deps (git, build tools optional)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Optional: set MLflow artifact/port defaults
ENV MLFLOW_TRACKING_URI=file:/workspace/mlruns
ENV PYTHONUNBUFFERED=1
ENV MPLCONFIGDIR=/workspace/.mplconfig

COPY . /workspace
CMD ["/bin/bash"]
