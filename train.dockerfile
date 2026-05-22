# -----------------------------------------------------------------------------
# Training image for the phishing email MLOps project.
#
# Build:   docker build -f train.dockerfile . -t train:latest
# Run (data + config mounted from host):
#   docker run --name train --rm \
#       -v ${PWD}/data:/app/data \
#       -v ${PWD}/configs:/app/configs \
#       train:latest
# Verified outcome: successfully trained dummy, logistic_regression,
# linear_svc, and complement_nb; best model by validation F2 was linear_svc.
# -----------------------------------------------------------------------------

# Start from a base image
FROM python:3.11-slim-bookworm

# Install build tools needed for Python wheels
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV MLOPS_CREW_PROJECT_ROOT=/app

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY configs/config.yaml configs/config.yaml
COPY src/mlops_crew/__init__.py src/mlops_crew/__init__.py
COPY src/mlops_crew/config.py src/mlops_crew/config.py
COPY src/mlops_crew/logging_config.py src/mlops_crew/logging_config.py
COPY src/mlops_crew/data/__init__.py src/mlops_crew/data/__init__.py
COPY src/mlops_crew/evaluation/__init__.py src/mlops_crew/evaluation/__init__.py
COPY src/mlops_crew/evaluation/metrics.py src/mlops_crew/evaluation/metrics.py
COPY src/mlops_crew/monitoring/__init__.py src/mlops_crew/monitoring/__init__.py
COPY src/mlops_crew/monitoring/resource_monitor.py src/mlops_crew/monitoring/resource_monitor.py
COPY src/mlops_crew/models/__init__.py src/mlops_crew/models/__init__.py
COPY src/mlops_crew/models/train_model.py src/mlops_crew/models/train_model.py
COPY src/mlops_crew/models/text_classifiers.py src/mlops_crew/models/text_classifiers.py
COPY src/mlops_crew/tracking/__init__.py src/mlops_crew/tracking/__init__.py
COPY src/mlops_crew/tracking/mlflow_tracking.py src/mlops_crew/tracking/mlflow_tracking.py
COPY src/mlops_crew/utils/__init__.py src/mlops_crew/utils/__init__.py
COPY src/mlops_crew/utils/io.py src/mlops_crew/utils/io.py
COPY src/mlops_crew/utils/seed.py src/mlops_crew/utils/seed.py

RUN pip install --no-cache-dir uv && \
    uv pip install --system -r requirements.txt && \
    uv pip install --system --no-deps .

ENTRYPOINT ["python", "-u", "-m", "mlops_crew.models.train_model"]