# -----------------------------------------------------------------------------
# Prediction image for the phishing email MLOps project.
#
# Build:   docker build -f predict.dockerfile . -t predict:latest
# Run (model, input, and output mounted from host):
#   docker run --name pred --rm \
#       -v ${PWD}/models:/app/models \
#       -v ${PWD}/data:/app/data \
#       -v ${PWD}/reports:/app/reports \
#       predict:latest \
#       --model-path /app/models/best_model.joblib \
#       --input /app/data/processed/test.csv \
#       --output /app/reports/predictions/batch_predictions.csv
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
COPY src/mlops_crew/models/__init__.py src/mlops_crew/models/__init__.py
COPY src/mlops_crew/models/text_classifiers.py src/mlops_crew/models/text_classifiers.py
COPY src/mlops_crew/models/predict_model.py src/mlops_crew/models/predict_model.py

RUN pip install --no-cache-dir uv && \
    uv pip install --system -r requirements.txt && \
    uv pip install --system --no-deps .

ENTRYPOINT ["python", "-u", "-m", "mlops_crew.models.predict_model"]