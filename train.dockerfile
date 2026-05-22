# -----------------------------------------------------------------------------
# Training image for the phishing email MLOps project.
#
# Build:   docker build -f train.dockerfile . -t train:latest
# Run (data + config mounted from host):
#   docker run --name train --rm \
#       -v ${PWD}/data:/app/data \
#       -v ${PWD}/configs:/app/configs \
#       -v ${PWD}/models:/app/models \
#       -v ${PWD}/reports:/app/reports \
#       -v ${PWD}/logs:/app/logs \
#       -v ${PWD}/mlruns:/app/mlruns \
#       train:latest
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
COPY src/mlops_crew src/mlops_crew

RUN pip install --no-cache-dir uv && \
    uv pip install --system -r requirements.txt && \
    uv pip install --system --no-deps .

ENTRYPOINT ["python", "-u", "-m", "mlops_crew.models.train_model"]
