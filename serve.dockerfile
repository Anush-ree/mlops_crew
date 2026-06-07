# -----------------------------------------------------------------------------
# Cloud Run serving image for the phishing email detection API.
#
# Build:
#   docker build -f serve.dockerfile . -t mlops-crew-api:latest
# Run locally:
#   docker run --rm -p 8080:8080 -e PORT=8080 -v "$PWD/models:/app/models" mlops-crew-api:latest
# -----------------------------------------------------------------------------

FROM python:3.11-slim-bookworm

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV MLOPS_CREW_PROJECT_ROOT=/app
ENV MODEL_PATH=/app/models/best_model.joblib
ENV MODEL_VERSION=phase2_linear_svc
ENV PORT=8080

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY configs/config.yaml configs/config.yaml
COPY api api
COPY src src

RUN pip install --no-cache-dir uv && \
    uv pip install --system -r requirements.txt && \
    uv pip install --system --no-deps .

EXPOSE 8080

CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
