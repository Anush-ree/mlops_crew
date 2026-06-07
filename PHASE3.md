# PHASE 3: Continuous Machine Learning (CML) & Deployment

## Overview

Phase 3 productionizes the phishing email detector built in Phases 1–2. It adds CI/CD pipelines, containerized training and serving, GCP cloud deployment (Cloud Run + Cloud Functions), a Gradio demo on Hugging Face Spaces, and a final holdout evaluation on data the model has never seen.

**Best model:** LinearSVC (TF-IDF pipeline)
**Phase 3 holdout F2:** 98.6% on 16,496 unseen emails

---

## 1. Continuous Integration & Testing

**Workflow:** [`.github/workflows/ci.yml`](.github/workflows/ci.yml)
Triggers on every push to `main` and every pull request.

**Steps run in CI:**
- `ruff check` — linting
- `ruff format --check` — formatting
- `mypy src` — type checking
- `pytest tests/ --cov=mlops_crew --cov-report=xml` — tests with coverage

**Python version:** 3.11 (pinned via matrix)

**Test files:**
| File | What it tests |
|---|---|
| `tests/test_api.py` | FastAPI `/health` and `/predict` endpoints |
| `tests/test_serving.py` | `ModelService` text normalization and prediction |
| `tests/test_hf_space_app.py` | Gradio UI helpers |
| `tests/test_phase1_pipeline.py` | Data pipeline stages |
| `tests/test_phase2_monitoring.py` | Divergence and latency monitoring |
| `tests/test_logging_and_hydra.py` | Logging config and Hydra integration |

**Pre-commit hooks:** [`.pre-commit-config.yaml`](.pre-commit-config.yaml)
- `ruff` (lint + autofix)
- `ruff-format`
- `mypy` (type checking)
- `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`

Run locally with `make dev` to install hooks.

---

## 2. Docker Automation & CML

### Docker Build and Push

**Workflow:** [`.github/workflows/docker-publish.yml`](.github/workflows/docker-publish.yml)

- **On pull requests:** builds the serving image only (no push) to validate the Dockerfile
- **On push to main:** builds and pushes to Docker Hub using `docker/metadata-action` tags (`sha-<hash>` + `latest`)
- Uses Docker Buildx and GitHub Actions layer cache (`type=gha`)
- Image: `serve.dockerfile` → FastAPI serving container

**Build locally:**
```bash
make docker-serve
# or
docker build -f serve.dockerfile . -t mlops-crew-api:latest
docker run --rm -p 8080:8080 -e PORT=8080 -v "$PWD/models:/app/models" mlops-crew-api:latest
```

### CML (Continuous Machine Learning)

**Workflow:** [`.github/workflows/cml.yml`](.github/workflows/cml.yml)
Triggers on pull requests to `main`.

Generates a markdown report posted as a PR comment containing:
- Model comparison table from `reports/metrics/model_comparison.csv`
- Best model metrics from `reports/metrics/best_model_metrics.json`
- Model comparison plot `reports/metrics/model_comparison.png`

Uses `cml comment update --publish` to update the same comment on each push (no duplicate bot comments).

**Model comparison summary:**

| Model | Val F2 | Test F2 | Test Recall |
|---|---|---|---|
| linear_svc | 99.2% | 99.1% | 99.2% |
| logistic_regression | 98.8% | 99.0% | 99.2% |
| complement_nb | 94.5% | 95.1% | 94.2% |
| dummy | 84.5% | 84.5% | 100% |

Best model selected by validation F2: **LinearSVC**

---

## 3. GCP Deployment

### Infrastructure

- **Project region:** us-central1
- **Artifact Registry:** Docker-format repository for serving image
- **GCS bucket:** versioned bucket for training data and model artifacts
- **IAM:** service account with roles for Cloud Run, Cloud Functions, Artifact Registry, and Storage

Evidence screenshots: [`reports/gcp/`](reports/gcp/)

### Cloud Training

Training image built from [`train.dockerfile`](train.dockerfile).

- Mounts data from GCS to `/app/data` before training
- Copies model artifacts back to GCS after training
- Entrypoint: `python -m mlops_crew.models.train_model`

Evidence: [`reports/gcp/gcp_runs.png`](reports/gcp/gcp_runs.png)

### FastAPI Inference Service

**Code:** [`api/main.py`](api/main.py) | **Schemas:** [`api/schemas.py`](api/schemas.py)

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Service readiness and model load status |
| `/predict` | POST | Classify a single email as phishing or legitimate |

**Request:**
```json
{"text": "Verify your account now or it will be suspended."}
```

**Response:**
```json
{
  "label": "phishing",
  "score_type": "decision_function",
  "score": 1.85,
  "model_version": "phase2_linear_svc",
  "latency_ms": 3.2,
  "is_phishing": true,
  "prediction": 1,
  "normalized_text_length": 52
}
```

### Cloud Run Deployment

Deployed from Artifact Registry using [`serve.dockerfile`](serve.dockerfile).
Listens on `$PORT` (default 8080). Model loaded from GCS via `MODEL_GCS_URI` env var.

Evidence:
- [`reports/gcp/gcp_live_api.png`](reports/gcp/gcp_live_api.png) — live API response
- [`reports/gcp/gcp_logs.png`](reports/gcp/gcp_logs.png) — Cloud Run logs
- [`reports/gcp/gcp_metrics.png`](reports/gcp/gcp_metrics.png) — request metrics

### Cloud Functions Deployment

**Code:** [`functions/predict/main.py`](functions/predict/main.py)

Thin HTTP wrapper that forwards requests to the Cloud Run FastAPI backend.
Configured via `BACKEND_PREDICT_URL` environment variable.

Evidence:
- [`reports/gcp/cloud_function.png`](reports/gcp/cloud_function.png)
- [`reports/gcp/cloud_function_log.png`](reports/gcp/cloud_function_log.png)
- [`reports/gcp/cloud_function_local_run.png`](reports/gcp/cloud_function_local_run.png)

### Load Testing

**Script:** [`scripts/load_test_api.py`](scripts/load_test_api.py)

```bash
make load-test-api
# or
python3 scripts/load_test_api.py --endpoint http://localhost:8080/predict --requests 10
```

Reports mean latency and p95 latency across N requests.

---

## 4. Hugging Face Spaces Demo

**Live demo:** https://huggingface.co/spaces/manas01AI/phishing-email-detector

**Code:** [`hf_space/app.py`](hf_space/app.py) — Gradio interface
**Auto-deploy workflow:** [`.github/workflows/deploy_hf_space.yml`](.github/workflows/deploy_hf_space.yml)

Triggers on push to `main` when `hf_space/` changes. Pushes the `hf_space/` directory to the HF Space repo using `HF_TOKEN`, `HF_USERNAME`, and `HF_SPACE_NAME` secrets.

The app reads `BACKEND_PREDICT_URL` from the environment. When the Cloud Run service URL is set as a Space secret, predictions go through the real model. Falls back to a mock prediction during development.

---

## 5. Phase 3 Holdout Evaluation

The holdout set (20% of data, reserved since Phase 2 sampling) was evaluated once on the final model with no subsequent tuning.

**Script:** [`src/mlops_crew/evaluation/phase3_holdout_eval.py`](src/mlops_crew/evaluation/phase3_holdout_eval.py)
**Results:** [`reports/metrics/phase3_holdout_metrics.json`](reports/metrics/phase3_holdout_metrics.json)

| Metric | Value |
|---|---|
| Holdout rows | 16,496 |
| Accuracy | 98.4% |
| Precision | 98.2% |
| Recall | 98.7% |
| F1 | 98.4% |
| **F2** | **98.6%** |
| False Negative Rate | 1.28% |
| False Positive Rate | 2.01% |
| ROC-AUC | 99.84% |

The model generalizes well — F2 drops only 0.6 points from test (99.1%) to holdout (98.6%), confirming no overfitting.

---

## 6. Team Contributions

| Workstream | Primary | Support |
|---|---|---|
| CI, Docker, CML workflows | Kirtan | Anushree |
| GCP infrastructure, Cloud Run, Cloud Functions, FastAPI | Krishna (Kirtan) | Anushree |
| Gradio UI, Hugging Face Spaces | Anas | — |
