# Phase 3 FastAPI Service

This directory contains the HTTP API used for Phase 3 deployment.

## Endpoints

- `GET /health` confirms the service can load the model artifact.
- `POST /predict` accepts JSON `{"text": "..."}` and returns the phishing
  prediction, score or margin, model version, and latency.

The route code delegates model loading and prediction to
`src/mlops_crew/models/serving.py`, which applies the same conservative text
cleaning used before training.

## Local Usage

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8080
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Urgent account verification required. Click the secure link now."}'
```

## Environment Variables

- `MODEL_PATH` - local path to `best_model.joblib`; defaults to
  `models/best_model.joblib`.
- `MODEL_GCS_URI` - optional `gs://bucket/path/model.joblib` source for Cloud
  Run or Cloud Functions.
- `MODEL_CACHE_DIR` - optional local cache for GCS model downloads.
- `MODEL_VERSION` - response metadata, default `phase2_linear_svc`.
