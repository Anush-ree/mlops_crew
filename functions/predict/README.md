# Cloud Functions Prediction Wrapper

This function provides the Cloud Functions HTTP endpoint. It forwards
JSON prediction requests to the deployed FastAPI `/predict` endpoint on Cloud
Run so both serving paths use the same model logic.

Deploy after Cloud Run is live:

```bash
gcloud functions deploy mlops-crew-predict \
  --gen2 \
  --runtime=python311 \
  --region=us-central1 \
  --source=functions/predict \
  --entry-point=predict \
  --trigger-http \
  --allow-unauthenticated \
  --set-env-vars BACKEND_PREDICT_URL="<cloud-run-url>/predict"
```

Test:

```bash
curl -X POST "<function-url>" \
  -H "Content-Type: application/json" \
  -d '{"text":"Urgent account verification required."}'
```
