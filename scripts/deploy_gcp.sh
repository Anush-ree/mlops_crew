#!/usr/bin/env bash
set -euo pipefail

# Deploy the Phase 3 FastAPI serving stack to GCP:
# 1. Build and push the Cloud Run image to Artifact Registry.
# 2. Upload the local trained model to Cloud Storage.
# 3. Deploy Cloud Run with MODEL_GCS_URI.
# 4. Deploy the Cloud Functions HTTP wrapper.
# 5. Run smoke tests against both endpoints.
#
# Usage:
#   GCP_PROJECT_ID=ml-ops-497304 scripts/deploy_gcp.sh
#
# Optional overrides:
#   GCP_REGION=us-central1
#   GCP_AR_REPO=mlops-crew
#   GCP_IMAGE_NAME=api
#   IMAGE_TAG=latest
#   CLOUD_RUN_SERVICE=mlops-crew-api
#   CLOUD_FUNCTION_NAME=mlops-crew-predict
#   MODEL_BUCKET=<project-id>-mlops-crew-models
#   MODEL_PATH=models/best_model.joblib
#   MODEL_VERSION=phase3
#   RUN_SA=mlops-crew-runner

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf 'ERROR: required command not found: %s\n' "$1" >&2
    exit 1
  fi
}

require_command gcloud
require_command docker
require_command curl

GCP_PROJECT_ID="${GCP_PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || true)}"
if [[ -z "$GCP_PROJECT_ID" ]]; then
  printf 'ERROR: set GCP_PROJECT_ID or run `gcloud config set project <project-id>`.\n' >&2
  exit 1
fi

GCP_REGION="${GCP_REGION:-us-central1}"
GCP_AR_REPO="${GCP_AR_REPO:-mlops-crew}"
GCP_IMAGE_NAME="${GCP_IMAGE_NAME:-api}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
CLOUD_RUN_SERVICE="${CLOUD_RUN_SERVICE:-mlops-crew-api}"
CLOUD_FUNCTION_NAME="${CLOUD_FUNCTION_NAME:-mlops-crew-predict}"
MODEL_BUCKET="${MODEL_BUCKET:-${GCP_PROJECT_ID}-mlops-crew-models}"
MODEL_PATH="${MODEL_PATH:-models/best_model.joblib}"
MODEL_VERSION="${MODEL_VERSION:-phase3}"
RUN_SA="${RUN_SA:-mlops-crew-runner}"

IMAGE_URI="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${GCP_AR_REPO}/${GCP_IMAGE_NAME}:${IMAGE_TAG}"
MODEL_GCS_URI="gs://${MODEL_BUCKET}/models/$(basename "$MODEL_PATH")"
RUN_SA_EMAIL="${RUN_SA}@${GCP_PROJECT_ID}.iam.gserviceaccount.com"
PROJECT_NUMBER="$(
  gcloud projects describe "$GCP_PROJECT_ID" \
    --format='value(projectNumber)'
)"
DEFAULT_COMPUTE_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

printf '==> Project: %s\n' "$GCP_PROJECT_ID"
printf '==> Region: %s\n' "$GCP_REGION"
printf '==> Image: %s\n' "$IMAGE_URI"
printf '==> Model: %s -> %s\n' "$MODEL_PATH" "$MODEL_GCS_URI"

if [[ ! -f "$MODEL_PATH" ]]; then
  printf 'ERROR: model file not found: %s\n' "$MODEL_PATH" >&2
  printf 'Run `make train` or copy the trained model into models/ first.\n' >&2
  exit 1
fi

printf '\n==> Enabling required GCP APIs\n'
gcloud services enable \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  cloudfunctions.googleapis.com \
  iam.googleapis.com \
  run.googleapis.com \
  storage.googleapis.com \
  --project="$GCP_PROJECT_ID"

printf '\n==> Ensuring Artifact Registry repository exists\n'
if ! gcloud artifacts repositories describe "$GCP_AR_REPO" \
  --location="$GCP_REGION" \
  --project="$GCP_PROJECT_ID" >/dev/null 2>&1; then
  gcloud artifacts repositories create "$GCP_AR_REPO" \
    --repository-format=docker \
    --location="$GCP_REGION" \
    --description="MLOps Crew Phase 3 API images" \
    --project="$GCP_PROJECT_ID"
fi

printf '\n==> Configuring Docker authentication for Artifact Registry\n'
gcloud auth configure-docker "${GCP_REGION}-docker.pkg.dev" --quiet

printf '\n==> Building and pushing linux/amd64 serving image\n'
docker buildx create --use --name mlops-crew-builder >/dev/null 2>&1 || \
  docker buildx use mlops-crew-builder
docker buildx build \
  --platform linux/amd64 \
  -f serve.dockerfile \
  -t "$IMAGE_URI" \
  --push \
  .

printf '\n==> Ensuring Cloud Storage bucket exists\n'
if ! gcloud storage buckets describe "gs://${MODEL_BUCKET}" >/dev/null 2>&1; then
  gcloud storage buckets create "gs://${MODEL_BUCKET}" \
    --location="$GCP_REGION" \
    --uniform-bucket-level-access
fi

printf '\n==> Uploading model artifact\n'
gcloud storage cp "$MODEL_PATH" "$MODEL_GCS_URI"
gcloud storage ls "$MODEL_GCS_URI"

printf '\n==> Ensuring Cloud Run runtime service account exists\n'
if ! gcloud iam service-accounts describe "$RUN_SA_EMAIL" \
  --project="$GCP_PROJECT_ID" >/dev/null 2>&1; then
  gcloud iam service-accounts create "$RUN_SA" \
    --display-name="MLOps Crew Cloud Run runtime" \
    --project="$GCP_PROJECT_ID"
fi

printf '\n==> Granting model read access to runtime service account\n'
gcloud projects add-iam-policy-binding "$GCP_PROJECT_ID" \
  --member="serviceAccount:${RUN_SA_EMAIL}" \
  --role="roles/storage.objectViewer" \
  --condition=None >/dev/null

printf '\n==> Ensuring Cloud Functions build service account has Cloud Build permissions\n'
gcloud projects add-iam-policy-binding "$GCP_PROJECT_ID" \
  --member="serviceAccount:${DEFAULT_COMPUTE_SA}" \
  --role="roles/cloudbuild.builds.builder" \
  --condition=None >/dev/null

printf '\n==> Deploying Cloud Run service\n'
gcloud run deploy "$CLOUD_RUN_SERVICE" \
  --image="$IMAGE_URI" \
  --region="$GCP_REGION" \
  --project="$GCP_PROJECT_ID" \
  --allow-unauthenticated \
  --port=8080 \
  --memory=1Gi \
  --service-account="$RUN_SA_EMAIL" \
  --set-env-vars="MODEL_GCS_URI=${MODEL_GCS_URI},MODEL_VERSION=${MODEL_VERSION}"

SERVICE_URL="$(
  gcloud run services describe "$CLOUD_RUN_SERVICE" \
    --region="$GCP_REGION" \
    --project="$GCP_PROJECT_ID" \
    --format='value(status.url)'
)"
printf '==> Cloud Run URL: %s\n' "$SERVICE_URL"

printf '\n==> Smoke testing Cloud Run\n'
curl -fsS "${SERVICE_URL}/health"
printf '\n'
curl -fsS -X POST "${SERVICE_URL}/predict" \
  -H "Content-Type: application/json" \
  -d '{"text":"Urgent account verification required. Click this link now."}'
printf '\n'

printf '\n==> Deploying Cloud Functions wrapper\n'
gcloud functions deploy "$CLOUD_FUNCTION_NAME" \
  --gen2 \
  --runtime=python311 \
  --region="$GCP_REGION" \
  --project="$GCP_PROJECT_ID" \
  --source=functions/predict \
  --entry-point=predict \
  --trigger-http \
  --allow-unauthenticated \
  --set-env-vars="BACKEND_PREDICT_URL=${SERVICE_URL}/predict"

FUNCTION_URL="$(
  gcloud functions describe "$CLOUD_FUNCTION_NAME" \
    --gen2 \
    --region="$GCP_REGION" \
    --project="$GCP_PROJECT_ID" \
    --format='value(serviceConfig.uri)'
)"
printf '==> Cloud Function URL: %s\n' "$FUNCTION_URL"

printf '\n==> Smoke testing Cloud Function\n'
curl -fsS -X POST "$FUNCTION_URL" \
  -H "Content-Type: application/json" \
  -d '{"text":"Please review the attached team meeting notes."}'
printf '\n'

cat <<EOF

Deployment complete.

Cloud Run:
  ${SERVICE_URL}

Cloud Function:
  ${FUNCTION_URL}

Model artifact:
  ${MODEL_GCS_URI}

EOF
