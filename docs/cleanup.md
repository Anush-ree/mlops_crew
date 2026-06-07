# GCP Resource Cleanup

Run these commands after screenshots and demo are captured. All resources are in `us-central1` under project `ml-ops-497304`.
You don't run this file except you run it after you're done with the project.
---

## 1. Delete Cloud Run Service

```bash
gcloud run services delete mlops-crew-api \
  --region=us-central1 \
  --project=ml-ops-497304 \
  --quiet
```

---

## 2. Delete Cloud Function

```bash
gcloud functions delete mlops-crew-predict \
  --gen2 \
  --region=us-central1 \
  --project=ml-ops-497304 \
  --quiet
```

---

## 3. Delete Artifact Registry Repository

```bash
gcloud artifacts repositories delete mlops-crew \
  --location=us-central1 \
  --project=ml-ops-497304 \
  --quiet
```

---

## 4. Delete GCS Bucket

```bash
gsutil -m rm -r gs://mlops-crew-data
```

> **Warning:** This permanently deletes all training data and model artifacts stored in GCS. Make sure local copies exist before running.

---

## 5. Delete Service Account

```bash
gcloud iam service-accounts delete mlops-crew-sa@ml-ops-497304.iam.gserviceaccount.com \
  --project=ml-ops-497304 \
  --quiet
```

---

## 6. Disable APIs (optional — stops future charges)

```bash
gcloud services disable \
  run.googleapis.com \
  cloudfunctions.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  aiplatform.googleapis.com \
  --project=ml-ops-497304 \
  --quiet
```

> Only disable if you are done with the project entirely. Re-enabling APIs takes a few minutes.

---

## 7. Verify No Billable Resources Remain

Check the GCP Console billing dashboard or run:

```bash
gcloud run services list --region=us-central1 --project=ml-ops-497304
gcloud functions list --project=ml-ops-497304
gcloud artifacts repositories list --location=us-central1 --project=ml-ops-497304
gsutil ls
```

All commands should return empty lists after cleanup.
