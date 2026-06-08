# Phishing Email Detection

**SE489 · ML Engineering for Production (MLOps) · DePaul University**

**Live Demo:** [Try the phishing email detector](https://huggingface.co/spaces/mlops-crew-depaul/phishing-email-detector)

**Video Demo** [▶ Watch Demo Video](https://drive.google.com/file/d/11pSo_hvUUvmee0k4NPlNiJT7gbRFSxBW/view)

## 1. Team

- Team Name: MLOps Crew
- Team Members (Name & Email):
  1. Anushree Bachhav ([abachhav@depaul.edu](mailto:abachhav@depaul.edu))
  2. Krishna Kalakonda ([kkalakon@depaul.edu](mailto:kkalakon@depaul.edu))
  3. Muhammad Anas ([MuhammadAnasPSI2@gmail.com](MuhammadAnasPSI2@gmail.com))
  4. Kirtankumar Parekh ([kparekh2@depaul.edu](mailto:kparekh2@depaul.edu))
- Course & Section: [SE489] ML Engineering for Production (MLOps)

## 2. Project overview

Phishing emails are a leading cybersecurity threat. This project trains a
reproducible binary classifier that decides whether an incoming email is
phishing or legitimate, and ships it as part of an end-to-end MLOps pipeline:
data versioning (DVC), reproducible stages, deterministic training, and tracked
metrics. We optimize for **recall** — missing a phishing email is worse than
flagging a legitimate one — and select models by **F2**.

Phase 1 trained on a stratified 60% sample of `phishing_email.csv`. Phase 2
uses 80% of the raw data, preserves the final 20% as a Phase 3 holdout, tracks
experiments with MLflow, and adds monitoring, profiling, and divergence reports.
Phase 3 deploys the model as a live Gradio app on Hugging Face Spaces with a
FastAPI backend on GCP Cloud Run.

## 3. Architecture

<img width="776" height="662" alt="image" src="https://github.com/user-attachments/assets/2aa3ed2a-427e-4ddb-b2e8-58e3d4a225c6" />

## 4. Phase deliverables

- [PHASE1.md](./PHASE1.md) — Project design & baseline model
- [PHASE2.md](./PHASE2.md) — Enhancing ML operations
- [PHASE3.md](./PHASE3.md) — Continuous ML & deployment
- [Live Demo](https://huggingface.co/spaces/mlops-crew-depaul/phishing-email-detector) —
  Gradio UI on Hugging Face Spaces


## 5. Phase 3 — Continuous ML & Deployment

For full evidence and explanations see [PHASE3.md](./PHASE3.md).

### What's new in Phase 3

| Area | What was added |
| --- | --- |
| CI/CD | GitHub Actions workflows for lint, type check, test, Docker build/push, and CML reporting |
| Docker | `serve.dockerfile` — Cloud Run-ready FastAPI serving image, auto-pushed to Docker Hub on merge to main |
| CML | Automated model metrics report posted as a PR comment on every pull request |
| FastAPI | `/health` and `/predict` endpoints — [`api/main.py`](api/main.py) |
| GCP Cloud Run | Live inference API — <https://mlops-crew-api-1043076962701.us-central1.run.app> |
| GCP Cloud Functions | Serverless HTTP wrapper — <https://us-central1-ml-ops-497304.cloudfunctions.net/mlops-crew-predict> |
| Hugging Face Spaces | Live Gradio demo — <https://huggingface.co/spaces/mlops-crew-depaul/phishing-email-detector> |
| Phase 3 holdout | Final evaluation on 16,496 unseen emails — **98.6% F2**, results in [`reports/metrics/phase3_holdout_metrics.json`](reports/metrics/phase3_holdout_metrics.json) |

### Demo recording

> Recording will be added here.

### Quick API test

- **Health check** (open in browser): <https://mlops-crew-api-1043076962701.us-central1.run.app/health>
- **Interactive docs** (open in browser): <https://mlops-crew-api-1043076962701.us-central1.run.app/docs>
- **Predict** (POST only, use curl):

```bash
curl -X POST https://mlops-crew-api-1043076962701.us-central1.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Urgent: verify your account now or it will be suspended."}'
```

---

## 6. Setup

### Windows, macOS, and Linux

We support **native Windows (PowerShell)**, **Git Bash**, **WSL**, and Unix
shells. Graders on Windows can follow [docs/windows_setup.md](docs/windows_setup.md)
for Chocolatey/`make`, AWS CLI, and verification without bash.

**PowerShell (Windows):**

```powershell
git clone https://github.com/Anush-ree/mlops_crew.git
cd mlops_crew
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements_dev.txt
pip install -e .
aws configure          # region: us-east-2
dvc pull
.\scripts\verify_phase2.ps1
```

**Bash (Linux / macOS / WSL / Git Bash):**

```bash
git clone https://github.com/Anush-ree/mlops_crew.git
cd mlops_crew
python -m venv .venv && source .venv/bin/activate   # Windows Git Bash: .venv/Scripts/activate
make install            # or: pip install -r requirements.txt && pip install -e .
make dev                # adds dev tools and pre-commit hooks
```

`make` is optional on Windows; see [docs/windows_setup.md](docs/windows_setup.md)
for direct `pip` / `dvc` / `pytest` equivalents.

### Data access (DVC + AWS S3)

Data is versioned with DVC and stored on S3 (Google Drive is kept as a backup
remote). Request AWS credentials from a teammate, then:

Bash:

```bash
git pull
pip install dvc-s3
aws configure          # region: us-east-2
dvc pull               # download raw + processed data
```

### Containerized training and prediction

The repository includes two code-only Docker images:

- [train.dockerfile](./train.dockerfile) builds the training image and runs
  `mlops_crew.models.train_model`.
- [predict.dockerfile](./predict.dockerfile) builds the prediction image and
  runs `mlops_crew.models.predict_model`.

Both images rely on host-mounted project files instead of copying data into the
image. This keeps the build small and makes the container use the same DVC-
tracked inputs as the local workflow.

Build them from the repo root:

```bash
docker build -f train.dockerfile . -t train:latest
docker build -f predict.dockerfile . -t predict:latest
```

Run training with the local DVC data and config mounted into `/app`:

```bash
docker run --rm \
  -e MLOPS_CREW_PROJECT_ROOT=/app \
  -v "$PWD/data:/app/data" \
  -v "$PWD/configs:/app/configs" \
  train:latest
```

Run prediction with the saved model, processed test data, and output folder
mounted from the host:

```bash
docker run --rm \
  -e MLOPS_CREW_PROJECT_ROOT=/app \
  -v "$PWD/data:/app/data" \
  -v "$PWD/configs:/app/configs" \
  -v "$PWD/models:/app/models" \
  -v "$PWD/reports:/app/reports" \
  predict:latest \
  --model-path /app/models/best_model.joblib \
  --input /app/data/processed/test.csv \
  --output /app/reports/predictions/batch_predictions.csv
```

If you prefer the one-command wrappers, use `make docker-train` and
`make docker-predict`.

### Phase 3 local API and UI

Phase 3 adds a FastAPI service and a Gradio UI on top of the Phase 2 saved
model. The API uses the same cleaning settings from `configs/config.yaml`
before calling the saved sklearn `Pipeline(TfidfVectorizer -> classifier)`.

Run the API locally:

```bash
make api
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Urgent account verification required. Click the secure link now."}'
```

Run the Gradio UI locally in another terminal:

```bash
make ui
```

Build and run the Cloud Run-ready serving image:

```bash
make docker-serve
curl http://localhost:8080/health
```

The serving image is code-only like the Phase 2 images. Local Docker runs mount
`models/`; Cloud Run should set `MODEL_GCS_URI` to a GCS-hosted
`best_model.joblib`. For Hugging Face Spaces, deploy the contents of
`hf_space/` and set the Space secret `BACKEND_PREDICT_URL` to the deployed
`/predict` endpoint.

### Tool Documentation

For complete setup and usage instructions for every tool integrated into this project,
see [docs/TOOL_DOCUMENTATION.md](./docs/TOOL_DOCUMENTATION.md). This guide covers:

- Purpose and setup for each tool
- How to use tools in the project
- Integration summary and quick reference
- External collaborators should start here before cloning and reproducing.

### Common commands

Bash:

```bash
make data       # sample, source manifest, clean, split, validate, transformer export
make train      # train all configured models, write metrics + predictions
make predict    # score the test split with the saved best model
make source-manifest      # build raw source metadata for divergence analysis
make transformer-data     # export JSONL train/val/test splits for transformer work
make latency              # benchmark saved-model inference latency
make divergence           # compare Phase 1 reference vs Phase 2 increment
make profile-train        # cProfile the training entrypoint
make profile-predict      # cProfile saved-model inference
make hydra-demo           # run two Hydra-configured MLflow experiments
make mlflow-ui            # open local MLflow UI on port 5001
make docker-train         # build and run the training container
make docker-predict       # build and run the prediction container
make api                  # start the Phase 3 FastAPI service on port 8080
make ui                   # start the Gradio UI from hf_space/
make docker-serve         # build/run the Cloud Run-ready API container
make load-test-api        # send 10 prediction requests to BACKEND_PREDICT_URL
make repro      # reproduce the full DVC pipeline end to end
scripts/verify_phase2.sh  # Bash: DVC repro + CI checks + Phase 2 smoke checks
scripts/verify_phase2.ps1 # PowerShell (Windows): same checks as the .sh script
scripts/verify_phase2.sh --replay-mlflow  # populate local MLflow from scratch outputs
make test       # pytest
make lint       # ruff check
make format     # ruff fix + format
```

### Reproduce results

Bash:

```bash
make install
dvc pull
make repro
scripts/verify_phase2.sh
```

This fits the configured dummy and TF-IDF model family, writes artifacts under
`models/` and `reports/`, and runs the full Phase 2 DVC graph:

```text
sample -> clean -> split -> validate -> transformer_dataset
                              -> train -> inference_latency
                          -> plot_model_comparison
sample + source_manifest + train -> divergence
```

## 7. Repo layout

See **§6 Setup → Data access** for `dvc pull`. Key paths:

```markdown
configs/config.yaml                 single source of truth for the pipeline
configs/hydra/                      Hydra experiment overrides
dvc.yaml                            DVC stages (sample → … → train → monitoring)
data/processed/validation_report.json   DVC validate artifact (row/label snapshot)
src/mlops_crew/
  data/
    sample.py                       phase partitions + 80% modeling sample
    source_manifest.py              raw source-block metadata (divergence)
    clean.py, split.py, validate.py
    export_transformer_dataset.py   JSONL export for future transformer work
    make_dataset.py                 local equivalent of data stages through validate
  models/text_classifiers.py        TF-IDF + classifier sklearn pipelines
  models/train_model.py             Phase 2 training CLI + implementation
  models/predict_model.py           Phase 2 prediction CLI + implementation
  train_hydra.py                    Hydra experiment wrapper
  evaluation/, monitoring/, tracking/
api/                                FastAPI service entrypoint and schemas
hf_space/                           Gradio app for Hugging Face Spaces
serve.dockerfile                    Cloud Run-ready API serving image
scripts/verify_phase2.ps1|.sh       grader verification (Windows + Bash)
PHASE2.md                           Phase 2 deliverable narrative
PHASE3.md                           Phase 3 deliverable narrative
docs/TOOL_DOCUMENTATION.md          Complete tool setup and usage guide
docs/windows_setup.md               Windows reproduction guide
```

## 8. Contributions

- **Anushree Bachhav** — proposal, repo setup, cookiecutter setup, DVC remotes, Dockerfiles, documentation; holdout evaluation, model comparison plot, PHASE3.md evidence report, README updates, cleanup docs, phase3 evidence screenshots
- **Muhammad Anas** — data cleaning, EDA, normalization, train/val/test splits, data documentation; Gradio UI, Hugging Face Spaces deployment
- **Krishna Kalakonda** — model evaluation, baseline metrics, architecture diagram, code organization; FastAPI backend, Cloud Run, Cloud Functions, GCP deployment, CI/Docker/CML workflows, load testing
- **Kirtankumar Parekh** — Phase 2 integration review, Windows verification (`verify_phase2.ps1`, `docs/windows_setup.md`), DVC validate stage, docs and docstrings, CI/CD pipelines, repo maintenance

## 9. References

- Dataset: Phishing Email Dataset (SpamAssassin, Enron, Nazario, Ling, CEAS, Nigerian)
- Source: <https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset?select=SpamAssasin.csv>
- Use: Primary training & evaluation data
