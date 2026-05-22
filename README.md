# Phishing Email Detection

**SE489 · ML Engineering for Production (MLOps) · DePaul University**

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

## 3. Architecture

<img width="776" height="662" alt="image" src="https://github.com/user-attachments/assets/2aa3ed2a-427e-4ddb-b2e8-58e3d4a225c6" />

## 4. Phase deliverables

- [PHASE1.md](./PHASE1.md) — Project design & baseline model
- [PHASE2.md](./PHASE2.md) — Enhancing ML operations
- [PHASE3.md](./PHASE3.md) — Continuous ML & deployment

## 5. Setup

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
make docker-train         # build/run the training container with host-mounted artifacts
make docker-predict       # build/run the prediction container with host-mounted artifacts
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

### Docker training and prediction

Phase 2 includes separate containers for training and prediction:

```bash
docker build -f train.dockerfile . -t train:latest
docker build -f predict.dockerfile . -t predict:latest
```

After `dvc pull` has restored the data and model artifacts, the Make targets run
the containers with the required host mounts:

```bash
make docker-train
make docker-predict
```

`make docker-train` mounts `data/`, `configs/`, `models/`, `reports/`, `logs/`,
and `mlruns/` so training outputs are preserved on the host. `make
docker-predict` mounts `data/`, `configs/`, `models/`, and `reports/`, and
writes `reports/predictions/batch_predictions.csv`.

## 6. Repo layout

See **§5 Setup → Data access** for `dvc pull`. Key paths:

```
configs/config.yaml                 single source of truth for the pipeline
conf/                               Hydra experiment overrides
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
scripts/verify_phase2.ps1|.sh       grader verification (Windows + Bash)
train.dockerfile                    containerized model training
predict.dockerfile                  containerized batch prediction
PHASE2.md                           Phase 2 deliverable narrative
docs/windows_setup.md               Windows reproduction guide
```

## 7. Contributions

- **Anushree Bachhav** — proposal, repo/cookiecutter setup, environment
  configuration, DVC + S3 + Google Drive remotes
- **Muhammad Anas** — data cleaning, EDA, normalization, train/val/test splits,
  data documentation
- **Krishna Kalakonda** — model evaluation, baseline metrics, architecture
  diagram, code organization
- **Kirtankumar Parekh** — Phase 2 integration review, Windows verification
  (`verify_phase2.ps1`, `docs/windows_setup.md`), DVC validate stage, docs and
  docstrings, repo maintenance

## 8. References

- Dataset: Phishing Email Dataset (SpamAssassin, Enron, Nazario, Ling, CEAS, Nigerian)
- Source: https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset?select=SpamAssasin.csv
- Use: Primary training & evaluation data
