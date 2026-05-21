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

Bash:
```bash
git clone https://github.com/Anush-ree/mlops_crew.git
cd mlops_crew
python -m venv .venv && source .venv/bin/activate
make install            # installs runtime deps + the package in editable mode
make dev                # adds dev tools and pre-commit hooks
```

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
make data       # sample, source manifest, clean, split, transformer export, validate
make train      # train all configured models, write metrics + predictions
make predict    # score the test split with the saved best model
make source-manifest      # build raw source metadata for divergence analysis
make transformer-data     # export JSONL train/val/test splits for transformer work
make latency              # benchmark saved-model inference latency
make divergence           # compare Phase 1 reference vs Phase 2 increment
make profile-train        # cProfile the training entrypoint
make profile-predict      # cProfile saved-model inference
make mlflow-ui            # open local MLflow UI on port 5001
make repro      # reproduce the full DVC pipeline end to end
scripts/verify_phase2.sh  # run DVC repro + CI checks + Phase 2 smoke checks
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
sample -> clean -> split -> transformer_dataset
                 -> train -> inference_latency
                          -> plot_model_comparison
sample + source_manifest + train -> divergence
```

## 6. Repo layout

Data is versioned with DVC and stored on AWS S3. Request AWS credentials then run.
Bash:
```bash
git pull                 # get latest config
uv pip install dvc-s3   # install DVC S3 plugin
aws configure           # enter credentials + region: us-east-2
dvc pull                # download data from S3
```
File Structure:
```
configs/config.yaml            single source of truth for the pipeline
src/mlops_crew/
  config.py                    project paths + YAML loader
  logging_config.py            shared logger setup
  data/
    sample.py                  stage 1: stratified sample of the raw CSV
    clean.py                   stage 2: schema, labels, text cleaning
    split.py                   stage 3: stratified train/val/test split
    validate.py                stage 4: post-split sanity checks
    source_manifest.py         source-block metadata for divergence monitoring
    export_transformer_dataset.py
    make_dataset.py            run all four stages locally
  models/text_classifiers.py   TF-IDF + classifier sklearn pipeline factory
  evaluation/metrics.py        recall-oriented binary metrics (F2 + confusion)
  evaluation/plot_model_comparison.py
  monitoring/                  resource usage, latency, divergence reports
  tracking/                    MLflow wrapper
  utils/                       seed + JSON helpers
  train_model.py               train every configured model, save artifacts
  predict_model.py             batch inference with the saved pipeline
dvc.yaml                       DVC stages for data, training, monitoring, reports
```

## 7. Contributions

- **Anushree Bachhav** — proposal, repo/cookiecutter setup, environment
  configuration, DVC + S3 + Google Drive remotes
- **Muhammad Anas** — data cleaning, EDA, normalization, train/val/test splits,
  data documentation
- **Krishna Kalakonda** — model evaluation, baseline metrics, architecture
  diagram, code organization
- **Kirtankumar Parekh** — model training, experiment tracking, Makefile,
  contribution guidelines, repo maintenance

## 8. References

- Dataset: Phishing Email Dataset (SpamAssassin, Enron, Nazario, Ling, CEAS, Nigerian)
- Source: https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset?select=SpamAssasin.csv
- Use: Primary training & evaluation data
