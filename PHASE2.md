# Phase 2 Report: Enhancing ML Operations

This report summarizes the Phase 2 implementation and how to reproduce the
outputs.

## What Phase 2 Adds

Phase 2 moves the project from the Phase 1 60% baseline to an 80% data workflow.
The final 20% of the data is still reserved for Phase 3.

Implemented in this phase:

- deterministic DVC data partitions for Phase 1 reference, Phase 2 increment,
  Phase 2 sample, and Phase 3 holdout
- split validation and source-manifest generation
- monitoring reports for resource usage, inference latency, and divergence
- cProfile scripts for training and inference
- MLflow experiment tracking for model runs and artifacts
- Rich console logging and rotating runtime file logs
- Hydra experiment runs for config override demonstrations
- TF-IDF model comparison across dummy, logistic regression, linear SVC, and
  Complement NB
- transformer-ready JSONL dataset export for future fine-tuning
- Docker containerization

## How to Reproduce

Run these commands from the repository root.

### 1. Create the Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements_dev.txt
pip install -e .
```

Windows PowerShell users should follow `docs/windows_setup.md` and run the
PowerShell verification command instead of the Bash script shown later:

```powershell
.\scripts\verify_phase2.ps1 -ReplayMlflow -CheckRemote
```

### 2. Restore DVC Artifacts

```bash
dvc pull
dvc status
dvc status -c
```

`dvc pull` and `dvc status -c` require access to the configured DVC remote.

Expected status:

```text
Data and pipelines are up to date.
Cache and remote 'storage' are in sync.
```

### 3. Reproduce the Pipeline

```bash
dvc repro
```

This regenerates the data partitions, cleaned data, train/validation/test
splits, validation report, transformer dataset, trained models, metrics,
monitoring reports, plots, and divergence report if any dependency changed.

### 4. Run Code Quality and Tests

```bash
ruff check --no-cache .
ruff format --check .
mypy src
pytest tests/ --cov=mlops_crew --cov-report=xml
```

### 5. Run the Phase 2 Verification Script

```bash
scripts/verify_phase2.sh --replay-mlflow --check-remote
```



Expected summary:

```text
Best model: linear_svc
Validation F2: 0.992381
Test F2: 0.991182
Test false-negative rate: 0.007778
Label JS distance: 0.000007
Source JS distance: 0.014327
Transformer JSONL rows: test=9862, train=46020, val=9861
```

## Optional Commands

Run these only when the related evidence needs to be regenerated.

### Docker Training and Prediction

Build both images:

```bash
docker build -f train.dockerfile . -t train:latest
docker build -f predict.dockerfile . -t predict:latest
```

Run the containerized training and prediction flows:

```bash
make docker-train
make docker-predict
```

`make docker-train` expects DVC data artifacts to already exist from `dvc pull`.
It mounts `data/`, `configs/`, `models/`, `reports/`, `logs/`, and `mlruns/`
so training outputs are saved back to the host.

`make docker-predict` expects `models/best_model.joblib` and
`data/processed/test.csv` to exist. It writes:

```text
reports/predictions/batch_predictions.csv
```

### MLflow UI

```bash
scripts/verify_phase2.sh --replay-mlflow
make mlflow-ui
```

Open:

```text
http://localhost:5001
```

`make mlflow-ui` starts a long-running local server. Stop it with `Ctrl+C`.

### Hydra Demo

```bash
make hydra-demo
```

This runs:

```bash
python -m mlops_crew.train_hydra experiment=phase2_default
python -m mlops_crew.train_hydra experiment=phase2_experimental
```

Hydra outputs are written under ignored `outputs/hydra/...` directories and are
logged to MLflow.

### Profiling

```bash
make profile-train
make profile-predict
```

`make profile-train` is slow because it profiles model training. The
verification script runs only the lighter profiling smoke check unless
`--include-slow-profile` is passed.

### Individual Monitoring Commands

```bash
python -m mlops_crew.data.validate
python -m mlops_crew.monitoring.inference_latency
python -m mlops_crew.monitoring.divergence
```

## Results

### Data Partitions

| Partition | Rows | Use |
| --- | ---: | --- |
| Phase 1 reference | 49,492 | Previous 60% baseline reference |
| Phase 2 increment | 16,497 | New 20% added in Phase 2 |
| Phase 2 sample | 65,989 | Data used for this phase |
| Phase 3 holdout | 16,497 | Reserved for final phase |

The existing combined raw dataset is used as the source of truth. Phase 2 does
not merge the raw source CSVs again. Source membership is tracked through
`data/interim/source_manifest.csv`.

### Model Summary

All models use the same splits, TF-IDF settings, and evaluation metrics. The
primary selection metric is validation F2 because phishing false negatives are
costly.

| Model | Val F2 | Test F2 | Test Recall | Test FNR |
| --- | ---: | ---: | ---: | ---: |
| Dummy | 0.8449 | 0.8449 | 1.0000 | 0.0000 |
| Logistic Regression | 0.9882 | 0.9903 | 0.9918 | 0.0082 |
| Linear SVC | **0.9924** | **0.9912** | 0.9922 | 0.0078 |
| Complement NB | 0.9453 | 0.9512 | 0.9417 | 0.0583 |

Selected model: `linear_svc`

Phase comparison:

| Phase | Best model | Val F2 | Test F2 | Test Recall | Test FNR |
| --- | --- | ---: | ---: | ---: | ---: |
| Phase 1, 60% data | logistic_regression | 0.9882 | 0.9867 | 0.9881 | 0.0119 |
| Phase 2, 80% data | linear_svc | 0.9924 | 0.9912 | 0.9922 | 0.0078 |

### Divergence Summary

| Check | Value |
| --- | ---: |
| Label Jensen-Shannon distance | 0.000007 |
| Source Jensen-Shannon distance | 0.014327 |
| Text length KS statistic | 0.005134 |
| New-token rate in Phase 2 increment | 0.039855 |
| Prediction Jensen-Shannon distance | 0.000207 |

The Phase 2 increment is close to the Phase 1 reference by label distribution
and prediction distribution. Source distribution is measured and should continue
to be reviewed when new data is added.

## Artifact Index

| Output | Path |
| --- | --- |
| Training Dockerfile | `train.dockerfile` |
| Prediction Dockerfile | `predict.dockerfile` |
| Docker ignore file | `.dockerignore` |
| DVC pipeline | `dvc.yaml`, `dvc.lock` |
| Phase partitions | `data/interim/phishing_email_phase*.csv` |
| Source manifest | `data/interim/source_manifest.csv` |
| Split validation report | `data/processed/validation_report.json` |
| Train/validation/test splits | `data/processed/train.csv`, `val.csv`, `test.csv` |
| Best model | `models/best_model.joblib` |
| Per-model artifacts | `models/*.joblib` |
| Metrics | `reports/metrics/*_metrics.json` |
| Model comparison table | `reports/metrics/model_comparison.csv` |
| Model comparison plot | `reports/metrics/model_comparison.png` |
| Predictions | `reports/predictions/*_predictions.csv` |
| Training resource usage | `reports/monitoring/training_resource_usage.csv` |
| Inference latency | `reports/monitoring/inference_latency.csv` |
| Divergence report | `reports/divergence/phase2_divergence_report.json` |
| Divergence summary | `reports/divergence/phase2_divergence_summary.md` |
| Training profile | `reports/profiling/train_model_cprofile.txt` |
| Inference profile | `reports/profiling/predict_model_cprofile.txt` |
| MLflow screenshots | `reports/mlflow/` |
| Logging screenshot | `reports/logging/rich_logging.png` |
| Transformer JSONL dataset | `data/processed/transformer/` |

## Transformer Dataset

Transformer or LLM fine-tuning was not run in Phase 2. The dataset was prepared
for future use.

| Split | Rows | Path |
| --- | ---: | --- |
| Train | 46,020 | `data/processed/transformer/train.jsonl` |
| Validation | 9,861 | `data/processed/transformer/val.jsonl` |
| Test | 9,862 | `data/processed/transformer/test.jsonl` |

The JSONL records contain `text` and `label` fields and will be used with
Hugging Face Datasets in a next transformer experiment.

## Phase 3 Notes

Phase 3 should use the reserved 20% holdout only after the final model family is
selected. Transformer fine-tuning can start from the JSONL dataset prepared in
this phase.
