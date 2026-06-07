# API Reference

The package is importable as `mlops_crew` after running `pip install -e .`.

## `mlops_crew.config`

Project-wide path constants and the YAML config loader.

```python
from mlops_crew.config import CONFIG_PATH, PROJECT_ROOT, load_project_config, resolve_project_path

config = load_project_config(CONFIG_PATH)
raw_path = resolve_project_path(config["data"]["raw_dir"]) / config["data"]["raw_file"]
```

Use `resolve_project_path` instead of hard-coded relative paths; it resolves
against the repo root regardless of the current working directory.

## `mlops_crew.logging_config`

```python
from mlops_crew.logging_config import setup_logging_from_config, get_logger

setup_logging_from_config(config)
logger = get_logger(__name__)
```

## `mlops_crew.data`

| Function | Purpose |
|---|---|
| `sample.run(config)` | Create Phase 1 reference, Phase 2 increment, Phase 2 sample, and Phase 3 holdout CSVs |
| `source_manifest.run(config)` | Create source-block metadata for divergence monitoring |
| `clean.run(config)` | Normalize schema, labels, and email text |
| `split.run(config)` | Create deterministic train/validation/test splits |
| `validate.run(config)` | Sanity-check cleaned and split CSVs; writes `data/processed/validation_report.json` on success |
| `validate.validation_report_path(config)` | Path to the DVC-tracked validation report |
| `export_transformer_dataset.export_transformer_dataset(config)` | Export HF-compatible JSONL train/validation/test splits |
| `make_dataset.process_data(config_path)` | Run sample → source manifest → clean → split → validate → transformer export (same order as `dvc repro` through validate) |

CLI: `python -m mlops_crew.data.make_dataset`

## `mlops_crew.models`

```python
from mlops_crew.models import build_text_classifier

model = build_text_classifier("logistic_regression", config)
```

The returned object is an unfitted sklearn
`Pipeline(TfidfVectorizer -> classifier)`.

## `mlops_crew.evaluation`

```python
from mlops_crew.evaluation import binary_classification_report

metrics = binary_classification_report(y_true, y_pred, y_score)
```

CLI: `python -m mlops_crew.evaluation.plot_model_comparison`

## `mlops_crew.monitoring`

| Module | `run(config)` writes |
|---|---|
| `inference_latency` | `reports/monitoring/inference_latency.csv` |
| `divergence` | `reports/divergence/phase2_divergence_report.json` and `phase2_divergence_summary.md` |

`ResourceMonitor` in `resource_monitor.py` samples CPU/memory during training;
`models/train_model.py` writes `reports/monitoring/training_resource_usage.csv`.

## `mlops_crew.tracking`

`mlops_crew.tracking.mlflow_tracking` wraps MLflow setup, nested model runs,
metric logging, and artifact logging for `models/train_model.py`.

## `mlops_crew.train_hydra`

Hydra entrypoint for Section 6 experiment configs. It loads the normal
`configs/config.yaml`, applies a `configs/hydra/experiment/*.yaml` override, routes
scratch artifacts to `outputs/hydra/...`, and calls the same `train(config)`
function used by `models/train_model.py`.

```bash
python -m mlops_crew.train_hydra experiment=phase2_default
python -m mlops_crew.train_hydra experiment=phase2_experimental
```

## `mlops_crew.utils`

```python
from mlops_crew.utils import set_seed, save_json, load_json

set_seed(42)
```

## Training / Prediction CLIs

```bash
python -m mlops_crew.models.train_model
python -m mlops_crew.train_hydra experiment=phase2_default
python -m mlops_crew.models.predict_model --model-path models/best_model.joblib --input data/processed/test.csv
```

## Phase 3 Serving API

The Phase 3 FastAPI app lives in `api/main.py` and uses
`mlops_crew.models.serving.ModelService` for model loading, text normalization,
and prediction. Incoming text is cleaned with the same `clean_text` behavior
used during training: lowercase, whitespace collapse, and minimum length
validation. The saved sklearn pipeline still owns TF-IDF transformation and
classification.

Run locally:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8080
```

Health check:

```bash
curl http://localhost:8080/health
```

Prediction request:

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Urgent account verification required. Click the secure link now."}'
```

Response fields:

| Field | Meaning |
| --- | --- |
| `label` | `phishing` or `legitimate` |
| `prediction` | integer class ID, where `1` is phishing |
| `is_phishing` | boolean convenience field |
| `score` | phishing probability when available, otherwise LinearSVC decision margin |
| `score_type` | `probability`, `decision_function`, or null |
| `model_version` | deployment metadata from `MODEL_VERSION` |
| `latency_ms` | API-side model latency |
| `normalized_text_length` | text length after serving-time cleaning |

Cloud Run-ready container:

```bash
docker build -f serve.dockerfile . -t mlops-crew-api:latest
docker run --rm -p 8080:8080 -e PORT=8080 -v "$PWD/models:/app/models" mlops-crew-api:latest
```

---

**phishing_email_detection** · Version see `mlops_crew.__version__`
