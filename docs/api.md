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
from mlops_crew.logging_config import setup_logging, get_logger

setup_logging(level="INFO")
logger = get_logger(__name__)
```

## `mlops_crew.data`

| Function | Purpose |
|---|---|
| `sample.run(config)` | Create the configured 60% raw-data sample |
| `clean.run(config)` | Normalize schema, labels, and email text |
| `split.run(config)` | Create deterministic train/validation/test splits |
| `validate.run(config)` | Sanity-check cleaned and split CSVs |
| `make_dataset.process_data(config_path)` | Run sample → clean → split → validate |

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

## `mlops_crew.utils`

```python
from mlops_crew.utils import set_seed, save_json, load_json

set_seed(42)
```

## Training / Prediction CLIs

```bash
python -m mlops_crew.train_model
python -m mlops_crew.predict_model --model-path models/best_model.joblib --input data/processed/test.csv
```

---

**phishing_email_detection** · Version see `mlops_crew.__version__`
