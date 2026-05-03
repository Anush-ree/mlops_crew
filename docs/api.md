# API Reference

The package is importable as `mlops_crew` after running `pip install -e .`.

## `mlops_crew.config`

Project-wide path constants and typed config dataclasses.

```python
from mlops_crew.config import (
    PROJECT_ROOT,
    DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
    MODELS_DIR, REPORTS_DIR, FIGURES_DIR,
    Config, TrainingConfig, DataConfig, DEFAULT_CONFIG,
)
```

Use these constants instead of hard-coded relative paths — they resolve against the repo root regardless of the current working directory.

## `mlops_crew.logging_config`

```python
from mlops_crew.logging_config import setup_logging, get_logger

setup_logging(level="INFO")
logger = get_logger(__name__)
```

## `mlops_crew.data`

| Function | Purpose |
|---|---|
| `load_raw(filename)` | Read CSV from `data/raw/` |
| `load_processed(filename)` | Read CSV from `data/processed/` |
| `save_processed(df, filename)` | Write CSV to `data/processed/` |
| `process_data(input_dir, output_dir)` | Raw → processed pipeline |

CLI: `python -m mlops_crew.data.make_dataset [--input PATH] [--output PATH]`

## `mlops_crew.features`

```python
from mlops_crew.features import build_features

df_features = build_features(df_processed)
```

## `mlops_crew.models`

### `BaseModel` (abstract)

Abstract interface with `fit`, `predict`, `save`, `load`. Extend this for any new estimator.

### `Model`

Reference implementation scaffold. Serializes via `joblib`.

```python
from pathlib import Path
from mlops_crew.models import Model

model = Model(config={"lr": 0.01})
# model.fit(X_train, y_train)
model.save(Path("models/model.joblib"))
reloaded = Model.load(Path("models/model.joblib"))
```

## `mlops_crew.evaluation`

```python
from mlops_crew.evaluation import classification_report, regression_report

metrics = classification_report(y_true, y_pred)
# -> {"accuracy": ..., "precision": ..., "recall": ..., "f1": ...}
```

## `mlops_crew.visualization`

```python
from mlops_crew.visualization import plot_training_history, plot_confusion_matrix
```

## `mlops_crew.utils`

```python
from mlops_crew.utils import set_seed, save_json, load_json

set_seed(42)
```

## Training / Prediction CLIs

```bash
python -m mlops_crew.train_model --epochs 100 --batch-size 64
python -m mlops_crew.predict_model --model-path models/model.joblib --input data/processed/test.csv
```

## Hydra Configuration

Configuration is managed through Hydra — see `configs/config.yaml` for defaults and override at runtime:

```bash
python -m mlops_crew.train_model model.name=custom_model training.epochs=200
```

---

**phishing_email_detection** · Version see `mlops_crew.__version__`
