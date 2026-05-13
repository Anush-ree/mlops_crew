"""Train each model listed in the config, evaluate, and save artifacts.

For every entry in `modeling.models`, we fit a TF-IDF + classifier pipeline on
the training split and score it on validation and test. Outputs:

- `models/<name>.joblib`   — fitted pipeline per model
- `models/best_model.joblib` — copy of the model with the best validation score
- `reports/metrics/*.json` — per-model and comparison metrics (DVC-tracked)
- `reports/predictions/*.csv` — row-level predictions for error analysis

To swap in a new dataset, point `data.raw_dir` / `data.raw_file` in
`configs/config.yaml` and re-run `dvc repro`. The training stage will pick up
the new splits automatically.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from mlops_crew.config import (
    CONFIG_PATH,
    PROJECT_ROOT,
    load_project_config,
    resolve_project_path,
)
from mlops_crew.data import LABEL_COLUMN, TEXT_COLUMN
from mlops_crew.evaluation.metrics import binary_classification_report
from mlops_crew.logging_config import get_logger, setup_logging
from mlops_crew.models.text_classifiers import build_text_classifier
from mlops_crew.utils.io import save_json
from mlops_crew.utils.seed import set_seed

logger = get_logger(__name__)


def _project_relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _artifact_paths(config: dict[str, Any]) -> dict[str, Path]:
    return {
        "processed_dir": resolve_project_path(config["data"]["processed_dir"]),
        "model_dir": resolve_project_path(config["modeling"]["output_dir"]),
        "metrics_dir": resolve_project_path(config["reports"]["metrics_dir"]),
        "predictions_dir": resolve_project_path(config["reports"]["predictions_dir"]),
    }


def _load_splits(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    data_config = config["data"]
    processed_dir = resolve_project_path(data_config["processed_dir"])
    paths = {
        "train": processed_dir / data_config["train_file"],
        "val": processed_dir / data_config["val_file"],
        "test": processed_dir / data_config["test_file"],
    }
    frames = {}
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name} split at {path}. Run `make data`.")
        frames[name] = pd.read_csv(path)
        logger.info("Loaded %s split with shape %s", name, frames[name].shape)
    return frames


def _scores_for(model: Any, texts: pd.Series) -> Any | None:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(texts)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(texts)
    return None


def _evaluate(
    model: Any, data: pd.DataFrame, *, positive_label: int
) -> tuple[dict[str, float], pd.DataFrame]:
    texts, labels = data[TEXT_COLUMN], data[LABEL_COLUMN]
    predictions = model.predict(texts)
    scores = _scores_for(model, texts)

    metrics = binary_classification_report(
        labels, predictions, scores, positive_label=positive_label
    )
    frame = pd.DataFrame({TEXT_COLUMN: texts, "label": labels, "prediction": predictions})
    if scores is not None:
        frame["score"] = scores
    return metrics, frame


def _train_one(
    model_name: str,
    frames: dict[str, pd.DataFrame],
    config: dict[str, Any],
    paths: dict[str, Path],
) -> tuple[dict[str, Any], Path]:
    logger.info("Training model: %s", model_name)
    model = build_text_classifier(model_name, config)
    model.fit(frames["train"][TEXT_COLUMN], frames["train"][LABEL_COLUMN])

    paths["model_dir"].mkdir(parents=True, exist_ok=True)
    model_path = paths["model_dir"] / f"{model_name}.joblib"
    joblib.dump(model, model_path)
    logger.info("Saved model to %s", model_path)

    positive_label = int(config["modeling"].get("positive_label", 1))
    val_metrics, val_predictions = _evaluate(model, frames["val"], positive_label=positive_label)
    test_metrics, test_predictions = _evaluate(model, frames["test"], positive_label=positive_label)

    pred_dir = paths["predictions_dir"]
    pred_dir.mkdir(parents=True, exist_ok=True)
    val_predictions.to_csv(pred_dir / f"{model_name}_val_predictions.csv", index=False)
    test_predictions.to_csv(pred_dir / f"{model_name}_test_predictions.csv", index=False)

    metrics = {
        "model_name": model_name,
        "model_path": _project_relative(model_path),
        "validation": val_metrics,
        "test": test_metrics,
    }
    save_json(metrics, paths["metrics_dir"] / f"{model_name}_metrics.json")
    logger.info("%s | val=%s", model_name, val_metrics)
    logger.info("%s | test=%s", model_name, test_metrics)
    return metrics, model_path


def train(config: dict[str, Any]) -> dict[str, Any]:
    """Train every configured model and write per-model and comparison artifacts."""
    set_seed(int(config["project"]["seed"]))
    paths = _artifact_paths(config)
    frames = _load_splits(config)

    all_metrics: list[dict[str, Any]] = []
    model_paths: dict[str, Path] = {}
    for model_name in config["modeling"]["models"]:
        metrics, model_path = _train_one(model_name, frames, config, paths)
        all_metrics.append(metrics)
        model_paths[model_name] = model_path

    primary_metric = config["modeling"].get("primary_metric", "f2")
    best = max(all_metrics, key=lambda m: m["validation"][primary_metric])
    best_model_path = paths["model_dir"] / "best_model.joblib"
    shutil.copyfile(model_paths[best["model_name"]], best_model_path)

    paths["metrics_dir"].mkdir(parents=True, exist_ok=True)
    comparison_rows = [
        {
            "model_name": m["model_name"],
            **{f"val_{k}": v for k, v in m["validation"].items()},
            **{f"test_{k}": v for k, v in m["test"].items()},
        }
        for m in all_metrics
    ]
    pd.DataFrame(comparison_rows).to_csv(paths["metrics_dir"] / "model_comparison.csv", index=False)
    save_json(
        {
            "primary_metric": primary_metric,
            "best_model": best["model_name"],
            "best_model_path": _project_relative(best_model_path),
            "models": all_metrics,
        },
        paths["metrics_dir"] / "model_comparison.json",
    )
    save_json(best, paths["metrics_dir"] / "best_model_metrics.json")
    logger.info("Best model by val %s: %s", primary_metric, best["model_name"])
    return {"best_model": best, "all_metrics": all_metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train phishing email classifiers")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    setup_logging()
    train(load_project_config(args.config))
    logger.info("Training complete")


if __name__ == "__main__":
    main()
