"""Small MLflow wrapper used by the Phase 2 training pipeline."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd

from mlops_crew.config import PROJECT_ROOT, resolve_project_path


def tracking_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("tracking", {}).get("enabled", False))


def setup_mlflow(config: dict[str, Any]) -> None:
    tracking = config.get("tracking", {})
    mlflow.set_tracking_uri(tracking.get("tracking_uri", "file:./mlruns"))
    mlflow.set_experiment(tracking.get("experiment_name", config["project"]["name"]))


def _flatten(prefix: str, value: Any, output: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten(next_prefix, nested, output)
    elif isinstance(value, list):
        output[prefix] = ",".join(str(item) for item in value)
    else:
        output[prefix] = value


def flatten_config(config: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    _flatten("", config, flattened)
    return flattened


@contextmanager
def training_run(config: dict[str, Any]) -> Iterator[Any]:
    """Open the parent MLflow run when tracking is enabled."""
    if not tracking_enabled(config):
        yield None
        return

    setup_mlflow(config)
    run_name = f"phase{config['project'].get('phase', 2)}-training"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(
            {
                "project": config["project"]["name"],
                "phase": config["project"].get("phase", 2),
                "primary_metric": config["modeling"].get("primary_metric", "f2"),
                "sample_fraction": config["data"]["sample"]["fraction"],
            }
        )
        config_path = PROJECT_ROOT / "configs" / "config.yaml"
        if config_path.exists():
            mlflow.log_artifact(str(config_path), artifact_path="config")
        yield run


@contextmanager
def model_run(config: dict[str, Any], model_name: str) -> Iterator[Any]:
    """Open a nested MLflow run for one trained model."""
    if not tracking_enabled(config):
        yield None
        return

    with mlflow.start_run(run_name=model_name, nested=True) as run:
        params = {
            "model_name": model_name,
            "phase": config["project"].get("phase", 2),
            **{
                f"tfidf.{key}": value
                for key, value in config["features"]["tfidf"].items()
                if isinstance(value, str | int | float | bool)
            },
        }
        for key, value in config["modeling"].get(model_name, {}).items():
            if isinstance(value, str | int | float | bool):
                params[f"model.{key}"] = value
        mlflow.log_params(params)
        yield run


def log_dataset_info(frames: dict[str, pd.DataFrame]) -> None:
    for split_name, frame in frames.items():
        mlflow.log_metric(f"{split_name}_rows", len(frame))
        for label, count in frame["label"].value_counts().items():
            mlflow.log_metric(f"{split_name}_label_{label}", int(count))


def log_metrics(metrics: dict[str, Any]) -> None:
    for split_name in ("validation", "test"):
        for metric_name, value in metrics[split_name].items():
            if isinstance(value, int | float):
                mlflow.log_metric(f"{split_name}_{metric_name}", float(value))


def log_artifacts(paths: list[Path], artifact_path: str) -> None:
    for path in paths:
        resolved = resolve_project_path(path)
        if resolved.exists():
            mlflow.log_artifact(str(resolved), artifact_path=artifact_path)


def log_model_artifacts(
    config: dict[str, Any],
    model_name: str,
    model: Any,
    model_path: Path,
    metrics: dict[str, Any],
) -> None:
    """Log metrics, model flavor, and local artifacts for one model."""
    if not tracking_enabled(config):
        return

    log_metrics(metrics)
    log_artifacts([model_path], artifact_path="joblib")
    prediction_paths = [
        Path(path)
        for key, path in metrics.items()
        if key.endswith("_prediction_path") and isinstance(path, str)
    ]
    log_artifacts(prediction_paths, artifact_path="predictions")

    if config.get("tracking", {}).get("log_models", True):
        mlflow.sklearn.log_model(model, name=f"{model_name}_pipeline")


def maybe_nullcontext() -> nullcontext[None]:
    return nullcontext()
