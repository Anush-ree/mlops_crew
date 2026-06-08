"""

Shared model-serving helpers for the API and cloud wrappers.

"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from mlops_crew.config import CONFIG_PATH, load_project_config, resolve_project_path
from mlops_crew.data.clean import clean_text
from mlops_crew.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL_VERSION = "phase2_linear_svc"
LABELS = {0: "legitimate", 1: "phishing"}


@dataclass(frozen=True)
class PredictionResult:
    """Structured response returned by the serving layer."""

    label: str
    prediction: int
    is_phishing: bool
    score: float | None
    score_type: str | None
    model_version: str
    latency_ms: float
    normalized_text_length: int


class ModelService:
    """Load a trained sklearn pipeline once and serve single-email predictions."""

    def __init__(self, model_path: Path, config: dict[str, Any], model_version: str) -> None:
        self.model_path = model_path
        self.config = config
        self.model_version = model_version
        logger.info("Loading serving model from %s", model_path)
        self.model: Any = joblib.load(model_path)

    def health(self) -> dict[str, Any]:
        """Return lightweight model readiness details for health checks."""
        return {
            "status": "ok",
            "model_loaded": True,
            "model_path": str(self.model_path),
            "model_version": self.model_version,
        }

    def normalize_text(self, text: str) -> str:
        """Apply the same conservative cleaning used before model training."""
        cleaning = self.config.get("cleaning", {})
        normalized = clean_text(
            text,
            lowercase=bool(cleaning.get("lowercase", True)),
            normalize_whitespace=bool(cleaning.get("normalize_whitespace", True)),
        )
        min_length = int(cleaning.get("min_text_length", 3))
        if len(normalized) < min_length:
            raise ValueError(f"Email text must be at least {min_length} characters after cleaning")
        return normalized

    def predict(self, text: str) -> PredictionResult:
        """Return a single-email phishing prediction."""
        started = time.perf_counter()
        normalized = self.normalize_text(text)
        prediction = int(self.model.predict([normalized])[0])
        score, score_type = _score_for(self.model, normalized)
        latency_ms = (time.perf_counter() - started) * 1000
        return PredictionResult(
            label=LABELS.get(prediction, str(prediction)),
            prediction=prediction,
            is_phishing=prediction == 1,
            score=score,
            score_type=score_type,
            model_version=self.model_version,
            latency_ms=latency_ms,
            normalized_text_length=len(normalized),
        )


_SERVICE_CACHE: ModelService | None = None


def _score_for(model: Any, text: str) -> tuple[float | None, str | None]:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([text])[0]
        if len(probabilities) > 1:
            return _finite_float(probabilities[1]), "probability"
    if hasattr(model, "decision_function"):
        score = model.decision_function([text])
        return _finite_float(score[0]), "decision_function"
    return None, None


def _finite_float(value: object) -> float | None:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _download_gcs_model(uri: str, cache_dir: Path) -> Path:
    """Download ``gs://bucket/path`` to ``cache_dir`` and return the local path."""
    if not uri.startswith("gs://"):
        raise ValueError("MODEL_GCS_URI must start with gs://")
    bucket_name, blob_name = uri[5:].split("/", 1)
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / Path(blob_name).name

    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(destination)
    logger.info("Downloaded model artifact from %s to %s", uri, destination)
    return destination


def resolve_model_path() -> Path:
    """Resolve the configured local or GCS model artifact path."""
    model_gcs_uri = os.getenv("MODEL_GCS_URI")
    if model_gcs_uri:
        cache_dir = resolve_project_path(os.getenv("MODEL_CACHE_DIR", "/tmp/mlops_crew_models"))
        return _download_gcs_model(model_gcs_uri, cache_dir)

    configured = os.getenv("MODEL_PATH")
    if configured:
        model_path = Path(configured)
        return model_path if model_path.is_absolute() else resolve_project_path(model_path)

    config = load_project_config(CONFIG_PATH)
    return resolve_project_path(config["modeling"]["output_dir"]) / "best_model.joblib"


def get_model_service(*, force_reload: bool = False) -> ModelService:
    """Return the cached model service, loading it on first use."""
    global _SERVICE_CACHE
    if _SERVICE_CACHE is None or force_reload:
        config = load_project_config(CONFIG_PATH)
        model_path = resolve_model_path()
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {model_path}")
        model_version = os.getenv("MODEL_VERSION", DEFAULT_MODEL_VERSION)
        _SERVICE_CACHE = ModelService(
            model_path=model_path, config=config, model_version=model_version
        )
    return _SERVICE_CACHE
