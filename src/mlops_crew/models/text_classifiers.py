"""Build the sklearn text classification pipeline used in training.

The saved model is a `Pipeline(TfidfVectorizer -> classifier)`, so the
vectorizer learned during training travels with the model. That avoids
train/test leakage and makes batch inference a one-line `model.predict`.

Adding a new classifier is a matter of adding a branch to `_build_estimator`
and listing the name under `modeling.models` in `configs/config.yaml`.
"""

from __future__ import annotations

from typing import Any

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_tfidf_vectorizer(config: dict[str, Any]) -> TfidfVectorizer:
    cfg = config["features"]["tfidf"]
    return TfidfVectorizer(
        max_features=int(cfg["max_features"]),
        ngram_range=tuple(cfg["ngram_range"]),
        min_df=cfg["min_df"],
        max_df=cfg["max_df"],
        sublinear_tf=bool(cfg["sublinear_tf"]),
        strip_accents=cfg.get("strip_accents"),
        token_pattern=cfg["token_pattern"],
    )


def _build_estimator(model_name: str, config: dict[str, Any]) -> Any:
    seed = int(config["project"]["seed"])
    model_config = config["modeling"].get(model_name, {})

    if model_name == "dummy":
        return DummyClassifier(
            strategy=model_config.get("strategy", "most_frequent"),
            random_state=seed,
        )
    if model_name == "logistic_regression":
        return LogisticRegression(
            C=float(model_config.get("C", 1.0)),
            class_weight=model_config.get("class_weight", "balanced"),
            max_iter=int(model_config.get("max_iter", 1000)),
            random_state=seed,
            solver=model_config.get("solver", "liblinear"),
        )

    raise ValueError(f"Unsupported model: {model_name}")


def build_text_classifier(model_name: str, config: dict[str, Any]) -> Pipeline:
    """Return an unfitted TF-IDF + classifier pipeline."""
    return Pipeline(
        steps=[
            ("tfidf", build_tfidf_vectorizer(config)),
            ("classifier", _build_estimator(model_name, config)),
        ]
    )
