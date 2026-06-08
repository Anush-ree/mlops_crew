"""Tests for shared Phase 3 serving helpers."""

from __future__ import annotations

from pathlib import Path

import joblib

from mlops_crew.models.serving import ModelService


class ConstantModel:
    def predict(self, texts: list[str]) -> list[int]:
        return [1 for _ in texts]

    def decision_function(self, texts: list[str]) -> list[float]:
        return [1.5 for _ in texts]


def test_model_service_normalizes_text_like_training(tmp_path: Path) -> None:
    model_path = tmp_path / "model.joblib"
    joblib.dump(ConstantModel(), model_path)
    config = {
        "cleaning": {
            "lowercase": True,
            "normalize_whitespace": True,
            "min_text_length": 3,
        }
    }
    service = ModelService(model_path=model_path, config=config, model_version="test")

    assert service.normalize_text("  CLICK   HTTP://Example.com  ") == "click http://example.com"
    result = service.predict("  CLICK   HTTP://Example.com  ")

    assert result.label == "phishing"
    assert result.prediction == 1
    assert result.score == 1.5
    assert result.score_type == "decision_function"
