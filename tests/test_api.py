"""Tests for the Phase 3 FastAPI prediction service."""

from __future__ import annotations

from dataclasses import asdict

from fastapi.testclient import TestClient

import api.main as api_main
from mlops_crew.models.serving import PredictionResult


class FakeService:
    def health(self) -> dict[str, object]:
        return {
            "status": "ok",
            "model_loaded": True,
            "model_path": "models/fake.joblib",
            "model_version": "test",
        }

    def predict(self, text: str) -> PredictionResult:
        return PredictionResult(
            label="phishing",
            prediction=1,
            is_phishing=True,
            score=0.75,
            score_type="probability",
            model_version="test",
            latency_ms=1.25,
            normalized_text_length=len(text.strip().lower()),
        )


# type: ignore[no-untyped-def]
def test_health_reports_loaded_model(monkeypatch) -> None:
    monkeypatch.setattr(api_main, "get_model_service", lambda: FakeService())
    client = TestClient(api_main.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


# type: ignore[no-untyped-def]
def test_health_returns_503_without_internal_details(monkeypatch) -> None:
    def fail_to_load_model() -> None:
        raise FileNotFoundError("/internal/path/model.joblib")

    monkeypatch.setattr(api_main, "get_model_service", fail_to_load_model)
    client = TestClient(api_main.app)

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json() == {"detail": "Service unavailable"}


# type: ignore[no-untyped-def]
def test_predict_returns_schema(monkeypatch) -> None:
    monkeypatch.setattr(api_main, "get_model_service", lambda: FakeService())
    client = TestClient(api_main.app)

    response = client.post(
        "/predict", json={"text": "Verify your account now"})

    assert response.status_code == 200
    assert response.json() == asdict(FakeService().predict("Verify your account now"))


# type: ignore[no-untyped-def]
def test_predict_rejects_blank_text(monkeypatch) -> None:
    monkeypatch.setattr(api_main, "get_model_service", lambda: FakeService())
    client = TestClient(api_main.app)

    response = client.post("/predict", json={"text": "   "})

    assert response.status_code == 422
