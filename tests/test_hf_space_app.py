"""Tests for the Hugging Face Space UI helpers."""

from __future__ import annotations

from typing import Any

from hf_space import app as hf_app


# type: ignore[no-untyped-def]
def test_backend_url_accepts_service_root(monkeypatch) -> None:
    monkeypatch.setenv("BACKEND_PREDICT_URL", "https://example.run.app")

    assert hf_app.backend_url() == "https://example.run.app/predict"


# type: ignore[no-untyped-def]
def test_backend_url_accepts_predict_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("BACKEND_PREDICT_URL",
                       "https://example.run.app/predict")

    assert hf_app.backend_url() == "https://example.run.app/predict"


class FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {
            "label": "phishing",
            "prediction": 1,
            "is_phishing": True,
            "score": 1.23456,
            "score_type": "decision_function",
            "latency_ms": 12.3456,
        }


# type: ignore[no-untyped-def]
def test_classify_email_formats_numeric_response(monkeypatch) -> None:
    monkeypatch.setattr(hf_app.requests, "post",
                        lambda *args, **kwargs: FakeResponse())

    verdict, score_text, latency_text, payload = hf_app.classify_email(
        "Verify account now")

    assert verdict == "Phishing"
    assert score_text == "77.5%"
    assert latency_text == "12.35 ms"
    assert payload["is_phishing"] is True


def test_confidence_text_formats_negative_margin_as_confidence() -> None:
    assert hf_app.confidence_text(-1.5, "decision_function") == "81.8%"


def test_confidence_text_formats_probability_as_percent() -> None:
    assert hf_app.confidence_text(0.91, "probability") == "91.0%"
