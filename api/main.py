"""
API implementation for the MLOps Crew Phishing Email Detection project.
This module defines the FastAPI application, including endpoints for health checks and predictions.

Endpoints:
- GET /health: Check the health of the service and model loading status.
- POST /predict: Predict whether a given email text is phishing or legitimate.
"""

from __future__ import annotations

from dataclasses import asdict

from fastapi import FastAPI, HTTPException

from api.schemas import HealthResponse, PredictRequest, PredictResponse
from mlops_crew.config import load_project_config
from mlops_crew.logging_config import get_logger, setup_logging_from_config
from mlops_crew.models.serving import get_model_service

config = load_project_config()
setup_logging_from_config(config)
logger = get_logger(__name__)

app = FastAPI(
    title="MLOps Crew Phishing Email Detection API",
    version="0.1.0",
    description="Phase 3 inference API for the trained phishing email classifier.",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return service readiness and model-loading status.
    Input: None

    Output:
    ---
     - status: "ok" if the service is healthy, otherwise "error"
     - model_loaded: True if the model is loaded successfully, otherwise False
     - detail: Optional error message if the health check fails

     Success Status Example:
     ---
     {
        "status": "ok",
        "model_loaded": true,
        "detail": null
     }

     Failure Status Example:
     ---
     {
        "status": "error",
        "model_loaded": false,
        "detail": "Model file not found"
     }
    """
    try:
        service = get_model_service()
        return HealthResponse(**service.health())
    except Exception:
        logger.exception("Health check failed")
        raise HTTPException(status_code=503, detail="Service unavailable") from None


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Predict whether a single email is phishing or legitimate.

    Input:
    ---
     - text: The raw email text to classify.

    Output:
    ---
     - label: 'phishing' or 'legitimate'
     - score_type: The type of score returned (e.g., 'probability', 'confidence')
     - latency_ms: The time taken to make the prediction in milliseconds
    """
    try:
        result = get_model_service().predict(request.text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc

    logger.info(
        "Served prediction label=%s score_type=%s latency_ms=%.2f",
        result.label,
        result.score_type,
        result.latency_ms,
    )
    return PredictResponse(**asdict(result))
