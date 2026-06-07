"""
Pydantic request and response models for the prediction API.

Models:
- PredictRequest: email text input.
- PredictResponse: predicted label, score, and metadata.
- HealthResponse: service status and model loading details.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    """Single email body sent to the phishing classifier."""

    # The only input we get from user is raw email text.
    text: str = Field(..., min_length=1, max_length=50_000)

    # Ensure the text is not just whitespace and strip leading/trailing spaces
    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("text must not be blank")
        return cleaned


class PredictResponse(BaseModel):
    """
    Prediction response returned to API clients and the Hugging Face UI.

    Fields:
    ---
    - label: The predicted class label ('phishing' or 'legitimate').
    - prediction: Integer class id, 0 for legitimate and 1 for phishing.
    - is_phishing: Boolean indicating if the email is classified as phishing.
    - score: The confidence score for the prediction (optional).
    - score_type: The type of score returned (e.g., 'probability', 'confidence') (optional).
    - model_version: The version of the model used for prediction.
    - latency_ms: The time taken to make the prediction in milliseconds.
    - normalized_text_length: Length after normalization.
    """

    label: str
    prediction: int
    is_phishing: bool
    score: float | None = None
    score_type: str | None = None
    model_version: str
    latency_ms: float
    normalized_text_length: int


class HealthResponse(BaseModel):
    """
    Readiness response for deployment probes.

    Fields:
    ---
    - status: "ok" if the service is healthy, otherwise "error"
    - model_loaded: True if the model is loaded successfully, otherwise False
    - model_path: The file path of the loaded model (optional)
    - model_version: The version of the loaded model (optional)
    - detail: Optional error message if the health check fails
    """

    status: str
    model_loaded: bool
    model_path: str | None = None
    model_version: str | None = None
    detail: str | None = None
