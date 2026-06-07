"""
Cloud Functions HTTP wrapper for the prediction endpoint.

This module defines a single Cloud Function that receives HTTP requests,
extracts email text, and forwards it to the deployed FastAPI endpoint.
The response from the FastAPI service is returned directly to the client.
"""

from __future__ import annotations

import json
import os
from typing import Any

import requests

REQUEST_TIMEOUT_SECONDS = 20


def predict(request: Any) -> tuple[str, int, dict[str, str]]:
    """
    Forward Cloud Functions requests to the deployed FastAPI backend.

    Input:
    ---
    - request: Cloud Functions HTTP request object with a JSON body.

    Output:
    ---
    - A response body, HTTP status code, and response headers.
    """
    backend_url = os.getenv("BACKEND_PREDICT_URL")
    if not backend_url:
        return _json_error("BACKEND_PREDICT_URL is not configured", 500)

    request_json = request.get_json(silent=True) or {}
    if "text" not in request_json:
        return _json_error("Request JSON must include a text field", 422)

    try:
        response = requests.post(
            backend_url,
            json={"text": request_json["text"]},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        return _json_error(f"Backend request failed: {exc}", 502)

    return (
        response.text,
        response.status_code,
        {"Content-Type": response.headers.get("Content-Type", "application/json")},
    )


def _json_error(message: str, status_code: int) -> tuple[str, int, dict[str, str]]:
    return (json.dumps({"detail": message}), status_code, {"Content-Type": "application/json"})
