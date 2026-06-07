"""
Gradio UI for phishing email detector.

This app provides a simple interface to test the deployed FastAPI prediction endpoint.
Users paste an email body and see the predicted label, score, and API latency.
"""

from __future__ import annotations

import os
from typing import Any

import gradio as gr
import requests

DEFAULT_BACKEND_URL = "http://localhost:8080/predict"
REQUEST_TIMEOUT_SECONDS = 20


def backend_url() -> str:
    """Return the configured prediction endpoint."""
    return os.getenv("BACKEND_PREDICT_URL", DEFAULT_BACKEND_URL).rstrip("/")


def classify_email(text: str) -> tuple[str, str, str, dict[str, Any]]:
    """
    Call the deployed FastAPI endpoint and format the response for the UI.

    Input:
    ---
    - text: The raw email text to classify.

    Output:
    ---
    - verdict: The predicted label and recommendation for the email.
    - score_text: The confidence score and its type (e.g., probability).
    - latency_text: The time taken for the API call in milliseconds.
    - payload: The raw JSON response from the API for debugging purposes.
    """
    if not text or not text.strip():
        return "Missing input", "Enter an email body first.", "N/A", {}

    try:
        response = requests.post(
            backend_url(),
            json={"text": text},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        return "Request failed", str(exc), "N/A", {"backend_url": backend_url()}

    payload: dict[str, Any] = response.json()
    label = str(payload.get("label", "unknown")).title()
    prediction = int(payload.get("prediction", -1))
    is_phishing = bool(payload.get("is_phishing", False))
    verdict = f"{label} (class {prediction})"
    if is_phishing:
        verdict = f"{verdict} - review before trusting this email"

    score = payload.get("score")
    score_type = payload.get("score_type") or "not available"
    if isinstance(score, (int, float)):
        score_text = f"{score:.4f} ({score_type})"
    else:
        score_text = f"N/A ({score_type})"

    latency = payload.get("latency_ms")
    latency_text = f"{float(latency):.2f} ms" if isinstance(latency, (int, float)) else "N/A"
    return verdict, score_text, latency_text, payload


EXAMPLES = [
    [
        "Your mailbox storage is full. Verify your password immediately at "
        "http://example.com/account to avoid suspension."
    ],
    [
        "Hi team, attached are the notes from today's project sync. Please review before "
        "tomorrow's meeting."
    ],
    [
        "Urgent security alert: unusual login detected. Confirm your identity now to keep "
        "your account active."
    ],
]


with gr.Blocks(title="Phishing Email Detection") as demo:
    gr.Markdown("# Phishing Email Detection")
    gr.Markdown(
        "Paste an email body below. The app calls the deployed Phishing Email Detection endpoint "
        "and returns the model prediction."
    )
    with gr.Row():
        with gr.Column(scale=2):
            email_text = gr.Textbox(
                label="Email text",
                lines=10,
                placeholder="Paste the email body here...",
            )
            classify = gr.Button("Classify Email", variant="primary")
        with gr.Column(scale=1):
            verdict = gr.Textbox(label="Prediction")
            score = gr.Textbox(label="Score")
            latency = gr.Textbox(label="API latency")
    raw_json = gr.JSON(label="Raw API response")
    gr.Examples(examples=EXAMPLES, inputs=email_text)

    classify.click(
        fn=classify_email,
        inputs=email_text,
        outputs=[verdict, score, latency, raw_json],
    )


if __name__ == "__main__":
    demo.launch()
