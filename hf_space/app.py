"""Gradio UI for phishing email detector."""

from __future__ import annotations

import os
from typing import Any

import requests

DEFAULT_BACKEND_URL = "http://localhost:8080/predict"
REQUEST_TIMEOUT_SECONDS = 20

PHISHING_EXAMPLE = """Subject: Urgent: Your account has been compromised

Dear Customer,

We have detected suspicious activity on your account. Your account will be suspended
within 24 hours unless you verify your information immediately.

Click here to verify: http://secure-login-verify.suspicious-domain.com/verify?id=12345

Please provide your username, password, and credit card details to restore access.

Regards,
Security Team"""

LEGITIMATE_EXAMPLE = """Subject: Team meeting rescheduled to Thursday

Hi everyone,

Just a quick note that our weekly team sync has been moved from Wednesday to Thursday
at 2pm due to a conflict with the client presentation.

Please update your calendars. The Zoom link remains the same.

Let me know if you have any questions.

Thanks,
Sarah"""


def backend_url() -> str:
    """Return the configured prediction endpoint."""
    configured_url = os.getenv("BACKEND_PREDICT_URL", DEFAULT_BACKEND_URL).rstrip("/")
    return configured_url if configured_url.endswith("/predict") else f"{configured_url}/predict"


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


def analyze(email_text: str) -> tuple[str, str, str, str]:
    """Run the backend prediction and return values for visible UI fields."""
    verdict, score_text, latency_text, payload = classify_email(email_text)
    model_version = str(payload.get("model_version", "N/A")) if payload else "N/A"
    return verdict, score_text, model_version, latency_text


def build_demo() -> Any:
    """Build the Gradio app lazily so helper tests do not require Gradio."""
    import gradio as gr

    with gr.Blocks(
        title="Phishing Email Detector",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
            # 🛡️ Phishing Email Detector
            **MLOps Crew · SE489 · DePaul University**

            Paste an email (subject + body) below to check whether it's a
            phishing attempt or a legitimate message. The model is a TF-IDF +
            Linear SVC classifier trained on 65,000+ emails.
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                email_text = gr.Textbox(
                    label="Email Content",
                    lines=12,
                    placeholder="Paste the full email text here (subject + body)...",
                )
                with gr.Row():
                    submit = gr.Button("Analyze", variant="primary")
                    clear = gr.Button("Clear")
            with gr.Column(scale=1):
                verdict = gr.Textbox(label="Prediction", interactive=False)
                score = gr.Textbox(label="Confidence", interactive=False)
                model_version = gr.Textbox(label="Model Version", interactive=False)
                latency = gr.Textbox(label="Latency", interactive=False)

        gr.Markdown("### Try these examples:")
        with gr.Row():
            gr.Examples(
                examples=[
                    [PHISHING_EXAMPLE],
                    [LEGITIMATE_EXAMPLE],
                ],
                inputs=email_text,
                label="Example emails",
            )

        gr.Markdown(
            """
            ---
            **Note:** The deployed Space reads `BACKEND_PREDICT_URL` from
            Hugging Face variables or secrets. It can point either to the Cloud
            Run service root or directly to the `/predict` endpoint.

            **Source:** [GitHub Repository](https://github.com/Anush-ree/mlops_crew)
            """
        )

        submit.click(
            fn=analyze,
            inputs=email_text,
            outputs=[verdict, score, model_version, latency],
        )
        clear.click(
            fn=lambda: ("", "", "", ""),
            outputs=[verdict, score, model_version, latency],
        )

    return demo


if __name__ == "__main__":
    build_demo().launch()
