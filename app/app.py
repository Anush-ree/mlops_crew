import os
import gradio as gr
import requests

# Backend URL - reads from environment variable, falls back to mock
BACKEND_URL = os.environ.get("BACKEND_PREDICT_URL", "https://mlops-crew-api-1043076962701.us-central1.run.app")

# Example emails for the interface
PHISHING_EXAMPLE = """Subject: Urgent: Your account has been compromised

Dear Customer,

We have detected suspicious activity on your account. Your account will be suspended within 24 hours unless you verify your information immediately.

Click here to verify: http://secure-login-verify.suspicious-domain.com/verify?id=12345

Please provide your username, password, and credit card details to restore access.

Regards,
Security Team"""

LEGITIMATE_EXAMPLE = """Subject: Team meeting rescheduled to Thursday

Hi everyone,

Just a quick note that our weekly team sync has been moved from Wednesday to Thursday at 2pm due to a conflict with the client presentation.

Please update your calendars. The Zoom link remains the same.

Let me know if you have any questions.

Thanks,
Sarah"""


def predict_email(email_text: str) -> dict:
    """Send email text to backend or run mock prediction."""
    if not email_text.strip():
        return {
            "prediction": "Please enter an email to analyze.",
            "confidence": "",
            "label": "",
        }

    # If backend URL is configured, call it
    if BACKEND_URL:
        try:
            response = requests.post(
                f"{BACKEND_URL}/predict",
                json={"text": email_text},
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            label = data.get("prediction", "unknown")
            confidence = data.get("confidence", data.get("phishing_score", "N/A"))
            model_version = data.get("model_version", "N/A")
            latency = data.get("latency_ms", "N/A")
            return {
                "prediction": "🚨 PHISHING" if label == 1 else "✅ LEGITIMATE",
                "confidence": f"{confidence:.2%}" if isinstance(confidence, float) else str(confidence),
                "model_version": model_version,
                "latency_ms": str(latency),
            }
        except Exception as e:
            return {
                "prediction": f"Backend error: {str(e)}",
                "confidence": "N/A",
                "model_version": "N/A",
                "latency_ms": "N/A",
            }

    # Mock prediction when no backend is configured
    text_lower = email_text.lower()
    phishing_signals = [
        "click here", "verify your", "suspended", "urgent",
        "account compromised", "provide your", "credit card",
        "suspicious-domain", "immediate action", "login verify",
        "confirm your password", "update your payment"
    ]
    score = sum(1 for signal in phishing_signals if signal in text_lower)
    is_phishing = score >= 2

    return {
        "prediction": "🚨 PHISHING" if is_phishing else "✅ LEGITIMATE",
        "confidence": f"{min(0.95, 0.6 + score * 0.1):.2%}" if is_phishing else "91.30%",
        "model_version": "linear_svc_v2 (mock)",
        "latency_ms": "12ms (mock)",
    }


def analyze(email_text: str) -> tuple[str, str, str, str]:
    """Run prediction and return formatted outputs."""
    result = predict_email(email_text)
    prediction = result.get("prediction", "Error")
    confidence = result.get("confidence", "N/A")
    model_version = result.get("model_version", "N/A")
    latency = result.get("latency_ms", "N/A")
    return prediction, confidence, model_version, latency


# Build the Gradio interface
with gr.Blocks(
    title="Phishing Email Detector",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        """
        # 🛡️ Phishing Email Detector
        **MLOps Crew · SE489 · DePaul University**

        Paste an email (subject + body) below to check whether it's a phishing attempt or a legitimate message.
        The model is a TF-IDF + Linear SVC classifier trained on 65,000+ emails.
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            email_input = gr.Textbox(
                label="Email Content",
                placeholder="Paste the full email text here (subject + body)...",
                lines=12,
            )
            with gr.Row():
                submit_btn = gr.Button("Analyze", variant="primary")
                clear_btn = gr.Button("Clear")

        with gr.Column(scale=1):
            prediction_output = gr.Textbox(label="Prediction", interactive=False)
            confidence_output = gr.Textbox(label="Confidence", interactive=False)
            model_output = gr.Textbox(label="Model Version", interactive=False)
            latency_output = gr.Textbox(label="Latency", interactive=False)

    gr.Markdown("### Try these examples:")
    with gr.Row():
        gr.Examples(
            examples=[
                [PHISHING_EXAMPLE],
                [LEGITIMATE_EXAMPLE],
            ],
            inputs=email_input,
            label="Example emails",
        )

    gr.Markdown(
        """
        ---
        **Note:** When the Cloud Run backend is live, set `BACKEND_PREDICT_URL` in Space secrets to use the real model.
        Until then the app runs a keyword-based mock for demonstration.

        **Source:** [GitHub Repository](https://github.com/Anush-ree/mlops_crew)
        """
    )

    submit_btn.click(
        fn=analyze,
        inputs=email_input,
        outputs=[prediction_output, confidence_output, model_output, latency_output],
    )
    clear_btn.click(
        fn=lambda: ("", "", "", ""),
        outputs=[prediction_output, confidence_output, model_output, latency_output],
    )


if __name__ == "__main__":
    demo.launch()
