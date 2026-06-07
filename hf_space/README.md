---
title: Phishing Email Detection
emoji: ""
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
license: mit
---

# Phishing Email Detection

This Hugging Face Space provides the user interface for the MLOps Crew
phishing email classifier. The app sends email text to the deployed FastAPI
backend through the `BACKEND_PREDICT_URL` Space secret.

## Local Development

```bash
pip install -r requirements.txt
BACKEND_PREDICT_URL=http://localhost:8080/predict python app.py
```
