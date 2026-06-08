# Phase 3: Continuous ML & Deployment

Phase 3 productionizes the Phase 2 **LinearSVC** TF-IDF pipeline: CI/CD,
containerized serving, GCP deployment, a Hugging Face Gradio demo, and a
one-time evaluation on the **20% holdout** reserved since Phase 2.

**Best model:** LinearSVC (saved `models/best_model.joblib`)
**Phase 3 holdout F2:** **98.6%** on 16,496 unseen emails (see root report for
full metrics)

## CI & testing

- Workflow: [`.github/workflows/ci.yml`](../.github/workflows/ci.yml)
- Runs on push/PR to `main`: `ruff check`, `ruff format --check`, `mypy src`,
  `pytest tests/ --cov=mlops_crew`
- Phase 3 tests: `test_api.py`, `test_serving.py`, `test_hf_space_app.py` plus
  Phase 1/2 suites
- Evidence: [`docs/phase3_evidence/ci_green.png`](phase3_evidence/ci_green.png)

CI does **not** run `dvc pull` — unit tests use fixtures; graders restore data
locally with `dvc pull` (see root PHASE3.md).

## Docker & CML

- **Serving image:** [`serve.dockerfile`](../serve.dockerfile) — Cloud Run / local
  FastAPI
- **Publish workflow:** [`.github/workflows/docker-publish.yml`](../.github/workflows/docker-publish.yml)
- **CML:** [`.github/workflows/cml.yml`](../.github/workflows/cml.yml) posts model
  comparison tables on PRs
- Local: `make docker-serve`, `make load-test-api`

## FastAPI & GCP

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/health` | GET | Readiness + model load status |
| `/predict` | POST | Classify one email body |

- **Code:** [`api/main.py`](../api/main.py), [`api/schemas.py`](../api/schemas.py)
- **Cloud Run:** live inference API (URL in root [PHASE3.md](../PHASE3.md))
- **Cloud Function:** HTTP wrapper in [`functions/predict/`](../functions/predict/)
- **GCP evidence:** [`reports/gcp/`](../reports/gcp/)

## Hugging Face Spaces

- **Live demo:** [mlops-crew-depaul/phishing-email-detector](https://huggingface.co/spaces/mlops-crew-depaul/phishing-email-detector)
- **UI code:** [`hf_space/app.py`](../hf_space/app.py)
- **Deploy workflow:** [`.github/workflows/deploy_hf_space.yml`](../.github/workflows/deploy_hf_space.yml)

## Holdout evaluation

Script: [`src/mlops_crew/evaluation/phase3_holdout_eval.py`](../src/mlops_crew/evaluation/phase3_holdout_eval.py)

| Metric | Value |
| --- | ---: |
| Holdout rows | 16,496 |
| F2 | **98.6%** |
| Recall | 98.7% |
| False-negative rate | 1.28% |
| ROC-AUC | 99.84% |

F2 drops only ~0.6 points from the Phase 2 test split (99.1%) to holdout (98.6%).

## Documentation & cleanup

- Full deliverable + screenshots: root [PHASE3.md](../PHASE3.md)
- API usage: [api.md](api.md)
- GCP teardown: [cleanup.md](cleanup.md)

See the root [PHASE3.md](../PHASE3.md) for workflow explanations, live URLs,
and team contribution table.
