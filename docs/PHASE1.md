# Phase 1: Project Design & Baseline Model

Phase 1 establishes a reproducible phishing-email classifier on a stratified
60% sample of the Kaggle combined dataset. The goal is a working DVC pipeline,
a sklearn baseline, and metrics oriented toward **recall** (selection metric:
**F2**).

## Dataset & sample

- **Source:** Kaggle Phishing Email Dataset (SpamAssassin, Enron, Nazario, Ling,
  CEAS, Nigerian) in `data/raw/archive/`, versioned with DVC (S3 + Drive backup).
- **Columns:** `text_combined`, `label` (1 = phishing, 0 = legitimate).
- **Phase 1 training sample:** 60% stratified; remaining 40% reserved for later
  phases.

## Pipeline

`configs/config.yaml` drives all stages:

```text
sample → clean → split → train
```

- **sample** — stratified interim CSV under `data/interim/`
- **clean** — schema/label normalization; conservative text cleaning (keeps URLs,
  numbers, punctuation as phishing signals)
- **split** — 70% / 15% / 15% train / val / test, stratified, fixed seed
- **train** — sklearn `Pipeline(TfidfVectorizer → classifier)` per model name

## Models

- Dummy (`most_frequent`) — sanity floor
- TF-IDF + Logistic Regression (`class_weight=balanced`) — Phase 1 baseline

## Baseline result (test set)

| Metric | Value |
| --- | ---: |
| F2 | 0.9867 |
| Recall | 0.9881 |
| Accuracy | 0.9839 |
| False-negative rate | 0.0119 |

Artifacts: `models/best_model.joblib`, metrics under `reports/metrics/`,
predictions under `reports/predictions/`.

## Reproduce

```bash
make install
dvc pull
make repro
```

## Tooling

DVC, scikit-learn, ruff, mypy, pytest, pre-commit.

See the root [PHASE1.md](../PHASE1.md) for the full Phase 1 deliverable.
