# Phase 1: Project Design & Model Development

## Problem & Goal

Phishing emails are one of the most common attack vectors. Rule-based filters
miss modern phishing content, so we build a binary classifier that flags
phishing vs. legitimate email. Phase 1 lands a reproducible end-to-end pipeline
with a baseline model. Recall is the metric we care about most because missing
a phishing email is worse than flagging a legitimate one; we use **F2** for
model selection.

## Dataset

- **Source:** Kaggle "Phishing Email Dataset" (SpamAssassin, Enron, Nazario,
  Ling, CEAS, Nigerian). Stored in `data/raw/archive/` and versioned with DVC
  (S3 remote, Google Drive backup).
- **Schema used in Phase 1:** `phishing_email.csv` with two columns —
  `text_combined` (email text) and `label` (1 = phishing, 0 = legitimate).
- **Phase 1 sample:** stratified 60% of `phishing_email.csv` so iteration is
  fast. The remaining 40% is held back for later phases.

## Pipeline (`dvc.yaml`)

`configs/config.yaml` is the single source of truth for paths, sample/split
ratios, TF-IDF settings, and model hyperparameters.

1. **sample** — stratified 60% sample of the raw CSV → `data/interim/`.
2. **clean** — normalize schema/labels, lowercase, collapse whitespace, drop
   empty/duplicate rows. URLs, emails, numbers, and punctuation are kept on
   purpose: they're useful phishing signals for later feature engineering.
3. **split** — stratified train (70%) / val (15%) / test (15%) with a fixed
   seed for reproducibility.
4. **train** — fit each model in `modeling.models` as a single sklearn
   `Pipeline(TfidfVectorizer -> classifier)`. The vectorizer travels with the
   model, which avoids train/test leakage and makes inference one line.

## Models

- **Dummy classifier** (`most_frequent`) — sanity floor.
- **TF-IDF + Logistic Regression** (`class_weight=balanced`) — Phase 1
  baseline.

## Baseline result (test set)

| metric              | value    |
| ------------------- | -------- |
| F2                  | 0.9867   |
| Recall              | 0.9881   |
| Accuracy            | 0.9839   |
| False-negative rate | 0.0119   |

Per-model JSON metrics live under `reports/metrics/`; row-level predictions
under `reports/predictions/`. The chosen artifact is `models/best_model.joblib`.

## Reproduce

```bash
make install
dvc pull        # fetch raw data from S3
make repro      # runs sample -> clean -> split -> train via DVC
```

## Tooling

- **DVC** for data and pipeline versioning (S3 remote)
- **scikit-learn** for the model pipeline
- **ruff** + **mypy** + **pytest** for code quality
- **pre-commit** for local hooks

## Next iteration

Switching datasets in future phases is intentionally a config change: point
`data.raw_dir` / `data.raw_file` at the new file (or override the sample
fraction) and re-run `dvc repro`. New model types are added by extending
`mlops_crew.models.text_classifiers._build_estimator` and listing the name
under `modeling.models`.
