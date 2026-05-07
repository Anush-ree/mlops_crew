# Phishing Email Detection

**SE489 · ML Engineering for Production (MLOps) · DePaul University**

## 1. Team

- [✅] Team Name: MLOps Crew
- [✅] Team Members (Name & Email):
    1. Anushree Bachhav ([abachhav@depaul.edu](mailto:abachhav@depaul.edu))
    2. Krishna Kalakonda ([kkalakon@depaul.edu](mailto:kkalakon@depaul.edu))
    3. Muhammad Anas ([MuhammadAnasPSI2@gmail.com](MuhammadAnasPSI2@gmail.com))
    4. Kirtankumar Parekh ([kparekh2@depaul.edu](mailto:kparekh2@depaul.edu))
- [✅] Course & Section: [SE489] ML Engineering for Production (MLOps)

## 2. Project overview

Phishing emails are a leading cybersecurity threat. This project trains a
reproducible binary classifier that decides whether an incoming email is
phishing or legitimate, and ships it as part of an end-to-end MLOps pipeline:
data versioning (DVC), reproducible stages, deterministic training, and tracked
metrics. We optimize for **recall** — missing a phishing email is worse than
flagging a legitimate one — and select models by **F2**.

Phase 1 trains on a stratified 60% sample of `phishing_email.csv` so iteration
is fast; later phases will pull in more data and additional model families.

## 3. Architecture

<img width="776" height="662" alt="image" src="https://github.com/user-attachments/assets/2aa3ed2a-427e-4ddb-b2e8-58e3d4a225c6" />

## 4. Phase deliverables

- [PHASE1.md](./PHASE1.md) — Project design & baseline model
- [PHASE2.md](./PHASE2.md) — Enhancing ML operations
- [PHASE3.md](./PHASE3.md) — Continuous ML & deployment

## 5. Setup

```bash
git clone https://github.com/Anush-ree/mlops_crew.git
cd mlops_crew
python -m venv .venv && source .venv/bin/activate
make install            # installs runtime deps + the package in editable mode
make dev                # adds dev tools and pre-commit hooks
```

### Data access (DVC + AWS S3)

Data is versioned with DVC and stored on S3 (Google Drive is kept as a backup
remote). Request AWS credentials from a teammate, then:

```bash
git pull
pip install dvc-s3
aws configure          # region: us-east-2
dvc pull               # download raw + processed data
```

### Common commands

```bash
make data       # sample -> clean -> split -> validate
make train      # train all configured models, write metrics + predictions
make predict    # score the test split with the saved best model
make repro      # reproduce the full DVC pipeline end to end
make test       # pytest
make lint       # ruff check
make format     # ruff fix + format
```

### Reproduce results

```bash
make install
dvc pull
make repro
```

This runs sample → clean → split → train, fits the dummy and TF-IDF + Logistic
Regression baselines, and writes the artifacts under `models/` and `reports/`.

## 6. Repo layout

```
configs/config.yaml            single source of truth for the pipeline
src/mlops_crew/
  config.py                    project paths + YAML loader
  logging_config.py            shared logger setup
  data/
    sample.py                  stage 1: stratified sample of the raw CSV
    clean.py                   stage 2: schema, labels, text cleaning
    split.py                   stage 3: stratified train/val/test split
    validate.py                stage 4: post-split sanity checks
    make_dataset.py            run all four stages locally
  models/text_classifiers.py   TF-IDF + classifier sklearn pipeline factory
  evaluation/metrics.py        recall-oriented binary metrics (F2 + confusion)
  utils/                       seed + JSON helpers
  train_model.py               train every configured model, save artifacts
  predict_model.py             batch inference with the saved pipeline
dvc.yaml                       DVC stages (sample, clean, split, train)
```

## 7. Contributions

- **Anushree Bachhav** — proposal, repo/cookiecutter setup, environment
  configuration, DVC + S3 + Google Drive remotes
- **Muhammad Anas** — data cleaning, EDA, normalization, train/val/test splits,
  data documentation
- **Krishna Kalakonda** — model evaluation, baseline metrics, architecture
  diagram, code organization
- **Kirtankumar Parekh** — model training, experiment tracking, Makefile,
  contribution guidelines, repo maintenance

## 8. References

- Phishing Email Dataset (Kaggle) — primary training and evaluation data,
  combining SpamAssassin, Enron, Nazario, Ling, CEAS, and Nigerian Fraud sets.
