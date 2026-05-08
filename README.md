# Phishing Email Detection

## SE489 · ML Engineering for Production (MLOps) · DePaul University

## 1. Team Informaton

- Team Name: MLOps Crew
- Team Members (Name & Email):
    1. Anushree Bachhav ([abachhav@depaul.edu](mailto:abachhav@depaul.edu))
    2. Krishna Kalakonda ([kkalakon@depaul.edu](mailto:kkalakon@depaul.edu))
    3. Muhammad Anas ([MuhammadAnasPSI2@gmail.com](MuhammadAnasPSI2@gmail.com))
    4. Kirtankumar Parekh ([kparekh2@depaul.edu](mailto:kparekh2@depaul.edu))
- Course & Section: [SE489] ML Engineering for Production (MLOps)

## 2. Project Overview

This repository implements a production-grade classification system for phishing email detection. 
It combines several email datasets, cleans and splits the data, converts text to features (TF‑IDF now, transformers later), trains models, and measures performance with an emphasis on recall and low latency..
Data is versioned with DVC on S3 (s3://mlops-crew-data/dvc-store, region us-east-2) with a Google Drive fallback.


Objectives:

Train a high-recall classifier (minimize missed phishing emails) using the SpamAssassin + Enron + Nazario corpus
Build a fully reproducible ML pipeline with data versioning, experiment tracking, and CI/CD
Deploy the model as a low-latency inference service (Phase 3)
Monitor for data drift and model degradation over time (Phase 3)

Success metrics: Recall, F1 score, Accuracy, Inference latency

## 3. Project Architecture Diagram

<img width="776" height="662" alt="image" src="https://github.com/user-attachments/assets/2aa3ed2a-427e-4ddb-b2e8-58e3d4a225c6" />

## 4. Phase Deliverables

- [ ] [PHASE1.md](./PHASE1.md): Project Design & Model Development
- [ ] [PHASE2.md](./PHASE2.md): Enhancing ML Operations
- [ ] [PHASE3.md](./PHASE3.md): Continuous ML & Deployment

## 5. Setup Instructions

### Prerequisites

Python 3.11+<br>
Git

#### Install

Bash:
```bash
git clone https://github.com/Anush-ree/mlops_crew.git
cd mlops_crew
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

#### Pre-commit hooks

Bash:
```bash
pre-commit install
```

#### Common commands

Bash:
```bash
make setup    # install all dependencies
make train    # run the training pipeline
make test     # run tests
make lint     # run ruff linter
make format   # auto-format code
```

#### Reproduce results

Bash:
```bash
make setup
make train
```

This will preprocess the data, train the baseline model, and print evaluation metrics to the console. MLflow logs will appear in mlruns/

### Data Access (DVC + AWS S3)

Data is versioned with DVC and stored on AWS S3. Request AWS credentials then run.
Bash:
```bash
git pull                 # get latest config
uv pip install dvc-s3   # install DVC S3 plugin
aws configure           # enter credentials + region: us-east-2
dvc pull                # download data from S3
```
Google Drive is kept as a backup remote (gdrive_backup) in case S3 is unavailable.

## 6. Contribution Summary

- Anushree Bachhav: Project proposal, repository structure, cookiecutter setup, environment configuration, data versioning with DVC, cloud storage setup (AWS S3 + Google Drive backup)
- Muhammad Anas: Data cleaning, EDA, normalization, train/val/test splits, data documentation 
- Krishna Kalakonda: Model evaluation, baseline performance documentation, architecture diagram 
- Kirtankumar Parekh: Model training, experiment tracking, Makefile, CONTRIBUTING.md, repo maintenance

## 7. References

- Dataset: Phishing Email Dataset (SpamAssassin, Enron, Nazario, Ling, CEAS, Nigerian)
- Source: https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset?select=SpamAssasin.csv
- Use: Primary training & evaluation data
