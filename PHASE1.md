# PHASE 1: Project Design & Model Development

## Overview

Phase 1 establishes the foundation for your MLOps project. This phase covers project planning, initial code organization, team collaboration setup, data handling, baseline model development, and comprehensive documentation. By the end of this phase, you should have a well-organized repository with a trained baseline model and clear documentation for future team members.

---

## 1. Project Proposal

- **Scope & Objectives**: 
      
  We are building a system to detect phishing emails using machine learning. The goal is to classify each email as either phishing or legitimate.

  Problem statement: Rule-based email filters fail against sophisticated modern phishing content. We need a data-driven, reproducible ML pipeline that can scale, be monitored, and improve as attack patterns evolve.

  Success metrics:
  - Recall (primary — minimize missed phishing emails)
  - F1 Score
  - Accuracy
  - Inference latency

---
- **Detailed Description**:

Phishing emails are one of the most common and damaging cybersecurity threats. Attackers craft messages that look legitimate to trick users into revealing passwords, financial information, or installing malware. Traditional rule-based filters struggle to keep up with the evolving tactics used in modern phishing campaigns. This motivates a machine learning approach that can learn patterns from large amounts of labeled email data.

This project builds a production-grade binary classifier to detect phishing emails. The input is raw email text. The output is a label: phishing or legitimate. The full pipeline covers data ingestion, cleaning, splitting, vectorization, model training, evaluation, and (in later phases) deployment and monitoring.

In Phase 1, we focused on data preparation and baseline modeling. The primary dataset is the Phishing Email Dataset from Kaggle, a combined corpus of around 82,000 emails sourced from SpamAssassin, Enron, Nazario, Ling, CEAS, and Nigerian email collections. We cleaned the data by removing duplicates, nulls, very short texts, URLs, and email addresses. We then split it into train, validation, and test sets using stratified sampling to preserve class balance.

For modeling, Phase 1 uses baseline approaches: logistic regression and decision tree classifiers trained on TF-IDF features. These give us a performance reference point. In Phase 2 and beyond, we will fine-tune transformer models from Hugging Face to improve detection quality.

The pipeline is fully reproducible. We use a fixed random seed, DVC for data versioning, MLflow for experiment tracking, and config files for hyperparameter management. This means any team member can reproduce results exactly and swap in new models or datasets without rewriting code.

We expect the final system to catch the majority of phishing emails (high recall) while remaining fast enough for real-world email filtering use cases. Monitoring for data drift and model degradation will be addressed in Phase 3.


- **Dataset Selection**: 
Dataset: [Phishing Email Dataset](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset?select=SpamAssasin.csv) — Kaggle

We chose this dataset because:
  - It is large (~82,000 emails), giving the model enough examples to learn from
  - It combines multiple well-known sources (SpamAssassin, Enron, Nazario, Ling, CEAS, Nigerian), making it more diverse than any single corpus
  - The class balance is healthy (imbalance ratio: 1.09), so the model is not biased toward one label
  - It is publicly available and well-documented, making it reproducible for others

 In Phase 2, individual source CSVs (Enron, CEAS, Nazario, SpamAssassin) will be added separately to increase variety and allow source-level analysis.

  ---

- **Dataset Description**: 

| Property | Details |
  |---|---|
  | Source | Kaggle — Phishing Email Dataset (combined corpus) |
  | Sub-sources | SpamAssassin, Enron, Nazario, Ling, CEAS, Nigerian |
  | File | phishing_email.csv |
  | Size after cleaning | 82,074 rows |
  | Format | CSV |
  | Columns | Email text, label (phishing / legitimate) |
  | Label type | Binary |
  | Class imbalance ratio | 1.09 (well balanced) |
  | Train split | 57,451 rows (70%) |
  | Validation split | 12,311 rows (15%) |
  | Test split | 12,312 rows (15%) |
  | Random seed | 42 |

---

- **Model Considerations**: 

Phase 1 — Baseline models:
  - Logistic Regression (trained on TF-IDF features)
  - Decision Tree

  These are fast to train, interpretable, and give a clear performance baseline.

Phase 2 — Advanced models:
  - Fine-tuned transformer models from Hugging Face (e.g., DistilBERT, RoBERTa)

  We will use config files to manage hyperparameters so we can swap settings without changing code. We will also track different data versions (e.g., case-sensitive vs. case-insensitive text) to monitor data drift over time.

---

- **Open-Source Tools**: 


  | Tool | Purpose | Justification |
  |---|---|---|
  | scikit-learn | TF-IDF vectorization, baseline models, stratified splits | Industry-standard ML library, well-documented |
  | pandas | Data cleaning and processing | Standard for tabular data in Python |
  | Hugging Face Transformers | Fine-tuned models in Phase 2 | Large library of pretrained NLP models |
  | DVC | Data versioning | Tracks dataset versions alongside code in Git |
  | MLflow | Experiment tracking | Logs metrics, parameters, and model artifacts |
  | ruff | Code linting | Fast, modern Python linter |
  | mypy | Static type checking | Catches type errors before runtime |
  | pytest | Testing | Standard Python testing framework |
  | Docker | Containerization (Phase 2+) | Ensures consistent environments across machines |

---

## 2. Code Organization & Setup

- **GitHub Repository**: Created at https://github.com/Anush-ree/mlops_crew using cookiecutter MLOps structure
- **Environment Setup**: Python environment configured using `uv` (fast Python package and environment manager)
- **Dependency Management**: `pyproject.toml` maintained with all dependencies
- **Project Structure**:Code organized with `src/`, `tests/`, `data/`, `configs/` separation
- **Version Pinning**: Pin all critical dependencies to specific versions
- **Installation Documentation**: Setup documented in README.md

---

## 3. Version Control & Collaboration

- **Regular Commits**: Established commit discipline with descriptive, atomic commits
- **Branching Strategy**: Implemented feature branching (e.g., git-flow or GitHub Flow)
- **Pull Request Process**: Established PR template and review requirements
- **Team Roles**: Clearly define responsibilities (author: kirtan, team members, reviewers)
- **Code Review Guidelines**: Documented code review expectations and checklist
- **Commit History**: Maintained clean, readable git history for project traceability

---

## 4. Data Handling

- **Data Cleaning Scripts**: `clean.py` removes duplicate rows, drops nulls, lowercases 
  all text, strips URLs and email addresses, and removes very short text entries. 
  Output saved to `data/processed/cleaned.csv`
- **Normalization**: TF-IDF vectorization implemented in `preprocess.py` with max 10,000 
  features and bigrams (1,2). Sublinear TF scaling used for normalization. Note: will be 
  updated in Phase 2 to fit only on training data to prevent data leakage.
- **Data Augmentation**: Not applicable for Phase 1. Dataset is already well balanced 
  (class imbalance ratio: 1.09) so no augmentation was needed.
- **Data Documentation**: 
  - Source: [Phishing Email Dataset](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset) — Kaggle
  - File: `phishing_email.csv` — 82,486 rows
  - `text_combined` (string) — full email text
  - `label` (int) — 1 for phishing, 0 for legitimate
- **Data Splits**: `split.py` uses stratified sampling to preserve class balance.
  - Train: 57,451 rows (70%)
  - Val: 12,311 rows (15%)
  - Test: 12,312 rows (15%)
  - Random seed = 42 for reproducibility
- **Data Validation**:  `validate.py` checks for nulls, invalid labels, empty text, 
  and class imbalance. All checks passed.
- **DVC Setup (Optional)**: DVC initialized with AWS S3 as primary remote and Google Drive as backup.
---

## 5. Model Training

- **Training Environment**: Local CPU environment using scikit-learn. GPU not required 
  for baseline models. Hydra used for configuration management.
- **Baseline Model**: Implement and train a baseline model
- **Hyperparameter Configuration**: Managed via Hydra config file. Key parameters:
  - Random state: 42
  - Train/test split: 0.8, val split: 0.1
  - Early stopping patience: 10
  - Models saved to `models/`
- **Evaluation Metrics**: Recall, F1 Score, Accuracy, Inference latency. 
  Cross-validation with 5 folds.
- **Model Persistence**: Trained models saved to `models/` directory with logs 
  in `logs/`
- **Training Reproducibility**: Random seed = 42 set in config. MLflow logs all 
  parameters and metrics. Hydra config ensures consistent runs.
- **Performance Baseline**: - Logistic Regression: F1 = 98%
  - Dummy classifier (baseline): F1 = 52%
  - Logistic Regression significantly outperforms the dummy baseline, confirming 
  - the model is learning real patterns and not just guessing the majority class.

---
