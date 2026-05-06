# Phishing Email Detection
##SE489 · ML Engineering for Production (MLOps) · DePaul University

### 1. Team Informaton

- Team NameMLOps Crew
- Team Members (Name & Email):
       Anushree Bachhav []
       Krishna Kalakonda []
       Muhammad Anas [MuhammadAnasPSI2@gmail.com]
       Kirtankumar Parekh [[kparekh2@depaul.edu](mailto:kparekh2@depaul.edu)]

## 2. Project Overview

Phishing emails are one of the most common and damaging cybersecurity threats, tricking users into revealing sensitive information or installing malware. This project builds a production-grade binary classifier that detects whether an incoming email is a phishing attempt or a legitimate message.
Problem statement: Automated phishing detection at scale requires a robust, reproducible ML pipeline that can be monitored and continuously improved as attack patterns evolve. Rule-based filters fail against sophisticated modern phishing content, motivating a data-driven approach.
Main objectives:

Train a high-recall classifier (minimize missed phishing emails) using the SpamAssassin + Enron + Nazario corpus
Build a fully reproducible ML pipeline with data versioning, experiment tracking, and CI/CD
Deploy the model as a low-latency inference service (Phase 3)
Monitor for data drift and model degradation over time (Phase 3)

Success metrics: Recall, F1 score, Accuracy, Inference latency

## 3. Project Architecture Diagram

-<img width="776" height="662" alt="image" src="https://github.com/user-attachments/assets/2aa3ed2a-427e-4ddb-b2e8-58e3d4a225c6" />


## 4. Phase Deliverables

- [ ] [PHASE1.md](./PHASE1.md): Project Design & Model Development
- [ ] [PHASE2.md](./PHASE2.md): Enhancing ML Operations
- [ ] [PHASE3.md](./PHASE3.md): Continuous ML & Deployment

## 5. Setup Instructions

### Prerequisites
Python 3.11+
Git
###Install
bashgit clone https://github.com/Anush-ree/mlops_crew.git
cd mlops_crew
python -m venv .venv
source .venv/bin/activate        
pip install -e ".[dev]"
####Pre-commit hooks
bashpre-commit install
### Common commands
bashmake setup    # install all dependencies
make train    # run the training pipeline
make test     # run tests
make lint     # run ruff linter
make format   # auto-format code
### Reproduce results
bashmake setup
make train
This will preprocess the data, train the baseline model, and print evaluation metrics to the console. MLflow logs will appear in mlruns/

## 6. Contribution Summary

- Anushree Bachhav: Project proposal, repository structure, cookiecutter setup, environment configuration 
- Muhammad Anas: Data cleaning, EDA, normalization, train/val/test splits, data documentation 
- Krishna Kalakonda: Model evaluation, baseline performance documentation, architecture diagram 
- Kirtankumar Parekh: Model training, experiment tracking, Makefile, CONTRIBUTING.md, repo maintenance

## 7. References

- Dataset: Phishing Email Dataset (SpamAssassin, Enron, Nazario, Ling, CEAS, Nigerian)
- Source: Kaggle
- Use: Primary training & evaluation data


