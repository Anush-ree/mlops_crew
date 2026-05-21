# Phase 2: Model Development and Operations

Phase 2 trains on an 80% reproducible sample, tracks experiments with MLflow,
adds divergence and latency monitoring, and exports JSONL splits for later
transformer fine-tuning.

## Implemented

- Deterministic 60/20/20 phase partitioning.
- Source-block manifest for source-level divergence reporting.
- TF-IDF experiments: dummy, Logistic Regression, Linear SVC, Complement NB.
- MLflow tracking for params, metrics, models, and prediction artifacts.
- Resource usage, inference latency, and divergence reports.
- cProfile scripts for training and prediction.
- Model comparison CSV/JSON plus a generated comparison chart.
- Transformer-ready JSONL train/validation/test export.

## Best Model

`linear_svc` is selected by validation F2.

| Metric | Validation | Test |
| --- | ---: | ---: |
| F2 | 0.9924 | 0.9912 |
| Recall | 0.9930 | 0.9922 |
| False negative rate | 0.0070 | 0.0078 |

## Commands

```bash
make repro
make mlflow-ui
make divergence
make latency
make profile-train
make profile-predict
```

See the root [PHASE2.md](../PHASE2.md) for the full deliverable details.
