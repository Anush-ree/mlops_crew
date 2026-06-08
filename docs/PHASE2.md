# Phase 2: Model Development and Operations

Phase 2 extends the Phase 1 60% baseline to an **80%** reproducible modeling run.
The final **20%** is reserved as the Phase 3 holdout. This phase adds monitoring,
profiling, MLflow tracking, Rich logging, Hydra experiment overlays, and
Docker-ready training/serving images.

## Implemented

- Deterministic 60/20/20 phase partitions (Phase 1 ref / Phase 2 increment /
  Phase 3 holdout)
- DVC stages: sample → clean → split → **validate** → transformer export → train
  → latency → plot → divergence
- Source-block manifest for source-level divergence
- Four TF-IDF models: dummy, logistic regression, linear SVC, complement NB
- MLflow experiment tracking (params, metrics, models, predictions)
- Resource usage, inference latency, and divergence reports
- cProfile scripts for training and prediction
- Rich console + rotating file logs (`logs/pipeline.log`)
- Hydra demos via `conf/` and `train_hydra.py` (DVC still uses `configs/config.yaml`)
- Transformer-ready JSONL export (no LLM training in this phase)
- Code-only Docker images (`train.dockerfile`, `predict.dockerfile`, `serve.dockerfile`)

## Best model

`linear_svc` selected by validation F2.

| Metric | Validation | Test |
| --- | ---: | ---: |
| F2 | 0.9924 | 0.9912 |
| Recall | 0.9930 | 0.9922 |
| False-negative rate | 0.0070 | 0.0078 |

## Common commands

```bash
dvc pull
make repro
scripts/verify_phase2.ps1    # Windows
scripts/verify_phase2.sh     # Bash
make mlflow-ui
make hydra-demo
make divergence
make profile-train
```

See the root [PHASE2.md](../PHASE2.md) for reproduction steps, MLflow wiring,
profiling notes, and evidence screenshots.
