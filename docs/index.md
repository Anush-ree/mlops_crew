# Phishing Email Detection

A reproducible MLOps pipeline for classifying emails as phishing or legitimate.

## Overview

| Phase | Focus |
| --- | --- |
| [Phase 1](PHASE1.md) | 60% baseline, DVC pipeline, TF-IDF + logistic regression |
| [Phase 2](PHASE2.md) | 80% data, MLflow, monitoring, Hydra, Docker images |
| [Phase 3](PHASE3.md) | CI/CD, CML, GCP + FastAPI, HF Spaces, holdout eval |

**Live demo:** [Hugging Face Space](https://huggingface.co/spaces/mlops-crew-depaul/phishing-email-detector)

Full submission write-ups live at the repository root:

- [PHASE1.md](../PHASE1.md)
- [PHASE2.md](../PHASE2.md)
- [PHASE3.md](../PHASE3.md)

## Quick start

```bash
pip install -r requirements.txt -r requirements_dev.txt
pip install -e .

aws configure          # teammates provide S3 credentials; region us-east-2
dvc pull

make repro             # full DVC pipeline
pytest tests/          # CI-equivalent tests (no full dataset required)
```

Windows graders: `.\scripts\verify_phase2.ps1` after `dvc pull`.

## Phase 3 serving (local)

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8080
curl http://localhost:8080/health
```

Or: `make docker-serve` with `models/best_model.joblib` mounted.

## Documentation

| Doc | Contents |
| --- | --- |
| [Getting Started](getting_started.md) | Install, pipeline, troubleshooting |
| [API Reference](api.md) | Package modules + FastAPI `/health` and `/predict` |
| [Windows setup](windows_setup.md) | PowerShell, Git Bash, WSL |
| [Phase 2 reproduction](phase2_reproduction_commands.md) | Grader command sequence |
| [GCP cleanup](cleanup.md) | Teardown commands after demos |
| [Phase 1 summary](PHASE1.md) | Brief Phase 1 overview |
| [Phase 2 summary](PHASE2.md) | Brief Phase 2 overview |
| [Phase 3 summary](PHASE3.md) | Brief Phase 3 overview |

## Project structure

```markdown
mlops_crew/                  # Repository root
├── src/
│   └── mlops_crew/          # Importable package (src/ layout)
│       ├── config.py                  # Repo paths + YAML config loader
│       ├── logging_config.py
│       ├── data/                      # sample, manifest, clean, split, export
│       ├── models/                    # TF-IDF classifier pipeline factory
│       ├── evaluation/                # Metric helpers and plots
│       ├── monitoring/                # divergence, latency, resource usage
│       ├── tracking/                  # MLflow helpers
│       ├── utils/                     # seed, io
│       ├── models/train_model.py      # Training CLI
│       ├── train_hydra.py             # Hydra experiment CLI
│       └── models/predict_model.py    # Inference CLI
├── data/                              # raw/ and processed/
├── models/                            # Trained artifacts
├── tests/                             # Unit tests
├── docs/                              # MkDocs docs
├── configs/hydra/                     # Hydra experiment overrides
├── Makefile                           # Common commands
└── pyproject.toml                     # Packaging & deps
```

## License

MIT — see [LICENSE](../LICENSE).
