# Phishing Email Detection

A reproducible MLOps pipeline for classifying emails as phishing or legitimate.

## Overview

Phase 2 trains on an 80% DVC-tracked sample, keeps the remaining 20% reserved
for Phase 3, compares multiple TF-IDF classifiers, logs experiments to MLflow,
and writes monitoring, profiling, and divergence reports.

## Quick Start

### Installation

```bash
# Using pip
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .
```

### Running the Pipeline

```bash
# Prepare data locally
make data

# Train configured models
make train

# Generate predictions
make predict

# Reproduce the full DVC pipeline
make repro
```

## Documentation

- [Getting Started](getting_started.md)
- [API Reference](api.md)

## Project Structure

```
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
├── conf/                              # Hydra experiment overrides
├── Makefile                           # Common commands
└── pyproject.toml                     # Packaging & deps
```

## License

This project is licensed under the MIT License. See LICENSE for details.
