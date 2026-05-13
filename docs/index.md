# Phishing Email Detection

A reproducible MLOps pipeline for classifying emails as phishing or legitimate.

## Overview

Phase 1 keeps the system intentionally small: sample the DVC-tracked raw data,
clean it, create deterministic train/validation/test splits, train baseline
models, and save metrics plus model artifacts.

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
# Prepare data: sample -> clean -> split -> validate
make data

# Train configured baseline models
make train

# Generate predictions
make predict

# Reproduce the DVC pipeline: sample -> clean -> split -> train
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
│       ├── data/                      # sample, clean, split, validate
│       ├── models/                    # TF-IDF classifier pipeline factory
│       ├── evaluation/                # Metric helpers
│       ├── utils/                     # seed, io
│       ├── train_model.py             # Training CLI
│       └── predict_model.py           # Inference CLI
├── data/                              # raw/ and processed/
├── models/                            # Trained artifacts
├── tests/                             # Unit tests
├── docs/                              # MkDocs docs
├── Makefile                           # Common commands
└── pyproject.toml                     # Packaging & deps
```

## License

This project is licensed under the MIT License. See LICENSE for details.
