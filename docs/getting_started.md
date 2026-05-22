# Getting Started with phishing_email_detection

## Prerequisites

- Python 3.11 or higher
- pip
- Git (for version control)
- DVC access to the project S3 remote

## Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd mlops_crew
```

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -U pip
make install
```

### Step 4: Set Up Development Environment

```bash
make dev
make test
```

## Running the Project

### Data Processing

Fetch DVC-tracked data from S3, then prepare the Phase 2 sample and splits:

```bash
dvc pull
make data
```

Or directly:

```bash
python -m mlops_crew.data.make_dataset
```

### Model Training

Train the configured models:

```bash
make train
```

### Model Prediction

Generate predictions on new data:

```bash
make predict
```

With custom paths:

```bash
python -m mlops_crew.models.predict_model \
  --model-path models/best_model.joblib \
  --input data/processed/test.csv \
  --output reports/predictions/manual_predictions.csv
```

## Development Workflow

### Running Tests

```bash
# Run all tests
make test

# Run tests with coverage
pytest tests/ --cov=mlops_crew

# Run the pipeline tests
pytest tests/test_phase1_pipeline.py -v
```

### Code Quality

```bash
# Check for linting issues
make lint

# Auto-format and fix issues
make format

# Type checking
mypy src
```

### Pre-commit Hooks

Pre-commit hooks automatically run before each commit:

```bash
# Manually run pre-commit checks
pre-commit run --all-files

# Update hooks
pre-commit autoupdate
```

## Project Structure

```
mlops_crew/                  # Repository root
├── src/
│   └── mlops_crew/          # Importable package (src/ layout)
│       ├── config.py                  # Repo paths + YAML config loader
│       ├── logging_config.py
│       ├── data/                      # sample, manifest, clean, split, export
│       ├── models/                    # TF-IDF classifier pipeline factory
│       ├── evaluation/                # Metrics and comparison plots
│       ├── monitoring/                # Divergence, latency, resource usage
│       ├── tracking/                  # MLflow helpers
│       ├── utils/                     # seed, io
│       ├── models/                    # train/predict CLI wrappers
│       │   ├── train_model.py
│       │   └── predict_model.py
│       ├── train_model.py
│       ├── train_hydra.py
│       └── predict_model.py
├── tests/                             # Unit tests
├── data/                              # raw/ and processed/
├── models/                            # Trained model artifacts
├── docs/                              # MkDocs documentation
├── configs/                           # Project configuration
├── conf/                              # Hydra experiment overrides
├── pyproject.toml
├── requirements.txt
└── Makefile
```

## Configuration

Edit `configs/config.yaml` to change paths, the 60/20/20 phase partitioning,
split ratios, TF-IDF settings, tracking settings, monitoring outputs, or the
list of models to train. Re-run `make repro` after config changes so DVC
updates the pipeline outputs.

Use `conf/` only for Hydra experiment overlays. For example, `make hydra-demo`
runs two MLflow-tracked training experiments without changing DVC-tracked
artifacts.

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError`, ensure:
1. Virtual environment is activated
2. Dependencies are installed: `pip install -r requirements.txt`
3. Package is installed in editable mode: `pip install -e .`

### Pre-commit Hook Failures

If pre-commit hooks fail:

```bash
# See what failed
pre-commit run --all-files

# Fix issues manually or with auto-fix
make format

# Try committing again
```

## Next Steps

1. Review the [documentation](index.md)
2. Start with the root [Phase 2 deliverable](../PHASE2.md)
3. Check the [API Reference](api.md)

## Support

For issues and questions:
- Check existing [documentation](index.md)
- Review the root [Phase deliverable](../PHASE2.md)
- Contact kirtan (kparekh2@depaul.edu)
