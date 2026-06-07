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

### Phase 3 API and UI

Start the FastAPI prediction service:

```bash
make api
```

Send a JSON request:

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Urgent account verification required. Click the secure link now."}'
```

Start the Gradio UI:

```bash
make ui
```

Build the Cloud Run-ready serving container:

```bash
make docker-serve
```

### Docker Workflow

The repo also includes Docker images for training and batch prediction. Use
these when you want a reproducible container that mounts the same host data and
config files used by the local workflow.

Build the images from the repository root:

```bash
docker build -f train.dockerfile . -t train:latest
docker build -f predict.dockerfile . -t predict:latest
```

Run training:

```bash
docker run --rm \
  -e MLOPS_CREW_PROJECT_ROOT=/app \
  -v "$PWD/data:/app/data" \
  -v "$PWD/configs:/app/configs" \
  train:latest
```

Run prediction:

```bash
docker run --rm \
  -e MLOPS_CREW_PROJECT_ROOT=/app \
  -v "$PWD/data:/app/data" \
  -v "$PWD/configs:/app/configs" \
  -v "$PWD/models:/app/models" \
  -v "$PWD/reports:/app/reports" \
  predict:latest \
  --model-path /app/models/best_model.joblib \
  --input /app/data/processed/test.csv \
  --output /app/reports/predictions/batch_predictions.csv
```

If you prefer the Makefile wrappers, run `make docker-train` or
`make docker-predict`.

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
│       ├── models/                    # Classifier factory + train/predict CLIs
│       │   ├── train_model.py
│       │   └── predict_model.py
│       └── train_hydra.py
├── tests/                             # Unit tests
├── data/                              # raw/ and processed/
├── models/                            # Trained model artifacts
├── docs/                              # MkDocs documentation
├── configs/                           # Project and Hydra configuration
│   └── hydra/                         # Hydra experiment overrides
├── pyproject.toml
├── requirements.txt
└── Makefile
```

## Configuration

Edit `configs/config.yaml` to change paths, the 60/20/20 phase partitioning,
split ratios, TF-IDF settings, tracking settings, monitoring outputs, or the
list of models to train. Re-run `make repro` after config changes so DVC
updates the pipeline outputs.

Use `configs/hydra/` only for Hydra experiment overlays. For example,
`make hydra-demo` runs two MLflow-tracked training experiments without changing
DVC-tracked artifacts.

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
