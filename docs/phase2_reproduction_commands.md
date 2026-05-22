# Phase 2 Reproduction Commands

This is the command sequence used to verify the Phase 2 model-development work
on `model_v2`. It covers Sections 2-4: monitoring/debugging, profiling, and
experiment tracking.

## One-command local verification

```bash
cd mlops_crew
python -m venv .venv
# Linux / macOS / WSL:
source .venv/bin/activate
# Git Bash on Windows:
# source .venv/Scripts/activate
make install
make dev
dvc pull

scripts/verify_phase2.sh
```

**Windows graders:** use PowerShell and `.\scripts\verify_phase2.ps1` — see
[windows_setup.md](./windows_setup.md). Git Bash can use `verify_phase2.sh`
(`.venv/Scripts` is detected automatically). WSL uses the Bash script with
`.venv/bin`.

`scripts/verify_phase2.sh` auto-detects `.venv/bin` (Unix) vs `.venv/Scripts`
(Windows). If `make` is missing, lint falls back to `ruff` directly.

The script runs:

```bash
dvc status
dvc repro   # includes validate -> data/processed/validation_report.json
make lint
mypy src
pytest tests/ --cov=mlops_crew --cov-report=xml
python scripts/profile_predict.py --output-dir reports/profiling/scratch/verify_predict
dvc status
```

To also regenerate the slower training profile:

```bash
scripts/verify_phase2.sh --include-slow-profile
```

To populate the local MLflow UI after `dvc pull` restored outputs and
`dvc repro` skipped training:

```bash
scripts/verify_phase2.sh --replay-mlflow
```

To replay MLflow from a clean local tracking directory:

```bash
scripts/verify_phase2.sh --clean-mlflow
```

To check that local DVC cache contents are synced to the default remote:

```bash
scripts/verify_phase2.sh --check-remote
```

Flags can be combined:

```bash
scripts/verify_phase2.sh --clean-mlflow --include-slow-profile --check-remote
```

## Manual command sequence

Use this when you want to inspect each step separately:

```bash
cd mlops_crew
source .venv/bin/activate

dvc status
dvc repro

make lint
mypy src
pytest tests/ --cov=mlops_crew --cov-report=xml

python scripts/profile_predict.py
python scripts/profile_train.py

dvc status
dvc status -c
```

The profiling scripts write readable cProfile summaries to
`reports/profiling/*_cprofile.txt`. They write temporary train/latency outputs
under `reports/profiling/scratch/`, which is ignored, so profiling does not
dirty DVC-tracked model, metric, prediction, monitoring, or MLflow artifacts by
default. The verification script also writes its profile output to scratch so a
clean worktree stays clean after verification.

## Expected final artifacts

After `dvc repro`, these are the key files a grader should inspect:

- `models/best_model.joblib`
- `reports/metrics/best_model_metrics.json`
- `reports/metrics/model_comparison.csv`
- `reports/monitoring/training_resource_usage.csv`
- `reports/monitoring/inference_latency.csv`
- `reports/divergence/phase2_divergence_report.json`
- `reports/divergence/phase2_divergence_summary.md`
- `data/processed/transformer/train.jsonl`
- `data/processed/transformer/val.jsonl`
- `data/processed/transformer/test.jsonl`
- `data/processed/transformer/dataset_info.json`
- local regenerated MLflow runs under `mlruns/` when training executes or
  `scripts/verify_phase2.sh --replay-mlflow` is used

The transformer files are dataset exports only. This branch does not fine-tune
or train a transformer/LLM model.
