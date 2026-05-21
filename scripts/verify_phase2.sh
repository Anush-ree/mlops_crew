#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PATH="$PWD/.venv/bin:$PATH"

include_slow_profile=0
clean_mlflow=0
check_remote=0
replay_mlflow=0

usage() {
  cat <<'USAGE'
Usage: scripts/verify_phase2.sh [--include-slow-profile] [--replay-mlflow] [--clean-mlflow] [--check-remote]

Runs the Phase 2 reproducibility checks in the same order used for local
verification. By default this does not delete local MLflow runs and does not
require DVC remote credentials.

Options:
  --include-slow-profile  Also run training cProfile. This retrains all models.
  --replay-mlflow         Populate MLflow using scratch outputs, without dirtying DVC.
  --clean-mlflow          Delete local mlruns, then replay MLflow with scratch outputs.
  --check-remote          Run `dvc status -c` after local verification.
USAGE
}

for arg in "$@"; do
  case "$arg" in
    --include-slow-profile)
      include_slow_profile=1
      ;;
    --replay-mlflow)
      replay_mlflow=1
      ;;
    --clean-mlflow)
      clean_mlflow=1
      replay_mlflow=1
      ;;
    --check-remote)
      check_remote=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

run() {
  printf '\n==> %s\n' "$*"
  "$@"
}

run_mlflow_replay() {
  printf '\n==> Replaying MLflow tracking with scratch outputs\n'
  python - <<'PY'
from copy import deepcopy
from pathlib import Path

from mlops_crew.config import CONFIG_PATH, load_project_config
from mlops_crew.train_model import train

config = deepcopy(load_project_config(CONFIG_PATH))
scratch_root = Path(config["reports"]["profiling_dir"]) / "scratch" / "mlflow_replay"
config["modeling"]["output_dir"] = str(scratch_root / "models")
config["reports"]["metrics_dir"] = str(scratch_root / "metrics")
config["reports"]["predictions_dir"] = str(scratch_root / "predictions")
config["reports"]["monitoring_dir"] = str(scratch_root / "monitoring")
config["tracking"]["enabled"] = True

train(config)
PY
}

if [[ "$clean_mlflow" == "1" ]]; then
  printf '\n==> Removing local mlruns for a fresh experiment-tracking replay\n'
  rm -rf mlruns
fi

run dvc status
run dvc repro
run make lint
run mypy src
run pytest tests/ --cov=mlops_crew --cov-report=xml
run python scripts/profile_predict.py

if [[ "$include_slow_profile" == "1" ]]; then
  run python scripts/profile_train.py
fi

if [[ "$replay_mlflow" == "1" ]]; then
  run_mlflow_replay
fi

run dvc status

if [[ "$check_remote" == "1" ]]; then
  run dvc status -c
fi

python - <<'PY'
import json
from pathlib import Path

best = json.loads(Path("reports/metrics/best_model_metrics.json").read_text())
divergence = json.loads(Path("reports/divergence/phase2_divergence_report.json").read_text())
transformer = json.loads(Path("data/processed/transformer/dataset_info.json").read_text())

print("\n==> Phase 2 summary")
print(f"Best model: {best['model_name']}")
print(f"Validation F2: {best['validation']['f2']:.6f}")
print(f"Test F2: {best['test']['f2']:.6f}")
print(f"Test false-negative rate: {best['test']['false_negative_rate']:.6f}")
print(
    "Label JS distance: "
    f"{divergence['label_distribution']['jensen_shannon_distance']:.6f}"
)
print(
    "Source JS distance: "
    f"{divergence['source_distribution']['jensen_shannon_distance']:.6f}"
)
print(
    "Transformer JSONL rows: "
    + ", ".join(
        f"{split}={details['rows']}" for split, details in transformer["splits"].items()
    )
)
PY
