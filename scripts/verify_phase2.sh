#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Prefer the venv layout that exists on this machine (Windows vs Unix).
# Git Bash / native Windows: .venv/Scripts — Linux, macOS, WSL: .venv/bin
resolve_venv_bin() {
  if [[ -x "$PWD/.venv/Scripts/python.exe" ]] || [[ -x "$PWD/.venv/Scripts/python" ]]; then
    echo "$PWD/.venv/Scripts"
  elif [[ -x "$PWD/.venv/bin/python" ]]; then
    echo "$PWD/.venv/bin"
  else
    printf 'ERROR: No usable .venv found. Create one first, e.g.:\n' >&2
    printf '  python -m venv .venv\n' >&2
    printf '  source .venv/bin/activate    # Linux / macOS / WSL\n' >&2
    printf '  source .venv/Scripts/activate  # Git Bash on Windows\n' >&2
    exit 1
  fi
}

VENV_BIN="$(resolve_venv_bin)"
export PATH="${VENV_BIN}:${PATH}"

if [[ -x "${VENV_BIN}/python.exe" ]]; then
  PYTHON="${VENV_BIN}/python.exe"
elif [[ -x "${VENV_BIN}/python" ]]; then
  PYTHON="${VENV_BIN}/python"
else
  printf 'ERROR: No python executable in %s\n' "$VENV_BIN" >&2
  exit 1
fi

# Windows venv exposes console_scripts as *.exe; Git Bash/WSL often cannot run `dvc`
# without the .exe suffix. `python -m dvc` works everywhere we support.
dvc() {
  "$PYTHON" -m dvc "$@"
}

printf '==> Using venv tools from: %s\n' "$VENV_BIN"
printf '==> Python: %s\n' "$PYTHON"
if [[ "$VENV_BIN" == *Scripts* ]] && grep -qi microsoft /proc/version 2>/dev/null; then
  printf 'NOTE: WSL + Windows .venv detected. Prefer: .\\scripts\\verify_phase2.ps1 in PowerShell,\n' >&2
  printf '      or recreate the venv inside WSL (python3 -m venv .venv → .venv/bin).\n' >&2
fi

include_slow_profile=0
clean_mlflow=0
check_remote=0
replay_mlflow=0

usage() {
  cat <<'USAGE'
Usage: scripts/verify_phase2.sh [--include-slow-profile] [--replay-mlflow] [--clean-mlflow] [--check-remote]

Runs the Phase 2 reproducibility checks in the same order used for local
verification. Works on Linux, macOS, WSL, and Git Bash on Windows (auto-detects
.venv/bin vs .venv/Scripts). Native Windows PowerShell users should run
scripts/verify_phase2.ps1 instead (see docs/windows_setup.md).

By default this does not delete local MLflow runs and does not require DVC
remote credentials.

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

run_lint() {
  printf '\n==> lint\n'
  # Always use the venv Python. `make lint` often fails on Windows/WSL because
  # Make invokes `ruff` without .venv/Scripts on PATH.
  "$PYTHON" -m ruff check .
  "$PYTHON" -m ruff format --check .
}

run_dvc() {
  printf '\n==> dvc %s\n' "$*"
  dvc "$@"
}

run_mlflow_replay() {
  printf '\n==> Replaying MLflow tracking with scratch outputs\n'
  "$PYTHON" - <<'PY'
from copy import deepcopy
from pathlib import Path

from mlops_crew.config import CONFIG_PATH, load_project_config
from mlops_crew.logging_config import setup_logging_from_config
from mlops_crew.train_model import train

config = deepcopy(load_project_config(CONFIG_PATH))
setup_logging_from_config(config)
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

run_dvc status
run_dvc repro
run_lint
run "$PYTHON" -m mypy src
run "$PYTHON" -m pytest tests/ --cov=mlops_crew --cov-report=xml
run "$PYTHON" scripts/profile_predict.py --output-dir reports/profiling/scratch/verify_predict

if [[ "$include_slow_profile" == "1" ]]; then
  run "$PYTHON" scripts/profile_train.py --output-dir reports/profiling/scratch/verify_train
fi

if [[ "$replay_mlflow" == "1" ]]; then
  run_mlflow_replay
fi

run_dvc status

if [[ "$check_remote" == "1" ]]; then
  run_dvc status -c
fi

"$PYTHON" - <<'PY'
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