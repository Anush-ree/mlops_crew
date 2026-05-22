# Phase 2 verification for native Windows PowerShell (same steps as verify_phase2.sh).
# Usage:
#   .\scripts\verify_phase2.ps1
#   .\scripts\verify_phase2.ps1 -IncludeSlowProfile
#   .\scripts\verify_phase2.ps1 -ReplayMlflow
#   .\scripts\verify_phase2.ps1 -CleanMlflow -ReplayMlflow
#   .\scripts\verify_phase2.ps1 -CheckRemote

param(
    [switch]$IncludeSlowProfile,
    [switch]$ReplayMlflow,
    [switch]$CleanMlflow,
    [switch]$CheckRemote
)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$venvScripts = Join-Path $PWD ".venv\Scripts"
$pythonExe = Join-Path $venvScripts "python.exe"
if (-not (Test-Path $pythonExe)) {
    Write-Error @"
No Windows venv at .venv\Scripts\python.exe.
Create and install first:
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install -r requirements.txt -r requirements_dev.txt
  pip install -e .
"@
}

$env:PATH = "$venvScripts;$env:PATH"
Write-Host "==> Using venv tools from: $venvScripts"

if ($CleanMlflow) {
    $ReplayMlflow = $true
}

function Invoke-Step {
    param(
        [string]$Label,
        [scriptblock]$Action
    )
    Write-Host ""
    Write-Host "==> $Label"
    & $Action
}

function Invoke-Lint {
    Write-Host ""
    Write-Host "==> lint"
    if (Get-Command make -ErrorAction SilentlyContinue) {
        make lint
    } else {
        ruff check --no-cache .
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        ruff format --check .
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
}

function Invoke-MlflowReplay {
    Write-Host ""
    Write-Host "==> Replaying MLflow tracking with scratch outputs"
    & $pythonExe -c @"
from copy import deepcopy
from pathlib import Path

from mlops_crew.config import CONFIG_PATH, load_project_config
from mlops_crew.logging_config import setup_logging_from_config
from mlops_crew.models.train_model import train

config = deepcopy(load_project_config(CONFIG_PATH))
setup_logging_from_config(config)
scratch_root = Path(config['reports']['profiling_dir']) / 'scratch' / 'mlflow_replay'
config['modeling']['output_dir'] = str(scratch_root / 'models')
config['reports']['metrics_dir'] = str(scratch_root / 'metrics')
config['reports']['predictions_dir'] = str(scratch_root / 'predictions')
config['reports']['monitoring_dir'] = str(scratch_root / 'monitoring')
config['tracking']['enabled'] = True

train(config)
"@
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if ($CleanMlflow) {
    Write-Host ""
    Write-Host "==> Removing local mlruns for a fresh experiment-tracking replay"
    if (Test-Path "mlruns") {
        Remove-Item -Recurse -Force "mlruns"
    }
}

Invoke-Step "dvc status" { dvc status; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE } }
Invoke-Step "dvc repro" { dvc repro; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE } }
Invoke-Lint
Invoke-Step "mypy src" { mypy src; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE } }
Invoke-Step "pytest" {
    pytest tests/ --cov=mlops_crew --cov-report=xml
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
Invoke-Step "profile_predict" {
    & $pythonExe scripts/profile_predict.py --output-dir reports/profiling/scratch/verify_predict
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if ($IncludeSlowProfile) {
    Invoke-Step "profile_train" {
        & $pythonExe scripts/profile_train.py --output-dir reports/profiling/scratch/verify_train
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
}

if ($ReplayMlflow) {
    Invoke-MlflowReplay
}

Invoke-Step "dvc status (final)" { dvc status; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE } }

if ($CheckRemote) {
    Invoke-Step "dvc status -c" { dvc status -c; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE } }
}

& $pythonExe -c @"
import json
from pathlib import Path

best = json.loads(Path('reports/metrics/best_model_metrics.json').read_text())
divergence = json.loads(Path('reports/divergence/phase2_divergence_report.json').read_text())
transformer = json.loads(Path('data/processed/transformer/dataset_info.json').read_text())

print('\n==> Phase 2 summary')
print(f"Best model: {best['model_name']}")
print(f"Validation F2: {best['validation']['f2']:.6f}")
print(f"Test F2: {best['test']['f2']:.6f}")
print(f"Test false-negative rate: {best['test']['false_negative_rate']:.6f}")
print(
    'Label JS distance: '
    f"{divergence['label_distribution']['jensen_shannon_distance']:.6f}"
)
print(
    'Source JS distance: '
    f"{divergence['source_distribution']['jensen_shannon_distance']:.6f}"
)
print(
    'Transformer JSONL rows: '
    + ', '.join(
        f"{split}={details['rows']}" for split, details in transformer['splits'].items()
    )
)
"@
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "==> Phase 2 verification finished successfully."
