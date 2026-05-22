# Windows setup and grading

This project is developed and graded on **Windows**, **macOS**, and **Linux**.
You do not need WSL if you use native Windows Python and PowerShell; WSL is an
optional path some teammates prefer.

## Recommended paths on Windows

| Path | Tools | Verification script |
| --- | --- | --- |
| **A — PowerShell (recommended)** | Python 3.11+, venv, AWS CLI | `.\scripts\verify_phase2.ps1` |
| **B — Git Bash** | Git for Windows includes Bash | `bash scripts/verify_phase2.sh` |
| **C — WSL2** | Ubuntu in WSL, Linux venv | `bash scripts/verify_phase2.sh` |

Both verification scripts run the same checks as the course reproduction list
(`dvc status`, `dvc repro`, lint, mypy, pytest, profiling smoke test).

`scripts/verify_phase2.sh` auto-detects `.venv/Scripts` (Windows) vs `.venv/bin`
(Unix). `scripts/verify_phase2.ps1` is for graders who stay in PowerShell.

## One-time prerequisites

1. **Python 3.11+** — [python.org](https://www.python.org/downloads/) or
   `winget install Python.Python.3.12`
2. **Git for Windows** (optional, for Git Bash) —
   [git-scm.com](https://git-scm.com/download/win)
3. **AWS CLI v2** (for `dvc pull` from S3) — MSI installer or
   `choco install awscli` (Chocolatey)
4. **`make`** (optional) — only if you want Makefile shortcuts:
   - Chocolatey: `choco install make`
   - Or use the command equivalents in [README.md](../README.md) and skip `make`

Restart the terminal after installing so `aws` and `make` are on `PATH`.

## Path A — Native Windows (PowerShell)

```powershell
cd path\to\mlops_crew
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt -r requirements_dev.txt
pip install -e .

aws configure   # region: us-east-2; keys from the team
dvc pull        # add --force if prompted about local outputs

.\scripts\verify_phase2.ps1
```

Open MLflow UI (no `make` required):

```powershell
mlflow ui --backend-store-uri ./mlruns --port 5001
```

## Path B — Git Bash on Windows

Create the venv in **Windows** (so tools live under `.venv/Scripts`):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements_dev.txt
pip install -e .
```

Then in **Git Bash**:

```bash
cd /d/path/to/mlops_crew
export PATH="$PWD/.venv/Scripts:$PATH"
bash scripts/verify_phase2.sh
```

## Path C — WSL2

Clone or access the repo under `/mnt/d/...` and use a **Linux** venv only in WSL:

```bash
cd /mnt/d/path/to/mlops_crew
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements_dev.txt
pip install -e .
dvc pull
bash scripts/verify_phase2.sh
```

Do not mix a venv created in WSL (`/usr/bin` in `pyvenv.cfg`) with PowerShell
on the same folder — recreate `.venv` in the environment you actually use.

**WSL + Windows `.venv`:** If `bash scripts/verify_phase2.sh` shows
`Using venv tools from: .../Scripts` but failed on `dvc` before, the script now
runs `python -m dvc` via your Windows `python.exe`. For fewer surprises, use
`.\scripts\verify_phase2.ps1` in PowerShell or create a Linux venv inside WSL
(`python3 -m venv .venv` → `.venv/bin`).

## Makefile without `make`

| `make` target | PowerShell / direct command |
| --- | --- |
| `make install` | `pip install -r requirements.txt` then `pip install -e .` |
| `make repro` | `dvc repro` |
| `make test` | `pytest tests/` |
| `make lint` | `ruff check .` then `ruff format --check .` |
| `make mlflow-ui` | `mlflow ui --backend-store-uri ./mlruns --port 5001` |
| `make train` | `python -m mlops_crew.models.train_model` |

## Verification script options

Bash (`scripts/verify_phase2.sh`) and PowerShell (`scripts/verify_phase2.ps1`)
support the same flags:

| Flag | Bash | PowerShell |
| --- | --- | --- |
| Slow training profile | `--include-slow-profile` | `-IncludeSlowProfile` |
| MLflow replay | `--replay-mlflow` | `-ReplayMlflow` |
| Clean `mlruns` + replay | `--clean-mlflow` | `-CleanMlflow` |
| Remote cache check | `--check-remote` | `-CheckRemote` |

## Common Windows issues

| Symptom | Fix |
| --- | --- |
| `make: not recognized` | Use PowerShell equivalents or `choco install make` |
| `dvc: command not found` in Git Bash | Use `verify_phase2.ps1`, or ensure `.venv/Scripts` is on `PATH` |
| `No Python at '/usr/bin/...'` | Venv was built in WSL; delete `.venv` and recreate in Windows |
| `aws: not recognized` | Install AWS CLI; restart terminal; or set `AWS_*` env vars |
| `dvc pull` overwrite prompt | `dvc pull --force` after backing up local `models/` if needed |

## Grader quick check (minimal)

After `dvc pull` and a successful verify script:

- `models/best_model.joblib` exists
- `reports/metrics/best_model_metrics.json` exists
- `reports/divergence/phase2_divergence_report.json` exists
- MLflow screenshots under `reports/` (if committed) or local `mlruns/` after replay

See [phase2_reproduction_commands.md](./phase2_reproduction_commands.md) for the
full artifact list.
