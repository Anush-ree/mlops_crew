# Tool Documentation

## Docker

Purpose: provide reproducible containerized training and prediction workflows for collaborators who want to run the project without installing the full local stack.

Setup: build the images from the repository root with `docker build -f train.dockerfile . -t train:latest` and `docker build -f predict.dockerfile . -t predict:latest`. The containers expect the local project files and DVC-tracked data to be mounted from the host.

Use: mount `data/`, `configs/`, `models/`, and `reports/` into the container, then run the training image for model fitting or the prediction image for batch inference. The repo also includes `make docker-train` and `make docker-predict` wrappers for the same flows.


## DVC

Purpose: version and reproduce the data pipeline outputs, including raw data partitions, processed splits, metrics, and reports.

Setup: install DVC with the S3 remote support package, then configure AWS credentials so `dvc pull` can access the shared cache.

Use: run `dvc pull` to restore tracked artifacts, `dvc repro` to rebuild the pipeline, and `dvc status` or `dvc status -c` to verify local and remote state.

## MLflow

Purpose: track experiments, compare runs, and store training artifacts such as metrics and model outputs.

Setup: run the training workflow once so MLflow artifacts are produced locally, then open the UI with `make mlflow-ui` or `mlflow ui --backend-store-uri ./mlruns --port 5001`.

Use: review the run parameters, metrics, and artifacts when comparing model families or replaying Phase 2 experiments.

## Hydra

Purpose: run configuration-driven experiment variants without editing the main pipeline config.

Setup: ensure the project is installed in editable mode so the `mlops_crew.train_hydra` module can be imported.

Use: run `make hydra-demo` or invoke `python -m mlops_crew.train_hydra experiment=phase2_default` and `python -m mlops_crew.train_hydra experiment=phase2_experimental` to compare overrides.

## Pytest

Purpose: validate the pipeline, reporting, monitoring, and logging behavior.

Setup: install the dev dependencies from `requirements_dev.txt`.

Use: run `make test` or `pytest tests/ --cov=mlops_crew --cov-report=xml` to confirm the project still behaves as expected after changes.

## Ruff

Purpose: enforce formatting and linting standards.

Setup: install the development dependencies.

Use: run `make lint` to check style and `make format` to auto-fix formatting issues.

## cProfile

Purpose: measure training and inference performance so runtime regressions are visible.

Setup: no extra setup is required beyond a working project environment.

Use: run `make profile-train` and `make profile-predict` when you need profiling evidence or want to compare runtime cost across changes.

## Phase 2 Verification Scripts

Purpose: provide the quickest end-to-end reproducibility check for graders and external collaborators.

Setup: use the Bash script on Unix-like systems or the PowerShell script on Windows.

Use: run `scripts/verify_phase2.sh --replay-mlflow --check-remote` on Bash-compatible shells, or `./scripts/verify_phase2.ps1 -ReplayMlflow -CheckRemote` on Windows PowerShell.
