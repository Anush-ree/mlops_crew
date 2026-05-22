# PHASE 2: Enhancing ML Operations

## Scope

Phase 2 extends the Phase 1 phishing classifier from a 60% baseline run to an
80% reproducible modeling run. This branch implements the model-development and
experiment-operations portion of Phase 2:

- Section 2: Monitoring and debugging
- Section 3: Profiling Python and ML code
- Section 4: Experiment management and tracking
- Section 5: Application logging with Rich and rotating log files
- Section 6: Hydra configuration experiments

Containerization is tracked separately. The DVC production pipeline keeps
`configs/config.yaml` as the source of truth, while Hydra uses `conf/` as an
experiment wrapper that overlays config changes and writes scratch artifacts
under ignored `outputs/hydra/` directories.

## Data Versioning

The raw combined dataset has 82,486 rows. Phase 2 uses 80% of that data and
holds the final 20% for Phase 3.

| Partition | Rows | Purpose |
| --- | ---: | --- |
| Phase 1 reference | 49,492 | Previous 60% baseline/reference |
| Phase 2 increment | 16,497 | New 20% added in Phase 2 |
| Phase 2 sample | 65,989 | Training input for this phase |
| Phase 3 holdout | 16,497 | Reserved for the final phase |

The pipeline also writes `data/interim/source_manifest.csv`, a DVC-tracked
source-block manifest for the combined raw file. The manifest validates source
block row counts and label distributions for Enron, Ling, CEAS 2008,
SpamAssassin, Nigerian Fraud, and Nazario. The source order was confirmed from
source-specific text prefixes in the combined file. This avoids recombining or
duplicating the existing dataset.

After `split`, the `validate` DVC stage writes `data/processed/validation_report.json`
(a row-count and label-distribution snapshot). Training and transformer export
depend on that artifact so `dvc repro` cannot skip validation.

### Pipeline versioning note

Phase 1 used a single 60% stratified sample on `main`. Phase 2 replaces that
with deterministic phase partitions (`sample.py`), explicit holdout reservation,
and DVC stages in `dvc.yaml`. There is no separate legacy preprocess script in
this branch; `reports/metrics/phase1_baseline/` is the frozen Phase 1 metric
snapshot for comparisons.

## Monitoring and Debugging

Implemented monitoring/debugging checks:

- Deterministic 60/20/20 phase partitioning with non-overlap checks.
- Post-split validation writes `data/processed/validation_report.json` (DVC stage).
- Source manifest validation checks source block row counts and label distributions.
- Training resource sampling writes `reports/monitoring/training_resource_usage.csv`.
- Inference latency benchmark writes `reports/monitoring/inference_latency.csv`.
- Data and prediction divergence report writes:
  - `reports/divergence/phase2_divergence_report.json`
  - `reports/divergence/phase2_divergence_summary.md`

Current divergence summary:

| Check | Result |
| --- | ---: |
| Label Jensen-Shannon distance | 0.000007 |
| Source Jensen-Shannon distance | 0.014327 |
| Text length KS statistic | 0.005134 |
| New-token rate in Phase 2 increment | 0.039855 |
| Prediction Jensen-Shannon distance | 0.000207 |

The Phase 2 increment is very close to the Phase 1 reference distribution by
label and prediction distribution. The source distribution is also stable enough
for this phase, but it is now measurable and should be reviewed again whenever a
new source is added.

Debugging entry points:

```bash
python -m pdb -m mlops_crew.models.train_model
python -m pdb -m mlops_crew.monitoring.divergence
PATH="$PWD/.venv/bin:$PATH" dvc repro
PATH="$PWD/.venv/bin:$PATH" dvc status
```

## Profiling and Optimization

Profiling scripts:

```bash
make profile-train
make profile-predict
```

By default the profiling scripts read the normal DVC inputs but write temporary
outputs under `reports/profiling/scratch/` and disable MLflow tracking. This
prevents profiling from creating duplicate experiment runs or dirtying the
DVC-tracked model, prediction, metric, and monitoring outputs. Use
`python scripts/profile_train.py --with-tracking` only when you explicitly want
to profile MLflow logging overhead too. Use `--output-dir` to write profile
summaries to ignored scratch space during verification.

Generated artifacts:

- `reports/profiling/train_model_cprofile.txt`
- `reports/profiling/predict_model_cprofile.txt`
- Raw `.prof` files are generated locally and ignored by Git.

Baseline profiling findings:

- Training under cProfile took about 147 seconds on the local machine used for
  this run. Normal DVC training can be faster because cProfile adds overhead.
- TF-IDF `fit_transform` dominates training time, which is expected for this
  pipeline because every classifier owns a full TF-IDF pipeline.
- The committed profile disables MLflow tracking so the profiler output stays
  focused on model training and does not create duplicate local runs.
- Saved-model inference is fast after the model is loaded: the 1,024-row batch
  benchmark is roughly 8k records/second on this machine.

### Optimization evaluated (deferred)

| Candidate change | Expected effect | Decision |
| --- | --- | --- |
| Shared TF-IDF matrix across the four sklearn pipelines | Large training-time reduction (TF-IDF dominates cProfile) | **Deferred** — adds coupling between models and complicates MLflow per-model artifacts |
| Smaller `max_features` or fewer n-grams | Faster vectorization, lower memory | **Deferred** — risks recall/F2 on phishing detection |
| MLflow disabled during profiling (already done) | Cleaner profiler signal | **Applied** in `scripts/profile_train.py` default |

No other code change was applied in this phase because model quality is already
strong and the current per-model pipelines are easier to reproduce and compare.
Revisit shared TF-IDF caching if repeated experiment sweeps make training time a
blocker.

## Experiment Tracking

MLflow tracking is configured in `configs/config.yaml`:

```yaml
tracking:
  enabled: true
  experiment_name: phishing-email-phase2
  tracking_uri: file:./mlruns
```

The local `mlruns/` directory is gitignored. A training run populates it when
the train stage actually executes. If `dvc pull` already restored every DVC
output and `dvc repro` skips training, populate the UI with the scratch replay:

```bash
scripts/verify_phase2.sh --replay-mlflow
```

Open the UI with:

```bash
make mlflow-ui              # serves http://localhost:5001
```

Exact wiring (where logging happens in the code):

| What gets logged | Code site |
| --- | --- |
| Parent run open + close + config artifact | `src/mlops_crew/tracking/mlflow_tracking.py:27-48` (`training_run`) |
| Per-model nested run with model params + TF-IDF params | `src/mlops_crew/tracking/mlflow_tracking.py:51-72` (`model_run`) |
| Dataset rows + label counts | `src/mlops_crew/tracking/mlflow_tracking.py:75-79` (`log_dataset_info`) |
| Validation + test metrics | `src/mlops_crew/tracking/mlflow_tracking.py:82-86` (`log_metrics`) |
| Joblib + sklearn flavor + optional prediction CSVs + monitoring CSV | `src/mlops_crew/tracking/mlflow_tracking.py:89-118` (`log_artifacts`, `log_model_artifacts`) |
| Call sites that drive all of the above | `src/mlops_crew/models/train_model.py` (`train`) |

What a grader will see in the UI after a training run or `--replay-mlflow` +
`make mlflow-ui`:

- Experiment `phishing-email-phase2`.
- One parent run named `phase2-training` per training invocation, with
  `config/config.yaml` attached as an artifact and four nested runs.
- Each nested run (`dummy`, `logistic_regression`, `linear_svc`,
  `complement_nb`) has params (`model_name`, `model.*`, `tfidf.*`), metrics
  prefixed `validation_*` / `test_*`, the saved `<model>.joblib` under
  `joblib/`, the corresponding `val_predictions.csv` / `test_predictions.csv`
  under `predictions/`, and the sklearn model flavor.
- The parent run also has `metrics/model_comparison.csv`,
  `metrics/model_comparison.json`, `metrics/best_model_metrics.json`, and
  `monitoring/training_resource_usage.csv` as artifacts.
- Select all four nested runs in the UI and click **Compare** to see the
  per-metric side-by-side table the professor expects.

If you want the UI to show only one fresh invocation, use
`scripts/verify_phase2.sh --clean-mlflow`. The replay writes models, metrics,
predictions, and monitoring CSVs under ignored scratch paths, so it does not
affect DVC artifacts or Git-tracked files.

The filesystem MLflow backend (`file:./mlruns`) is acceptable for this
class/local workflow — MLflow 3 prints a deprecation note in favor of a SQLite
or PostgreSQL backend for shared use; for Phase 3 deployment we will switch the
URI accordingly without touching any of the call sites above.

## Application Logging

Application logging is configured in `configs/config.yaml` and implemented in
`src/mlops_crew/logging_config.py`.

```yaml
logging:
  level: INFO
  log_dir: logs
  log_file: pipeline.log
  max_bytes: 10485760
  backup_count: 5
  rich_tracebacks: true
```

Every pipeline entrypoint calls `setup_logging_from_config(config)`, which
attaches:

- a Rich stdout handler for readable terminal output and tracebacks
- a rotating structured file handler under `logs/pipeline.log`

The log file format is:

```text
timestamp | level | module | message
```

Actual log files are local runtime artifacts and are ignored by Git; `logs/.gitkeep`
keeps the directory visible in the repository.

## Hydra Configuration Experiments

Hydra config files live under `conf/`:

```text
conf/config.yaml
conf/experiment/phase2_default.yaml
conf/experiment/phase2_experimental.yaml
```

This `conf/` directory is intentionally not a replacement for `configs/`.
`configs/config.yaml` remains the source of truth for the normal DVC pipeline.
`conf/` is the required Hydra experiment layer from the assignment: it loads the
base config, applies only experiment overrides, and records the exact Hydra run
configuration in `outputs/hydra/.../.hydra/`.

Hydra is used for experiment sweeps through:

```bash
make hydra-train
make hydra-demo
```

`make hydra-demo` runs the same Hydra training script twice:

```bash
python -m mlops_crew.train_hydra experiment=phase2_default
python -m mlops_crew.train_hydra experiment=phase2_experimental
```

The Hydra wrapper logs both runs to the `phishing-email-phase2-hydra` MLflow
experiment. It writes models, metrics, predictions, monitoring CSVs, and the
effective config under ignored `outputs/hydra/...` paths so demo runs do not
modify DVC-tracked artifacts. Each run logs tags such as `config_source=hydra`,
`hydra_experiment`, `data_version`, and `model_version`, plus the effective YAML
config artifact.

## Model Experiments

All models use the same cleaned splits, TF-IDF settings, and evaluation metrics.
The primary selection metric is validation F2 because recall matters more than
precision for phishing detection.

| Model | Val F2 | Test F2 | Test Recall | Test FNR |
| --- | ---: | ---: | ---: | ---: |
| Dummy | 0.8449 | 0.8449 | 1.0000 | 0.0000 |
| Logistic Regression | 0.9882 | 0.9903 | 0.9918 | 0.0082 |
| Linear SVC | **0.9924** | **0.9912** | 0.9922 | 0.0078 |
| Complement NB | 0.9453 | 0.9512 | 0.9417 | 0.0583 |

Selected model: `linear_svc`

Artifacts:

- `models/best_model.joblib`
- `reports/metrics/model_comparison.csv`
- `reports/metrics/model_comparison.json`
- `reports/metrics/model_comparison.png`
- `reports/predictions/*_predictions.csv`

### Phase 1 → Phase 2 comparison

`reports/metrics/phase1_baseline/` holds an immutable snapshot of the Phase 1
metrics (committed from the `main` branch). The Phase 2 model-development
notebook joins them with the live Phase 2 metrics for a side-by-side view.

| Phase | Best model | Val F2 | Test F2 | Test Recall | Test FNR |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 (60% sample, LR only) | logistic_regression | 0.9882 | 0.9867 | 0.9881 | 0.0119 |
| 2 (80% sample, 4 models) | linear_svc | 0.9924 | 0.9912 | 0.9922 | **0.0078** |

False-negative rate on the test split drops by roughly **35 %** between Phase 1
and Phase 2. That gain comes from two changes pulling in the same direction:
the additional 20 % of training data, and the addition of `linear_svc` as a
candidate model family. Per-model deltas are in
`notebooks/phase2_model_development.ipynb` (cells 11–14).

### Notebooks (developer experimentation view)

Two Phase 2 notebooks were added, both committed with executed outputs so
graders can review without re-running:

- `notebooks/phase2_model_development.ipynb` — partition counts, per-model
  metrics, Phase 1 vs Phase 2 delta table + chart, cross-evaluation of the
  Phase 1-style LR on the Phase 2 increment, MLflow run enumeration, false
  negative spot-check.
- `notebooks/phase2_divergence_analysis.ipynb` — label / source / text-length /
  vocabulary / prediction drift, plus a per-source false-negative breakdown on
  the Phase 2 increment.

Run with `jupyter notebook notebooks/` or read inline on GitHub. Both consume
only artifacts produced by `make repro`.

## Transformer Dataset Export

Fine-tuning a transformer/LLM is not run in this phase because it would add a
larger training dependency and compute requirement. The pipeline does prepare a
versioned dataset for that work:

```bash
make transformer-data
```

Outputs:

- `data/processed/transformer/train.jsonl`
- `data/processed/transformer/val.jsonl`
- `data/processed/transformer/test.jsonl`
- `data/processed/transformer/dataset_info.json`

Each JSONL record has `text` and `label` fields and can be loaded with Hugging
Face `datasets.load_dataset("json", data_files={...})`.

Embedding-based experiments are also deferred. They make sense after the TF-IDF
baseline is locked because embeddings add external model/version management and
should be compared under the same metrics and tracking structure.

## Reproduction

Use the project virtualenv so DVC stages run with the same dependencies as CI:

```bash
python -m venv .venv
source .venv/bin/activate
make install
dvc pull
make repro
make test
make lint
mypy src/mlops_crew
scripts/verify_phase2.sh
```

> The Phase 2 DVC pipeline has been pushed to the S3 remote `storage`
> (`s3://mlops-crew-data/dvc-store`) so `dvc pull` retrieves every interim,
> processed, model, and report artifact listed in `dvc.lock`. The Google Drive
> remote `gdrive_backup` is kept as a fallback. If you are without AWS
> credentials, run `dvc pull -r gdrive_backup` instead.

Refer to this file for commands to execute: `docs/phase2_reproduction_commands.md`.
Windows-specific setup (PowerShell, Git Bash, WSL, Chocolatey, optional `make`):
`docs/windows_setup.md`. Run `scripts/verify_phase2.ps1` in PowerShell or
`scripts/verify_phase2.sh` in Bash — same checks.

The full DVC pipeline now runs:

```text
sample -> clean -> split -> validate -> transformer_dataset
                              -> train -> inference_latency
                                       -> plot_model_comparison
sample + source_manifest + train -> divergence
```

Current `dvc status` result:

```text
Data and pipelines are up to date.
```

## Phase 3 Notes

- Use the reserved 20% Phase 3 holdout only when the final model family is
  selected.
- If transformer fine-tuning is pursued, start from the JSONL export and log it
  as a separate MLflow experiment family.
- For production/shared tracking, replace the file-based MLflow backend with a
  SQLite or remote tracking backend.
