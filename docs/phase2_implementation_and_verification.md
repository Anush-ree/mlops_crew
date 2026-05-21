# Phase 2 (Sections 2–4) — Implementation and Verification Guide

Branch: `model_v2`  
Scope: Sections 2 (monitoring & debugging), 3 (profiling), 4 (experiment tracking)  
Out of scope on this branch: Sections 1 (Docker), 5 (logging/rich), 6 (Hydra) — owned by other teammates on their branches.

This document has two halves:

- **Part A — Implementation overview.** What was added, where it lives, how it is wired, and why each choice was made. Specifically the model-development story, the data versioning state (local + remote), and the experiment-tracking setup.
- **Part B — Verification script.** The exact commands a grader runs to reproduce the work, with the expected output of each step. Designed so a Phase-1-style failure ("the metric claimed in PHASE2.md cannot be reproduced from anything in the repo") cannot recur.

The Phase 1 professor feedback (`phase1_professor_comments.txt`) is the rubric:

> "Nothing in the repo could have produced that number."  
> "Commit the actual notebook or script that trained the LR/Dummy baselines with metrics output and a saved joblib in models/."  
> "Tick PHASE1.md checkboxes only after the corresponding artifact is committed."

Everything below is engineered against that lesson — every number we claim is reproducible from a committed file, a committed notebook output, or a one-command pipeline.

---

## Part A — Implementation overview

### A.1 Sections owned by this branch

| Section | Status | Where to look |
| --- | --- | --- |
| 2.1 Monitoring | done | `src/mlops_crew/monitoring/resource_monitor.py`, `src/mlops_crew/monitoring/inference_latency.py`, `src/mlops_crew/monitoring/divergence.py`, `reports/monitoring/*.csv`, `reports/divergence/*` |
| 2.2 Debugging | covered in PHASE2.md "Debugging entry points" + this doc | `PHASE2.md`, this file (§A.4) |
| 3.1 cProfile | done | `scripts/profile_train.py`, `scripts/profile_predict.py`, `reports/profiling/*_cprofile.txt` |
| 3.2 Optimization decision | documented (no aggressive opt) | `PHASE2.md` §Profiling and Optimization |
| 4 Experiment tracking | done | `src/mlops_crew/tracking/mlflow_tracking.py`, `src/mlops_crew/train_model.py:154,161,165`, `mlruns/` (regenerated per machine by training or scratch replay) |

### A.2 Data versioning — local and remote

The grader needs to reproduce the same data the team trained on. That requires both the local pipeline definitions (Git) and the actual data bytes (DVC + S3 / Google Drive).

**What Git versions** (`git ls-tree -r model_v2`):

- All source code under `src/`, `scripts/`, `tests/`, `notebooks/`.
- The single config: `configs/config.yaml`.
- Pipeline definitions: `dvc.yaml`, `dvc.lock`, `Makefile`.
- The raw-data pointer: `data/raw/archive.dvc`.
- Lightweight metric JSONs: `reports/metrics/*.json`, `model_comparison.csv` (DVC `metrics: cache: false`).
- The Phase 1 baseline metric snapshot: `reports/metrics/phase1_baseline/`.

**What DVC versions** (`dvc list -R --dvc-only .`):

- `data/raw/archive/` — the 7 raw source CSVs, 261 MB total, hashed in `data/raw/archive.dvc`.
- Every stage output declared in `dvc.yaml`: phase partitions, source manifest, cleaned + split CSVs, transformer JSONLs, model joblibs, prediction CSVs, monitoring CSVs, divergence report.

**Where the bytes live** (`dvc remote list`):

```
storage         s3://mlops-crew-data/dvc-store   (default)
gdrive_backup   gdrive://1Ai0Mp59YmxdesufVcR0mZEHb1bDFOib9
```

**Current sync state** (verified with `dvc status -c`): the local pipeline outputs are committed in `dvc.lock` and pushed to the `storage` remote. A fresh `dvc pull` brings everything down. The `gdrive_backup` mirror exists in case the grader has no AWS credentials.

**What is gitignored** (`reports/.gitignore`, `models/.gitignore`, `data/interim/.gitignore`, `data/processed/.gitignore`, root `.gitignore`): every byte that DVC owns, plus `mlruns/` and `logs/`. Nothing important is lost — `dvc pull` rehydrates it all.

**Transformer dataset** (`make transformer-data` / the `transformer_dataset` DVC stage): writes one JSONL file per split under `data/processed/transformer/` plus a `dataset_info.json`. Row counts are committed to the DVC lock so any drift is caught:

| Split | Rows | Label 0 | Label 1 |
| --- | ---: | ---: | ---: |
| train | 46,020 | 22,022 | 23,998 |
| val | 9,861 | 4,719 | 5,142 |
| test | 9,862 | 4,719 | 5,143 |

The export keeps only `text` and `label` so it loads cleanly with `datasets.load_dataset("json", data_files={...})` — the Hugging Face contract.

### A.3 The 60 / 20 / 20 partition (Phase 1 vs Phase 2 vs Phase 3)

The full raw file is 82,486 rows. `src/mlops_crew/data/sample.py:partition_phase_data` produces four artifacts deterministically and stratified by label:

| Partition | Rows | Purpose | File |
| --- | ---: | --- | --- |
| `phase1_reference` | 49,492 | Reproduces the Phase 1 60 % slice | `data/interim/phishing_email_phase1_reference.csv` |
| `phase2_increment` | 16,497 | The extra 20 % added in Phase 2 | `data/interim/phishing_email_phase2_increment.csv` |
| `phase2_sample` | 65,989 | What Phase 2 trains on (= phase1 ∪ increment) | `data/interim/phishing_email_phase2_sample.csv` |
| `phase3_holdout` | 16,497 | Reserved 20 % for Phase 3 | `data/interim/phishing_email_phase3_holdout.csv` |

A test in `tests/test_phase1_pipeline.py` (`test_partition_phase_data_creates_non_overlapping_phase_slices`) locks in the non-overlap invariant.

### A.4 Model development story

Phase 2 expands the model family from {dummy, logistic_regression} to {dummy, logistic_regression, linear_svc, complement_nb}, and trains all four on the Phase 2 80 % sample.

**Committed metrics** (`reports/metrics/`):

| Model | Val F2 | Test F2 | Test Recall | Test FNR |
| --- | ---: | ---: | ---: | ---: |
| dummy | 0.8449 | 0.8449 | 1.0000 | 0.0000 |
| logistic_regression | 0.9882 | 0.9903 | 0.9918 | 0.0082 |
| **linear_svc** | **0.9924** | **0.9912** | 0.9922 | **0.0078** |
| complement_nb | 0.9453 | 0.9512 | 0.9417 | 0.0583 |

**Phase 1 → Phase 2 head-to-head** (Phase 1 numbers from `reports/metrics/phase1_baseline/`, snapshotted from `main`):

| Phase | Best | Test F2 | Test Recall | Test FNR |
| --- | --- | ---: | ---: | ---: |
| 1 (60 %, LR only) | logistic_regression | 0.9867 | 0.9881 | 0.0119 |
| 2 (80 %, 4 models) | linear_svc | 0.9912 | 0.9922 | **0.0078** |

Test FNR drops by **~35 %**. F2 lifts by ~0.45 pp. Both deltas are real and reproducible from committed CSVs.

**Notebooks** that exercise the full developer experimentation loop:

- `notebooks/phase2_model_development.ipynb` — partitions, per-model metrics, Phase 1 vs Phase 2 delta table + chart, cross-evaluation of a Phase-1-style LR on the Phase 2 increment, MLflow run enumeration, false-negative deep dive. **All 11 code cells executed; outputs committed.**
- `notebooks/phase2_divergence_analysis.ipynb` — label / source / text-length / vocabulary / prediction drift between Phase 1 reference and Phase 2 increment, per-source false-negative breakdown. **All 10 code cells executed; outputs committed.**

Both notebooks read pre-computed pipeline artifacts only — they do not re-run training. The professor's complaint last phase was "nothing in the repo could have produced that number"; here every metric in the notebooks resolves to either `reports/metrics/*.json` or a one-line recomputation from `data/processed/*.csv` plus `models/best_model.joblib`.

### A.5 Monitoring (Section 2.1)

Three artifacts feed Section 2.1:

1. **Resource monitor.** `ResourceMonitor` (background thread, psutil, 1 Hz) wraps the entire training run. Output: `reports/monitoring/training_resource_usage.csv`. Columns: timestamp, process_cpu_percent, system_cpu_percent, rss_mb, available_memory_mb. Current runs produce roughly 70–120 samples depending on local scheduling and MLflow serialization time.

2. **Inference latency.** `inference_latency.py` measures batch sizes [1, 32, 256, 1024] × 3 repeats on `models/best_model.joblib`. Output: `reports/monitoring/inference_latency.csv` with `milliseconds_per_record` and `records_per_second`. Current numbers: batch=1024 at roughly 8k rec/s (~0.12 ms/record) on the local machine used for this run.

3. **Divergence.** `divergence.py` compares Phase 1 reference vs Phase 2 increment along five axes — label JS, source JS (joined on the manifest), text-length KS, vocabulary new-token rate, prediction JS. Outputs: `reports/divergence/phase2_divergence_report.json` (machine-readable) and `phase2_divergence_summary.md` (human-readable). Current numbers — label JS=0.000007, source JS=0.014327, KS=0.005134, new-token rate=0.039855, prediction JS=0.000207 — meaning the Phase 2 increment is **not** out-of-distribution.

### A.6 Debugging (Section 2.2)

Four worked scenarios — actual commands you can run today:

1. **Missing DVC data.**  
   Symptom: `FileNotFoundError: data/raw/archive/phishing_email.csv` when `make data` starts.  
   Diagnose: `python -m pdb -c "b src/mlops_crew/data/sample.py:134" -c c -m mlops_crew.data.sample` then `p paths["raw"]` shows the resolved path. The fix is `dvc pull` (or `dvc pull -r gdrive_backup`).

2. **One-class split.**  
   Symptom: `validate.run(config)` returns False with `[train] missing expected labels: [0]`.  
   Diagnose: open the offending split CSV, `df["label"].value_counts(dropna=False)`. Root cause is usually a too-small `sample.fraction` combined with `stratify=False` — config should always keep `stratify: true`.

3. **Label normalization mismatch.**  
   Symptom: cleaning step drops most rows. Reason: source labels were strings (`"spam"`/`"ham"`) but `_normalize_label` only accepts numeric `{0, 1}`. Quick check: `breakpoint()` inside `clean_dataset` and `print(cleaned[LABEL_COLUMN].isna().sum())`. For the current dataset all sources are already numeric; the `source_manifest` stage asserts this so a string-labeled source would fail fast.

4. **F2 regression after adding Phase 2 data.**  
   Diagnose flow: `notebooks/phase2_model_development.ipynb` prints the Phase 1 to Phase 2 delta table per model. If the delta is negative for the winning model, cross-check `phase2_divergence_analysis.ipynb` for a vocabulary jump or source-mix shift. The false-negatives-by-source output tells you whether one source is dragging recall down.

### A.7 Profiling (Section 3)

`scripts/profile_train.py` runs the training path inside `cProfile.Profile`. `scripts/profile_predict.py` does the same for `monitoring.inference_latency.run`. Both write a `.prof` (machine-readable, gitignored) plus a `*_cprofile.txt` (committed). By default they write temporary models/metrics/predictions/latency CSVs under `reports/profiling/scratch/` and disable MLflow tracking, so profiling does not create duplicate runs or dirty DVC outputs. Use `python scripts/profile_train.py --with-tracking` only when profiling MLflow overhead explicitly. The committed text files show:

- **Training:** ~147 s under cProfile on the local machine used for this run. Top of the cumulative-time table: `train` → `_train_one` → `pipeline.fit` → `TfidfVectorizer.fit_transform`. TF-IDF dominates, as expected for sklearn text classifiers. Normal DVC training can be faster because cProfile adds overhead.
- **Inference:** ~2.6 s for the full latency sweep. The top cost is `joblib.load` (one-time pickle deserialization) — `predict` itself is fast.

**Optimization decision documented** in PHASE2.md: keep TF-IDF inside the sklearn `Pipeline` rather than precomputing feature pickles. The professor's PDF specifically asks for the decision-and-reason, not optimization for its own sake.

### A.8 Experiment tracking (Section 4)

The MLflow wiring is small but covers every line item the PDF lists:

| PDF requirement | Where it is satisfied |
| --- | --- |
| Tool integrated | `src/mlops_crew/tracking/mlflow_tracking.py` |
| Tracking URI | `tracking.tracking_uri: file:./mlruns` in `configs/config.yaml` |
| Experiment name | `tracking.experiment_name: phishing-email-phase2` |
| Hyperparameters logged | `model_run()` logs `model.*` + `tfidf.*` per nested run |
| Metrics logged | `log_metrics()` writes `validation_*` + `test_*` to each nested run |
| Trained model artifacts | `mlflow.sklearn.log_model(...)` + joblib uploaded to `joblib/` |
| Useful artifacts | `config/config.yaml`, optional `predictions/*.csv`, `metrics/model_comparison.csv`, `monitoring/training_resource_usage.csv` |
| Exact training code line numbers | `src/mlops_crew/train_model.py:154` (parent run), `:161` (nested run), `:165-167` (model artifact log), `:174-175` (resource csv), `:205-209` (comparison artifacts) |
| Compare runs in UI | parent run + 4 nested runs; select-all-and-Compare in the MLflow UI |
| How to access | `make mlflow-ui` (port 5001 — macOS AirPlay uses 5000) |

The `mlruns/` directory is gitignored. A fresh training run populates it locally, but if `dvc pull` already restored every DVC output, `dvc repro` can correctly skip training and therefore not create new MLflow runs. Use `scripts/verify_phase2.sh --replay-mlflow` to populate the UI with scratch outputs without dirtying DVC artifacts. Class 7 mentioned committing the directory, but the PDF rubric accepts "screenshot or link" instead, and a clean local replay is a more honest reproducibility signal than committing one person's local store.

### A.9 What was checked against `main` and pre-PR-#5

- `comm -23 main model_v2` (committed-file diff) returns **empty** — no Git-tracked file from `main` was dropped on `model_v2`. The branch is a strict superset.
- Pre-PR-#5 main (commit `1ddb060`) was the cookiecutter scaffold that the professor tested for Phase 1. Its `train_model.py` was a stub, `models/model.py` raised `NotImplementedError`, and `notebooks/` only had `.gitkeep`. PR #5 (the `model` branch) replaced the scaffold with the real `Pipeline(tfidf -> classifier)` training code, the metrics module, the validators, and the LR/Dummy training. `model_v2` builds Sections 2–4 directly on top of that real foundation; the Phase 1 critique cannot apply to the present branch because every metric is now backed by a committed `*_metrics.json`, a notebook with executed outputs, and a saved `*.joblib`.

### A.10 Known gaps (out of scope on this branch)

These are deliberately not addressed here because the team agreed they go on other branches:

- **No `Dockerfile`** — Section 1 of the PDF (Containerization).
- **No `conf/` Hydra tree** — Section 6 (Configuration Management).
- **`logging_config.py` is plain stdlib only, no Rich, no rotating file** — Section 5 (Application logging). `rich` is also missing from `requirements.txt`.

The team must merge those three branches into `main` before tagging the Phase 2 submission. Until then, `model_v2` is a "graded Sections 2–4" branch, not a full Phase 2 submission on its own.

---

## Part B — Reproduction script (what the professor will run)

A grader gets the repo URL and a Phase 2 deadline. This is the exact sequence they will execute. For each step the expected output is shown. Anything else is a regression.

### B.0 Environment setup

```bash
git clone https://github.com/Anush-ree/mlops_crew.git
cd mlops_crew
git checkout model_v2

python3.11 -m venv .venv
source .venv/bin/activate
make install            # runtime deps + editable install of mlops_crew
make dev                # adds pytest, ruff, mypy, pre-commit
```

Expected: `make install` completes; `pip list` shows `mlops-crew`, `mlflow`, `dvc`, `dvc-s3`, `psutil`, `scipy`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `pyyaml`, `joblib`.

### B.1 Pull data + pipeline outputs from the remote

```bash
pip install dvc-s3      # required for the s3:// remote
aws configure           # region: us-east-2
dvc pull                # uses the default `storage` remote
# Fallback if no AWS creds:
# dvc pull -r gdrive_backup
```

Expected: `dvc pull` reports rehydrating the raw archive plus interim, processed, model, and report outputs. After it finishes, `dvc status -c` should report that the cache and default remote are in sync.

### B.1a One-command verification shortcut

```bash
scripts/verify_phase2.sh
```

Expected: runs `dvc status`, `dvc repro`, lint, mypy, pytest with coverage,
inference profiling, and a final `dvc status`. Use
`scripts/verify_phase2.sh --include-slow-profile` to also regenerate the
training profile, `--replay-mlflow` to populate local MLflow from scratch
outputs when DVC skips training, `--clean-mlflow` to delete local runs before
that replay, and `--check-remote` to include `dvc status -c`.

### B.2 Static checks (≈ 30 seconds total)

```bash
ruff check .
ruff format --check .
mypy src/mlops_crew
make test
```

Expected:

```
ruff check .                       -> All checks passed!
ruff format --check .              -> (no diff)
mypy src/mlops_crew                -> Success: no issues found
pytest                             -> 12 passed
```

### B.3 DVC pipeline integrity

```bash
dvc status
dvc dag
```

Expected:

- `dvc status` → `Data and pipelines are up to date.`
- `dvc dag` shows: `sample → clean → split → {transformer_dataset, train}`, `train → {inference_latency, plot_model_comparison}`, and `sample + source_manifest + train → divergence`.

### B.4 Full pipeline reproduction (≈ 2–4 min on a laptop)

```bash
make repro
```

Expected: 8 stages run (or skip if cached). After completion these files exist and are non-empty:

- `data/interim/{phishing_email_phase1_reference.csv, phishing_email_phase2_increment.csv, phishing_email_phase2_sample.csv, phishing_email_phase3_holdout.csv, source_manifest.csv, sample_summary.json, source_manifest_summary.json}`
- `data/processed/{cleaned.csv, train.csv, val.csv, test.csv, cleaning_summary.json, split_summary.json}`
- `data/processed/transformer/{train.jsonl, val.jsonl, test.jsonl, dataset_info.json}`
- `models/{dummy, logistic_regression, linear_svc, complement_nb, best_model}.joblib`
- `reports/metrics/*_metrics.json`, `best_model_metrics.json`, `model_comparison.{csv,json,png}`
- `reports/predictions/*_val_predictions.csv`, `*_test_predictions.csv`
- `reports/monitoring/{training_resource_usage.csv, inference_latency.csv}`
- `reports/divergence/{phase2_divergence_report.json, phase2_divergence_summary.md}`
- `mlruns/<exp_id>/<run_id>/...` populated when the train stage executes; if DVC skips training from pulled artifacts, run `scripts/verify_phase2.sh --replay-mlflow`

### B.5 Best-model metrics check

```bash
python - <<'PY'
import json
d = json.load(open("reports/metrics/best_model_metrics.json"))
print("model        :", d["model_name"])
print("val F2       :", round(d["validation"]["f2"], 4))
print("test F2      :", round(d["test"]["f2"], 4))
print("test recall  :", round(d["test"]["recall"], 4))
print("test FNR     :", round(d["test"]["false_negative_rate"], 4))
PY
```

Expected:

```
model        : linear_svc
val F2       : 0.9924
test F2      : 0.9912
test recall  : 0.9922
test FNR     : 0.0078
```

### B.6 Phase 1 vs Phase 2 comparison

```bash
python - <<'PY'
import pandas as pd
p1 = pd.read_csv("reports/metrics/phase1_baseline/model_comparison.csv").set_index("model_name")
p2 = pd.read_csv("reports/metrics/model_comparison.csv").set_index("model_name")
common = p1.index.intersection(p2.index)
cols = ["val_f2", "test_f2", "test_recall", "test_false_negative_rate"]
print("Phase 1 (60%):"); print(p1.loc[common, cols].round(4))
print()
print("Phase 2 (80%):"); print(p2.loc[common, cols].round(4))
print()
print("Delta (Phase 2 - Phase 1):"); print((p2.loc[common, cols] - p1.loc[common, cols]).round(4))
PY
```

Expected: dummy stays at F2 ≈ 0.845 (unchanged baseline). logistic_regression test F2 lifts by ~0.0036; test FNR drops by ~0.004. This is the lift from the additional 20 % of training data with the same model family.

### B.7 Divergence summary

```bash
cat reports/divergence/phase2_divergence_summary.md
```

Expected:

```
- Phase 1 reference rows: 49492
- Phase 2 increment rows: 16497
- Label JS distance: 0.000007
- Text length KS statistic: 0.005134
- New-token rate in Phase 2 increment: 0.039855
- Prediction JS distance: 0.000207
- (per-source counts)
```

Interpretation: Phase 2 increment is in-distribution; observed model lift is from more data + new model family, not from a data-regime shift.

### B.8 Inference latency

```bash
cat reports/monitoring/inference_latency.csv | column -ts ','
```

Expected: rows for batch_size ∈ {1, 32, 256, 1024} × repeat ∈ {1, 2, 3}. Batch=1024 is roughly 8k records/sec on the local machine used for the committed report; exact values vary by CPU.

### B.9 Resource monitoring

```bash
head -3 reports/monitoring/training_resource_usage.csv
wc -l reports/monitoring/training_resource_usage.csv
```

Expected: header is `timestamp,process_cpu_percent,system_cpu_percent,rss_mb,available_memory_mb`. Row count roughly tracks training duration; current local runs are around 70–120 rows depending on scheduling and MLflow logging time.

### B.10 Profiling artifacts

```bash
head -25 reports/profiling/train_model_cprofile.txt
head -25 reports/profiling/predict_model_cprofile.txt
```

Expected: training cProfile shows `train` → `_train_one` → `pipeline.fit` → `text.py:fit_transform` near the top. Inference cProfile shows `joblib.load` + `predict` dominating.

### B.11 MLflow UI

```bash
make mlflow-ui          # http://localhost:5001
```

Expected: experiment `phishing-email-phase2`. If no run appears because DVC skipped training from pulled artifacts, run `scripts/verify_phase2.sh --replay-mlflow` and refresh the UI. Click into a `phase2-training` run → it shows 4 nested runs (`dummy`, `logistic_regression`, `linear_svc`, `complement_nb`), each with params (`model.*`, `tfidf.*`), metrics (`validation_f2`, `test_f2`, `validation_recall`, …), artifacts (`joblib/<model>.joblib`, `predictions/*.csv`, `config/config.yaml`), and the sklearn model flavor. Parent run additionally carries `metrics/model_comparison.csv` and `monitoring/training_resource_usage.csv`. Select the 4 nested runs and click **Compare** for the side-by-side metrics table.

### B.12 Notebooks (developer experimentation evidence)

```bash
jupyter notebook notebooks/
```

Open and re-run if desired:

- `notebooks/phase2_model_development.ipynb` — 11 code cells. It prints the Phase 1 → Phase 2 delta table, cross-evaluates a Phase-1-style LR on the Phase 2 increment, summarizes the latest completed MLflow training invocation, and spot-checks false negatives.
- `notebooks/phase2_divergence_analysis.ipynb` — 10 code cells. It prints the drift report and the per-source false-negative breakdown on the Phase 2 increment.

Both notebooks were committed with executed outputs, so the grader can read them without a kernel.

### B.13 Debugging spot-check (Section 2.2)

```bash
python -m pdb -m mlops_crew.train_model
(Pdb) b src/mlops_crew/train_model.py:142
(Pdb) c
(Pdb) p config["modeling"]["models"]
(Pdb) p [f.shape for f in frames.values()]
(Pdb) c
```

Expected: breakpoint hits, prints `['dummy', 'logistic_regression', 'linear_svc', 'complement_nb']` and shape list `[(46020, 2), (9861, 2), (9862, 2)]`, then continues to completion.

### B.14 What "pass" looks like

The grader sees, in order:

- `dvc pull` succeeds.
- `ruff`, `mypy`, `pytest` all green.
- `dvc status` clean.
- `make repro` finishes in a few minutes.
- `best_model_metrics.json` matches the table in `PHASE2.md` (linear_svc, val F2 ≈ 0.992).
- Phase 1 → Phase 2 comparison shows the FNR drop.
- `mlflow ui` shows the experiment with the parent + 4 nested runs.
- Both Phase 2 notebooks render with the committed outputs.
- The divergence and inference-latency CSVs exist with the expected shapes.

The Phase 1 critique ("nothing in the repo could have produced that number") cannot recur — every metric in `PHASE2.md`, the notebooks, and this document is backed by a JSON / CSV file on disk and a one-command path that regenerates it.

---

## Appendix — Quick reference of commands

| Goal | Command |
| --- | --- |
| Install runtime + dev deps | `make install && make dev` |
| Pull DVC outputs from S3 | `dvc pull` (or `dvc pull -r gdrive_backup`) |
| Reproduce full pipeline | `make repro` |
| Re-train only | `make train` |
| Re-run divergence only | `make divergence` |
| Re-run latency benchmark | `make latency` |
| Profile training | `make profile-train` |
| Profile inference | `make profile-predict` |
| Full local verification | `scripts/verify_phase2.sh` |
| Open MLflow UI | `make mlflow-ui` (port 5001) |
| Run tests | `make test` |
| Lint | `make lint` |
| Pipeline graph | `dvc dag` |
| Sync state vs remote | `dvc status -c` |
