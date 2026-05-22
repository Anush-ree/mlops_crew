# Phase 1 Baseline Metrics Snapshot

These JSON / CSV files are an **immutable copy** of the Phase 1 metrics that were
produced on the `main` branch before Phase 2 began. They are committed here so
the Phase 2 notebooks and `PHASE2.md` can compare the new 80% Phase 2 run
against the original 60% Phase 1 run **without** needing to check out the
`main` branch.

## Provenance

- Source branch: `main` at the time of the Phase 2 fork.
- Producing run: `dvc repro` against `configs/config.yaml` with
  `data.sample.fraction = 0.60` and `modeling.models = [dummy, logistic_regression]`.

## What is in this directory

| File | Meaning |
| --- | --- |
| `best_model_metrics.json` | Phase 1 winner (`logistic_regression`) full metrics |
| `logistic_regression_metrics.json` | Phase 1 LR val + test metrics |
| `dummy_metrics.json` | Phase 1 dummy baseline |
| `model_comparison.json` | Phase 1 per-model comparison (verbose) |
| `model_comparison.csv` | Phase 1 per-model comparison (flat) |

## How it is used

- `notebooks/phase2_model_development.ipynb` loads `model_comparison.csv` and
  the Phase 2 `reports/metrics/model_comparison.csv` to produce the
  side-by-side comparison table.
- `PHASE2.md` cites the deltas directly from these files so the claim is
  reproducible.

Do not regenerate these files inside the Phase 2 pipeline. They are a
historical snapshot, not a live artifact.
