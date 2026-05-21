# Configs Directory

`config.yaml` is the single source of truth for the Phase 2 pipeline.

It controls raw/interim/processed paths, the 60/20/20 phase partitioning,
train/validation/test split ratios, TF-IDF settings, model hyperparameters,
MLflow tracking, monitoring outputs, and artifact locations. After changing it,
run `make repro` so DVC rebuilds the affected stages.
