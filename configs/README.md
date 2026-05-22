# Configs Directory

`config.yaml` is the single source of truth for the normal Phase 2 DVC pipeline.

It controls raw/interim/processed paths, the 60/20/20 phase partitioning,
train/validation/test split ratios, TF-IDF settings, model hyperparameters,
MLflow tracking, application logging, monitoring outputs, and artifact
locations. After changing it, run `make repro` so DVC rebuilds the affected
stages.

Hydra experiment configs live under `../conf/`. They overlay selected values on
top of this base config for MLflow comparison runs without changing the
DVC-tracked production artifacts.
