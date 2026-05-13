# Configs Directory

`config.yaml` is the single source of truth for the Phase 1 pipeline.

It controls raw/interim/processed paths, the 60% Phase 1 sample fraction,
train/validation/test split ratios, TF-IDF settings, model hyperparameters, and
artifact locations. After changing it, run `make repro` so DVC rebuilds the
affected stages.
