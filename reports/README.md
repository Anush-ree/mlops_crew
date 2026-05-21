# Reports Directory

Store generated analysis reports, findings, and visualizations here.

## Structure

- **`metrics/`** — DVC-tracked JSON/CSV model metrics
- **`predictions/`** — DVC-tracked row-level validation/test predictions
- **`divergence/`** — DVC-tracked Phase 1 vs Phase 2 divergence reports
- **`monitoring/`** — DVC-tracked resource and latency reports
- **`profiling/`** — local cProfile outputs and tracked text summaries

## Guidelines

- Generate reports and figures programmatically from notebooks
- Include timestamps and dataset versions in report names
- Store high-quality figures for presentations and documentation
