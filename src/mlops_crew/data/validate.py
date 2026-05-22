"""Stage 4: sanity-check the cleaned and split CSVs before training.

Failures here mean a bad pipeline run, not a flaky model — so this returns a
non-zero exit code so CI or `make` can stop early.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from mlops_crew.config import CONFIG_PATH, PROJECT_ROOT, load_project_config, resolve_project_path
from mlops_crew.data import LABEL_COLUMN, TEXT_COLUMN
from mlops_crew.logging_config import get_logger, setup_logging_from_config
from mlops_crew.utils.io import save_json

logger = get_logger(__name__)

EXPECTED_COLUMNS = {TEXT_COLUMN, LABEL_COLUMN}
VALID_LABELS = {0, 1}
HIGH_IMBALANCE_RATIO = 10.0
VALIDATION_REPORT_FILE = "validation_report.json"


def dataset_paths(config: dict[str, Any]) -> dict[str, Path]:
    """Resolve cleaned and split CSV paths checked by validation."""
    data_config = config["data"]
    processed_dir = resolve_project_path(data_config["processed_dir"])
    return {
        "cleaned": processed_dir / data_config["cleaned_file"],
        "train": processed_dir / data_config["train_file"],
        "val": processed_dir / data_config["val_file"],
        "test": processed_dir / data_config["test_file"],
    }


def validate_dataset(path: Path, name: str, min_text_length: int) -> bool:
    """Return True when the CSV at `path` passes every check."""
    if not path.exists():
        logger.error("[%s] file not found: %s", name, path)
        return False

    data = pd.read_csv(path)
    logger.info("[%s] shape=%s", name, data.shape)
    if data.empty:
        logger.error("[%s] file is empty", name)
        return False

    missing = EXPECTED_COLUMNS - set(data.columns)
    if missing:
        logger.error("[%s] missing columns: %s", name, sorted(missing))
        return False

    nulls = data[list(EXPECTED_COLUMNS)].isnull().sum()
    if nulls.any():
        logger.error("[%s] null counts: %s", name, nulls.to_dict())
        return False

    invalid_labels = set(data[LABEL_COLUMN].unique()) - VALID_LABELS
    if invalid_labels:
        logger.error("[%s] invalid labels: %s", name, sorted(invalid_labels))
        return False
    missing_labels = VALID_LABELS - set(data[LABEL_COLUMN].unique())
    if missing_labels:
        logger.error("[%s] missing expected labels: %s", name, sorted(missing_labels))
        return False

    short_count = int((data[TEXT_COLUMN].str.len() < min_text_length).sum())
    if short_count:
        logger.error("[%s] %d rows shorter than %d chars", name, short_count, min_text_length)
        return False

    counts = data[LABEL_COLUMN].value_counts()
    ratio = float(counts.max() / counts.min())
    if ratio > HIGH_IMBALANCE_RATIO:
        logger.warning("[%s] high class imbalance: %.2f", name, ratio)
    logger.info("[%s] label_distribution=%s", name, counts.to_dict())
    return True


def validation_report_path(config: dict[str, Any]) -> Path:
    """Return the DVC-tracked validation report path under ``data/processed/``."""
    processed_dir = resolve_project_path(config["data"]["processed_dir"])
    return processed_dir / VALIDATION_REPORT_FILE


def _build_validation_report(config: dict[str, Any]) -> dict[str, Any]:
    """Summarize validated datasets for the DVC stage output artifact."""
    paths = dataset_paths(config)
    datasets: dict[str, Any] = {}
    for name, path in paths.items():
        frame = pd.read_csv(path)
        counts = frame[LABEL_COLUMN].value_counts().to_dict()
        try:
            relative_path = str(path.relative_to(PROJECT_ROOT))
        except ValueError:
            relative_path = str(path)
        datasets[name] = {
            "path": relative_path,
            "rows": int(len(frame)),
            "label_distribution": {str(label): int(count) for label, count in counts.items()},
        }
    return {
        "status": "passed",
        "min_text_length": int(config.get("cleaning", {}).get("min_text_length", 3)),
        "datasets": datasets,
    }


def run(config: dict[str, Any]) -> bool:
    """Validate splits; on success write ``validation_report.json`` for DVC."""
    min_length = int(config.get("cleaning", {}).get("min_text_length", 3))
    paths = dataset_paths(config)
    passed = all(validate_dataset(path, name, min_length) for name, path in paths.items())
    if passed:
        save_json(_build_validation_report(config), validation_report_path(config))
        logger.info("Wrote validation report to %s", validation_report_path(config))
    return passed


def main() -> None:
    """CLI entrypoint for the DVC ``validate`` stage."""
    parser = argparse.ArgumentParser(description="Validate cleaned and split datasets")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    config = load_project_config(args.config)
    setup_logging_from_config(config)
    if not run(config):
        logger.error("Validation failed")
        sys.exit(1)
    logger.info("All validation checks passed")


if __name__ == "__main__":
    main()
