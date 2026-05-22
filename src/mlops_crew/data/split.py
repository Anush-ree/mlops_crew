"""Stage 3: split cleaned data into train, validation, and test CSVs.

Stratification on the label keeps class balance consistent across splits, and
the random seed makes the split reproducible across runs and machines.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from mlops_crew.config import CONFIG_PATH, load_project_config, resolve_project_path
from mlops_crew.data import LABEL_COLUMN
from mlops_crew.logging_config import get_logger, setup_logging_from_config
from mlops_crew.utils.io import save_json

logger = get_logger(__name__)


def split_paths(config: dict[str, Any]) -> dict[str, Path]:
    """Resolve cleaned input and train/val/test output paths."""
    data_config = config["data"]
    processed_dir = resolve_project_path(data_config["processed_dir"])
    return {
        "cleaned": processed_dir / data_config["cleaned_file"],
        "train": processed_dir / data_config["train_file"],
        "val": processed_dir / data_config["val_file"],
        "test": processed_dir / data_config["test_file"],
        "summary": processed_dir / "split_summary.json",
    }


def split_data(
    data: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train / val / test using the proportions in config."""
    cfg = config["data"]["split"]
    train_size = float(cfg["train_size"])
    val_size = float(cfg["val_size"])
    test_size = float(cfg["test_size"])
    if abs(train_size + val_size + test_size - 1.0) > 1e-8:
        raise ValueError("Split proportions must sum to 1.0")
    if min(train_size, val_size, test_size) <= 0:
        raise ValueError("Split proportions must all be positive")

    random_state = int(cfg["random_state"])
    stratify = cfg.get("stratify", True)
    holdout_size = val_size + test_size

    train, holdout = train_test_split(
        data,
        test_size=holdout_size,
        random_state=random_state,
        stratify=data[LABEL_COLUMN] if stratify else None,
    )
    val, test = train_test_split(
        holdout,
        test_size=test_size / holdout_size,
        random_state=random_state,
        stratify=holdout[LABEL_COLUMN] if stratify else None,
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def run(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split cleaned data into train, validation, and test CSVs."""
    paths = split_paths(config)
    if not paths["cleaned"].exists():
        raise FileNotFoundError(f"Cleaned data not found at {paths['cleaned']}. Run `make data`.")

    logger.info("Loading cleaned data from %s", paths["cleaned"])
    cleaned = pd.read_csv(paths["cleaned"])
    train, val, test = split_data(cleaned, config)

    paths["train"].parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(paths["train"], index=False)
    val.to_csv(paths["val"], index=False)
    test.to_csv(paths["test"], index=False)
    logger.info("Split sizes: train=%d val=%d test=%d", len(train), len(val), len(test))

    summary = {
        name: {
            "rows": int(len(frame)),
            "label_distribution": {
                str(label): int(count)
                for label, count in frame[LABEL_COLUMN].value_counts().items()
            },
        }
        for name, frame in {"train": train, "val": val, "test": test}.items()
    }
    save_json(summary, paths["summary"])
    return train, val, test


def main() -> None:
    """CLI entrypoint for the DVC ``split`` stage."""
    parser = argparse.ArgumentParser(description="Create train/val/test splits")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    config = load_project_config(args.config)
    setup_logging_from_config(config)
    run(config)


if __name__ == "__main__":
    main()
