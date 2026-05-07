"""Stage 1: take a stratified sample of the raw CSV.

We train Phase 1 on a fraction of `phishing_email.csv` so iteration is fast and
reproducible. Sampling is stratified on the label so class balance is preserved.
The output is a separate DVC artifact, which means every downstream stage runs
on the exact same rows.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from mlops_crew.config import CONFIG_PATH, load_project_config, resolve_project_path
from mlops_crew.data import LABEL_COLUMN
from mlops_crew.logging_config import get_logger, setup_logging
from mlops_crew.utils.io import save_json

logger = get_logger(__name__)


def sample_paths(config: dict[str, Any]) -> dict[str, Path]:
    data_config = config["data"]
    raw_path = resolve_project_path(
        data_config["raw_dir"]) / data_config["raw_file"]
    interim_dir = resolve_project_path(data_config["interim_dir"])
    return {
        "raw": raw_path,
        "sampled": interim_dir / data_config["sampled_file"],
        "summary": interim_dir / "sample_summary.json",
    }


def sample_dataset(data: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Return a stratified sample of `data` according to `config["data"]["sample"]`."""
    sample_config = config["data"]["sample"]
    fraction = float(sample_config["fraction"])
    if not 0 < fraction <= 1:
        raise ValueError(f"Sample fraction must be in (0, 1], got {fraction}")

    random_state = int(sample_config["random_state"])
    label_column = config["data"].get("label_column", LABEL_COLUMN)

    if sample_config.get("stratify", True):
        if label_column not in data.columns:
            raise ValueError(
                f"Cannot stratify; missing label column `{label_column}`")
        parts = [
            group.sample(frac=fraction, random_state=random_state)
            for _, group in data.groupby(label_column)
        ]
        sampled = pd.concat(parts, axis=0)
    else:
        sampled = data.sample(frac=fraction, random_state=random_state)

    return sampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def run(config: dict[str, Any]) -> pd.DataFrame:
    paths = sample_paths(config)
    if not paths["raw"].exists():
        raise FileNotFoundError(
            f"Raw data not found at {paths['raw']}. Run `dvc pull data/raw/archive.dvc`."
        )

    logger.info("Loading raw data from %s", paths["raw"])
    raw = pd.read_csv(paths["raw"])
    logger.info("Raw shape: %s", raw.shape)

    sampled = sample_dataset(raw, config)
    paths["sampled"].parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(paths["sampled"], index=False)
    logger.info("Saved sample (%s) to %s", sampled.shape, paths["sampled"])

    label_column = config["data"].get("label_column", LABEL_COLUMN)
    summary = {
        "raw_rows": int(len(raw)),
        "sampled_rows": int(len(sampled)),
        "sample_fraction": float(config["data"]["sample"]["fraction"]),
        "label_distribution": {
            str(label): int(count) for label, count in sampled[label_column].value_counts().items()
        },
    }
    save_json(summary, paths["summary"])
    return sampled


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample the raw phishing email CSV")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    setup_logging()
    run(load_project_config(args.config))


if __name__ == "__main__":
    main()
