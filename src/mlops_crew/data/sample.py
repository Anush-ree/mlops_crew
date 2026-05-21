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
from mlops_crew.data import LABEL_COLUMN, RAW_INDEX_COLUMN
from mlops_crew.logging_config import get_logger, setup_logging
from mlops_crew.utils.io import save_json

logger = get_logger(__name__)


def sample_paths(config: dict[str, Any]) -> dict[str, Path]:
    data_config = config["data"]
    raw_path = resolve_project_path(data_config["raw_dir"]) / data_config["raw_file"]
    interim_dir = resolve_project_path(data_config["interim_dir"])
    return {
        "raw": raw_path,
        "sampled": interim_dir / data_config["sampled_file"],
        "phase1_reference": interim_dir / data_config["phase1_reference_file"],
        "phase2_increment": interim_dir / data_config["phase2_increment_file"],
        "phase3_holdout": interim_dir / data_config["phase3_holdout_file"],
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
            raise ValueError(f"Cannot stratify; missing label column `{label_column}`")
        parts = [
            group.sample(frac=fraction, random_state=random_state)
            for _, group in data.groupby(label_column)
        ]
        sampled = pd.concat(parts, axis=0)
    else:
        sampled = data.sample(frac=fraction, random_state=random_state)

    return sampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def _label_distribution(data: pd.DataFrame, label_column: str) -> dict[str, int]:
    return {str(label): int(count) for label, count in data[label_column].value_counts().items()}


def _shuffle_group(group: pd.DataFrame, random_state: int) -> pd.DataFrame:
    return group.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def partition_phase_data(data: pd.DataFrame, config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Return deterministic Phase 1, Phase 2 increment, Phase 2 sample, and holdout data.

    Phase 1 used 60% of the raw data. Phase 2 adds the next 20%, while Phase 3
    keeps the remaining 20% untouched. Partitions are stratified by label and
    retain the raw row index for source/divergence analysis.
    """
    sample_config = config["data"]["sample"]
    label_column = config["data"].get("label_column", LABEL_COLUMN)
    random_state = int(sample_config["random_state"])
    phase1_fraction = float(sample_config.get("phase1_fraction", 0.60))
    phase2_fraction = float(sample_config.get("phase2_increment_fraction", 0.20))
    holdout_fraction = float(sample_config.get("phase3_holdout_fraction", 0.20))
    sample_fraction = float(sample_config.get("fraction", phase1_fraction + phase2_fraction))

    if abs(phase1_fraction + phase2_fraction + holdout_fraction - 1.0) > 1e-8:
        raise ValueError("Phase partition fractions must sum to 1.0")
    if abs(sample_fraction - (phase1_fraction + phase2_fraction)) > 1e-8:
        raise ValueError("Sample fraction must equal phase1_fraction + phase2_increment_fraction")
    if not sample_config.get("stratify", True):
        raise ValueError("Phase partitioning must be stratified for reproducible class balance")
    if label_column not in data.columns:
        raise ValueError(f"Cannot partition; missing label column `{label_column}`")

    indexed = data.copy()
    indexed[RAW_INDEX_COLUMN] = indexed.index

    phase1_parts = []
    phase2_parts = []
    holdout_parts = []
    for _, group in indexed.groupby(label_column, sort=True):
        shuffled = _shuffle_group(group, random_state)
        total_sample_rows = int(round(len(shuffled) * sample_fraction))
        phase1_rows = int(round(len(shuffled) * phase1_fraction))
        phase2_rows = total_sample_rows - phase1_rows

        phase1_parts.append(shuffled.iloc[:phase1_rows])
        phase2_parts.append(shuffled.iloc[phase1_rows : phase1_rows + phase2_rows])
        holdout_parts.append(shuffled.iloc[phase1_rows + phase2_rows :])

    phase1 = pd.concat(phase1_parts, axis=0)
    phase2_increment = pd.concat(phase2_parts, axis=0)
    holdout = pd.concat(holdout_parts, axis=0)
    phase2_sample = pd.concat([phase1, phase2_increment], axis=0)

    return {
        "phase1_reference": phase1.sample(frac=1.0, random_state=random_state).reset_index(
            drop=True
        ),
        "phase2_increment": phase2_increment.sample(
            frac=1.0, random_state=random_state
        ).reset_index(drop=True),
        "phase2_sample": phase2_sample.sample(frac=1.0, random_state=random_state).reset_index(
            drop=True
        ),
        "phase3_holdout": holdout.sample(frac=1.0, random_state=random_state).reset_index(
            drop=True
        ),
    }


def run(config: dict[str, Any]) -> pd.DataFrame:
    paths = sample_paths(config)
    if not paths["raw"].exists():
        raise FileNotFoundError(
            f"Raw data not found at {paths['raw']}. Run `dvc pull data/raw/archive.dvc`."
        )

    logger.info("Loading raw data from %s", paths["raw"])
    raw = pd.read_csv(paths["raw"])
    logger.info("Raw shape: %s", raw.shape)

    partitions = partition_phase_data(raw, config)
    sampled = partitions["phase2_sample"]
    paths["sampled"].parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(paths["sampled"], index=False)
    logger.info("Saved sample (%s) to %s", sampled.shape, paths["sampled"])

    partitions["phase1_reference"].to_csv(paths["phase1_reference"], index=False)
    partitions["phase2_increment"].to_csv(paths["phase2_increment"], index=False)
    partitions["phase3_holdout"].to_csv(paths["phase3_holdout"], index=False)
    logger.info(
        "Saved phase partitions: phase1=%d phase2_increment=%d phase3_holdout=%d",
        len(partitions["phase1_reference"]),
        len(partitions["phase2_increment"]),
        len(partitions["phase3_holdout"]),
    )

    label_column = config["data"].get("label_column", LABEL_COLUMN)
    summary = {
        "raw_rows": int(len(raw)),
        "sampled_rows": int(len(sampled)),
        "sample_fraction": float(config["data"]["sample"]["fraction"]),
        "phase1_reference_rows": int(len(partitions["phase1_reference"])),
        "phase2_increment_rows": int(len(partitions["phase2_increment"])),
        "phase3_holdout_rows": int(len(partitions["phase3_holdout"])),
        "label_distribution": _label_distribution(sampled, label_column),
        "partition_label_distribution": {
            name: _label_distribution(frame, label_column) for name, frame in partitions.items()
        },
    }
    save_json(summary, paths["summary"])
    return sampled


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample the raw phishing email CSV")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    setup_logging()
    run(load_project_config(args.config))


if __name__ == "__main__":
    main()
