"""Build row-level source metadata for the already-combined phishing dataset.

The Kaggle archive includes the individual source CSVs and `phishing_email.csv`,
which is the concatenated modeling file used by Phase 1. We do not recombine the
text data here. Instead, this stage records which source block each combined row
came from so Phase 2 can report source-level divergence without duplicating the
dataset. The `source_row_index` value is the index inside the combined-file
source block, not a guaranteed pointer back to the original source CSV row.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from mlops_crew.config import CONFIG_PATH, load_project_config, resolve_project_path
from mlops_crew.data import LABEL_COLUMN, RAW_INDEX_COLUMN
from mlops_crew.logging_config import get_logger, setup_logging_from_config
from mlops_crew.utils.io import save_json

logger = get_logger(__name__)


def source_manifest_paths(config: dict[str, Any]) -> dict[str, Path]:
    data_config = config["data"]
    interim_dir = resolve_project_path(data_config["interim_dir"])
    raw_dir = resolve_project_path(data_config["raw_dir"])
    return {
        "raw_dir": raw_dir,
        "combined": raw_dir / data_config["raw_file"],
        "manifest": interim_dir / data_config["source_manifest_file"],
        "summary": interim_dir / "source_manifest_summary.json",
    }


def _read_labels(path: Path) -> pd.Series:
    data = pd.read_csv(path, usecols=[LABEL_COLUMN])
    labels = pd.to_numeric(data[LABEL_COLUMN], errors="coerce")
    if labels.isna().any():
        raise ValueError(f"Source labels must be numeric 0/1 in {path}")
    invalid = sorted(set(labels.astype(int).unique()) - {0, 1})
    if invalid:
        raise ValueError(f"Unexpected labels in {path}: {invalid}")
    return labels.astype("int64")


def build_source_manifest(config: dict[str, Any]) -> pd.DataFrame:
    """Create a row-index to source mapping matching `phishing_email.csv` order."""
    paths = source_manifest_paths(config)
    if not paths["combined"].exists():
        raise FileNotFoundError(f"Combined raw data not found: {paths['combined']}")

    combined_labels = _read_labels(paths["combined"])
    frames = []
    start = 0
    for source in config["data"]["source_order"]:
        source_name = source["name"]
        source_path = paths["raw_dir"] / source["file"]
        if not source_path.exists():
            raise FileNotFoundError(f"Source CSV not found: {source_path}")

        labels = _read_labels(source_path)
        row_count = len(labels)
        combined_segment = combined_labels.iloc[start : start + row_count].reset_index(drop=True)
        source_distribution = labels.value_counts().sort_index().to_dict()
        combined_distribution = combined_segment.value_counts().sort_index().to_dict()
        if combined_distribution != source_distribution:
            raise ValueError(
                f"Combined row segment for {source_name} does not match the source label "
                f"distribution: combined={combined_distribution}, source={source_distribution}"
            )
        frame = pd.DataFrame(
            {
                RAW_INDEX_COLUMN: range(start, start + row_count),
                "source": source_name,
                "source_row_index": range(row_count),
                LABEL_COLUMN: combined_segment,
            }
        )
        frames.append(frame)
        logger.info("%s: %d rows", source_name, row_count)
        start += row_count

    manifest = pd.concat(frames, ignore_index=True)
    if len(manifest) != len(combined_labels):
        raise ValueError(
            f"Source rows ({len(manifest)}) do not match combined rows ({len(combined_labels)})"
        )
    return manifest


def run(config: dict[str, Any]) -> pd.DataFrame:
    paths = source_manifest_paths(config)
    manifest = build_source_manifest(config)
    paths["manifest"].parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(paths["manifest"], index=False)

    summary = {
        "rows": int(len(manifest)),
        "source_distribution": {
            str(source): int(count) for source, count in manifest["source"].value_counts().items()
        },
        "label_by_source": {
            str(source): {
                str(label): int(count)
                for label, count in group[LABEL_COLUMN].value_counts().items()
            }
            for source, group in manifest.groupby("source")
        },
    }
    save_json(summary, paths["summary"])
    logger.info("Saved source manifest to %s", paths["manifest"])
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build source metadata for raw rows")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    config = load_project_config(args.config)
    setup_logging_from_config(config)
    run(config)


if __name__ == "__main__":
    main()
