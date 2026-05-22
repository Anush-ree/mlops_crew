"""Stage 2: clean the sampled CSV into a model-ready table.

The schema is reduced to two columns (`text_combined`, `label`). Cleaning is
deliberately conservative: we lowercase and collapse whitespace for the TF-IDF
baseline but keep URLs, addresses, numbers, and punctuation, since those are
useful phishing signals for later feature engineering.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from mlops_crew.config import CONFIG_PATH, load_project_config, resolve_project_path
from mlops_crew.data import LABEL_COLUMN, TEXT_COLUMN
from mlops_crew.logging_config import get_logger, setup_logging_from_config
from mlops_crew.utils.io import save_json

logger = get_logger(__name__)

VALID_LABELS = {0, 1}


def clean_paths(config: dict[str, Any]) -> dict[str, Path]:
    data_config = config["data"]
    interim_dir = resolve_project_path(data_config["interim_dir"])
    processed_dir = resolve_project_path(data_config["processed_dir"])
    return {
        "sampled": interim_dir / data_config["sampled_file"],
        "cleaned": processed_dir / data_config["cleaned_file"],
        "summary": processed_dir / "cleaning_summary.json",
    }


def _normalize_label(value: object) -> int | None:
    if pd.isna(value):
        return None
    try:
        as_float = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if not as_float.is_integer():
        return None
    label = int(as_float)
    return label if label in VALID_LABELS else None


def clean_text(text: object, *, lowercase: bool, normalize_whitespace: bool) -> str:
    """Lowercase and collapse whitespace; keep URLs/addresses/numbers/punctuation."""
    cleaned = "" if pd.isna(text) else str(text)
    if lowercase:
        cleaned = cleaned.lower()
    if normalize_whitespace:
        cleaned = " ".join(cleaned.split())
    return cleaned.strip()


def clean_dataset(data: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict]:
    """Apply schema, label, and text cleaning. Returns the cleaned frame plus a summary."""
    data_config = config["data"]
    cleaning = config.get("cleaning", {})

    text_source = data_config.get("text_column", TEXT_COLUMN)
    label_source = data_config.get("label_column", LABEL_COLUMN)
    missing = {text_source, label_source} - set(data.columns)
    if missing:
        raise ValueError(f"Input data missing required columns: {sorted(missing)}")

    cleaned = data[[text_source, label_source]].rename(
        columns={text_source: TEXT_COLUMN, label_source: LABEL_COLUMN}
    )

    raw_rows = len(cleaned)
    cleaned[LABEL_COLUMN] = cleaned[LABEL_COLUMN].apply(_normalize_label)
    cleaned[TEXT_COLUMN] = cleaned[TEXT_COLUMN].apply(
        clean_text,
        lowercase=bool(cleaning.get("lowercase", True)),
        normalize_whitespace=bool(cleaning.get("normalize_whitespace", True)),
    )

    cleaned = cleaned.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])
    min_length = int(cleaning.get("min_text_length", 3))
    cleaned = cleaned[cleaned[TEXT_COLUMN].str.len() >= min_length]

    if cleaning.get("drop_duplicates", True):
        cleaned = cleaned.drop_duplicates(subset=[TEXT_COLUMN, LABEL_COLUMN])

    cleaned[LABEL_COLUMN] = cleaned[LABEL_COLUMN].astype("int64")
    cleaned = cleaned.reset_index(drop=True)

    summary = {
        "raw_rows": int(raw_rows),
        "cleaned_rows": int(len(cleaned)),
        "rows_removed": int(raw_rows - len(cleaned)),
        "label_distribution": {
            str(label): int(count) for label, count in cleaned[LABEL_COLUMN].value_counts().items()
        },
    }
    return cleaned, summary


def run(config: dict[str, Any]) -> pd.DataFrame:
    paths = clean_paths(config)
    if not paths["sampled"].exists():
        raise FileNotFoundError(
            f"Sampled data not found at {paths['sampled']}. Run `python -m mlops_crew.data.sample`."
        )

    logger.info("Loading sampled data from %s", paths["sampled"])
    sampled = pd.read_csv(paths["sampled"])
    cleaned, summary = clean_dataset(sampled, config)

    paths["cleaned"].parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(paths["cleaned"], index=False)
    save_json(summary, paths["summary"])
    logger.info("Cleaned %s rows -> %s rows", summary["raw_rows"], summary["cleaned_rows"])
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean the sampled phishing email CSV")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    config = load_project_config(args.config)
    setup_logging_from_config(config)
    run(config)


if __name__ == "__main__":
    main()
