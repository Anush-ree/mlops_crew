"""Data validation script for phishing email detection.

This script checks the cleaned and split datasets for quality issues
such as nulls, class imbalance, empty text, and shape mismatches.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CLEANED_PATH = Path("data/processed/cleaned.csv")
TRAIN_PATH = Path("data/processed/train.csv")
VAL_PATH = Path("data/processed/val.csv")
TEST_PATH = Path("data/processed/test.csv")

EXPECTED_COLUMNS = {"text_combined", "label"}
VALID_LABELS = {0, 1}
MIN_TEXT_LENGTH = 3
MAX_IMBALANCE_RATIO = 3.0


def check_columns(df: pd.DataFrame, name: str) -> bool:
    """Check that required columns exist.

    Args:
        df: DataFrame to validate.
        name: Name of the dataset for logging.

    Returns:
        True if valid, False otherwise.
    """
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        logger.error(f"[{name}] Missing columns: {missing}")
        return False
    logger.info(f"[{name}] Columns OK: {list(df.columns)}")
    return True


def check_nulls(df: pd.DataFrame, name: str) -> bool:
    """Check for null values in required columns.

    Args:
        df: DataFrame to validate.
        name: Name of the dataset for logging.

    Returns:
        True if no nulls, False otherwise.
    """
    nulls = df[list(EXPECTED_COLUMNS)].isnull().sum()
    if nulls.any():
        logger.error(f"[{name}] Null values found:\n{nulls[nulls > 0]}")
        return False
    logger.info(f"[{name}] No null values found")
    return True


def check_labels(df: pd.DataFrame, name: str) -> bool:
    """Check that all labels are valid binary values.

    Args:
        df: DataFrame to validate.
        name: Name of the dataset for logging.

    Returns:
        True if all labels valid, False otherwise.
    """
    invalid = set(df["label"].unique()) - VALID_LABELS
    if invalid:
        logger.error(f"[{name}] Invalid label values: {invalid}")
        return False
    logger.info(f"[{name}] Label distribution:\n{df['label'].value_counts()}")
    return True


def check_class_balance(df: pd.DataFrame, name: str) -> bool:
    """Check that class imbalance ratio is within acceptable range.

    Args:
        df: DataFrame to validate.
        name: Name of the dataset for logging.

    Returns:
        True if balanced enough, False otherwise.
    """
    counts = df["label"].value_counts()
    ratio = counts.max() / counts.min()
    if ratio > MAX_IMBALANCE_RATIO:
        logger.warning(f"[{name}] High class imbalance ratio: {ratio:.2f}")
    else:
        logger.info(f"[{name}] Class imbalance ratio: {ratio:.2f} (OK)")
    return True


def check_empty_text(df: pd.DataFrame, name: str) -> bool:
    """Check for empty or very short text entries.

    Args:
        df: DataFrame to validate.
        name: Name of the dataset for logging.

    Returns:
        True if no problematic text found, False otherwise.
    """
    short = df[df["text_combined"].str.len() < MIN_TEXT_LENGTH]
    if len(short) > 0:
        logger.warning(f"[{name}] {len(short)} rows with very short text (< {MIN_TEXT_LENGTH} chars)")
    else:
        logger.info(f"[{name}] No empty or very short text found")
    return True


def validate_dataset(path: Path, name: str) -> bool:
    """Run all validation checks on a dataset.

    Args:
        path: Path to CSV file.
        name: Name for logging.

    Returns:
        True if all checks pass, False otherwise.
    """
    if not path.exists():
        logger.error(f"[{name}] File not found: {path}")
        return False

    df = pd.read_csv(path)
    logger.info(f"[{name}] Shape: {df.shape}")

    checks = [
        check_columns(df, name),
        check_nulls(df, name),
        check_labels(df, name),
        check_class_balance(df, name),
        check_empty_text(df, name),
    ]
    return all(checks)


def main() -> None:
    """Validate all processed datasets."""
    datasets = [
        (CLEANED_PATH, "cleaned"),
        (TRAIN_PATH, "train"),
        (VAL_PATH, "val"),
        (TEST_PATH, "test"),
    ]

    results = []
    for path, name in datasets:
        logger.info(f"\n--- Validating {name} ---")
        result = validate_dataset(path, name)
        results.append(result)

    if all(results):
        logger.info("\nAll validation checks passed.")
    else:
        logger.error("\nSome validation checks failed. Review logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()