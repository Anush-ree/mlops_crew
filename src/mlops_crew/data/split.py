"""Data splitting script for phishing email detection.

This script splits the cleaned dataset into train, validation, and test sets
using stratified sampling to preserve class distribution.
"""

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CLEANED_PATH = Path("data/processed/cleaned.csv")
TRAIN_PATH = Path("data/processed/train.csv")
VAL_PATH = Path("data/processed/val.csv")
TEST_PATH = Path("data/processed/test.csv")

RANDOM_SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15


def load_cleaned_data(path: Path) -> pd.DataFrame:
    """Load cleaned CSV data.

    Args:
        path: Path to the cleaned CSV file.

    Returns:
        Loaded DataFrame.
    """
    logger.info(f"Loading cleaned data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows")
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into train, validation, and test sets.

    Uses stratified splitting to preserve class distribution across splits.

    Args:
        df: Full cleaned DataFrame.
        test_size: Fraction of data to use for test set.
        val_size: Fraction of data to use for validation set.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label"],
    )

    val_fraction = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_fraction,
        random_state=seed,
        stratify=train_val["label"],
    )

    logger.info(f"Train: {len(train)} rows | Val: {len(val)} rows | Test: {len(test)} rows")
    logger.info(f"Train label dist:\n{train['label'].value_counts()}")
    logger.info(f"Val label dist:\n{val['label'].value_counts()}")
    logger.info(f"Test label dist:\n{test['label'].value_counts()}")

    return train, val, test


def save_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Save train, val, and test splits to CSV.

    Args:
        train: Training DataFrame.
        val: Validation DataFrame.
        test: Test DataFrame.
    """
    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_PATH, index=False)
    val.to_csv(VAL_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)
    logger.info(f"Saved train to {TRAIN_PATH}")
    logger.info(f"Saved val to {VAL_PATH}")
    logger.info(f"Saved test to {TEST_PATH}")


def main() -> None:
    """Run the full splitting pipeline."""
    df = load_cleaned_data(CLEANED_PATH)
    train, val, test = split_data(df)
    save_splits(train, val, test)
    logger.info("Splitting complete.")


if __name__ == "__main__":
    main()