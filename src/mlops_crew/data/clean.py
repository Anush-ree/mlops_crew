"""Data cleaning script for phishing email detection.

This script loads the raw phishing email dataset, performs cleaning steps,
and saves the cleaned data to the processed directory.
"""

import logging
import re
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RAW_PATH = Path("data/raw/phishing_email.csv")
PROCESSED_PATH = Path("data/processed/cleaned.csv")


def load_data(path: Path) -> pd.DataFrame:
    """Load raw CSV data from disk.

    Args:
        path: Path to the raw CSV file.

    Returns:
        Loaded DataFrame.
    """
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows, {df.shape[1]} columns")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows from the DataFrame.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with duplicates removed.
    """
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    logger.info(f"Removed {before - after} duplicate rows")
    return df


def drop_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing values.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with null rows removed.
    """
    before = len(df)
    df = df.dropna(subset=["text_combined", "label"])
    after = len(df)
    logger.info(f"Removed {before - after} rows with null values")
    return df


def clean_text(text: str) -> str:
    """Clean a single email text string.

    Applies the following transformations:
    - Lowercase
    - Remove URLs
    - Remove email addresses
    - Remove extra whitespace

    Args:
        text: Raw email text.

    Returns:
        Cleaned text string.
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text_column(df: pd.DataFrame) -> pd.DataFrame:
    """Apply text cleaning to the text_combined column.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with cleaned text column.
    """
    logger.info("Cleaning text column...")
    df["text_combined"] = df["text_combined"].apply(clean_text)
    return df


def drop_short_text(df: pd.DataFrame, min_length: int = 3) -> pd.DataFrame:
    """Drop rows where text is empty or too short after cleaning.

    Args:
        df: Input DataFrame.
        min_length: Minimum number of characters required.

    Returns:
        DataFrame with short text rows removed.
    """
    before = len(df)
    df = df[df["text_combined"].str.strip().str.len() > min_length]
    after = len(df)
    logger.info(
        f"Removed {before - after} rows with text shorter than {min_length} chars")
    return df


def validate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure labels are binary (0 or 1) and drop invalid rows.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with only valid label values.
    """
    before = len(df)
    df = df[df["label"].isin([0, 1])]
    after = len(df)
    logger.info(f"Removed {before - after} rows with invalid labels")
    logger.info(f"Label distribution:\n{df['label'].value_counts()}")
    return df


def save_data(df: pd.DataFrame, path: Path) -> None:
    """Save cleaned DataFrame to CSV.

    Args:
        df: Cleaned DataFrame.
        path: Output file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved cleaned data to {path} ({len(df)} rows)")


def main() -> None:
    """Run the full cleaning pipeline."""
    df = load_data(RAW_PATH)
    df = remove_duplicates(df)
    df = drop_nulls(df)
    df = clean_text_column(df)
    df = drop_short_text(df)
    df = validate_labels(df)
    save_data(df, PROCESSED_PATH)
    logger.info("Cleaning complete.")


if __name__ == "__main__":
    main()
