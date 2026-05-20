"""Data cleaning script for phishing email detection.

This script loads the raw phishing email dataset, performs cleaning steps,
and saves the cleaned data to the processed directory.

Uses Hydra for configuration management and rich for structured logging.
"""

import re
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from mlops_crew.utils.logging_setup import get_logger, setup_logging

logger = get_logger(__name__)


def load_data(path: Path) -> pd.DataFrame:
    """Load raw CSV data from disk.

    Args:
        path: Path to the raw CSV file.

    Returns:
        Loaded DataFrame.
    """
    logger.info(f"Loading data from [bold]{path}[/bold]")
    df = pd.read_csv(path)
    logger.info(f"Loaded [green]{len(df):,}[/green] rows, {df.shape[1]} columns")
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
    logger.info(f"Removed [yellow]{before - len(df):,}[/yellow] duplicate rows")
    return df


def drop_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing values in required columns.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with null rows removed.
    """
    before = len(df)
    df = df.dropna(subset=["text_combined", "label"])
    logger.info(f"Removed [yellow]{before - len(df):,}[/yellow] rows with null values")
    return df


def clean_text(text: str) -> str:
    """Clean a single email text string.

    Applies the following transformations:
    - Lowercase
    - Remove extra whitespace

    Note: URLs and email addresses are preserved as they are
    strong signals for phishing detection. Full feature
    engineering will be handled in Phase 2.

    Args:
        text: Raw email text.

    Returns:
        Cleaned text string.
    """
    text = str(text).lower()
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
    logger.info(f"Removed [yellow]{before - len(df):,}[/yellow] rows with text shorter than {min_length} chars")
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
    logger.info(f"Removed [yellow]{before - len(df):,}[/yellow] rows with invalid labels")
    logger.info(f"Label distribution:\n{df['label'].value_counts().to_string()}")
    return df


def save_data(df: pd.DataFrame, path: Path) -> None:
    """Save cleaned DataFrame to CSV.

    Args:
        df: Cleaned DataFrame.
        path: Output file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved cleaned data to [bold]{path}[/bold] ([green]{len(df):,}[/green] rows)")


@hydra.main(config_path="../../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the full cleaning pipeline using Hydra config.

    Args:
        cfg: Hydra configuration object.
    """
    setup_logging(
        log_dir=cfg.logging.log_dir,
        log_file=cfg.logging.log_file,
        log_level=cfg.logging.log_level,
    )
    logger.info("[bold blue]Starting data cleaning pipeline[/bold blue]")

    df = load_data(Path(cfg.data.raw_path))
    df = remove_duplicates(df)
    df = drop_nulls(df)
    df = clean_text_column(df)
    df = drop_short_text(df, min_length=cfg.data.min_text_length)
    df = validate_labels(df)
    save_data(df, Path(cfg.data.cleaned_path))

    logger.info("[bold green]Cleaning complete.[/bold green]")


if __name__ == "__main__":
    main()