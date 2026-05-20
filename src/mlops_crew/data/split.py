"""Data splitting script for phishing email detection.

This script splits the cleaned dataset into train, validation, and test sets
using stratified sampling to preserve class distribution.

Uses Hydra for configuration management and rich for structured logging.
"""

from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from mlops_crew.utils.logging_setup import get_logger, setup_logging

logger = get_logger(__name__)


def load_cleaned_data(path: Path) -> pd.DataFrame:
    """Load cleaned CSV data.

    Args:
        path: Path to the cleaned CSV file.

    Returns:
        Loaded DataFrame.
    """
    logger.info(f"Loading cleaned data from [bold]{path}[/bold]")
    df = pd.read_csv(path)
    logger.info(f"Loaded [green]{len(df):,}[/green] rows")
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float,
    val_size: float,
    seed: int,
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

    logger.info(
        f"Split complete — Train: [green]{len(train):,}[/green] | "
        f"Val: [yellow]{len(val):,}[/yellow] | "
        f"Test: [blue]{len(test):,}[/blue]"
    )
    return train, val, test


def save_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    train_path: Path,
    val_path: Path,
    test_path: Path,
) -> None:
    """Save train, val, and test splits to CSV.

    Args:
        train: Training DataFrame.
        val: Validation DataFrame.
        test: Test DataFrame.
        train_path: Output path for train split.
        val_path: Output path for val split.
        test_path: Output path for test split.
    """
    train_path.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)
    logger.info(f"Saved train → [bold]{train_path}[/bold]")
    logger.info(f"Saved val   → [bold]{val_path}[/bold]")
    logger.info(f"Saved test  → [bold]{test_path}[/bold]")


@hydra.main(config_path="../../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the full splitting pipeline using Hydra config.

    Args:
        cfg: Hydra configuration object.
    """
    setup_logging(
        log_dir=cfg.logging.log_dir,
        log_file=cfg.logging.log_file,
        log_level=cfg.logging.log_level,
    )
    logger.info("[bold blue]Starting data splitting pipeline[/bold blue]")

    df = load_cleaned_data(Path(cfg.data.cleaned_path))
    train, val, test = split_data(
        df,
        test_size=cfg.data.test_size,
        val_size=cfg.data.val_size,
        seed=cfg.data.random_seed,
    )
    save_splits(
        train, val, test,
        Path(cfg.data.train_path),
        Path(cfg.data.val_path),
        Path(cfg.data.test_path),
    )

    logger.info("[bold green]Splitting complete.[/bold green]")


if __name__ == "__main__":
    main()