"""Data combination script for phishing email detection.

This script loads all 7 raw CSV datasets, maps them to a unified schema,
and saves a single combined dataset for use in the pipeline.

Datasets handled:
- phishing_email.csv (already unified - used as base)
- CEAS_08.csv
- Enron.csv
- Ling.csv
- Nazario.csv
- Nigerian_Fraud.csv
- SpamAssasin.csv
"""

from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from mlops_crew.utils.logging_setup import get_logger, setup_logging
import csv
csv.field_size_limit(10_000_000)

logger = get_logger(__name__)

RAW_DIR = Path("data/raw")
COMBINED_PATH = Path("data/processed/combined.csv")


def load_phishing_email() -> pd.DataFrame:
    """Load the main phishing_email.csv dataset.

    Returns:
        DataFrame with text_combined, label, source columns.
    """
    path = RAW_DIR / "phishing_email.csv"
    if not path.exists():
        logger.warning("phishing_email.csv not found, skipping.")
        return pd.DataFrame()
    df = pd.read_csv(path, engine='python', encoding_errors='ignore')[["text_combined", "label"]].copy()
    df["source"] = "phishing_email"
    logger.info(f"phishing_email.csv: {len(df):,} rows")
    return df


def load_ceas() -> pd.DataFrame:
    """Load CEAS_08.csv and map to unified schema.

    Returns:
        DataFrame with text_combined, label, source columns.
    """
    path = RAW_DIR / "CEAS_08.csv"
    if not path.exists():
        logger.warning("CEAS_08.csv not found, skipping.")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df.rename(columns={"body": "text_combined"})
    df = df[["text_combined", "label"]].copy()
    df["source"] = "ceas_08"
    logger.info(f"CEAS_08.csv: {len(df):,} rows")
    return df


def load_enron() -> pd.DataFrame:
    """Load Enron.csv and map to unified schema.

    Returns:
        DataFrame with text_combined, label, source columns.
    """
    path = RAW_DIR / "Enron.csv"
    if not path.exists():
        logger.warning("Enron.csv not found, skipping.")
        return pd.DataFrame()
    df = pd.read_csv(path, engine='python', encoding='utf-8', encoding_errors='ignore')
    df = df.rename(columns={"body": "text_combined"})
    df["label"] = df["label"].map({"spam": 1, "ham": 0})
    df = df[["text_combined", "label"]].dropna().copy()
    df["source"] = "enron"
    logger.info(f"Enron.csv: {len(df):,} rows")
    return df


def load_ling() -> pd.DataFrame:
    """Load Ling.csv and map to unified schema.

    Returns:
        DataFrame with text_combined, label, source columns.
    """
    path = RAW_DIR / "Ling.csv"
    if not path.exists():
        logger.warning("Ling.csv not found, skipping.")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df.rename(columns={"body": "text_combined"})
    df = df[["text_combined", "label"]].dropna().copy()
    df["source"] = "ling"
    logger.info(f"Ling.csv: {len(df):,} rows")
    return df


def load_nazario() -> pd.DataFrame:
    """Load Nazario.csv and map to unified schema.

    Returns:
        DataFrame with text_combined, label, source columns.
    """
    path = RAW_DIR / "Nazario.csv"
    if not path.exists():
        logger.warning("Nazario.csv not found, skipping.")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df.rename(columns={"body": "text_combined"})
    df["label"] = 1  # Nazario is all phishing
    df = df[["text_combined", "label"]].dropna().copy()
    df["source"] = "nazario"
    logger.info(f"Nazario.csv: {len(df):,} rows")
    return df


def load_nigerian_fraud() -> pd.DataFrame:
    """Load Nigerian_Fraud.csv and map to unified schema.

    Returns:
        DataFrame with text_combined, label, source columns.
    """
    path = RAW_DIR / "Nigerian_Fraud.csv"
    if not path.exists():
        logger.warning("Nigerian_Fraud.csv not found, skipping.")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df.rename(columns={"body": "text_combined"})
    df["label"] = 1  # Nigerian fraud emails are all phishing
    df = df[["text_combined", "label"]].dropna().copy()
    df["source"] = "nigerian_fraud"
    logger.info(f"Nigerian_Fraud.csv: {len(df):,} rows")
    return df


def load_spamassassin() -> pd.DataFrame:
    """Load SpamAssasin.csv and map to unified schema.

    Returns:
        DataFrame with text_combined, label, source columns.
    """
    path = RAW_DIR / "SpamAssasin.csv"
    if not path.exists():
        logger.warning("SpamAssasin.csv not found, skipping.")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df.rename(columns={"body": "text_combined"})
    df["label"] = df["label"].map({"spam": 1, "ham": 0})
    df = df[["text_combined", "label"]].dropna().copy()
    df["source"] = "spamassassin"
    logger.info(f"SpamAssasin.csv: {len(df):,} rows")
    return df


def combine_all() -> pd.DataFrame:
    """Load and combine all datasets into a unified DataFrame.

    Returns:
        Combined DataFrame with text_combined, label, source columns.
    """
    loaders = [
        load_phishing_email,
        load_ceas,
        load_enron,
        load_ling,
        load_nazario,
        load_nigerian_fraud,
        load_spamassassin,
    ]

    frames = []
    for loader in loaders:
        df = loader()
        if not df.empty:
            frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    logger.info(f"Combined dataset: {len(combined):,} rows")
    logger.info(f"Label distribution:\n{combined['label'].value_counts().to_string()}")
    logger.info(f"Source distribution:\n{combined['source'].value_counts().to_string()}")
    return combined


def save_combined(df: pd.DataFrame, path: Path) -> None:
    """Save combined dataset to CSV.

    Args:
        df: Combined DataFrame.
        path: Output file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved combined data to {path} ({len(df):,} rows)")


@hydra.main(config_path="../../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the data combination pipeline.

    Args:
        cfg: Hydra configuration object.
    """
    setup_logging(
        log_dir=cfg.logging.log_dir,
        log_file=cfg.logging.log_file,
        log_level=cfg.logging.log_level,
    )
    logger.info("Starting data combination pipeline")
    combined = combine_all()
    save_combined(combined, COMBINED_PATH)
    logger.info("Combination complete.")


if __name__ == "__main__":
    main()