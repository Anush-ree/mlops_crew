"""Preprocessing script for phishing email detection.

This script applies TF-IDF vectorization to email text and saves
feature matrices for training, validation, and test sets.

Critically, the TF-IDF vectorizer is fit ONLY on the training set
to prevent data leakage into validation and test sets.
"""

from pathlib import Path

import hydra
import joblib
import pandas as pd
from omegaconf import DictConfig
from sklearn.feature_extraction.text import TfidfVectorizer

from mlops_crew.utils.logging_setup import get_logger, setup_logging

logger = get_logger(__name__)


def load_split(path: Path) -> pd.DataFrame:
    """Load a data split from CSV.

    Args:
        path: Path to the CSV file.

    Returns:
        Loaded DataFrame.
    """
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df):,} rows from {path}")
    return df


def fit_tfidf(
    train_texts: pd.Series,
    max_features: int,
    ngram_range: list[int],
    sublinear_tf: bool,
) -> TfidfVectorizer:
    """Fit TF-IDF vectorizer on training texts only.

    Fitting only on training data prevents data leakage —
    the vectorizer never sees vocabulary from val or test sets
    during fitting.

    Args:
        train_texts: Series of training email texts.
        max_features: Maximum number of TF-IDF features.
        ngram_range: N-gram range as [min_n, max_n].
        sublinear_tf: Whether to apply sublinear TF scaling.

    Returns:
        Fitted TF-IDF vectorizer.
    """
    logger.info(
        f"Fitting TF-IDF on training data only — "
        f"max_features={max_features}, ngram_range={tuple(ngram_range)}"
    )
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=tuple(ngram_range),
        sublinear_tf=sublinear_tf,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\b[a-zA-Z]{2,}\b",
    )
    vectorizer.fit(train_texts)
    logger.info("TF-IDF fitting complete")
    return vectorizer


def transform_and_save(
    vectorizer: TfidfVectorizer,
    df: pd.DataFrame,
    output_path: Path,
    split_name: str,
) -> None:
    """Transform a split using the fitted vectorizer and save.

    Args:
        vectorizer: Fitted TF-IDF vectorizer.
        df: DataFrame with text_combined and label columns.
        output_path: Path to save the feature matrix.
        split_name: Name of the split for logging.
    """
    X = vectorizer.transform(df["text_combined"])
    y = df["label"].values
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"X": X, "y": y}, output_path)
    logger.info(f"Saved {split_name} features to {output_path} — shape: {X.shape}")


@hydra.main(config_path="../../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the full preprocessing pipeline.

    Args:
        cfg: Hydra configuration object.
    """
    setup_logging(
        log_dir=cfg.logging.log_dir,
        log_file=cfg.logging.log_file,
        log_level=cfg.logging.log_level,
    )
    logger.info("Starting preprocessing pipeline")

    train = load_split(Path(cfg.data.train_path))
    val = load_split(Path(cfg.data.val_path))
    test = load_split(Path(cfg.data.test_path))

    vectorizer = fit_tfidf(
        train["text_combined"],
        max_features=cfg.data.tfidf.max_features,
        ngram_range=cfg.data.tfidf.ngram_range,
        sublinear_tf=cfg.data.tfidf.sublinear_tf,
    )

    processed_dir = Path("data/processed")
    transform_and_save(vectorizer, train, processed_dir / "train_features.pkl", "train")
    transform_and_save(vectorizer, val, processed_dir / "val_features.pkl", "val")
    transform_and_save(vectorizer, test, processed_dir / "test_features.pkl", "test")

    joblib.dump(vectorizer, processed_dir / "tfidf_vectorizer.pkl")
    logger.info(f"Saved vectorizer to {processed_dir / 'tfidf_vectorizer.pkl'}")
    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()