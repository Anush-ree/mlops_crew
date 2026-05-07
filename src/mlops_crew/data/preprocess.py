"""Preprocessing script for phishing email detection.

This script applies TF-IDF vectorization to the cleaned email text
and saves the feature matrix and labels for model training.
"""

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CLEANED_PATH = Path("data/processed/cleaned.csv")
FEATURES_PATH = Path("data/processed/features.pkl")
VECTORIZER_PATH = Path("data/processed/tfidf_vectorizer.pkl")

MAX_FEATURES = 10000
NGRAM_RANGE = (1, 2)


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


def fit_tfidf(texts: pd.Series) -> tuple[TfidfVectorizer, object]:
    """Fit a TF-IDF vectorizer on email texts.

    Args:
        texts: Series of cleaned email text strings.

    Returns:
        Tuple of (fitted vectorizer, feature matrix).
    """
    logger.info(
        f"Fitting TF-IDF with max_features={MAX_FEATURES}, ngram_range={NGRAM_RANGE}")
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\b[a-zA-Z]{2,}\b",
    )
    X = vectorizer.fit_transform(texts)
    logger.info(f"Feature matrix shape: {X.shape}")
    return vectorizer, X


def save_artifacts(vectorizer: TfidfVectorizer, X: object, y: pd.Series) -> None:
    """Save vectorizer and feature matrix to disk.

    Args:
        vectorizer: Fitted TF-IDF vectorizer.
        X: Feature matrix.
        y: Label series.
    """
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"X": X, "y": y.values}, FEATURES_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    logger.info(f"Saved features to {FEATURES_PATH}")
    logger.info(f"Saved vectorizer to {VECTORIZER_PATH}")


def main() -> None:
    """Run the full preprocessing pipeline."""
    df = load_cleaned_data(CLEANED_PATH)
    vectorizer, X = fit_tfidf(df["text_combined"])
    save_artifacts(vectorizer, X, df["label"])
    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
