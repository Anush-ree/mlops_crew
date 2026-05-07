"""Tests covering the data and model invariants we care about for Phase 1."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mlops_crew.data import LABEL_COLUMN, TEXT_COLUMN
from mlops_crew.data.clean import clean_dataset, clean_text
from mlops_crew.data.sample import sample_dataset
from mlops_crew.data.split import split_data
from mlops_crew.data.validate import validate_dataset
from mlops_crew.models.text_classifiers import build_text_classifier


def test_clean_text_keeps_urls_and_emails() -> None:
    """Phase 1 cleaning should not delete phishing-specific URL/email signals."""
    text = "Contact Admin@Example.com at HTTPS://Example.com/reset"

    cleaned = clean_text(text, lowercase=True, normalize_whitespace=True)

    assert "admin@example.com" in cleaned
    assert "https://example.com/reset" in cleaned


def test_clean_dataset_normalizes_schema_and_labels() -> None:
    """Cleaned output should preserve the modeling schema."""
    config = {
        "data": {"text_column": TEXT_COLUMN, "label_column": LABEL_COLUMN},
        "cleaning": {
            "lowercase": True,
            "normalize_whitespace": True,
            "drop_duplicates": True,
            "min_text_length": 3,
        },
    }
    raw = pd.DataFrame(
        {
            TEXT_COLUMN: ["Hello   WORLD", "Hello   WORLD", "Reset password now"],
            LABEL_COLUMN: [0, 0, 1],
        }
    )

    cleaned, summary = clean_dataset(raw, config)

    assert list(cleaned.columns) == [TEXT_COLUMN, LABEL_COLUMN]
    assert len(cleaned) == 2
    assert cleaned.loc[0, TEXT_COLUMN] == "hello world"
    assert summary["label_distribution"] == {"0": 1, "1": 1}


def test_split_data_is_reproducible_and_stratified() -> None:
    """Repeated splits with the same seed should produce the same labels."""
    config = {
        "data": {
            "split": {
                "train_size": 0.7,
                "val_size": 0.15,
                "test_size": 0.15,
                "random_state": 42,
                "stratify": True,
            }
        }
    }
    data = pd.DataFrame(
        {
            TEXT_COLUMN: [f"email {index}" for index in range(100)],
            LABEL_COLUMN: [0] * 70 + [1] * 30,
        }
    )

    first = split_data(data, config)
    second = split_data(data, config)

    assert [len(split) for split in first] == [70, 15, 15]
    assert first[0][LABEL_COLUMN].tolist() == second[0][LABEL_COLUMN].tolist()
    assert first[1][LABEL_COLUMN].sum() == second[1][LABEL_COLUMN].sum()


def test_sample_dataset_uses_configured_fraction() -> None:
    """Phase 1 should use a reproducible 60% stratified raw-data sample."""
    config = {
        "data": {
            "label_column": LABEL_COLUMN,
            "sample": {
                "fraction": 0.60,
                "random_state": 42,
                "stratify": True,
            },
        }
    }
    data = pd.DataFrame(
        {
            TEXT_COLUMN: [f"email {index}" for index in range(100)],
            LABEL_COLUMN: [0] * 70 + [1] * 30,
        }
    )

    sampled = sample_dataset(data, config)

    assert len(sampled) == 60
    assert sampled[LABEL_COLUMN].value_counts().to_dict() == {0: 42, 1: 18}


def test_validate_dataset_rejects_missing_class(tmp_path: Path) -> None:
    """A modeling split should contain both phishing and legitimate examples."""
    path = tmp_path / "one_class.csv"
    pd.DataFrame(
        {
            TEXT_COLUMN: ["reset password now", "verify account access"],
            LABEL_COLUMN: [1, 1],
        }
    ).to_csv(path, index=False)

    assert not validate_dataset(path, "one_class", min_text_length=3)


def test_build_text_classifier_contains_tfidf_and_classifier() -> None:
    """Model factories should build full sklearn pipelines."""
    config = {
        "project": {"seed": 42},
        "features": {
            "tfidf": {
                "max_features": 100,
                "ngram_range": [1, 2],
                "min_df": 1,
                "max_df": 1.0,
                "sublinear_tf": True,
                "strip_accents": "unicode",
                "token_pattern": r"(?u)\b\w\w+\b",
            }
        },
        "modeling": {
            "logistic_regression": {
                "C": 1.0,
                "class_weight": "balanced",
                "max_iter": 100,
                "solver": "liblinear",
            }
        },
    }

    pipeline = build_text_classifier("logistic_regression", config)

    assert list(pipeline.named_steps) == ["tfidf", "classifier"]
