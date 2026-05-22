"""Tests covering the data and model invariants we care about for Phase 1."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mlops_crew.data import LABEL_COLUMN, RAW_INDEX_COLUMN, TEXT_COLUMN
from mlops_crew.data.clean import clean_dataset, clean_text
from mlops_crew.data.export_transformer_dataset import export_transformer_dataset
from mlops_crew.data.sample import partition_phase_data, sample_dataset
from mlops_crew.data.source_manifest import build_source_manifest
from mlops_crew.data.split import split_data
from mlops_crew.data.validate import validate_dataset, validation_report_path
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


def test_partition_phase_data_creates_non_overlapping_phase_slices() -> None:
    """Phase 2 should add 20% while keeping Phase 3 data held out."""
    config = {
        "data": {
            "label_column": LABEL_COLUMN,
            "sample": {
                "fraction": 0.80,
                "phase1_fraction": 0.60,
                "phase2_increment_fraction": 0.20,
                "phase3_holdout_fraction": 0.20,
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

    partitions = partition_phase_data(data, config)

    assert len(partitions["phase1_reference"]) == 60
    assert len(partitions["phase2_increment"]) == 20
    assert len(partitions["phase2_sample"]) == 80
    assert len(partitions["phase3_holdout"]) == 20

    used_indexes = set(partitions["phase2_sample"][RAW_INDEX_COLUMN])
    holdout_indexes = set(partitions["phase3_holdout"][RAW_INDEX_COLUMN])
    assert not used_indexes & holdout_indexes


def test_build_source_manifest_uses_combined_file_source_blocks(tmp_path: Path) -> None:
    """Source metadata should map combined-file row blocks without recombining text."""
    raw_dir = tmp_path / "raw"
    interim_dir = tmp_path / "interim"
    raw_dir.mkdir()

    pd.DataFrame({LABEL_COLUMN: [0, 1, 1]}).to_csv(raw_dir / "phishing_email.csv", index=False)
    pd.DataFrame({LABEL_COLUMN: [1, 0]}).to_csv(raw_dir / "source_a.csv", index=False)
    pd.DataFrame({LABEL_COLUMN: [1]}).to_csv(raw_dir / "source_b.csv", index=False)
    config = {
        "data": {
            "raw_dir": str(raw_dir),
            "raw_file": "phishing_email.csv",
            "interim_dir": str(interim_dir),
            "source_manifest_file": "source_manifest.csv",
            "source_order": [
                {"name": "source_a", "file": "source_a.csv"},
                {"name": "source_b", "file": "source_b.csv"},
            ],
        }
    }

    manifest = build_source_manifest(config)

    assert manifest[RAW_INDEX_COLUMN].tolist() == [0, 1, 2]
    assert manifest["source"].tolist() == ["source_a", "source_a", "source_b"]
    assert manifest[LABEL_COLUMN].tolist() == [0, 1, 1]


def test_export_transformer_dataset_writes_jsonl_splits(tmp_path: Path) -> None:
    """Transformer export should keep only text and label fields."""
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    for split_name in ("train", "val", "test"):
        pd.DataFrame({TEXT_COLUMN: [f"{split_name} email"], LABEL_COLUMN: [1]}).to_csv(
            processed_dir / f"{split_name}.csv",
            index=False,
        )
    config = {
        "data": {
            "processed_dir": str(processed_dir),
            "train_file": "train.csv",
            "val_file": "val.csv",
            "test_file": "test.csv",
            "transformer_dir": "transformer",
        }
    }

    summary = export_transformer_dataset(config)

    train_jsonl = processed_dir / "transformer" / "train.jsonl"
    assert train_jsonl.exists()
    assert train_jsonl.read_text(encoding="utf-8").strip() == ('{"text":"train email","label":1}')
    assert summary["splits"]["train"]["rows"] == 1


def test_validate_run_writes_report_for_valid_splits(tmp_path: Path) -> None:
    """Successful validation should emit the DVC-tracked report artifact."""
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True)
    for name, labels in {
        "cleaned": [0, 1],
        "train": [0, 1, 0],
        "val": [0, 1],
        "test": [1, 0],
    }.items():
        pd.DataFrame(
            {
                TEXT_COLUMN: [f"{name} email text {index}" for index in range(len(labels))],
                LABEL_COLUMN: labels,
            }
        ).to_csv(processed_dir / f"{name}.csv", index=False)

    config = {
        "data": {
            "processed_dir": str(processed_dir),
            "cleaned_file": "cleaned.csv",
            "train_file": "train.csv",
            "val_file": "val.csv",
            "test_file": "test.csv",
        },
        "cleaning": {"min_text_length": 3},
    }

    from mlops_crew.data.validate import run

    assert run(config)
    report_path = validation_report_path(config)
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert set(report["datasets"]) == {"cleaned", "train", "val", "test"}


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
            },
            "linear_svc": {"C": 1.0, "class_weight": "balanced", "max_iter": 100},
            "complement_nb": {"alpha": 0.5},
        },
    }

    pipeline = build_text_classifier("logistic_regression", config)

    assert list(pipeline.named_steps) == ["tfidf", "classifier"]

    assert list(build_text_classifier("linear_svc", config).named_steps) == [
        "tfidf",
        "classifier",
    ]
    assert list(build_text_classifier("complement_nb", config).named_steps) == [
        "tfidf",
        "classifier",
    ]
