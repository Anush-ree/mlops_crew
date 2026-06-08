"""Tests for Phase 2 logging and Hydra experiment configuration."""

from __future__ import annotations

import logging
from pathlib import Path

from mlops_crew.logging_config import setup_logging
from mlops_crew.train_hydra import build_effective_config, deep_merge


def test_setup_logging_writes_rotating_file_without_duplicate_handlers(tmp_path: Path) -> None:
    log_file = "pipeline.log"

    setup_logging(log_dir=tmp_path, log_file=log_file, level="INFO")
    first_handler_count = len(logging.getLogger().handlers)
    setup_logging(log_dir=tmp_path, log_file=log_file, level="INFO")

    logger = logging.getLogger("tests.phase2.logging")
    logger.info("structured logging smoke test")
    for handler in logging.getLogger().handlers:
        handler.flush()

    assert first_handler_count == 2
    assert len(logging.getLogger().handlers) == first_handler_count
    assert "structured logging smoke test" in (tmp_path / log_file).read_text(encoding="utf-8")


def test_deep_merge_preserves_base_config_sections() -> None:
    base = {
        "features": {"tfidf": {"max_features": 20000, "min_df": 2}},
        "tracking": {"enabled": True, "tracking_uri": "file:./mlruns"},
    }
    overrides = {
        "features": {"tfidf": {"max_features": 10000}},
        "tracking": {"experiment_name": "override"},
    }

    merged = deep_merge(base, overrides)

    assert merged["features"]["tfidf"] == {"max_features": 10000, "min_df": 2}
    assert merged["tracking"]["enabled"] is True
    assert merged["tracking"]["tracking_uri"] == "file:./mlruns"
    assert merged["tracking"]["experiment_name"] == "override"


def test_hydra_effective_config_routes_artifacts_to_output_dir(tmp_path: Path) -> None:
    hydra_config = {
        "base_config_path": "configs/config.yaml",
        "features": {"tfidf": {"max_features": 10000, "ngram_range": [1, 3]}},
        "modeling": {"logistic_regression": {"C": 0.5}},
        "tracking": {"experiment_name": "phishing-email-phase2-hydra"},
    }

    config = build_effective_config(
        hydra_config,
        hydra_output_dir=tmp_path,
        experiment_name="phase2_experimental",
    )

    assert config["features"]["tfidf"]["max_features"] == 10000
    assert config["features"]["tfidf"]["ngram_range"] == [1, 3]
    assert config["features"]["tfidf"]["min_df"] == 2
    assert config["modeling"]["logistic_regression"]["C"] == 0.5
    assert Path(config["modeling"]["output_dir"]).parent == tmp_path / "artifacts"
    assert config["tracking"]["run_name"] == "hydra-phase2_experimental"
    assert config["tracking"]["tags"]["config_source"] == "hydra"
    assert config["tracking"]["tags"]["hydra_experiment"] == "phase2_experimental"
