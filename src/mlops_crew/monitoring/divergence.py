"""Phase 2 data and prediction divergence reporting."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp
from sklearn.feature_extraction.text import CountVectorizer

from mlops_crew.config import CONFIG_PATH, load_project_config, resolve_project_path
from mlops_crew.data import LABEL_COLUMN, RAW_INDEX_COLUMN, TEXT_COLUMN
from mlops_crew.logging_config import get_logger, setup_logging_from_config
from mlops_crew.utils.io import save_json

logger = get_logger(__name__)


def divergence_paths(config: dict[str, Any]) -> dict[str, Path]:
    """Resolve phase partitions, model, and divergence report output paths."""
    data_config = config["data"]
    interim_dir = resolve_project_path(data_config["interim_dir"])
    reports_dir = resolve_project_path(config["reports"]["divergence_dir"])
    return {
        "phase1": interim_dir / data_config["phase1_reference_file"],
        "phase2_increment": interim_dir / data_config["phase2_increment_file"],
        "source_manifest": interim_dir / data_config["source_manifest_file"],
        "model": resolve_project_path(config["modeling"]["output_dir"]) / "best_model.joblib",
        "report_json": reports_dir / "phase2_divergence_report.json",
        "summary_md": reports_dir / "phase2_divergence_summary.md",
    }


def _distribution(series: pd.Series) -> dict[str, int]:
    return {str(label): int(count) for label, count in series.value_counts().items()}


def _normalize_distribution(counts: dict[str, int], keys: list[str]) -> np.ndarray:
    values = np.array([counts.get(key, 0) for key in keys], dtype=float)
    total = values.sum()
    return values / total if total else values


def _js_distance(left: dict[str, int], right: dict[str, int]) -> float:
    keys = sorted(set(left) | set(right))
    return float(
        jensenshannon(
            _normalize_distribution(left, keys),
            _normalize_distribution(right, keys),
        )
    )


def _text_length_stats(frame: pd.DataFrame) -> dict[str, float]:
    lengths = frame[TEXT_COLUMN].astype(str).str.len()
    return {
        "mean": float(lengths.mean()),
        "median": float(lengths.median()),
        "p90": float(lengths.quantile(0.90)),
        "max": float(lengths.max()),
    }


def _token_counter(texts: pd.Series, token_pattern: str) -> Counter[str]:
    vectorizer = CountVectorizer(token_pattern=token_pattern, lowercase=False)
    matrix = vectorizer.fit_transform(texts.astype(str))
    counts = np.asarray(matrix.sum(axis=0)).ravel()
    return Counter(dict(zip(vectorizer.get_feature_names_out(), counts, strict=True)))


def _vocabulary_report(
    reference: pd.DataFrame, current: pd.DataFrame, config: dict[str, Any]
) -> dict[str, Any]:
    token_pattern = config["features"]["tfidf"]["token_pattern"]
    reference_counts = _token_counter(reference[TEXT_COLUMN], token_pattern)
    current_counts = _token_counter(current[TEXT_COLUMN], token_pattern)
    reference_vocab = set(reference_counts)
    current_vocab = set(current_counts)
    new_tokens = current_vocab - reference_vocab
    current_token_total = sum(current_counts.values())
    new_token_total = sum(current_counts[token] for token in new_tokens)
    top_n = int(config.get("monitoring", {}).get("top_new_tokens", 50))
    return {
        "reference_vocab_size": int(len(reference_vocab)),
        "current_vocab_size": int(len(current_vocab)),
        "new_token_count": int(len(new_tokens)),
        "new_token_rate": float(new_token_total / current_token_total)
        if current_token_total
        else 0.0,
        "top_new_tokens": [
            {"token": token, "count": int(count)}
            for token, count in Counter(
                {token: current_counts[token] for token in new_tokens}
            ).most_common(top_n)
        ],
    }


def _source_distribution(frame: pd.DataFrame, manifest: pd.DataFrame) -> dict[str, int]:
    if RAW_INDEX_COLUMN not in frame.columns:
        return {}
    merged = frame[[RAW_INDEX_COLUMN]].merge(
        manifest[[RAW_INDEX_COLUMN, "source"]],
        on=RAW_INDEX_COLUMN,
    )
    return _distribution(merged["source"])


def _prediction_distribution(model: Any, frame: pd.DataFrame) -> dict[str, int]:
    predictions = model.predict(frame[TEXT_COLUMN])
    return _distribution(pd.Series(predictions))


def build_divergence_report(config: dict[str, Any]) -> dict[str, Any]:
    """Compare Phase 1 reference vs Phase 2 increment label, source, and text drift."""
    paths = divergence_paths(config)
    phase1 = pd.read_csv(paths["phase1"])
    phase2_increment = pd.read_csv(paths["phase2_increment"])
    manifest = (
        pd.read_csv(paths["source_manifest"])
        if paths["source_manifest"].exists()
        else pd.DataFrame()
    )

    label_phase1 = _distribution(phase1[LABEL_COLUMN])
    label_phase2 = _distribution(phase2_increment[LABEL_COLUMN])
    source_phase1 = _source_distribution(phase1, manifest) if not manifest.empty else {}
    source_phase2 = _source_distribution(phase2_increment, manifest) if not manifest.empty else {}

    lengths_phase1 = phase1[TEXT_COLUMN].astype(str).str.len()
    lengths_phase2 = phase2_increment[TEXT_COLUMN].astype(str).str.len()
    length_ks_test = ks_2samp(lengths_phase1, lengths_phase2)
    report: dict[str, Any] = {
        "phase1_reference_rows": int(len(phase1)),
        "phase2_increment_rows": int(len(phase2_increment)),
        "label_distribution": {
            "phase1_reference": label_phase1,
            "phase2_increment": label_phase2,
            "jensen_shannon_distance": _js_distance(label_phase1, label_phase2),
        },
        "source_distribution": {
            "phase1_reference": source_phase1,
            "phase2_increment": source_phase2,
            "jensen_shannon_distance": _js_distance(source_phase1, source_phase2)
            if source_phase1 and source_phase2
            else None,
        },
        "text_length": {
            "phase1_reference": _text_length_stats(phase1),
            "phase2_increment": _text_length_stats(phase2_increment),
            "ks_statistic": float(length_ks_test.statistic),
            "ks_pvalue": float(length_ks_test.pvalue),
        },
        "vocabulary": _vocabulary_report(phase1, phase2_increment, config),
    }

    if paths["model"].exists():
        model = joblib.load(paths["model"])
        pred_phase1 = _prediction_distribution(model, phase1)
        pred_phase2 = _prediction_distribution(model, phase2_increment)
        report["prediction_distribution"] = {
            "phase1_reference": pred_phase1,
            "phase2_increment": pred_phase2,
            "jensen_shannon_distance": _js_distance(pred_phase1, pred_phase2),
        }
    return report


def _format_distribution(distribution: dict[str, int]) -> str:
    if not distribution:
        return "not available"
    return ", ".join(f"{key}: {value}" for key, value in sorted(distribution.items()))


def write_summary(report: dict[str, Any], path: Path) -> None:
    """Write a human-readable Markdown summary of the divergence JSON report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 2 Divergence Summary",
        "",
        f"- Phase 1 reference rows: {report['phase1_reference_rows']}",
        f"- Phase 2 increment rows: {report['phase2_increment_rows']}",
        "- Label distribution Phase 1: "
        f"{_format_distribution(report['label_distribution']['phase1_reference'])}",
        "- Label distribution Phase 2 increment: "
        f"{_format_distribution(report['label_distribution']['phase2_increment'])}",
        f"- Label JS distance: {report['label_distribution']['jensen_shannon_distance']:.6f}",
        "- Source distribution Phase 1: "
        f"{_format_distribution(report['source_distribution']['phase1_reference'])}",
        "- Source distribution Phase 2 increment: "
        f"{_format_distribution(report['source_distribution']['phase2_increment'])}",
        f"- Text length KS statistic: {report['text_length']['ks_statistic']:.6f}",
        f"- New-token rate in Phase 2 increment: {report['vocabulary']['new_token_rate']:.6f}",
    ]
    if "prediction_distribution" in report:
        lines.extend(
            [
                "- Prediction distribution Phase 1: "
                f"{_format_distribution(report['prediction_distribution']['phase1_reference'])}",
                "- Prediction distribution Phase 2 increment: "
                f"{_format_distribution(report['prediction_distribution']['phase2_increment'])}",
                "- Prediction JS distance: "
                f"{report['prediction_distribution']['jensen_shannon_distance']:.6f}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config: dict[str, Any]) -> dict[str, Any]:
    """Build divergence JSON and Markdown reports for the DVC stage."""
    paths = divergence_paths(config)
    report = build_divergence_report(config)
    save_json(report, paths["report_json"])
    write_summary(report, paths["summary_md"])
    logger.info("Saved divergence report to %s", paths["report_json"])
    return report


def main() -> None:
    """CLI entrypoint for the DVC ``divergence`` stage."""
    parser = argparse.ArgumentParser(description="Build Phase 2 divergence report")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    config = load_project_config(args.config)
    setup_logging_from_config(config)
    run(config)


if __name__ == "__main__":
    main()
