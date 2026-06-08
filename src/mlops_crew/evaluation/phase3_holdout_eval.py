"""Evaluate the best trained model once on the Phase 3 holdout set.

This script must be run at most once without retraining.
Output: reports/metrics/phase3_holdout_metrics.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from mlops_crew.config import CONFIG_PATH, PROJECT_ROOT, load_project_config, resolve_project_path
from mlops_crew.data import LABEL_COLUMN, TEXT_COLUMN
from mlops_crew.data.clean import clean_text
from mlops_crew.evaluation.metrics import binary_classification_report
from mlops_crew.logging_config import get_logger, setup_logging_from_config
from mlops_crew.utils.io import save_json

logger = get_logger(__name__)


def _project_relative(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def evaluate_holdout(config: dict) -> dict:
    model_path = resolve_project_path(config["modeling"]["output_dir"]) / "best_model.joblib"
    holdout_path = (
        resolve_project_path(config["data"]["interim_dir"]) / config["data"]["phase3_holdout_file"]
    )
    output_path = (
        resolve_project_path(config["reports"]["metrics_dir"]) / "phase3_holdout_metrics.json"
    )

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run `make train` first.")
    if not holdout_path.exists():
        raise FileNotFoundError(
            f"Holdout data not found: {holdout_path}. Run `python -m mlops_crew.data.sample`."
        )

    logger.info("Loading model from %s", model_path)
    model = joblib.load(model_path)

    logger.info("Loading holdout set from %s", holdout_path)
    df = pd.read_csv(holdout_path)

    cleaning = config.get("cleaning", {})
    df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(
        lambda t: clean_text(
            t,
            lowercase=bool(cleaning.get("lowercase", True)),
            normalize_whitespace=bool(cleaning.get("normalize_whitespace", True)),
        )
    )
    df = df[df[TEXT_COLUMN].str.len() >= int(cleaning.get("min_text_length", 3))]
    logger.info("Holdout rows after cleaning: %d", len(df))

    predictions = model.predict(df[TEXT_COLUMN])
    scores = None
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(df[TEXT_COLUMN])[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(df[TEXT_COLUMN])

    positive_label = int(config["modeling"].get("positive_label", 1))
    metrics = binary_classification_report(
        df[LABEL_COLUMN], predictions, scores, positive_label=positive_label
    )

    result = {
        "model_path": _project_relative(model_path),
        "holdout_path": _project_relative(holdout_path),
        "holdout_rows": len(df),
        "metrics": metrics,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(result, output_path)
    logger.info("Phase 3 holdout metrics saved to %s", output_path)

    logger.info("Holdout results:")
    for k, v in metrics.items():
        logger.info("  %s: %s", k, v)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate best model on Phase 3 holdout set")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    config = load_project_config(args.config)
    setup_logging_from_config(config)
    evaluate_holdout(config)


if __name__ == "__main__":
    main()
