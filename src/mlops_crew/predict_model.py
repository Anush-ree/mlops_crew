"""Score a CSV with the saved phishing classifier pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from mlops_crew.config import CONFIG_PATH, load_project_config, resolve_project_path
from mlops_crew.data import TEXT_COLUMN
from mlops_crew.logging_config import get_logger, setup_logging_from_config

logger = get_logger(__name__)


def predict(model_path: Path, input_path: Path, output_path: Path) -> None:
    logger.info("Loading model from %s", model_path)
    model = joblib.load(model_path)

    data = pd.read_csv(input_path)
    if TEXT_COLUMN not in data.columns:
        raise ValueError(f"Input must include `{TEXT_COLUMN}` column")

    output = data.copy()
    output["prediction"] = model.predict(data[TEXT_COLUMN])
    if hasattr(model, "predict_proba"):
        output["score"] = model.predict_proba(data[TEXT_COLUMN])[:, 1]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    logger.info("Wrote predictions to %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score a CSV file with the trained phishing classifier"
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = load_project_config(args.config)
    setup_logging_from_config(config)
    processed_dir = resolve_project_path(config["data"]["processed_dir"])
    model_dir = resolve_project_path(config["modeling"]["output_dir"])
    predictions_dir = resolve_project_path(config["reports"]["predictions_dir"])

    model_path = args.model_path or model_dir / "best_model.joblib"
    input_path = args.input or processed_dir / config["data"]["test_file"]
    output_path = args.output or predictions_dir / "batch_predictions.csv"
    predict(model_path, input_path, output_path)


if __name__ == "__main__":
    main()
