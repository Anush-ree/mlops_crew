"""Measure saved-model inference latency for Phase 2 monitoring."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from mlops_crew.config import CONFIG_PATH, load_project_config, resolve_project_path
from mlops_crew.data import TEXT_COLUMN
from mlops_crew.logging_config import get_logger, setup_logging_from_config

logger = get_logger(__name__)


def latency_paths(config: dict[str, Any]) -> dict[str, Path]:
    """Resolve best model, test input, and latency CSV output paths."""
    processed_dir = resolve_project_path(config["data"]["processed_dir"])
    monitoring_dir = resolve_project_path(config["reports"]["monitoring_dir"])
    return {
        "model": resolve_project_path(config["modeling"]["output_dir"]) / "best_model.joblib",
        "input": processed_dir / config["data"]["test_file"],
        "output": monitoring_dir / "inference_latency.csv",
    }


def measure_latency(
    model: Any,
    data: pd.DataFrame,
    *,
    batch_sizes: list[int],
    repeats: int = 3,
) -> pd.DataFrame:
    """Benchmark ``model.predict`` at several batch sizes and return timing rows."""
    rows = []
    for batch_size in batch_sizes:
        batch = data.head(batch_size)
        if batch.empty:
            continue
        for repeat in range(repeats):
            start = time.perf_counter()
            model.predict(batch[TEXT_COLUMN])
            elapsed = time.perf_counter() - start
            rows.append(
                {
                    "batch_size": int(len(batch)),
                    "repeat": repeat + 1,
                    "elapsed_seconds": float(elapsed),
                    "records_per_second": float(len(batch) / elapsed) if elapsed else 0.0,
                    "milliseconds_per_record": float((elapsed / len(batch)) * 1000),
                }
            )
    return pd.DataFrame(rows)


def run(config: dict[str, Any]) -> pd.DataFrame:
    """Load the best model, benchmark inference, and write the latency CSV."""
    paths = latency_paths(config)
    model = joblib.load(paths["model"])
    data = pd.read_csv(paths["input"])
    batch_sizes = [1, 32, 256, min(1024, len(data))]
    results = measure_latency(model, data, batch_sizes=batch_sizes)
    paths["output"].parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(paths["output"], index=False)
    logger.info("Saved inference latency report to %s", paths["output"])
    return results


def main() -> None:
    """CLI entrypoint for the DVC ``inference_latency`` stage."""
    parser = argparse.ArgumentParser(description="Measure inference latency")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    config = load_project_config(args.config)
    setup_logging_from_config(config)
    run(config)


if __name__ == "__main__":
    main()
