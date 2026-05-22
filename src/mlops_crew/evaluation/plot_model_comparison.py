"""Generate a compact Phase 2 model comparison chart."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd

from mlops_crew.config import CONFIG_PATH, load_project_config, resolve_project_path
from mlops_crew.logging_config import get_logger, setup_logging

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

logger = get_logger(__name__)


def plot_paths(config: dict[str, Any]) -> dict[str, Path]:
    """Resolve model comparison CSV input and PNG output paths."""
    metrics_dir = resolve_project_path(config["reports"]["metrics_dir"])
    return {
        "comparison": metrics_dir / "model_comparison.csv",
        "figure": metrics_dir / "model_comparison.png",
    }


def plot_model_comparison(config: dict[str, Any]) -> Path:
    """Build F2 and false-negative-rate bar charts from ``model_comparison.csv``."""
    paths = plot_paths(config)
    metrics = pd.read_csv(paths["comparison"])
    metrics = metrics.sort_values("val_f2", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].bar(metrics["model_name"], metrics["val_f2"], label="validation")
    axes[0].bar(metrics["model_name"], metrics["test_f2"], alpha=0.65, label="test")
    axes[0].set_title("F2 Score")
    axes[0].set_ylim(0.80, 1.01)
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].legend()

    axes[1].bar(
        metrics["model_name"],
        metrics["test_false_negative_rate"],
        color="#c44e52",
    )
    axes[1].set_title("Test False Negative Rate")
    axes[1].set_ylim(0, max(metrics["test_false_negative_rate"].max() * 1.25, 0.01))
    axes[1].tick_params(axis="x", rotation=25)

    fig.suptitle("Phase 2 Phishing Classifier Comparison")
    paths["figure"].parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths["figure"], dpi=160)
    plt.close(fig)
    logger.info("Saved model comparison chart to %s", paths["figure"])
    return paths["figure"]


def main() -> None:
    """CLI entrypoint for the DVC ``plot_model_comparison`` stage."""
    parser = argparse.ArgumentParser(description="Plot model comparison metrics")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    setup_logging()
    plot_model_comparison(load_project_config(args.config))


if __name__ == "__main__":
    main()
