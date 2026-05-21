"""Profile the training entrypoint with cProfile."""

from __future__ import annotations

import argparse
import cProfile
import pstats
from copy import deepcopy
from pathlib import Path
from typing import Any

from mlops_crew.config import CONFIG_PATH, load_project_config, resolve_project_path
from mlops_crew.train_model import train


def _isolated_config(config: dict[str, Any], *, with_tracking: bool) -> dict[str, Any]:
    """Profile the training path without changing DVC-tracked outputs."""
    profile_config = deepcopy(config)
    scratch_root = Path(config["reports"]["profiling_dir"]) / "scratch" / "train"
    profile_config["modeling"]["output_dir"] = str(scratch_root / "models")
    profile_config["reports"]["metrics_dir"] = str(scratch_root / "metrics")
    profile_config["reports"]["predictions_dir"] = str(scratch_root / "predictions")
    profile_config["reports"]["monitoring_dir"] = str(scratch_root / "monitoring")
    profile_config["tracking"]["enabled"] = bool(with_tracking)
    return profile_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile model training")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for cProfile outputs. Defaults to reports.profiling_dir.",
    )
    parser.add_argument("--sort", default="cumulative")
    parser.add_argument(
        "--with-tracking",
        action="store_true",
        help=(
            "Also create an MLflow run while profiling. Disabled by default "
            "to avoid duplicate runs."
        ),
    )
    args = parser.parse_args()

    config = load_project_config(args.config)
    output_dir = (
        resolve_project_path(args.output_dir)
        if args.output_dir is not None
        else resolve_project_path(config["reports"]["profiling_dir"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = output_dir / "train_model.prof"
    text_path = output_dir / "train_model_cprofile.txt"

    profiler = cProfile.Profile()
    profiler.enable()
    train(_isolated_config(config, with_tracking=args.with_tracking))
    profiler.disable()
    profiler.dump_stats(profile_path)

    with text_path.open("w", encoding="utf-8") as file:
        stats = pstats.Stats(profiler, stream=file).strip_dirs().sort_stats(args.sort)
        stats.print_stats(40)

    print(f"Wrote {profile_path}")
    print(f"Wrote {text_path}")


if __name__ == "__main__":
    main()
