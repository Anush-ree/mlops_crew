"""Profile saved-model batch inference with cProfile."""

from __future__ import annotations

import argparse
import cProfile
import pstats
from copy import deepcopy
from pathlib import Path
from typing import Any

from mlops_crew.config import CONFIG_PATH, load_project_config, resolve_project_path
from mlops_crew.monitoring.inference_latency import run as run_latency


def _isolated_config(config: dict[str, Any]) -> dict[str, Any]:
    """Profile inference without overwriting the DVC-tracked latency report."""
    profile_config = deepcopy(config)
    profile_config["reports"]["monitoring_dir"] = str(
        Path(config["reports"]["profiling_dir"]) / "scratch" / "predict" / "monitoring"
    )
    return profile_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile saved-model inference")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for cProfile outputs. Defaults to reports.profiling_dir.",
    )
    parser.add_argument("--sort", default="cumulative")
    args = parser.parse_args()

    config = load_project_config(args.config)
    output_dir = (
        resolve_project_path(args.output_dir)
        if args.output_dir is not None
        else resolve_project_path(config["reports"]["profiling_dir"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = output_dir / "predict_model.prof"
    text_path = output_dir / "predict_model_cprofile.txt"

    profiler = cProfile.Profile()
    profiler.enable()
    run_latency(_isolated_config(config))
    profiler.disable()
    profiler.dump_stats(profile_path)

    with text_path.open("w", encoding="utf-8") as file:
        stats = pstats.Stats(profiler, stream=file).strip_dirs().sort_stats(args.sort)
        stats.print_stats(40)

    print(f"Wrote {profile_path}")
    print(f"Wrote {text_path}")


if __name__ == "__main__":
    main()
