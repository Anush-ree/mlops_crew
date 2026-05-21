"""Profile the training entrypoint with cProfile."""

from __future__ import annotations

import argparse
import cProfile
import pstats
from pathlib import Path

from mlops_crew.config import CONFIG_PATH, load_project_config, resolve_project_path
from mlops_crew.train_model import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile model training")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--sort", default="cumulative")
    args = parser.parse_args()

    config = load_project_config(args.config)
    output_dir = resolve_project_path(config["reports"]["profiling_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = output_dir / "train_model.prof"
    text_path = output_dir / "train_model_cprofile.txt"

    profiler = cProfile.Profile()
    profiler.enable()
    train(config)
    profiler.disable()
    profiler.dump_stats(profile_path)

    with text_path.open("w", encoding="utf-8") as file:
        stats = pstats.Stats(profiler, stream=file).strip_dirs().sort_stats(args.sort)
        stats.print_stats(40)

    print(f"Wrote {profile_path}")
    print(f"Wrote {text_path}")


if __name__ == "__main__":
    main()
