"""Run the Phase 2 data preparation flow end to end.

Convenience entrypoint for `make data`. The DVC pipeline (`dvc.yaml`) runs the
same stages individually so artifacts can be cached.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mlops_crew.config import CONFIG_PATH, load_project_config
from mlops_crew.data import (
    clean,
    export_transformer_dataset,
    sample,
    source_manifest,
    split,
    validate,
)
from mlops_crew.logging_config import get_logger, setup_logging_from_config

logger = get_logger(__name__)


def process_data(config_path: Path = CONFIG_PATH) -> None:
    """Run the local data flow in the same order as ``dvc repro`` through validate."""
    config = load_project_config(config_path)
    sample.run(config)
    source_manifest.run(config)
    clean.run(config)
    split.run(config)
    if not validate.run(config):
        raise RuntimeError("Data validation failed")
    export_transformer_dataset.export_transformer_dataset(config)


def main() -> None:
    """CLI entrypoint for ``make data``."""
    parser = argparse.ArgumentParser(
        description="Run the full data preparation pipeline")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    setup_logging_from_config(load_project_config(args.config))
    process_data(args.config)
    logger.info("Data pipeline complete")


if __name__ == "__main__":
    main()
