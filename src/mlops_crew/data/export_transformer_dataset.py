"""Export processed splits as JSONL files for transformer fine-tuning."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from mlops_crew.config import CONFIG_PATH, PROJECT_ROOT, load_project_config, resolve_project_path
from mlops_crew.data import LABEL_COLUMN, TEXT_COLUMN
from mlops_crew.logging_config import get_logger, setup_logging_from_config
from mlops_crew.utils.io import save_json

logger = get_logger(__name__)


def transformer_dataset_paths(config: dict[str, Any]) -> dict[str, Path]:
    """Resolve processed split inputs and JSONL export directory paths."""
    data_config = config["data"]
    processed_dir = resolve_project_path(data_config["processed_dir"])
    transformer_dir = processed_dir / data_config.get("transformer_dir", "transformer")
    return {
        "processed_dir": processed_dir,
        "transformer_dir": transformer_dir,
        "summary": transformer_dir / "dataset_info.json",
    }


def _split_file(config: dict[str, Any], split_name: str) -> str:
    return str(config["data"][f"{split_name}_file"])


def _write_jsonl(input_path: Path, output_path: Path) -> dict[str, Any]:
    frame = pd.read_csv(input_path)
    missing = {TEXT_COLUMN, LABEL_COLUMN} - set(frame.columns)
    if missing:
        raise ValueError(f"{input_path} is missing required columns: {sorted(missing)}")

    output = frame[[TEXT_COLUMN, LABEL_COLUMN]].rename(columns={TEXT_COLUMN: "text"})
    output[LABEL_COLUMN] = output[LABEL_COLUMN].astype("int64")
    output.to_json(output_path, orient="records", lines=True, force_ascii=False)
    return {
        "rows": int(len(output)),
        "label_distribution": {
            str(label): int(count) for label, count in output[LABEL_COLUMN].value_counts().items()
        },
    }


def _project_relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def export_transformer_dataset(config: dict[str, Any]) -> dict[str, Any]:
    """Export train/val/test CSVs as Hugging Face-compatible JSONL splits."""
    paths = transformer_dataset_paths(config)
    paths["transformer_dir"].mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "format": "jsonl",
        "loader": 'datasets.load_dataset("json", data_files={...})',
        "columns": ["text", LABEL_COLUMN],
        "splits": {},
    }
    for split_name in ("train", "val", "test"):
        input_path = paths["processed_dir"] / _split_file(config, split_name)
        output_path = paths["transformer_dir"] / f"{split_name}.jsonl"
        summary["splits"][split_name] = {
            "path": _project_relative(output_path),
            **_write_jsonl(input_path, output_path),
        }
        logger.info("Saved %s transformer split to %s", split_name, output_path)

    save_json(summary, paths["summary"])
    return summary


def main() -> None:
    """CLI entrypoint for the DVC ``transformer_dataset`` stage."""
    parser = argparse.ArgumentParser(description="Export HF-compatible JSONL splits")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    config = load_project_config(args.config)
    setup_logging_from_config(config)
    export_transformer_dataset(config)


if __name__ == "__main__":
    main()
