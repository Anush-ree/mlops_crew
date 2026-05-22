"""Shared logging setup for command-line, DVC, and Hydra entrypoints."""

from __future__ import annotations

import logging
import logging.handlers
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install as install_rich_traceback

from mlops_crew.config import PROJECT_ROOT

_FILE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _resolve_log_dir(log_dir: str | Path) -> Path:
    path = Path(log_dir)
    return path if path.is_absolute() else PROJECT_ROOT / path


def setup_logging(
    level: str = "INFO",
    log_dir: str | Path = "logs",
    log_file: str = "pipeline.log",
    max_bytes: int = 10_485_760,
    backup_count: int = 5,
    rich_tracebacks: bool = True,
) -> None:
    """Configure Rich stdout logs and a rotating structured log file.

    The function is safe to call more than once; existing root handlers are
    removed before the new console and file handlers are attached.
    """
    if rich_tracebacks:
        install_rich_traceback(show_locals=False)

    log_path = _resolve_log_dir(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    parsed_level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(parsed_level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console_handler = RichHandler(
        console=Console(file=sys.stdout),
        level=parsed_level,
        rich_tracebacks=rich_tracebacks,
        show_time=True,
        show_level=True,
        show_path=True,
        markup=False,
    )
    console_handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))

    file_handler = logging.handlers.RotatingFileHandler(
        log_path / log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(parsed_level)
    file_handler.setFormatter(logging.Formatter(_FILE_FORMAT, datefmt=_DATEFMT))

    root.addHandler(console_handler)
    root.addHandler(file_handler)


def setup_logging_from_config(config: Mapping[str, Any]) -> None:
    """Configure logging from the project's ``logging`` config section."""
    logging_config = config.get("logging", {})
    setup_logging(
        level=str(logging_config.get("level", "INFO")),
        log_dir=logging_config.get("log_dir", "logs"),
        log_file=str(logging_config.get("log_file", "pipeline.log")),
        max_bytes=int(logging_config.get("max_bytes", 10_485_760)),
        backup_count=int(logging_config.get("backup_count", 5)),
        rich_tracebacks=bool(logging_config.get("rich_tracebacks", True)),
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
