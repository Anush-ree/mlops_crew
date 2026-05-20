"""Logging configuration for phishing email detection pipeline.

Sets up structured logging with rich for colored terminal output
and rotating file handler for persistent logs.
"""

import logging
import logging.handlers
from pathlib import Path

from rich.logging import RichHandler
from rich.traceback import install as install_rich_traceback


def setup_logging(
    log_dir: str = "logs",
    log_file: str = "pipeline.log",
    log_level: str = "INFO",
    max_bytes: int = 10_485_760,
    backup_count: int = 5,
) -> logging.Logger:
    """Set up application logging with rich terminal output and rotating file handler.

    Configures two handlers:
    - RichHandler: colored, readable output to stdout
    - RotatingFileHandler: persistent logs saved under log_dir/

    Args:
        log_dir: Directory to store log files.
        log_file: Name of the log file.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
        max_bytes: Maximum size of each log file before rotation.
        backup_count: Number of rotated log files to keep.

    Returns:
        Configured root logger.
    """
    install_rich_traceback(show_locals=True)

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    level = getattr(logging, log_level.upper(), logging.INFO)

    rich_handler = RichHandler(
        level=level,
        rich_tracebacks=True,
        show_time=True,
        show_level=True,
        show_path=True,
        markup=True,
    )
    rich_handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))

    file_handler = logging.handlers.RotatingFileHandler(
        log_path / log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(rich_handler)
    root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a named logger.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        Named logger instance.
    """
    return logging.getLogger(name)