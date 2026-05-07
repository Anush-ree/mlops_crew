"""Project paths and config loader.

The repository layout is fixed relative to this file, so paths resolve the same
way whether code runs from the repo root, a notebook, or a DVC stage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIG_PATH: Path = PROJECT_ROOT / "configs" / "config.yaml"


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a path against the repo root, leaving absolute paths unchanged."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def load_project_config(config_path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Load and return the YAML project config as a dict."""
    path = resolve_project_path(config_path)
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return config
