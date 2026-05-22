"""Small JSON helpers used by the data and training stages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_json(obj: Any, path: Path) -> None:
    """Write ``obj`` as sorted, indented JSON to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)


def load_json(path: Path) -> Any:
    """Load JSON from ``path``."""
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)
