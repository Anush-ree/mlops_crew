"""Compatibility CLI wrapper for Phase 2 model training."""

from __future__ import annotations

from mlops_crew.train_model import main, train

__all__ = ["main", "train"]


if __name__ == "__main__":
    main()
