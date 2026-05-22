"""Compatibility CLI wrapper for Phase 2 batch prediction."""

from __future__ import annotations

from mlops_crew.predict_model import main, predict

__all__ = ["main", "predict"]


if __name__ == "__main__":
    main()
