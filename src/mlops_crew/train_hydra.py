"""Hydra entrypoint for Phase 2 experiment sweeps.

The normal DVC pipeline still reads ``configs/config.yaml`` directly. This
module layers Hydra overrides on top of that base config for experiment runs so
configuration sweeps are traceable in MLflow without dirtying DVC-tracked model
and report artifacts.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import hydra
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from mlops_crew.config import PROJECT_ROOT, load_project_config, resolve_project_path
from mlops_crew.logging_config import get_logger, setup_logging_from_config
from mlops_crew.models.train_model import train

logger = get_logger(__name__)

_HYDRA_CONTROL_KEYS = {"base_config_path", "run_outputs_under_hydra_dir"}


def _project_relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _to_plain_dict(config: DictConfig | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(config, DictConfig):
        container = OmegaConf.to_container(config, resolve=True)
    else:
        container = dict(config)
    if not isinstance(container, dict):
        raise ValueError("Hydra config root must be a mapping")
    return cast(dict[str, Any], container)


def deep_merge(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Return a recursive merge without mutating either input mapping."""
    merged = deepcopy(dict(base))
    for key, value in overrides.items():
        if key in _HYDRA_CONTROL_KEYS:
            continue
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _route_outputs_to_hydra_dir(config: dict[str, Any], output_dir: Path) -> None:
    artifact_root = output_dir / "artifacts"
    config["modeling"]["output_dir"] = str(artifact_root / "models")

    reports = config["reports"]
    reports["metrics_dir"] = str(artifact_root / "reports" / "metrics")
    reports["predictions_dir"] = str(artifact_root / "reports" / "predictions")
    reports["monitoring_dir"] = str(artifact_root / "reports" / "monitoring")


def build_effective_config(
    hydra_config: DictConfig | Mapping[str, Any],
    *,
    hydra_output_dir: str | Path,
    experiment_name: str = "unknown",
) -> dict[str, Any]:
    """Load the base project config and apply Hydra overrides."""
    hydra_dict = _to_plain_dict(hydra_config)
    base_config_path = hydra_dict.get("base_config_path", "configs/config.yaml")
    base_config = load_project_config(str(base_config_path))

    effective_config = deep_merge(base_config, hydra_dict)
    output_dir = resolve_project_path(hydra_output_dir)
    if bool(hydra_dict.get("run_outputs_under_hydra_dir", True)):
        _route_outputs_to_hydra_dir(effective_config, output_dir)

    tracking = effective_config.setdefault("tracking", {})
    tracking["run_name"] = f"hydra-{experiment_name}"
    tracking["effective_config_path"] = str(output_dir / "effective_config.yaml")
    tags = tracking.setdefault("tags", {})
    tags.update(
        {
            "config_source": "hydra",
            "hydra_experiment": experiment_name,
            "hydra_output_dir": _project_relative(output_dir),
        }
    )
    return effective_config


def write_effective_config(config: Mapping[str, Any], output_dir: str | Path) -> Path:
    """Write the exact config used for a Hydra training run."""
    path = resolve_project_path(output_dir) / "effective_config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(dict(config), file, sort_keys=False)
    return path


@hydra.main(config_path="../../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    runtime = HydraConfig.get().runtime
    experiment_name = str(runtime.choices.get("experiment", "unknown"))
    output_dir = resolve_project_path(runtime.output_dir)
    config = build_effective_config(
        cfg,
        hydra_output_dir=output_dir,
        experiment_name=experiment_name,
    )
    effective_config_path = write_effective_config(config, output_dir)
    config["tracking"]["effective_config_path"] = str(effective_config_path)

    setup_logging_from_config(config)
    logger.info("Starting Hydra experiment %s", experiment_name)
    logger.info("Hydra artifacts will be written under %s", output_dir)
    train(config)
    logger.info("Hydra experiment complete: %s", experiment_name)


if __name__ == "__main__":
    main()
