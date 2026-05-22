# Hydra Configs

This directory is intentionally separate from `configs/`.

`configs/config.yaml` remains the source of truth for the normal DVC pipeline.
The DVC stages, Makefile `train` target, notebooks, and production-style
reproduction flow all read that file directly.

`conf/` is created for the Phase 2 Hydra requirement. It contains a small overlay
config that loads `configs/config.yaml`, applies experiment overrides, and runs
the same training function through `python -m mlops_crew.train_hydra`.

Run the demo with:

```bash
make hydra-demo
```

That command runs the same Hydra training entrypoint twice, once with
`experiment=phase2_default` and once with `experiment=phase2_experimental`, and
logs both runs to MLflow.