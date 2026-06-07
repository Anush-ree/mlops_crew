# Debugging Examples

This note addresses the Phase 2 debugging rubric item without changing normal
runtime behavior.

## Prediction Debug Session With `pdb`

Use Python's debugger when a prediction path fails or returns an unexpected
label. This runs the existing batch prediction CLI and drops into `pdb` before
the module starts:

```bash
python -m pdb -m mlops_crew.models.predict_model \
  --model-path models/best_model.joblib \
  --input data/processed/test.csv \
  --output reports/predictions/debug_predictions.csv
```

Useful commands once inside `pdb`:

- `b src/mlops_crew/models/predict_model.py:17` sets a breakpoint at
  `predict`.
- `c` continues to the breakpoint.
- `n` steps to the next line.
- `p model_path`, `p input_path`, and `p data.head()` inspect runtime state.
- `q` exits.

For a targeted code breakpoint during local debugging, temporarily add:

```python
breakpoint()
```

inside `predict_model.predict` after the input CSV is read. Remove it before
committing production code.
