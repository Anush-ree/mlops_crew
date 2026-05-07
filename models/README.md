# Models Directory

Store trained models, serialized artifacts, and predictions here.

## What Goes Here

- Trained/serialized models (`.pkl`, `.joblib`, `.h5`, `.pth`, etc.)
- The fitted sklearn pipelines (`dummy.joblib`, `logistic_regression.joblib`)
- `best_model.joblib`, copied from the best validation model

## Best Practices

- **Never commit** large model files to Git
- Use DVC/S3 to version and manage model artifacts
- Document model architecture, hyperparameters, and performance metrics
- Keep metrics and row-level predictions under `reports/`
