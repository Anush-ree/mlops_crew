# Data Directory

Store all data files for the phishing_email_detection project here.

## Structure

- **`raw/`** — Original, immutable data as received. Never modify files here.
- **`interim/`** — Reproducible phase partitions and source manifest data.
- **`processed/`** — Cleaned train/validation/test CSVs ready for modeling.
- **`processed/transformer/`** — JSONL splits for later transformer fine-tuning.

## Best Practices

- Use **DVC** to version large data files instead of Git
- Track `.dvc` files in Git; store actual data remotely
- **Never commit** large data files directly to Git
- Document data sources and transformations in notebooks or scripts
