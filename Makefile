.PHONY: install dev data train predict repro test lint format clean

install:
	pip install -U pip
	pip install -r requirements.txt
	pip install -e .

dev: install
	pip install -r requirements_dev.txt
	pre-commit install

# Run the full data pipeline (sample -> clean -> split -> validate)
data:
	python -m mlops_crew.data.make_dataset

# Train every model listed in configs/config.yaml under modeling.models
train:
	python -m mlops_crew.train_model

# Score the test set with the best saved model
predict:
	python -m mlops_crew.predict_model

# Reproduce the whole DVC pipeline end to end
repro:
	dvc repro

test:
	pytest tests/

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

clean:
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache -o -name .mypy_cache -o -name "*.egg-info" \) -exec rm -rf {} + 2>/dev/null || true
