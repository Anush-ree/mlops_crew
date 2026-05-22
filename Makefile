.PHONY: install dev data train predict repro test lint format clean docker-train docker-predict

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
	python -m mlops_crew.models.train_model

# Score the test set with the best saved model
predict:
	python -m mlops_crew.models.predict_model

# Build and run the training image with host-mounted DVC data
docker-train:
	docker build -f train.dockerfile . -t train:latest
	docker run --rm \
		-e MLOPS_CREW_PROJECT_ROOT=/app \
		-v "$$(pwd)/data:/app/data" \
		-v "$$(pwd)/configs:/app/configs" \
		train:latest

# Build and run the prediction image with host-mounted DVC data and saved models
docker-predict:
	docker build -f predict.dockerfile . -t predict:latest
	docker run --rm \
		-e MLOPS_CREW_PROJECT_ROOT=/app \
		-v "$$(pwd)/data:/app/data" \
		-v "$$(pwd)/configs:/app/configs" \
		-v "$$(pwd)/models:/app/models" \
		-v "$$(pwd)/reports:/app/reports" \
		predict:latest \
		--model-path /app/models/best_model.joblib \
		--input /app/data/processed/test.csv \
		--output /app/reports/predictions/batch_predictions.csv

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
