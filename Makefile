.PHONY: install dev data source-manifest transformer-data train hydra-train hydra-demo predict plot latency divergence profile-train profile-predict mlflow-ui repro test lint format clean docker-train docker-predict

install:
	pip install -U pip
	pip install -r requirements.txt
	pip install -e .

dev: install
	pip install -r requirements_dev.txt
	pre-commit install

# Run the full local data pipeline
data:
	python -m mlops_crew.data.make_dataset

# Build source metadata used for Phase 2 divergence reports
source-manifest:
	python -m mlops_crew.data.source_manifest

transformer-data:
	python -m mlops_crew.data.export_transformer_dataset

# Train every model listed in configs/config.yaml under modeling.models
train:
	python -m mlops_crew.models.train_model

# Train through Hydra using conf/ overrides. Outputs go under ignored outputs/hydra/.
hydra-train:
	python -m mlops_crew.train_hydra

hydra-demo:
	python -m mlops_crew.train_hydra experiment=phase2_default
	python -m mlops_crew.train_hydra experiment=phase2_experimental

# Score the test set with the best saved model
predict:
	python -m mlops_crew.models.predict_model

plot:
	python -m mlops_crew.evaluation.plot_model_comparison

latency:
	python -m mlops_crew.monitoring.inference_latency

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
	ruff check --no-cache .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

clean:
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache -o -name .mypy_cache -o -name "*.egg-info" \) -exec rm -rf {} + 2>/dev/null || true
