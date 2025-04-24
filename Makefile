.PHONY: install test lint format clean train api docker-api docker-train mlflow-ui mlflow-clean

# Installing dependencies
install:
	poetry install

# Running tests
test:
	pytest

# Code linting
lint:
	flake8 pneumonia_classifier scripts tests
	mypy pneumonia_classifier scripts tests
	
# Code formatting
format:
	black pneumonia_classifier scripts tests
	isort pneumonia_classifier scripts tests

# Cleaning temporary files
clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Training the model
train:
	python scripts/train.py

# Running the API server
api:
	python scripts/api.py

# Building and running Docker image for API
docker-api:
	docker build -f docker/Dockerfile.api -t pneumonia-classifier-api:latest .
	docker run -p 8000:8000 pneumonia-classifier-api:latest

# Building and running Docker image for training
docker-train:
	docker build -f docker/Dockerfile.training -t pneumonia-classifier-training:latest .
	docker run -v $(PWD)/models:/app/models pneumonia-classifier-training:latest

# Running MLflow UI
mlflow-ui:
	python scripts/mlflow_ui.py

# Removing MLflow data
mlflow-clean:
	rm -rf mlruns

# Training with MLflow tracking
train-mlflow:
	python scripts/train.py --run_name "$(shell date +%Y-%m-%d-%H-%M-%S)" 