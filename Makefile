SHELL := /bin/bash
PYTHON := .venv/bin/python
PIP := .venv/bin/pip
UVICORN := .venv/bin/uvicorn
MLFLOW := .venv/bin/mlflow
FLAKE8 := .venv/bin/flake8
PYTEST := .venv/bin/pytest

.PHONY: setup data train api docker-build docker-run lint test mlflow

setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

data:
	$(PYTHON) src/data.py

mlflow:
	$(MLFLOW) server --backend-store-uri sqlite:///mlflow.db \
	  --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
	# or: . .venv/bin/activate && ./scripts/run_mlflow.sh

train:
	MLFLOW_TRACKING_URI=http://127.0.0.1:5000 $(PYTHON) src/train.py

api:
	$(UVICORN) api.main:app --reload --port 8000

lint:
	$(FLAKE8)

test:
	$(PYTEST) -q

docker-build:
	docker build -t gregariousgovind/mlops-iris:latest .

docker-run:
	docker run --rm -p 8000:8000 gregariousgovind/mlops-iris:latest

dvc-init:
	dvc init

# Creates a local folder outside the repo as the default DVC remote
dvc-remote:
	mkdir -p ../mlops-iris-storage
	dvc remote add -d localremote ../mlops-iris-storage

# Build data via DVC pipeline (uses dvc.yaml)
dvc-repro:
	dvc repro

# Sync data artifacts to remote / from remote
dvc-push:
	dvc push
dvc-pull:
	dvc pull