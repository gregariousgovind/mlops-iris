# ---- Config & tool paths ------------------------------------------------------
SHELL := /bin/bash

# Virtualenv tools (created by `make setup`)
PYTHON := .venv/bin/python
PIP := .venv/bin/pip
UVICORN := .venv/bin/uvicorn
MLFLOW := .venv/bin/mlflow
FLAKE8 := .venv/bin/flake8
PYTEST := .venv/bin/pytest
DVC := .venv/bin/dvc

# App/Ports
API_PORT := 8000
MLFLOW_HOST := 127.0.0.1
MLFLOW_PORT := 5000

# Docker
DOCKER_IMAGE := gregariousgovind/mlops-iris:latest

# ---- Meta ---------------------------------------------------------------------
.PHONY: help setup data mlflow train api lint test \
        docker-build docker-run \
        dvc-init dvc-remote dvc-autostage dvc-repro dvc-push dvc-pull dvc-status \
        test-part1 qa-part1 clean

# Default goal
help:
	@echo ""
	@echo "Targets:"
	@echo "  setup          Create venv and install dependencies"
	@echo "  data           Generate data files (raw/processed/schema/metadata)"
	@echo "  mlflow         Run MLflow tracking server (SQLite backend)"
	@echo "  train          Train models and log to MLflow"
	@echo "  api            Run FastAPI locally"
	@echo "  lint           Run flake8"
	@echo "  test           Run full pytest"
	@echo "  docker-build   Build Docker image ($(DOCKER_IMAGE))"
	@echo "  docker-run     Run Docker image on port $(API_PORT)"
	@echo "  dvc-init       Initialize DVC (idempotent)"
	@echo "  dvc-remote     Configure local DVC remote (idempotent)"
	@echo "  dvc-autostage  Enable DVC autostage"
	@echo "  dvc-repro      Reproduce data pipeline"
	@echo "  dvc-push       Push artifacts to DVC remote"
	@echo "  dvc-pull       Pull artifacts from DVC remote"
	@echo "  dvc-status     Show DVC status (with remote)"
	@echo "  test-part1     Run Part-1 validation tests only"
	@echo "  qa-part1       Quick DVC status + push (Part-1 QA)"
	@echo "  clean          Remove caches/pyc (keeps venv)"
	@echo ""

# ---- Environment & installs ---------------------------------------------------
setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# ---- Data / MLflow / Training / API ------------------------------------------
data:
	$(PYTHON) src/data.py

mlflow:
	$(MLFLOW) server --backend-store-uri sqlite:///mlflow.db \
	  --default-artifact-root ./mlruns --host $(MLFLOW_HOST) --port $(MLFLOW_PORT)
	# or: . .venv/bin/activate && ./scripts/run_mlflow.sh

train:
	MLFLOW_TRACKING_URI=http://$(MLFLOW_HOST):$(MLFLOW_PORT) $(PYTHON) src/train.py

api:
	$(UVICORN) api.main:app --reload --port $(API_PORT)

# ---- Quality ------------------------------------------------------------------
lint:
	$(FLAKE8)

test:
	$(PYTEST) -q

# Focused Part-1 test (schema/metadata/determinism)
test-part1:
	$(PYTEST) -q tests/test_part1_validation.py

# Quick Part-1 DVC QA: show status + push
qa-part1:
	-$(DVC) status -c
	-$(DVC) push
	@echo "If status showed 'up to date', Part-1 artifacts are synced."

# ---- Docker -------------------------------------------------------------------
docker-build:
	docker build -t $(DOCKER_IMAGE) .

docker-run:
	docker run --rm -p $(API_PORT):8000 $(DOCKER_IMAGE)

# ---- DVC (idempotent) ---------------------------------------------------------
dvc-init:
	@{ [ -d .dvc ] && echo "DVC already initialized. Skipping."; } || $(DVC) init

dvc-remote:
	@mkdir -p ../mlops-iris-storage
	@{ $(DVC) remote list | grep -q '^localremote'; } \
		&& echo "DVC remote 'localremote' already exists. Skipping." \
		|| $(DVC) remote add -d localremote ../mlops-iris-storage

dvc-autostage:
	$(DVC) config core.autostage true

dvc-repro:
	$(DVC) repro

dvc-push:
	$(DVC) push

dvc-pull:
	$(DVC) pull

dvc-status:
	$(DVC) status -c

# ---- Housekeeping -------------------------------------------------------------
clean:
	@find . -name "__pycache__" -type d -exec rm -rf {} +
	@find . -name "*.pyc" -delete
	@echo "Cleaned caches."
