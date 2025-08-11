# ---- Config & tool paths ------------------------------------------------------
SHELL := /bin/bash

# Find a Python interpreter (prefer python3)
PY_CMD := $(or $(shell command -v python3),$(shell command -v python),$(shell command -v py))
ifeq ($(PY_CMD),)
$(error No Python interpreter found. Install Python 3.10+ and ensure 'python3' or 'python' is on PATH.)
endif

# ---- Virtualenv paths (cross-OS) ---------------------------------------------
ifeq ($(OS),Windows_NT)
  VENV_BIN := .venv/Scripts
else
  VENV_BIN := .venv/bin
endif

PYTHON  := $(VENV_BIN)/python
PIP     := $(VENV_BIN)/pip
UVICORN := $(VENV_BIN)/uvicorn
MLFLOW  := $(VENV_BIN)/mlflow
FLAKE8  := $(VENV_BIN)/flake8
PYTEST  := $(VENV_BIN)/pytest
DVC     := $(VENV_BIN)/dvc

# Auto-select requirements file: prefer lock if present
REQUIREMENTS_FILE := $(if $(wildcard requirements.lock.txt),requirements.lock.txt,requirements.txt)

# App/Ports
API_PORT := 8000
MLFLOW_HOST := 127.0.0.1
MLFLOW_PORT := 5000

# Docker
DOCKER_IMAGE := gregariousgovind/mlops-iris:latest

# ---- Meta ---------------------------------------------------------------------
.PHONY: help setup setup-lock lock update-lock ci-install \
        data mlflow train api lint test \
        docker-build docker-run \
        dvc-init dvc-remote dvc-autostage dvc-repro dvc-push dvc-pull dvc-status \
        test-part1 qa-part1 clean

help:
	@echo ""
	@echo "Targets:"
	@echo "  setup           Create venv and install deps (prefers requirements.lock.txt if present)"
	@echo "  setup-lock      Create venv and install EXACT deps from requirements.lock.txt"
	@echo "  lock            Install ranges, then freeze to requirements.lock.txt"
	@echo "  update-lock     Alias for 'lock' (refresh lock after changing requirements.txt)"
	@echo "  ci-install      Install EXACT deps from lock (for local CI parity)"
	@echo "  data            Generate data files (raw/processed/schema/metadata)"
	@echo "  mlflow          Run MLflow tracking server (SQLite backend)"
	@echo "  train           Train models and log to MLflow"
	@echo "  api             Run FastAPI locally"
	@echo "  lint            Run flake8"
	@echo "  test            Run full pytest (ensures data exists)"
	@echo "  docker-build    Build Docker image ($(DOCKER_IMAGE))"
	@echo "  docker-run      Run Docker image on port $(API_PORT)"
	@echo "  dvc-init        Initialize DVC (idempotent)"
	@echo "  dvc-remote      Configure local DVC remote (idempotent)"
	@echo "  dvc-autostage   Enable DVC autostage"
	@echo "  dvc-repro       Reproduce data pipeline"
	@echo "  dvc-push        Push artifacts to DVC remote"
	@echo "  dvc-pull        Pull artifacts from DVC remote"
	@echo "  dvc-status      Show DVC status (with remote)"
	@echo "  test-part1      Run Part-1 validation tests only (ensures data exists)"
	@echo "  qa-part1        Quick DVC status + push (Part-1 QA)"
	@echo "  clean           Remove caches/pyc (keeps venv)"
	@echo ""

# ---- Environment & installs ---------------------------------------------------
setup:
	@echo ">> Using $(REQUIREMENTS_FILE) for installation"
	$(PY_CMD) -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r $(REQUIREMENTS_FILE)

setup-lock:
	@test -f requirements.lock.txt || (echo "requirements.lock.txt not found. Run 'make lock' first." && exit 1)
	$(PY_CMD) -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.lock.txt

lock:
	$(PY_CMD) -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) freeze | sed '/^-e /d' > requirements.lock.txt
	@echo ">> Wrote requirements.lock.txt"

update-lock: lock

ci-install:
	@test -f requirements.lock.txt || (echo "requirements.lock.txt not found. Run 'make lock' first." && exit 1)
	$(PY_CMD) -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.lock.txt

# ---- Data / MLflow / Training / API ------------------------------------------
data:
	$(PYTHON) src/data.py

mlflow:
	$(MLFLOW) server --backend-store-uri sqlite:///mlflow.db \
	  --default-artifact-root ./mlruns --host $(MLFLOW_HOST) --port $(MLFLOW_PORT)

train:
	MLFLOW_TRACKING_URI=http://$(MLFLOW_HOST):$(MLFLOW_PORT) $(PYTHON) src/train.py

api:
	$(UVICORN) api.main:app --reload --port $(API_PORT)

# ---- Quality ------------------------------------------------------------------
lint:
	$(FLAKE8)

# Ensure data exists before running tests
test: data
	$(PYTEST) -q

test-part1: data
	$(PYTEST) -q tests/test_part1_validation.py

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
