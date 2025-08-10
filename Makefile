.PHONY: setup data train api docker-build docker-run lint test mlflow

setup:
	python3 -m venv .venv && . .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

data:
	python src/data.py

mlflow:
	./scripts/run_mlflow.sh

train:
	MLFLOW_TRACKING_URI=http://127.0.0.1:5000 python src/train.py

api:
	uvicorn api.main:app --reload --port 8000

lint:
	flake8

test:
	pytest -q

docker-build:
	docker build -t gregariousgovind/mlops-iris:latest .

docker-run:
	docker run --rm -p 8000:8000 gregariousgovind/mlops-iris:latest