# MLOps Iris: Build, Track, Package, Deploy & Monitor

**Dataset:** Iris (classification)  
**Stack:** Git/GitHub, (DVC optional), MLflow, FastAPI, Docker, GitHub Actions, Logging+SQLite, Prometheus metrics

## Architecture
1. **Data** → `src/data.py` saves Iris to CSV (`data/raw`, `data/processed`) [DVC optional].
2. **Training** → `src/train.py` trains LogisticRegression & RandomForest, logs params/metrics to MLflow, registers best model `iris_clf` (Production), and exports `artifacts/model/model.joblib` for packaging.
3. **Serving** → `api/main.py` exposes `/predict`, `/health`, `/metrics`; uses **Pydantic** for validation; logs to file + SQLite.
4. **Container** → `Dockerfile` bakes API and model; `uvicorn` serves port 8000.
5. **CI/CD** → `.github/workflows/ci.yml` lints, tests, builds, pushes image to Docker Hub.
6. **Monitoring** → `/metrics` exposes Prometheus counters & histograms (requests, latency).

## Local Run
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/data.py
./scripts/run_mlflow.sh     # new terminal tab
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
python src/train.py
uvicorn api.main:app --port 8000
```

**Test prediction:**
```bash
curl -s -X POST http://127.0.0.1:8000/predict   -H "Content-Type: application/json"   -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

## Docker
```bash
docker build -t <you>/mlops-iris:latest .
docker run --rm -p 8000:8000 <you>/mlops-iris:latest
```

## CI/CD (GitHub Actions)
- Add repo secrets: `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`.
- Push to `main` → lints, tests, builds and pushes image tags (`latest` and commit SHA).

## Logs & Monitoring
- File logs: `logs/app.log`
- SQLite: `logs/predictions.db` (table: `requests` with features, prediction, latency)
- Metrics: `GET /metrics` (Prometheus format)

## Notes
- API will **fallback** to a quick-initialized model if `artifacts/model/model.joblib` is missing (useful for CI tests).
- Prefer running MLflow server with SQLite backend: `./scripts/run_mlflow.sh` (UI at `http://127.0.0.1:5000`).