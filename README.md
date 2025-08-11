# MLOps Iris — Build, Track, Package, Deploy & Monitor

**Dataset:** Iris (classification)  
**Stack:** Git/GitHub · DVC · MLflow · FastAPI · Docker · GitHub Actions · Logging+SQLite · Prometheus

---

## How this meets the assignment

- **Part 1 — Repo & Data Versioning (4/4)**
  - Clean structure: `src/`, `api/`, `tests/`, `artifacts/`, `data/`, `scripts/`, `.github/`.
  - `src/data.py` writes  
    `data/raw/iris.csv`, `data/processed/iris.csv`, `data/schema.json`, `data/metadata.json`.
  - DVC pipeline: `dvc.yaml`; local remote via `make dvc-remote`; `make dvc-repro/push/pull`.

- **Part 2 — Model + Experiment Tracking (6/6)**
  - Trains **LogisticRegression** & **RandomForest**; logs params/metrics/artifacts in MLflow.
  - Best model exported to `artifacts/model/model.joblib`.
  - **Model Registry**: when MLflow server is running, best model is registered as `iris_clf`
    and given alias **production** (or stage “Production” if aliases not supported).

- **Part 3 — API & Docker (4/4)**
  - FastAPI + Pydantic v2: `/predict`, `/health`, `/metrics` (Prometheus).
  - Slim Dockerfile with lockfile installs; exposes port **8000**.

- **Part 4 — CI/CD (6/6)**
  - GitHub Actions: lockfile install, lint, tests, build & push image on `main`,
    then smoke-tests `/health`.

- **Part 5 — Logging & Monitoring (4/4)**
  - Rotating file logs: `logs/app.log`.
  - SQLite request log: `logs/predictions.db`.
  - Prometheus metrics at `/metrics` (counters + latency histogram).

- **Part 6 — Summary + Demo (2/2)**
  - This README is the summary; see “Demo script” below for a 5-minute walkthrough.

- **Bonus (+ up to 4)**
  - ✅ Input validation (Pydantic v2)
  - ✅ Prometheus endpoint
  - ✳️ Optional Prometheus/Grafana compose + retrain trigger available on request

---

## New system setup (tested on Python 3.11.11)

**Prereqs:** Git, Make, Python 3.11 on PATH (`python3`), Docker (optional), `curl`.

```bash
# 0) Clone & enter
git clone <your-repo-url> mlops-iris
cd mlops-iris

# 1) Create venv and install EXACT locked deps (preferred)
make setup-lock     # or: make setup (installs from requirements.txt)

# 2) Generate Part-1 data artifacts
make data
ls -1 data/raw/iris.csv data/processed/iris.csv data/schema.json data/metadata.json

# 3) Run tests & lint
make test
make lint
````

---

## One-command bootstrap (local)

This does: **venv → deps → data → tests → MLflow → train → API**.

```bash
./scripts/bootstrap.sh
# or via make alias:
make bootstrap
```

Options:

```bash
./scripts/bootstrap.sh --api-port 9000
./scripts/bootstrap.sh --mlflow-port 5000
./scripts/bootstrap.sh --mlflow-uri http://127.0.0.1:5002  # use an existing MLflow
./scripts/bootstrap.sh --skip-tests
./scripts/bootstrap.sh --skip-mlflow
./scripts/bootstrap.sh --train-only
```

MLflow logs: `logs/mlflow_bootstrap.log`. PID: `logs/mlflow.pid`.

---

## MLflow (tracking + registry)

**Auto-port helper (prints the URL):**

```bash
./scripts/run_mlflow.sh
# sample output line:
# [mlflow] UI            : http://127.0.0.1:5002
export MLFLOW_TRACKING_URI=http://127.0.0.1:5002
```

**Fixed port 5000:**

```bash
make mlflow
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

**Train & register (works with or without UI):**

```bash
python src/train.py
# - logs runs/metrics/artifacts
# - exports best model to artifacts/model/model.joblib
# - registers 'iris_clf' and sets alias 'production' (if registry available)
```

Open the printed MLflow URL in your browser to inspect runs and the **Model Registry**.

---

## Run the API locally

```bash
# Uses artifacts/model/model.joblib (created by train.py)
make api
# or explicitly:
.venv/bin/uvicorn api.main:app --reload --port 8000
```

Try it:

```bash
curl -s http://127.0.0.1:8000/health | jq
curl -s -X POST http://127.0.0.1:8000/predict -H "content-type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}' | jq
curl -s http://127.0.0.1:8000/metrics | head
```

---

## Docker (local)

```bash
# Build image
docker build -t <your-dockerhub-username>/mlops-iris:dev .

# Run container
docker run --rm -p 9000:8000 <your-dockerhub-username>/mlops-iris:dev
curl -s http://127.0.0.1:9000/health | jq
```

> On Apple Silicon, CI images are multi-arch. If you pull an amd64-only image, run with
> `--platform linux/amd64` or rebuild locally.

---

## DVC (optional for Iris, enabled here)

```bash
make dvc-init dvc-remote
make dvc-repro
make dvc-status
make dvc-push
make dvc-pull
```

---

## Make targets you’ll use a lot

```bash
make setup-lock   # venv + locked deps
make data         # Part-1 artifacts
make test         # run tests (ensures data exists)
make mlflow       # MLflow on :5000
make api          # FastAPI on :8000
make docker-build # build Docker image
make docker-run   # run image locally
make bootstrap    # one-command local demo
```

---

## Troubleshooting

* **`mlflow: command not found`**
  Run `make setup-lock`, or start via `make mlflow` (uses `.venv/bin/mlflow`).

* **Port busy (5000/8000)**
  Use a different port: `./scripts/run_mlflow.sh --port 5010`, or run API with `--port 9000`.

* **API says “Model not loaded”**
  Ensure `python src/train.py` ran and `artifacts/model/model.joblib` exists.
