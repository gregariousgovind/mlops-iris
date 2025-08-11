#!/usr/bin/env bash
# End-to-end local bootstrap for this repo:
#  - Creates venv and installs deps (prefers requirements.lock.txt)
#  - Generates data artifacts
#  - Runs tests
#  - Starts MLflow (auto-picks a free port) and exports MLFLOW_TRACKING_URI
#  - Trains models, registers best (if registry available)
#  - Starts the FastAPI server (foreground), and cleans up MLflow on exit
#
# Usage:
#   ./scripts/bootstrap.sh
#   ./scripts/bootstrap.sh --api-port 9000
#   ./scripts/bootstrap.sh --mlflow-port 5000
#   ./scripts/bootstrap.sh --mlflow-uri http://127.0.0.1:5002   # use an existing MLflow
#   ./scripts/bootstrap.sh --skip-tests
#   ./scripts/bootstrap.sh --skip-mlflow                        # train with local file store (no registry)
#   ./scripts/bootstrap.sh --train-only                         # stop after training
#   ./scripts/bootstrap.sh --no-api                             # alias of --train-only
#
# Logs:
#   logs/mlflow_bootstrap.log  (MLflow server output)
#   logs/mlflow.pid            (MLflow PID)
#
# Notes:
# - This script is for local dev; CI does not depend on it.

set -euo pipefail

# Ensure we run from repo root (script can be called from anywhere)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

API_PORT=8000
MLFLOW_PORT=""
MLFLOW_URI_OVERRIDE=""
RUN_TESTS=1
START_MLFLOW=1
TRAIN_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --api-port)      API_PORT="${2:?}"; shift 2 ;;
    --mlflow-port)   MLFLOW_PORT="${2:?}"; shift 2 ;;
    --mlflow-uri)    MLFLOW_URI_OVERRIDE="${2:?}"; START_MLFLOW=0; shift 2 ;;
    --skip-tests)    RUN_TESTS=0; shift ;;
    --skip-mlflow)   START_MLFLOW=0; shift ;;
    --train-only|--no-api) TRAIN_ONLY=1; shift ;;
    *) echo "Unknown option: $1"; exit 2 ;;
  esac
done

# ---- Resolve Python + venv paths ---------------------------------------------
PY_CMD="$(command -v python3 || true)"
if [[ -z "$PY_CMD" ]]; then PY_CMD="$(command -v python || true)"; fi
if [[ -z "$PY_CMD" ]]; then PY_CMD="$(command -v py || true)"; fi
if [[ -z "$PY_CMD" ]]; then
  echo "[bootstrap] No Python found on PATH. Install Python 3.10+."; exit 1
fi

if [[ "${OS:-}" == "Windows_NT" ]]; then
  VENV_BIN=".venv/Scripts"
else
  VENV_BIN=".venv/bin"
fi

PIP="$VENV_BIN/pip"
PYTHON="$VENV_BIN/python"
UVICORN="$VENV_BIN/uvicorn"
PYTEST="$VENV_BIN/pytest"

mkdir -p logs

# ---- Cleanup handler (stop MLflow if we started it) ---------------------------
cleanup() {
  if [[ -f logs/mlflow.pid ]]; then
    local pid
    pid="$(cat logs/mlflow.pid || true)"
    if [[ -n "$pid" ]] && ps -p "$pid" >/dev/null 2>&1; then
      echo "[bootstrap] Stopping MLflow (pid=$pid) ..."
      kill "$pid" >/dev/null 2>&1 || true
      # give it a moment then hard kill if needed
      sleep 1
      ps -p "$pid" >/dev/null 2>&1 && kill -9 "$pid" >/dev/null 2>&1 || true
    fi
    rm -f logs/mlflow.pid
  fi
}
trap cleanup EXIT INT TERM

# ---- Step 1: venv + deps -----------------------------------------------------
if [[ ! -x "$PYTHON" ]]; then
  echo "[bootstrap] Creating virtualenv ..."
  "$PY_CMD" -m venv .venv
fi

REQ_FILE="requirements.lock.txt"
if [[ ! -f "$REQ_FILE" ]]; then
  REQ_FILE="requirements.txt"
fi

echo "[bootstrap] Installing deps from $REQ_FILE ..."
"$PIP" install --upgrade pip >/dev/null
"$PIP" install -r "$REQ_FILE"

# ---- Step 2: data artifacts --------------------------------------------------
echo "[bootstrap] Generating data artifacts ..."
"$PYTHON" src/data.py

# ---- Step 3: tests -----------------------------------------------------------
if [[ "$RUN_TESTS" -eq 1 ]]; then
  echo "[bootstrap] Running tests ..."
  if ! "$PYTEST" -q; then
    echo "[bootstrap] Tests failed"; exit 1
  fi
else
  echo "[bootstrap] Skipping tests (--skip-tests)"
fi

# ---- Step 4: MLflow server (optional) ----------------------------------------
MLFLOW_URL=""
if [[ -n "$MLFLOW_URI_OVERRIDE" ]]; then
  export MLFLOW_TRACKING_URI="$MLFLOW_URI_OVERRIDE"
  echo "[bootstrap] Using provided MLflow URI: $MLFLOW_TRACKING_URI"
elif [[ "$START_MLFLOW" -eq 1 ]]; then
  echo "[bootstrap] Starting MLflow server ..."
  : > logs/mlflow_bootstrap.log
  rm -f logs/mlflow.pid

  if [[ -n "$MLFLOW_PORT" ]]; then
    ./scripts/run_mlflow.sh --port "$MLFLOW_PORT" > logs/mlflow_bootstrap.log 2>&1 &
  else
    ./scripts/run_mlflow.sh > logs/mlflow_bootstrap.log 2>&1 &
  fi
  MLFLOW_PID=$!
  echo "$MLFLOW_PID" > logs/mlflow.pid

  echo -n "[bootstrap] Waiting for MLflow UI URL"
  for i in {1..40}; do
    if grep -q "UI[[:space:]]*:" logs/mlflow_bootstrap.log; then
      MLFLOW_URL="$(grep -m1 -o 'http://127.0.0.1:[0-9]\+' logs/mlflow_bootstrap.log || true)"
      break
    fi
    echo -n "."
    sleep 1
  done
  echo

  if [[ -n "$MLFLOW_URL" ]]; then
    export MLFLOW_TRACKING_URI="$MLFLOW_URL"
    echo "[bootstrap] MLflow UI: $MLFLOW_TRACKING_URI"
  else
    echo "[bootstrap] Could not detect MLflow URL after 40s. Check logs/mlflow_bootstrap.log"
    echo "[bootstrap] Proceeding WITHOUT registry (file:./mlruns)."
  fi
else
  echo "[bootstrap] Skipping MLflow (--skip-mlflow). Using local file store."
fi

# ---- Step 5: training ---------------------------------------------------------
echo "[bootstrap] Training models (and registering best if registry available) ..."
"$PYTHON" src/train.py

if [[ "$TRAIN_ONLY" -eq 1 ]]; then
  echo "[bootstrap] Train-only complete."
  exit 0
fi

# ---- Step 6: API --------------------------------------------------------------
echo "[bootstrap] Starting API on :$API_PORT ..."
# Run API in foreground but keep shell to handle cleanup on Ctrl-C
"$UVICORN" api.main:app --reload --host 0.0.0.0 --port "$API_PORT"
