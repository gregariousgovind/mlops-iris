#!/usr/bin/env bash
# Start MLflow tracking server (SQLite backend).
# Usage:
#   ./scripts/run_mlflow.sh
#   ./scripts/run_mlflow.sh --port 5010
#   ./scripts/run_mlflow.sh --fresh

set -euo pipefail

BACKEND_URI="${MLFLOW_BACKEND_URI:-sqlite:///mlflow.db}"
ARTIFACT_ROOT="${MLFLOW_ARTIFACT_ROOT:-./mlruns}"
HOST="${MLFLOW_HOST:-127.0.0.1}"
PORT="${MLFLOW_PORT:-5000}"
FRESH=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fresh) FRESH=1; shift ;;
    --port)  PORT="${2:-$PORT}"; shift 2 ;;
    *) echo "Unknown option: $1" && exit 2 ;;
  esac
done

if [[ "$FRESH" -eq 1 ]]; then
  TS=$(date +%s)
  if [[ -f mlflow.db ]]; then
    mv mlflow.db "mlflow.db.bak.${TS}"
    echo "[mlflow] Backed up DB -> mlflow.db.bak.${TS}"
  fi
  if [[ -d "$ARTIFACT_ROOT" ]]; then
    rm -rf "$ARTIFACT_ROOT"
    echo "[mlflow] Cleared artifact root -> ${ARTIFACT_ROOT}"
  fi
fi

# --- find a free port without lsof (Python one-liner) ---
is_port_busy() {
python - <<'PY' "$1"
import socket, sys
port=int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    sys.exit(0 if s.connect_ex(("127.0.0.1", port))==0 else 1)
PY
# exit code: 0 => BUSY, nonzero => FREE
}

TRY=0
MAX_TRIES=20
while is_port_busy "$PORT"; do
  if [[ $TRY -ge $MAX_TRIES ]]; then
    echo "[mlflow] No free port found starting from $PORT within +$MAX_TRIES."
    echo "[mlflow] Try: ./scripts/run_mlflow.sh --port 6000"
    exit 1
  fi
  echo "[mlflow] Port $PORT is busy; trying $((PORT+1)) ..."
  PORT=$((PORT+1))
  TRY=$((TRY+1))
done

echo "[mlflow] Backend URI   : ${BACKEND_URI}"
echo "[mlflow] Artifact root : ${ARTIFACT_ROOT}"
echo "[mlflow] UI            : http://${HOST}:${PORT}"

# Best-effort DB upgrade (ok to continue if it fails; suggest --fresh)
if ! mlflow db upgrade "${BACKEND_URI}"; then
  echo "[mlflow] DB upgrade failed (likely migration mismatch)."
  echo "[mlflow] Tip: run './scripts/run_mlflow.sh --fresh' to reset local DB."
fi

exec mlflow server \
  --backend-store-uri "${BACKEND_URI}" \
  --default-artifact-root "${ARTIFACT_ROOT}" \
  --host "${HOST}" \
  --port "${PORT}"
