#!/usr/bin/env bash
# Deploy the Iris API container locally and run a /health smoke check.
# Usage:
#   ./scripts/deploy.sh
#   ./scripts/deploy.sh --image you/mlops-iris:latest --port 9000
#   ./scripts/deploy.sh --name iris-api --model ./artifacts/model/model.joblib
#   ./scripts/deploy.sh --platform linux/amd64   # helpful on Apple Silicon
#
# Flags:
#   --image <ref>     Docker image reference (default: gregariousgovind/mlops-iris:latest)
#   --name <name>     Container name (default: iris-api)
#   --port <port>     Host port to map to container :8000 (default: 8000)
#   --model <path>    Set MODEL_PATH env inside the container
#   --no-pull         Skip docker pull (use local image only)
#   --env KEY=VAL     Extra env var (can repeat)
#   --network <net>   Docker network to attach (optional)
#   --platform <p>    Platform to run (e.g., linux/amd64 or linux/arm64)
#   --help            Show help and exit
#
# Behavior:
#   - Replaces an existing container with the same name.
#   - Waits up to 25s for http://127.0.0.1:<port>/health to return 200.
#   - Prints container logs if health check fails, then exits non-zero.

set -euo pipefail

IMAGE="gregariousgovind/mlops-iris:latest"
NAME="iris-api"
PORT="8000"
MODEL_PATH=""
PULL=1
NETWORK=""
PLATFORM=""
declare -a EXTRA_ENV=()

print_help() {
  awk 'NR==1,/# Behavior:/{print}' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image)    IMAGE="$2"; shift 2 ;;
    --name)     NAME="$2"; shift 2 ;;
    --port)     PORT="$2"; shift 2 ;;
    --model)    MODEL_PATH="$2"; shift 2 ;;
    --no-pull)  PULL=0; shift ;;
    --env)      EXTRA_ENV+=("$2"); shift 2 ;;
    --network)  NETWORK="$2"; shift 2 ;;
    --platform) PLATFORM="$2"; shift 2 ;;
    --help|-h)  print_help; exit 0 ;;
    *) echo "Unknown option: $1"; exit 2 ;;
  esac
done

echo "[deploy] image=${IMAGE} name=${NAME} port=${PORT} ${PLATFORM:+platform=${PLATFORM}}"

if [[ "$PULL" -eq 1 ]]; then
  echo "[deploy] pulling image ..."
  if [[ -n "$PLATFORM" ]]; then
    docker pull --platform "$PLATFORM" "$IMAGE"
  else
    docker pull "$IMAGE"
  fi
else
  echo "[deploy] skipping docker pull (--no-pull)"
  if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "[deploy] ❌ image '$IMAGE' not found locally and --no-pull set; aborting."
    echo "       Try: docker pull ${PLATFORM:+--platform $PLATFORM }$IMAGE"
    exit 1
  fi
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true

RUN_ARGS=(run -d --rm --name "$NAME" -p "${PORT}:8000")
if [[ -n "$PLATFORM" ]]; then
  RUN_ARGS+=(--platform "$PLATFORM")
fi
if [[ -n "$NETWORK" ]]; then
  RUN_ARGS+=(--network "$NETWORK")
fi

# Env vars
if [[ -n "$MODEL_PATH" ]]; then
  RUN_ARGS+=(-e "MODEL_PATH=$MODEL_PATH")
fi
if ((${#EXTRA_ENV[@]})); then
  for kv in "${EXTRA_ENV[@]}"; do
    RUN_ARGS+=(-e "$kv")
  done
fi

echo "[deploy] starting container ..."
docker "${RUN_ARGS[@]}" "$IMAGE"

# Health check loop
echo "[deploy] waiting for health: http://127.0.0.1:${PORT}/health"
HEALTH_URL="http://127.0.0.1:${PORT}/health"
ATTEMPTS=25
SLEEP=1

ok=0
for i in $(seq 1 "$ATTEMPTS"); do
  if curl -sf "$HEALTH_URL" >/dev/null; then
    ok=1; break
  fi
  sleep "$SLEEP"
done

if [[ "$ok" -ne 1 ]]; then
  echo "[deploy] ❌ health check failed after $((ATTEMPTS*SLEEP))s"
  echo "---- container logs ----"
  docker logs "$NAME" || true
  echo "-------------------------"
  exit 1
fi

echo "[deploy] ✅ healthy at ${HEALTH_URL}"
echo "[deploy] try: curl -s ${HEALTH_URL}"
echo "[deploy] try: curl -s -X POST http://127.0.0.1:${PORT}/predict -H 'content-type: application/json' -d '{\"sepal_length\":5.1,\"sepal_width\":3.5,\"petal_length\":1.4,\"petal_width\":0.2}'"
