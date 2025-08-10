#!/usr/bin/env bash
set -euo pipefail
IMAGE="${1:-gregariousgovind/mlops-iris:latest}"
docker pull "$IMAGE"
docker rm -f iris-api || true
docker run -d --name iris-api -p 8000:8000 "$IMAGE"
echo "Running on http://127.0.0.1:8000"