FROM python:3.11-slim

# --- Runtime env ----------------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    MODEL_PATH=/app/artifacts/model/model.joblib

WORKDIR /app

# --- System deps (runtime only) -------------------------------------------------
# scikit-learn wheels often need OpenMP at runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# --- Python deps (deterministic via lock) --------------------------------------
COPY requirements.lock.txt .
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.lock.txt

# --- App code & artifacts ------------------------------------------------------
COPY src ./src
COPY api ./api
# The CI job ensures this exists (python src/train.py), so we can bake a real model
COPY artifacts/model ./artifacts/model

# Pre-create logs dir for the app
RUN mkdir -p /app/logs

# --- Security: run as non-root -------------------------------------------------
RUN useradd -m appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE ${PORT}

# --- Entrypoint ----------------------------------------------------------------
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
