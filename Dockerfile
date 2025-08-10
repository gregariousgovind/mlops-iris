FROM python:3.11-slim

# --- Runtime env ----------------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # App defaults (FastAPI/uvicorn uses this; you can override with -e PORT=...)
    PORT=8000 \
    # Our app reads this; override if needed
    MODEL_PATH=/app/artifacts/model/model.joblib

WORKDIR /app

# --- System deps (runtime only) -------------------------------------------------
# libgomp1 is often required by scikit-learn’s OpenMP-enabled wheels
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# --- Python deps (deterministic via lock) --------------------------------------
# If you ever need a fallback to requirements.txt, copy both and conditionally install.
COPY requirements.lock.txt .
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.lock.txt

# --- App code & artifacts ------------------------------------------------------
COPY src ./src
COPY api ./api
COPY artifacts/model ./artifacts/model

# Pre-create logs dir so a non-root user can write
RUN mkdir -p /app/logs

# --- Security: run as non-root -------------------------------------------------
# Create an unprivileged user and take ownership of the app dir
RUN useradd -m appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE ${PORT}

# --- Entrypoint ----------------------------------------------------------------
# Use python -m to ensure module resolution via PATH
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
