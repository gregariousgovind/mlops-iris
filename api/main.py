"""
FastAPI service for the Iris classifier.

Endpoints
---------
GET  /health    : basic liveness & model status
POST /predict   : predict label from sepal/petal measurements
GET  /metrics   : Prometheus metrics (counters + latency histogram)
GET  /docs      : Swagger UI

Features
--------
- Pydantic v2 input validation
- Rotating file logs (logs/app.log)
- SQLite request log (logs/predictions.db)
- Prometheus metrics (predict_requests_total, predict_latency_seconds)
- Reads model from MODEL_PATH (default: artifacts/model/model.joblib)
"""

from __future__ import annotations
import os
import time
from datetime import datetime
from typing import Dict, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.utils import (
    get_logger,
    init_db,
    log_prediction,
    PREDICT_REQUESTS,
    PREDICT_LATENCY,
)

# ------------------------------------------------------------------------------
# App init
# ------------------------------------------------------------------------------
app = FastAPI(title="Iris MLOps API", version="1.0.0", docs_url="/docs", redoc_url=None)
logger = get_logger()
init_db()  # ensure SQLite table exists at startup

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model/model.joblib")

_model = None  # loaded on startup
_feature_order = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
_target_names = {0: "setosa", 1: "versicolor", 2: "virginica"}


def _load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    model = joblib.load(path)
    logger.info("Loaded model from %s", path)
    return model


@app.on_event("startup")
def _startup():
    global _model
    try:
        _model = _load_model(MODEL_PATH)
    except Exception as e:
        logger.exception("Failed to load model: %s", e)
        # Don't crash the process; /health will report not ready


# ------------------------------------------------------------------------------
# Schemas (Pydantic v2)
# ------------------------------------------------------------------------------
class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., ge=0, description="sepal length (cm)")
    sepal_width: float = Field(..., ge=0, description="sepal width (cm)")
    petal_length: float = Field(..., ge=0, description="petal length (cm)")
    petal_width: float = Field(..., ge=0, description="petal width (cm)")

    def to_dataframe(self) -> pd.DataFrame:
        # Ensure correct column order for sklearn model
        return pd.DataFrame([
            [
                self.sepal_length,
                self.sepal_width,
                self.petal_length,
                self.petal_width
            ]
        ], columns=_feature_order)


class PredictResponse(BaseModel):
    label_id: int
    label_name: str
    probabilities: Optional[Dict[str, float]] = None
    model_path: str
    timestamp: str


# ------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------
@app.get("/health")
def health() -> JSONResponse:
    ready = _model is not None
    body = {
        "ok": True,
        "model_loaded": ready,
        "model_path": MODEL_PATH,
        "time": datetime.utcnow().isoformat() + "Z",
    }
    status = 200 if ready else 503
    return JSONResponse(content=body, status_code=status)


@app.post("/predict", response_model=PredictResponse)
def predict(payload: IrisFeatures):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    PREDICT_REQUESTS.inc()
    t0 = time.perf_counter()
    ts = datetime.utcnow().isoformat() + "Z"

    try:
        X = payload.to_dataframe()
        yhat = _model.predict(X)[0]
        resp = PredictResponse(
            label_id=int(yhat),
            label_name=_target_names.get(int(yhat), str(yhat)),
            probabilities=None,
            model_path=MODEL_PATH,
            timestamp=ts,
        )

        # If model supports probabilities, include them
        if hasattr(_model, "predict_proba"):
            proba = _model.predict_proba(X)[0]
            resp.probabilities = {_target_names[int(i)]: float(p) for i, p in enumerate(proba)}

        return resp
    finally:
        # Metrics + logging regardless of success
        latency = time.perf_counter() - t0
        PREDICT_LATENCY.observe(latency)
        try:
            log_prediction(
                ts_iso=ts,
                features=payload.model_dump(),
                prediction=int(yhat) if "_model" in globals() and _model is not None else -1,
                latency_ms=latency * 1000.0,
            )
        except Exception as e:
            logger.warning("Failed to log prediction to SQLite: %s", e)
        logger.info(
            "pred | features=%s | yhat=%s | latency_ms=%.2f",
            payload.model_dump(),
            int(yhat) if _model else None,
            latency * 1000.0
        )


@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
