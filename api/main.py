"""
FastAPI service for the Iris classifier.
"""

from __future__ import annotations
import os
import time
from datetime import datetime, timezone
from typing import Dict, Optional
from contextlib import asynccontextmanager

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
# App + infra init
# ------------------------------------------------------------------------------
logger = get_logger()
init_db()  # ensure SQLite table exists early

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model/model.joblib")
_model = None  # loaded during lifespan
_feature_order = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
_target_names = {0: "setosa", 1: "versicolor", 2: "virginica"}


def _load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    model = joblib.load(path)
    logger.info("Loaded model from %s", path)
    return model


def _ensure_model_loaded() -> None:
    global _model
    if _model is None:
        try:
            _model = _load_model(MODEL_PATH)
        except Exception as e:
            logger.exception("Failed to load model: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    _ensure_model_loaded()
    yield
    # shutdown (noop)


app = FastAPI(
    title="Iris MLOps API",
    version="1.0.1",
    docs_url="/docs",
    redoc_url=None,
    lifespan=lifespan,
)


# ------------------------------------------------------------------------------
# Schemas (Pydantic v2)
# ------------------------------------------------------------------------------
class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., ge=0, description="sepal length (cm)")
    sepal_width: float = Field(..., ge=0, description="sepal width (cm)")
    petal_length: float = Field(..., ge=0, description="petal length (cm)")
    petal_width: float = Field(..., ge=0, description="petal width (cm)")

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [[self.sepal_length, self.sepal_width, self.petal_length, self.petal_width]],
            columns=_feature_order,
        )


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
        "time": datetime.now(timezone.utc).isoformat(),
    }
    status = 200 if ready else 503
    return JSONResponse(content=body, status_code=status)


@app.post("/predict", response_model=PredictResponse)
def predict(payload: IrisFeatures):
    if _model is None:
        _ensure_model_loaded()
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    PREDICT_REQUESTS.inc()
    t0 = time.perf_counter()
    ts = datetime.now(timezone.utc).isoformat()

    yhat_int = -1
    try:
        # Use numpy arrays to avoid sklearn "feature names" warning
        X_df = payload.to_dataframe()
        X = X_df.values

        yhat_int = int(_model.predict(X)[0])
        resp = PredictResponse(
            label_id=yhat_int,
            label_name=_target_names.get(yhat_int, str(yhat_int)),
            probabilities=None,
            model_path=MODEL_PATH,
            timestamp=ts,
        )

        if hasattr(_model, "predict_proba"):
            proba = _model.predict_proba(X)[0]
            resp.probabilities = {
                _target_names[int(i)]: float(p) for i, p in enumerate(proba)
            }

        return resp
    finally:
        latency = time.perf_counter() - t0
        PREDICT_LATENCY.observe(latency)
        try:
            log_prediction(
                ts_iso=ts,
                features=payload.model_dump(),
                prediction=yhat_int,
                latency_ms=latency * 1000.0,
            )
        except Exception as e:
            logger.warning("Failed to log prediction to SQLite: %s", e)
        logger.info(
            "pred | features=%s | yhat=%s | latency_ms=%.2f",
            payload.model_dump(),
            None if yhat_int < 0 else yhat_int,
            latency * 1000.0,
        )


@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
