import os
import sqlite3
import time
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel, Field
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

from src.utils import ensure_sqlite, get_logger


# Optional: fallback model if artifact missing
def _load_or_init_model(path: str):
    if os.path.exists(path):
        return joblib.load(path)
    # Fallback (keeps CI green): small quick model on Iris
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression

    iris = load_iris()
    X, y = iris.data, iris.target
    model = LogisticRegression(max_iter=200).fit(X, y)
    return model


app = FastAPI(title="Iris MLOps API", version="1.0.0")
logger = get_logger()
ensure_sqlite()

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model/model.joblib")
model = _load_or_init_model(MODEL_PATH)

REQUESTS = Counter("predict_requests_total", "Total prediction requests")
LATENCY = Histogram("predict_latency_seconds", "Prediction latency (s)")

COLS: List[str] = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


class IrisPayload(BaseModel):
    sepal_length: float = Field(..., ge=0)
    sepal_width: float = Field(..., ge=0)
    petal_length: float = Field(..., ge=0)
    petal_width: float = Field(..., ge=0)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: IrisPayload):
    REQUESTS.inc()
    t0 = time.time()
    row = payload.model_dump()

    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None:
        ordered = [row[str(c)] for c in feature_names]
        X = pd.DataFrame([ordered], columns=list(feature_names))
    else:
        X = [[row[c] for c in COLS]]

    pred = int(model.predict(X)[0])
    dt = time.time() - t0
    LATENCY.observe(dt)
    logger.info(f"predict: {row} -> {pred} in {dt*1000:.2f}ms")

    # store into SQLite
    conn = sqlite3.connect("logs/predictions.db")
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO requests(
            ts, sepal_length, sepal_width, petal_length, petal_width, pred, latency_ms
        )
        VALUES(datetime('now'), ?, ?, ?, ?, ?, ?)
        """,
        (
            row["sepal_length"],
            row["sepal_width"],
            row["petal_length"],
            row["petal_width"],
            pred,
            dt * 1000,
        ),
    )
    conn.commit()
    conn.close()

    return {"prediction": pred, "latency_ms": round(dt * 1000, 2)}


@app.get("/metrics")
def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
