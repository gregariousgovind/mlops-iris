"""
Utilities for logging, metrics, and simple SQLite storage.

- get_logger(name): structured app logger (rotating file)
- init_db(db_path): ensure SQLite table exists for prediction logs
- log_prediction(...): append a prediction event to SQLite
- Prometheus metrics: PREDICT_REQUESTS, PREDICT_LATENCY

These are used by the FastAPI app to satisfy:
  * Part 5: Logging (file + SQLite) and Monitoring (/metrics)
"""

from __future__ import annotations
import logging
import os
import sqlite3
from logging.handlers import RotatingFileHandler
from typing import Dict

from prometheus_client import Counter, Histogram

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------

LOG_DIR = os.getenv("LOG_DIR", "logs")
LOG_FILE = os.path.join(LOG_DIR, "app.log")


def get_logger(name: str = "mlops-iris") -> logging.Logger:
    """
    Create or return a module-level logger that writes to a rotating file.

    Rotates at ~5MB, keeps 3 backups. Level defaults to INFO (override via LOG_LEVEL).
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.setLevel(level)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=3)
    file_handler.setFormatter(fmt)
    file_handler.setLevel(level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    stream_handler.setLevel(level)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    logger.info("Logger initialized (level=%s, file=%s)", level, LOG_FILE)
    return logger


# ------------------------------------------------------------------------------
# SQLite lightweight storage for prediction logs
# ------------------------------------------------------------------------------

DEFAULT_DB = os.getenv("PREDICTIONS_DB", os.path.join(LOG_DIR, "predictions.db"))


def init_db(db_path: str = DEFAULT_DB) -> None:
    """Create the SQLite DB and table if missing."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                sepal_length REAL NOT NULL,
                sepal_width  REAL NOT NULL,
                petal_length REAL NOT NULL,
                petal_width  REAL NOT NULL,
                prediction   INTEGER NOT NULL,
                latency_ms   REAL NOT NULL
            )
            """
        )
        conn.commit()


def log_prediction(
    *,
    db_path: str = DEFAULT_DB,
    ts_iso: str,
    features: Dict[str, float],
    prediction: int,
    latency_ms: float,
) -> None:
    """
    Append a single prediction event to SQLite.

    Parameters
    ----------
    db_path : str
        Database file path.
    ts_iso : str
        ISO8601 timestamp string.
    features : Dict[str, float]
        Must include: sepal_length, sepal_width, petal_length, petal_width.
    prediction : int
        Predicted class label.
    latency_ms : float
        End-to-end latency for the request.
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO predictions
            (ts, sepal_length, sepal_width, petal_length, petal_width, prediction, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts_iso,
                float(features["sepal_length"]),
                float(features["sepal_width"]),
                float(features["petal_length"]),
                float(features["petal_width"]),
                int(prediction),
                float(latency_ms),
            ),
        )
        conn.commit()


# ------------------------------------------------------------------------------
# Prometheus metrics
# ------------------------------------------------------------------------------

# Count total prediction requests
PREDICT_REQUESTS = Counter(
    "predict_requests_total",
    "Total number of prediction requests received.",
)

# Record prediction latency in seconds
PREDICT_LATENCY = Histogram(
    "predict_latency_seconds",
    "Latency of prediction requests in seconds.",
    buckets=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5),
)
