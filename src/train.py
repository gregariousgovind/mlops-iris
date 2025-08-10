"""
Train two Iris classifiers, track with MLflow, and (when a registry is available)
register the best model.

- Models: LogisticRegression, RandomForestClassifier
- Tracking: params, metrics, model (with signature + input example)
- Export: artifacts/model/model.joblib (used by the API & Docker image)
- Registry: registers 'iris_clf' and promotes to 'Production' if server/DB backend exists
- CI note: defaults to MLflow file store (no server required)

Usage
-----
$ python src/train.py
Optionally set MLFLOW_TRACKING_URI to an MLflow server to enable the registry:
$ export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
$ python src/train.py
"""

from __future__ import annotations
import json
import os
from typing import Dict, Tuple

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


# --- Paths & constants ---------------------------------------------------------
DATA = "data/processed/iris.csv"
METADATA_JSON = "data/metadata.json"
MODEL_EXPORT_DIR = "artifacts/model"

EXPERIMENT = "iris-exp"
MODEL_NAME = "iris_clf"


def _load_metadata(path: str) -> Dict:
    """Load dataset metadata to tag runs (checksum, git SHA, etc.)."""
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def train() -> None:
    # ---- Data -----------------------------------------------------------------
    df = pd.read_csv(DATA)
    X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y = df["label"]

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---- MLflow setup ---------------------------------------------------------
    # Default to local file store so CI doesn't need an MLflow server.
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment(EXPERIMENT)

    meta = _load_metadata(METADATA_JSON)
    dataset_checksum = meta.get("checksum_sha256", "unknown")
    code_version_git = meta.get("code_version_git", "unknown")

    candidates: Tuple[Tuple[str, object], ...] = (
        ("logreg", LogisticRegression(max_iter=200)),
        ("rf", RandomForestClassifier(n_estimators=200, random_state=42)),
    )

    best = None

    for name, model in candidates:
        with mlflow.start_run(run_name=name) as run:
            # Helpful tags for searchability
            mlflow.set_tags(
                {
                    "framework": "sklearn",
                    "dataset": "iris",
                    "dataset_checksum": dataset_checksum,
                    "code_version_git": code_version_git,
                    "stage_hint": "candidate",
                }
            )

            # Train
            model.fit(Xtr, ytr)

            # Evaluate
            ypred = model.predict(Xte)
            acc = accuracy_score(yte, ypred)
            f1m = f1_score(yte, ypred, average="macro")

            # Log params (explicit, simple)
            if name == "logreg":
                mlflow.log_param("max_iter", model.max_iter)
            if name == "rf":
                mlflow.log_param("n_estimators", model.n_estimators)
                mlflow.log_param("random_state", 42)

            # Log metrics
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_macro", f1m)

            # Log model with signature + input example
            signature = infer_signature(Xtr, model.predict(Xtr))
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                signature=signature,
                input_example=Xtr.head(3),
            )

            # Track best
            if (best is None) or (acc > best["acc"]):
                best = {"name": name, "acc": acc, "f1": f1m, "run_id": run.info.run_id}

    assert best is not None, "No candidate models were evaluated."

    # ---- Export best model for packaging -------------------------------------
    os.makedirs(MODEL_EXPORT_DIR, exist_ok=True)
    best_uri = f"runs:/{best['run_id']}/model"
    best_model = mlflow.sklearn.load_model(best_uri)
    joblib.dump(best_model, f"{MODEL_EXPORT_DIR}/model.joblib")
    print(
        f"[OK] Best={best['name']} acc={best['acc']:.4f} -> {MODEL_EXPORT_DIR}/model.joblib"
    )

    # ---- Register best (when registry available) ------------------------------
    # On a file store (CI), this raises; we skip gracefully.
    try:
        client = MlflowClient()
        reg = mlflow.register_model(model_uri=best_uri, name=MODEL_NAME)
        # Set/overwrite an alias instead of using stages (future-proof)
        client.set_registered_model_alias(MODEL_NAME, "production", reg.version)
        print(f"[OK] Registered {MODEL_NAME} v{reg.version} -> alias=production")
    except Exception as e:
        print(
            "[INFO] Registry not available in this environment; skipping registration:",
            e,
        )


if __name__ == "__main__":
    train()
