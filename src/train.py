"""
Train two Iris classifiers, track with MLflow, pick the best, and register.

What this script does
---------------------
- Loads processed Iris data from data/processed/iris.csv
- Trains two models:
    1) logreg : StandardScaler + LogisticRegression
    2) rf     : RandomForestClassifier
- Logs params, metrics (acc, precision/recall/f1 macro), and a confusion matrix artifact
- Logs models with signature + input example to MLflow
- Selects the best model (by accuracy) and:
    * exports it to artifacts/model/model.joblib (for the API)
    * registers it to the MLflow Model Registry and sets alias 'production'

Environment
-----------
- Set MLFLOW_TRACKING_URI to your MLflow server or leave unset to use local ./mlruns
- Optional: change experiment/model names via ENV:
    MLFLOW_EXPERIMENT=iris-exp
    MLFLOW_MODEL_NAME=iris_clf
"""

from __future__ import annotations
import json
import os
import pathlib
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

# ------------------------------
# Config
# ------------------------------
DATA_PROCESSED = "data/processed/iris.csv"
METADATA_JSON = "data/metadata.json"
ARTIFACT_OUT = "artifacts/model/model.joblib"

EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "iris-exp")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "iris_clf")

SEED = 42
TEST_SIZE = 0.2

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET = "label"


def _load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict]:
    df = pd.read_csv(DATA_PROCESSED)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    # optional metadata (checksum, git sha) for run tags
    meta: Dict = {}
    if os.path.exists(METADATA_JSON):
        try:
            meta = json.load(open(METADATA_JSON))
        except Exception:
            meta = {}
    return X_train, X_test, y_train, y_test, meta


def _confusion_matrix_png(y_true: np.ndarray, y_pred: np.ndarray, out_path: str) -> None:
    """Save a simple confusion matrix plot to out_path (PNG)."""
    import matplotlib.pyplot as plt

    labels = [0, 1, 2]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)

    # annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _build_models() -> Dict[str, Pipeline]:
    # Pipeline 1: Standardize -> LogisticRegression
    logreg = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=300, random_state=SEED)),
        ]
    )

    # Pipeline 2: RandomForest
    rf = Pipeline(
        steps=[
            ("clf", RandomForestClassifier(n_estimators=200, random_state=SEED)),
        ]
    )
    return {"logreg": logreg, "rf": rf}


def _log_run(
    name: str,
    model: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    meta_tags: Dict,
) -> Tuple[float, str]:
    """
    Train, evaluate, log to MLflow. Returns (accuracy, run_id).
    """
    with mlflow.start_run(run_name=name) as run:
        run_id = run.info.run_id

        # Tags for reproducibility
        mlflow.set_tags(
            {
                "mlops.dataset": "iris",
                "mlops.split.test_size": TEST_SIZE,
                "code.git_sha": meta_tags.get("code_version_git", "unknown"),
                "data.checksum": meta_tags.get("checksum_sha256", "unknown"),
            }
        )

        # Params
        # Extract final estimator for clearer params
        final_est = model.named_steps.get("clf", model)
        for p, v in getattr(final_est, "get_params", lambda: {})().items():
            # Log only simple params to avoid verbosity
            if isinstance(v, (int, float, str, bool, type(None))):
                mlflow.log_param(f"{name}.{p}", v)

        # Train
        model.fit(X_train, y_train)

        # Predict & metrics
        y_pred = model.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision_macro", float(prec))
        mlflow.log_metric("recall_macro", float(rec))
        mlflow.log_metric("f1_macro", float(f1))

        # Confusion matrix artifact
        cm_path = f"artifacts/{name}_confusion_matrix.png"
        _confusion_matrix_png(y_test.to_numpy(), y_pred, cm_path)
        mlflow.log_artifact(cm_path, artifact_path="eval")

        # Signature + input example
        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_train.iloc[:2]

        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=None,  # register best later
        )

        print(f"🏃 View run {name} at: {mlflow.get_tracking_uri()} (run_id={run_id})")
        return acc, run_id


def main() -> None:
    np.random.seed(SEED)

    # Prepare MLflow experiment
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"🧪 Using MLflow experiment: {EXPERIMENT_NAME}")

    # Load data
    X_train, X_test, y_train, y_test, meta = _load_data()

    # Train & log both models
    models = _build_models()
    results: Dict[str, Tuple[float, str]] = {}

    for name, model in models.items():
        acc, run_id = _log_run(name, model, X_train, X_test, y_train, y_test, meta)
        results[name] = (acc, run_id)
        print(f"✅ {name}: acc={acc:.4f} run_id={run_id}")

    # Select best by accuracy
    best_name = max(results, key=lambda k: results[k][0])
    best_acc, best_run = results[best_name]
    print(f"[OK] Best={best_name} acc={best_acc:.4f}")

    # Export best to artifacts/ for API
    # Load the logged model from the run artifact and re-save a lightweight copy
    best_model_uri = f"runs:/{best_run}/model"
    best_model = mlflow.sklearn.load_model(best_model_uri)
    pathlib.Path(ARTIFACT_OUT).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, ARTIFACT_OUT)
    print(f"[OK] Exported best model -> {ARTIFACT_OUT}")

    # Register best in MLflow Model Registry and set alias 'production'
    client = MlflowClient()
    reg = mlflow.register_model(model_uri=best_model_uri, name=MODEL_NAME)

    try:
        # Use aliases (future-proof vs. deprecated stages)
        client.set_registered_model_alias(MODEL_NAME, "production", reg.version)
        print(f"[OK] Registered {MODEL_NAME} v{reg.version} -> alias=production")
    except Exception as e:
        # Fallback: try stage API if alias not supported
        try:
            client.transition_model_version_stage(MODEL_NAME, reg.version, stage="Production")
            print(f"[OK] Registered {MODEL_NAME} v{reg.version} -> Production (stage)")
        except Exception as e2:
            print(f"[WARN] Could not set alias/stage: {e} | {e2}")


if __name__ == "__main__":
    main()
