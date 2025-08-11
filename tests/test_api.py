import sys
import importlib
from typing import Dict

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris


def _train_tiny_iris_model(tmp_model_path: str) -> None:
    """Train a very small model quickly and save it to tmp_model_path."""
    iris = load_iris(as_frame=True)
    X = iris.frame[["sepal length (cm)", "sepal width (cm)",
                    "petal length (cm)", "petal width (cm)"]].to_numpy()
    y = iris.target.to_numpy()
    clf = LogisticRegression(max_iter=300)
    clf.fit(X, y)
    joblib.dump(clf, tmp_model_path)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """
    TestClient that runs FastAPI lifespan (startup/shutdown),
    ensuring the model loads before requests.
    """
    model_path = tmp_path / "model.joblib"
    _train_tiny_iris_model(str(model_path))
    monkeypatch.setenv("MODEL_PATH", str(model_path))

    # Ensure a clean import so api.main picks up env + lifespan
    if "api.main" in sys.modules:
        del sys.modules["api.main"]
    api_main = importlib.import_module("api.main")

    # Use context manager so lifespan events run
    with TestClient(api_main.app) as c:
        yield c


def test_health_ok(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("ok") is True
    assert body.get("model_loaded") is True
    assert "model_path" in body


def test_predict_success(client: TestClient):
    payload: Dict[str, float] = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert "label_id" in body and "label_name" in body
    assert body["label_id"] in [0, 1, 2]
    if "probabilities" in body and body["probabilities"] is not None:
        assert isinstance(body["probabilities"], dict)
    assert "timestamp" in body


def test_predict_validation_error(client: TestClient):
    bad = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        # missing petal_width
    }
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_metrics_exposed(client: TestClient):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert r.headers.get("content-type", "").startswith("text/plain")
    assert "predict_requests_total" in r.text
