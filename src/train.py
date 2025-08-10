import os, time
import mlflow, mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature

DATA = "data/processed/iris.csv"
MODEL_EXPORT_DIR = "artifacts/model"

def train():
    df = pd.read_csv(DATA)
    X = df[["sepal_length","sepal_width","petal_length","petal_width"]]
    y = df["label"]

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    experiments = [
        ("logreg", LogisticRegression(max_iter=200)),
        ("rf", RandomForestClassifier(n_estimators=200, random_state=42)),
    ]

    best = None
    mlflow.set_experiment("iris-exp")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))

    for name, model in experiments:
        with mlflow.start_run(run_name=name) as run:
            model.fit(Xtr, ytr)
            ypred = model.predict(Xte)
            acc = accuracy_score(yte, ypred)
            f1m = f1_score(yte, ypred, average="macro")

            # params
            if name == "logreg":
                mlflow.log_param("max_iter", getattr(model, "max_iter", None))
            if name == "rf":
                mlflow.log_param("n_estimators", getattr(model, "n_estimators", None))

            # metrics
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_macro", f1m)

            # signature + input example (silences warnings, better lineage)
            signature = infer_signature(Xtr, model.predict(Xtr))
            input_example = Xtr.head(1)

            # Support newer MLflow API (name=) with fallback to artifact_path=
            try:
                mlflow.sklearn.log_model(sk_model=model, name="model",
                                         signature=signature, input_example=input_example)
            except TypeError:
                mlflow.sklearn.log_model(sk_model=model, artifact_path="model",
                                         signature=signature, input_example=input_example)

            if (best is None) or (acc > best["acc"]):
                best = {"name": name, "acc": acc, "f1": f1m, "run_id": run.info.run_id}

    client = MlflowClient()
    model_uri = f"runs:/{best['run_id']}/model"
    model_name = "iris_clf"

    # Register best model and set alias "prod" (instead of deprecated stages)
    try:
        reg = mlflow.register_model(model_uri=model_uri, name=model_name)
        # In some backends, registration is async; simple wait loop
        for _ in range(30):
            mv = client.get_model_version(model_name, reg.version)
            if getattr(mv, "status", "READY") == "READY":
                break
            time.sleep(1)
        client.set_registered_model_alias(model_name, "prod", reg.version)
    except Exception as e:
        print("Registration/alias step skipped:", e)

    # Export a copy for packaging
    os.makedirs(MODEL_EXPORT_DIR, exist_ok=True)
    best_model = mlflow.sklearn.load_model(model_uri)
    joblib.dump(best_model, f"{MODEL_EXPORT_DIR}/model.joblib")
    print(f"Best={best['name']} acc={best['acc']:.4f} -> {MODEL_EXPORT_DIR}/model.joblib")

if __name__ == "__main__":
    train()
