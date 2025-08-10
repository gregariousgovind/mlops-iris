import pandas as pd
from pathlib import Path
from sklearn.datasets import load_iris

def save_iris(raw_path="data/raw/iris.csv", processed_path="data/processed/iris.csv"):
    Path(raw_path).parent.mkdir(parents=True, exist_ok=True)
    Path(processed_path).parent.mkdir(parents=True, exist_ok=True)

    iris = load_iris(as_frame=True)
    df = iris.frame.rename(columns={
        "sepal length (cm)": "sepal_length",
        "sepal width (cm)": "sepal_width",
        "petal length (cm)": "petal_length",
        "petal width (cm)": "petal_width",
        "target": "label",
    })
    df.to_csv(raw_path, index=False)
    df.to_csv(processed_path, index=False)
    print(f"saved: {raw_path}, {processed_path}")

if __name__ == "__main__":
    save_iris()