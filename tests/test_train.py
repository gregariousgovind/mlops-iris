import os
from src.data import save_iris


def test_data_exists_or_create():
    raw = "data/raw/iris.csv"
    processed = "data/processed/iris.csv"
    if not (os.path.exists(raw) and os.path.exists(processed)):
        save_iris(raw, processed)
    assert os.path.exists(processed)
