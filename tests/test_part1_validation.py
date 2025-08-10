"""
Part 1 Validation Tests
- Verifies dataset files exist
- Validates schema & metadata contracts
- Confirms deterministic checksum
- Checks basic data assumptions (rows, columns, class balance)
Run:
    pytest -q tests/test_part1_validation.py
"""

from __future__ import annotations
import hashlib
import json
import os
from typing import Dict

import pandas as pd

RAW = "data/raw/iris.csv"
PROCESSED = "data/processed/iris.csv"
SCHEMA = "data/schema.json"
METADATA = "data/metadata.json"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def test_files_exist() -> None:
    assert os.path.exists(RAW), "Missing data/raw/iris.csv"
    assert os.path.exists(PROCESSED), "Missing data/processed/iris.csv"
    assert os.path.exists(SCHEMA), "Missing data/schema.json"
    assert os.path.exists(METADATA), "Missing data/metadata.json"


def test_schema_contract() -> None:
    schema: Dict = json.load(open(SCHEMA))
    assert "features" in schema and "target" in schema
    feats = schema["features"]
    assert list(feats.keys()) == [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
    ], "Feature names must match expected columns"
    assert "label" in schema["target"] or schema["target"].get("label") is not None


def test_metadata_contract_and_checksum() -> None:
    meta: Dict = json.load(open(METADATA))
    # Required keys
    for k in [
        "dataset",
        "created_at",
        "rows",
        "columns",
        "class_counts",
        "processed_csv",
        "checksum_sha256",
        "code_version_git",
    ]:
        assert k in meta, f"metadata.json missing key: {k}"
    # Sanity
    assert meta["dataset"] == "iris"
    assert meta["processed_csv"] == PROCESSED
    assert meta["rows"] == 150
    assert meta["columns"] == [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "label",
    ]
    # Deterministic checksum
    assert sha256_file(PROCESSED) == meta["checksum_sha256"], "Checksum mismatch"


def test_dataframe_expectations() -> None:
    df = pd.read_csv(PROCESSED)
    # Shape & columns
    assert df.shape[0] == 150
    assert list(df.columns) == [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "label",
    ]
    # Types: pandas will coerce to float64/int64; allow any numeric
    assert pd.api.types.is_numeric_dtype(df["sepal_length"])
    assert pd.api.types.is_numeric_dtype(df["sepal_width"])
    assert pd.api.types.is_numeric_dtype(df["petal_length"])
    assert pd.api.types.is_numeric_dtype(df["petal_width"])
    assert pd.api.types.is_integer_dtype(df["label"]) or pd.api.types.is_numeric_dtype(
        df["label"]
    )

    # Class balance (50 each for classic Iris)
    counts = df["label"].value_counts().to_dict()
    assert counts.get(0, 0) == 50 and counts.get(1, 0) == 50 and counts.get(2, 0) == 50
