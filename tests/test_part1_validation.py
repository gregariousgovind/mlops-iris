import os
import json
import hashlib
from typing import Dict

import pandas as pd

# ---- Paths expected from src/data.py (and dvc.yaml stage) ---------------------
RAW = "data/raw/iris.csv"
PROCESSED = "data/processed/iris.csv"
SCHEMA = "data/schema.json"
METADATA = "data/metadata.json"

FEATURE_COLS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET_COL = "label"
EXPECTED_ROWS = 150
EXPECTED_LABELS = {0, 1, 2}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------------
# Existence checks
# --------------------------------------------------------------------------------
def test_files_exist() -> None:
    assert os.path.exists(RAW), "Missing data/raw/iris.csv"
    assert os.path.exists(PROCESSED), "Missing data/processed/iris.csv"
    assert os.path.exists(SCHEMA), "Missing data/schema.json"
    assert os.path.exists(METADATA), "Missing data/metadata.json"


# --------------------------------------------------------------------------------
# Schema contract
# --------------------------------------------------------------------------------
def test_schema_contract() -> None:
    schema: Dict = json.load(open(SCHEMA))
    assert "features" in schema and isinstance(schema["features"], dict)
    assert "target" in schema and isinstance(schema["target"], dict)
    assert "primary_key" in schema

    for col in FEATURE_COLS:
        assert col in schema["features"], f"Schema missing feature: {col}"
        assert isinstance(schema["features"][col], str), "Feature types should be strings"

    assert TARGET_COL in schema["target"], "Schema missing target label"


# --------------------------------------------------------------------------------
# Metadata contract + checksum integrity
# --------------------------------------------------------------------------------
def test_metadata_contract_and_checksum() -> None:
    meta: Dict = json.load(open(METADATA))
    for key in ["dataset", "rows", "columns", "processed_csv", "checksum_sha256", "created_at"]:
        assert key in meta, f"metadata.json missing '{key}'"

    # Path should match constant
    assert meta["processed_csv"] == PROCESSED

    # Checksum must match actual processed csv
    actual = sha256_file(PROCESSED)
    assert actual == meta["checksum_sha256"], "Checksum mismatch: processed CSV changed unexpectedly"


# --------------------------------------------------------------------------------
# Dataframe expectations (shape, columns, basic types and values)
# --------------------------------------------------------------------------------
def test_dataframe_expectations() -> None:
    df = pd.read_csv(PROCESSED)

    # shape & columns
    assert df.shape[0] == EXPECTED_ROWS, f"Expected {EXPECTED_ROWS} rows"
    for col in FEATURE_COLS + [TARGET_COL]:
        assert col in df.columns, f"Missing column: {col}"

    # simple type checks (numeric features, integer labels)
    assert pd.api.types.is_numeric_dtype(df[FEATURE_COLS].dtypes).all(), "Features must be numeric"
    assert pd.api.types.is_integer_dtype(df[TARGET_COL].dtype), "Label should be integer-coded"

    # value domain checks
    unique_labels = set(df[TARGET_COL].unique().tolist())
    assert unique_labels.issubset(EXPECTED_LABELS), f"Unexpected labels: {unique_labels - EXPECTED_LABELS}"

    # no nulls
    assert not df.isna().any().any(), "Found missing values in processed data"
