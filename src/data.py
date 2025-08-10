"""
Data preparation for the Iris dataset.

What this script does
---------------------
1) Loads Iris from scikit-learn (public dataset).
2) Normalizes column names to snake_case.
3) Writes:
   - data/raw/iris.csv
   - data/processed/iris.csv   (same as raw for this simple dataset)
   - data/schema.json          (feature/target schema for validation/docs)
   - data/metadata.json        (rows, columns, class counts, checksum, git SHA, timestamp)

Why keep both raw and processed?
--------------------------------
Mirrors real pipelines and makes it easy to insert transformations later
without changing the repository structure.

How to run
----------
$ python src/data.py

This file is also the single command in the DVC stage defined in dvc.yaml.
"""

from __future__ import annotations
import hashlib
import json
import os
import subprocess
from datetime import datetime
from typing import Dict

from sklearn.datasets import load_iris

# Default output locations
RAW_PATH = "data/raw/iris.csv"
PROCESSED_PATH = "data/processed/iris.csv"
SCHEMA_PATH = "data/schema.json"
METADATA_PATH = "data/metadata.json"


def _git_commit_short_sha() -> str:
    """Return the short Git commit SHA if available; 'unknown' if not a git repo."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _sha256_file(path: str) -> str:
    """Compute SHA-256 checksum for a file (used in metadata for integrity checks)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def save_iris(
    raw_path: str = RAW_PATH,
    processed_path: str = PROCESSED_PATH,
    schema_path: str = SCHEMA_PATH,
    metadata_path: str = METADATA_PATH,
) -> None:
    """
    Load Iris, write CSVs, and produce schema + metadata.

    Parameters
    ----------
    raw_path : str
        Output path for the raw CSV export.
    processed_path : str
        Output path for the processed CSV (identical to raw in this project).
    schema_path : str
        Output path for the JSON schema (feature types, target info).
    metadata_path : str
        Output path for the JSON metadata (shape, checksum, code version).
    """
    # ---- Load & rename columns
    iris = load_iris(as_frame=True)
    df = iris.frame.rename(
        columns={
            "sepal length (cm)": "sepal_length",
            "sepal width (cm)": "sepal_width",
            "petal length (cm)": "petal_length",
            "petal width (cm)": "petal_width",
            "target": "label",
        }
    )

    # ---- Ensure directories exist
    for p in (raw_path, processed_path, schema_path, metadata_path):
        os.makedirs(os.path.dirname(p), exist_ok=True)

    # ---- Write CSVs (raw & processed)
    df.to_csv(raw_path, index=False)
    df.to_csv(processed_path, index=False)

    # ---- Schema for future validation (API & training)
    schema: Dict[str, object] = {
        "features": {
            "sepal_length": "float",
            "sepal_width": "float",
            "petal_length": "float",
            "petal_width": "float",
        },
        "target": {"label": "int (0=setosa, 1=versicolor, 2=virginica)"},
        "primary_key": None,
    }
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)

    # ---- Metadata for auditability
    checksum = _sha256_file(processed_path)
    class_counts = df["label"].value_counts().to_dict()
    metadata = {
        "dataset": "iris",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "rows": int(df.shape[0]),
        "columns": list(df.columns),
        "class_counts": class_counts,
        "processed_csv": processed_path,
        "checksum_sha256": checksum,
        "code_version_git": _git_commit_short_sha(),
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[OK] Saved raw: {raw_path}")
    print(f"[OK] Saved processed: {processed_path}")
    print(f"[OK] Saved schema: {schema_path}")
    print(f"[OK] Saved metadata: {metadata_path}")


if __name__ == "__main__":
    save_iris()
