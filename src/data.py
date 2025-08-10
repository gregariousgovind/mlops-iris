"""
Data preparation for the Iris dataset.

What this script does
---------------------
1) Loads the Iris dataset from scikit-learn (a well-known public dataset).
2) Renames columns to snake_case for consistency across the project.
3) Saves:
   - Raw CSV:        data/raw/iris.csv
   - Processed CSV:  data/processed/iris.csv  (same as raw for this dataset)
   - Metadata JSON:  data/metadata.json       (rows, columns, checksum, version info, etc.)
   - Schema JSON:    data/schema.json         (feature names and types)

Why so many outputs?
--------------------
- CSVs are used by training and CI tests.
- Metadata/Schema make the project professional and future-proof:
  * We can validate API inputs later (Part 3 & Bonus).
  * We can quickly see data drift or accidental changes by comparing checksums.

How to run
----------
$ python src/data.py

This file is called by DVC as well (see dvc.yaml).
"""

from __future__ import annotations
import hashlib
import json
import os
import subprocess
from datetime import datetime
from typing import Dict

from sklearn.datasets import load_iris


RAW_PATH = "data/raw/iris.csv"
PROCESSED_PATH = "data/processed/iris.csv"
METADATA_PATH = "data/metadata.json"
SCHEMA_PATH = "data/schema.json"


def _git_commit_short_sha() -> str:
    """Return the short Git commit SHA if available, else 'unknown'."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _sha256_file(path: str) -> str:
    """Compute SHA-256 checksum of a file (helps detect changes reliably)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def save_iris(
    raw_path: str = RAW_PATH,
    processed_path: str = PROCESSED_PATH,
    metadata_path: str = METADATA_PATH,
    schema_path: str = SCHEMA_PATH,
) -> None:
    """
    Load Iris dataset, normalize headers, write CSVs, and produce metadata/schema.

    Parameters
    ----------
    raw_path : str
        Output path for the raw CSV.
    processed_path : str
        Output path for the processed CSV (same content for this simple dataset).
    metadata_path : str
        Output path for metadata JSON.
    schema_path : str
        Output path for schema JSON.
    """
    # 1) Load dataset
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

    # Ensure folders exist
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

    # 2) Save CSVs
    df.to_csv(raw_path, index=False)
    # For this dataset, no heavy preprocessing is required (no missing values, tiny size).
    # Keeping "processed" identical to raw keeps the pipeline shape realistic.
    df.to_csv(processed_path, index=False)

    # 3) Schema (for future validation and docs)
    schema: Dict[str, str] = {
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

    # 4) Metadata (helpful in reviews & CI logs)
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
