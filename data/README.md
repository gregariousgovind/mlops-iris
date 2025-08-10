# Iris Dataset (Project Copy)

**Source**: `sklearn.datasets.load_iris()` – classic 150-sample flower dataset.  
**License**: Freely usable for research/teaching; distributed via scikit-learn.

## Files
- `data/raw/iris.csv` – direct export from scikit-learn with normalized headers.
- `data/processed/iris.csv` – identical to raw for this project (no additional cleaning needed).
- `data/schema.json` – feature names and simple types for API/training validation later.
- `data/metadata.json` – checksum, row/column counts, class distribution, timestamp, git version.

## Columns
- Features (float): `sepal_length`, `sepal_width`, `petal_length`, `petal_width`
- Target (int): `label` in {0=setosa, 1=versicolor, 2=virginica}

## Notes
- We keep raw and processed splits to mirror real-world pipelines.
- For heavier datasets, consider DVC remotes like S3/GDrive; here we use a simple local remote.
