"""Fit the primary L2-regularised logistic-regression pipeline and persist the artefact."""

from __future__ import annotations

from pathlib import Path

from sklearn.pipeline import Pipeline


def build_pipeline() -> Pipeline:
    """Build the ColumnTransformer + LogisticRegression(penalty='l2') pipeline.

    Numeric features are scaled; low-cardinality categoricals are one-hot encoded with
    infrequent-category handling; high-cardinality categoricals are target-encoded
    inside CV folds only.
    """
    raise NotImplementedError("Phase 2")


def fit(model: Pipeline, X, y) -> Pipeline:
    """Fit and return the fitted pipeline."""
    raise NotImplementedError("Phase 2")


def save(model: Pipeline, path: Path) -> None:
    """Persist the fitted pipeline via joblib."""
    raise NotImplementedError("Phase 2")


def main() -> None:
    """Entrypoint for ``python -m cancellation_logreg.modeling.train``."""
    raise NotImplementedError("Phase 2")


if __name__ == "__main__":
    main()
