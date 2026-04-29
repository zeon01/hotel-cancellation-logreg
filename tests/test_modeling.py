"""Modeling smoke tests. Real metric tests live in 03_modeling.ipynb / Phase 2."""

from __future__ import annotations

import pytest


def test_pipeline_builds() -> None:
    pytest.importorskip("cancellation_logreg.modeling.train")
    from cancellation_logreg.modeling.train import build_pipeline

    try:
        pipe = build_pipeline()
    except NotImplementedError:
        pytest.skip("modeling.train.build_pipeline not yet implemented")

    assert pipe is not None
    assert hasattr(pipe, "fit")
    assert hasattr(pipe, "predict_proba")


def test_time_series_cv_returns_cv_iterator() -> None:
    from cancellation_logreg.modeling.splits import time_series_cv

    cv = time_series_cv(n_splits=3)
    assert cv.n_splits == 3
