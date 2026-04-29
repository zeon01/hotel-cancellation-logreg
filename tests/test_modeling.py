"""Modeling smoke tests. Real metric tests live in 03_modeling.ipynb / Phase 2."""

from __future__ import annotations

import pytest


def test_pipeline_builds() -> None:
    pytest.importorskip("cancellation_logreg.modeling.train")
    import pandas as pd

    from cancellation_logreg.modeling.train import build_pipeline

    fake = pd.DataFrame(
        {
            "lead_time": [10, 200, 30],
            "adr": [80.0, 120.0, 100.0],
            "hotel": ["City Hotel", "Resort Hotel", "City Hotel"],
            "country": ["PRT", "GBR", "FRA"],
        }
    )
    pipe = build_pipeline(fake, C=1.0)
    assert pipe is not None
    assert hasattr(pipe, "fit")
    assert hasattr(pipe, "predict_proba")


def test_time_series_cv_returns_cv_iterator() -> None:
    from cancellation_logreg.modeling.splits import time_series_cv

    cv = time_series_cv(n_splits=3)
    assert cv.n_splits == 3
