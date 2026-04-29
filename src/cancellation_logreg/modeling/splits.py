"""Time-aware train/val/test splits and CV iterators."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def time_based_split(
    df: pd.DataFrame, date_col: str = "arrival_date"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by date: train ≤ TRAIN_END, val ≤ VAL_END, test > VAL_END.

    Per Antonio et al. (2019, *Data Science Journal*), a stratified random split is
    optimistic versus a time-based split for this dataset.
    """
    raise NotImplementedError("Phase 2")


def time_series_cv(n_splits: int = 5) -> TimeSeriesSplit:
    """Return a sklearn TimeSeriesSplit configured for the train fold."""
    return TimeSeriesSplit(n_splits=n_splits)


__all__ = ["time_based_split", "time_series_cv"]
