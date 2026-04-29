"""Time-aware train/val/test splits and CV iterators."""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from cancellation_logreg.config import TARGET, TRAIN_END, VAL_END

log = logging.getLogger(__name__)


# Columns that are NOT used as features (target, raw date columns replaced by engineered
# ones, and a couple of leakage-tagged surfaces that snuck in via feature engineering).
DROP_FROM_X: tuple[str, ...] = (
    "is_canceled",
    "arrival_date",
    "booking_date",
    "arrival_date_year",
    "arrival_date_month",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
)


def time_based_split(
    df: pd.DataFrame, date_col: str = "arrival_date"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by date: train ≤ TRAIN_END, val (TRAIN_END, VAL_END], test > VAL_END.

    Per Antonio et al. (2019, *Data Science Journal*), a stratified random split is
    optimistic versus a time-based split for this dataset.
    """
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])

    train_end = pd.Timestamp(TRAIN_END)
    val_end = pd.Timestamp(VAL_END)

    train = df.loc[df[date_col] <= train_end].copy()
    val = df.loc[(df[date_col] > train_end) & (df[date_col] <= val_end)].copy()
    test = df.loc[df[date_col] > val_end].copy()

    log.info(
        "time_based_split: train=%d (%s..%s) | val=%d | test=%d",
        len(train),
        train[date_col].min().date() if len(train) else "-",
        train[date_col].max().date() if len(train) else "-",
        len(val),
        len(test),
    )
    return train, val, test


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) with non-feature columns removed."""
    drop = [c for c in DROP_FROM_X if c in df.columns]
    X = df.drop(columns=drop)
    y = df[TARGET]
    return X, y


def time_series_cv(n_splits: int = 5) -> TimeSeriesSplit:
    """sklearn ``TimeSeriesSplit`` configured for the train fold."""
    return TimeSeriesSplit(n_splits=n_splits)


__all__ = ["DROP_FROM_X", "split_xy", "time_based_split", "time_series_cv"]
