"""Subgroup performance analysis.

Honest answer to "where is this model worst?". For an OTA Supply Ops deployment, the
practical follow-up to a subgroup-level deficit is a segment-specific recalibrator, not
a global retrain.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

log = logging.getLogger(__name__)


_LEAD_TIME_BUCKETS: list[tuple[str, int, int]] = [
    ("0-7d", 0, 7),
    ("8-30d", 8, 30),
    ("31-90d", 31, 90),
    ("91-180d", 91, 180),
    ("181d+", 181, 10**6),
]


def _safe(metric_fn, y, p) -> float:
    try:
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(metric_fn(y, p))
    except Exception:
        return float("nan")


def _row(label: str, sub_y: pd.Series, sub_p: np.ndarray) -> dict:
    y = sub_y.to_numpy()
    return {
        "subgroup": label,
        "n": len(y),
        "positive_rate": float(y.mean()) if len(y) else float("nan"),
        "pr_auc": _safe(average_precision_score, y, sub_p),
        "roc_auc": _safe(roc_auc_score, y, sub_p),
        "brier": _safe(brier_score_loss, y, sub_p),
    }


def lead_time_buckets(lead_time: pd.Series) -> pd.Series:
    """Map lead_time to the canonical bucket labels."""
    out = pd.Series("181d+", index=lead_time.index, dtype="object")
    for label, lo, hi in _LEAD_TIME_BUCKETS:
        out.loc[(lead_time >= lo) & (lead_time <= hi)] = label
    return pd.Categorical(
        out, categories=[label for label, _, _ in _LEAD_TIME_BUCKETS], ordered=True
    )


def subgroup_metrics(
    df_test: pd.DataFrame,
    y_proba: np.ndarray,
    target: str = "is_canceled",
    grouping_columns: tuple[str, ...] = (
        "hotel",
        "market_segment",
        "is_repeated_guest",
    ),
) -> pd.DataFrame:
    """Per-subgroup metric grid.

    Rows: one per (column, value) combination plus a per-lead-time-bucket breakdown.
    """
    test = df_test.reset_index(drop=True).copy()
    test["_proba"] = y_proba
    rows: list[dict] = []
    rows.append({**_row("OVERALL", test[target], test["_proba"].to_numpy()), "feature": "ALL"})

    for col in grouping_columns:
        if col not in test.columns:
            continue
        for value, frame in test.groupby(col, dropna=False, observed=True):
            label = f"{col}={value}"
            rows.append({**_row(label, frame[target], frame["_proba"].to_numpy()), "feature": col})

    if "lead_time" in test.columns:
        test["_lt_bucket"] = lead_time_buckets(test["lead_time"])
        for value, frame in test.groupby("_lt_bucket", dropna=False, observed=True):
            label = f"lead_time={value}"
            rows.append(
                {
                    **_row(label, frame[target], frame["_proba"].to_numpy()),
                    "feature": "lead_time_bucket",
                }
            )

    return pd.DataFrame(rows)


__all__ = ["lead_time_buckets", "subgroup_metrics"]
