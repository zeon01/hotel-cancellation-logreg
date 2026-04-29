"""Bootstrap confidence intervals on headline metrics.

Cluster-bootstrap by ``arrival_date_month`` to honour temporal correlation. For a
time-series-flavoured model, naive row-level bootstrap underestimates SE; clustering by
month is the cheap-and-defensible fix.
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


def _metric_dict(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def _top_decile_lift(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) < 100:
        return float("nan")
    df = pd.DataFrame({"y": y, "p": p})
    df["d"] = pd.qcut(df["p"], 10, labels=False, duplicates="drop")
    base = float(df["y"].mean())
    if base == 0:
        return float("nan")
    top = df.loc[df["d"] == df["d"].max(), "y"].mean()
    return float(top / base)


def cluster_bootstrap_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cluster_ids: np.ndarray | pd.Series,
    n_boot: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Cluster-bootstrap PR-AUC, ROC-AUC, Brier, top-decile lift.

    For each replicate, sample clusters with replacement and concatenate their members.
    Each replicate yields one row of metrics.
    """
    rng = np.random.default_rng(seed)
    cluster_ids = pd.Series(cluster_ids).reset_index(drop=True)
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_proba = pd.Series(y_proba).reset_index(drop=True)
    unique = cluster_ids.unique()
    grouped = {c: cluster_ids[cluster_ids == c].index.to_numpy() for c in unique}

    rows: list[dict[str, float]] = []
    for _ in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([grouped[c] for c in sampled])
        yb = y_true.iloc[idx].to_numpy()
        pb = y_proba.iloc[idx].to_numpy()
        if len(np.unique(yb)) < 2:
            continue
        m = _metric_dict(yb, pb)
        m["lift_top_decile"] = _top_decile_lift(yb, pb)
        rows.append(m)
    return pd.DataFrame(rows)


def summarise_ci(replicate_df: pd.DataFrame, lo: float = 2.5, hi: float = 97.5) -> pd.DataFrame:
    """Mean + percentile CI per metric."""
    out = []
    for col in replicate_df.columns:
        s = replicate_df[col].dropna()
        out.append(
            {
                "metric": col,
                "mean": float(s.mean()),
                "ci_lo": float(np.percentile(s, lo)),
                "ci_hi": float(np.percentile(s, hi)),
                "n": len(s),
            }
        )
    return pd.DataFrame(out)


def format_metric_with_ci(summary: pd.DataFrame, metric: str, fmt: str = "{:.3f}") -> str:
    """Render 'mean (95% CI [lo, hi])' for a single metric."""
    row = summary[summary["metric"] == metric].iloc[0]
    return (
        f"{fmt.format(row['mean'])} "
        f"(95% CI [{fmt.format(row['ci_lo'])}, {fmt.format(row['ci_hi'])}])"
    )


__all__ = [
    "cluster_bootstrap_metrics",
    "format_metric_with_ci",
    "summarise_ci",
]
