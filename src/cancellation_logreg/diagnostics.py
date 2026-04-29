"""Multicollinearity, calibration, and stability diagnostics."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

log = logging.getLogger(__name__)


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """Variance Inflation Factor per numeric column.

    Returns a frame sorted by VIF descending. Threshold convention: 5 = warn, 10 = drop.
    """
    if X.empty:
        return pd.DataFrame(columns=["feature", "vif"])

    numeric = X.select_dtypes(include=[np.number]).copy()
    numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all").fillna(0.0)

    if numeric.shape[1] < 2:
        return pd.DataFrame({"feature": numeric.columns, "vif": [np.nan] * numeric.shape[1]})

    arr = numeric.to_numpy()
    vifs = [variance_inflation_factor(arr, i) for i in range(arr.shape[1])]
    return (
        pd.DataFrame({"feature": numeric.columns, "vif": vifs})
        .sort_values("vif", ascending=False)
        .reset_index(drop=True)
    )


def calibration_curve_data(
    y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Reliability-diagram bins; mirrors evaluate.reliability_table."""
    from cancellation_logreg.modeling.evaluate import reliability_table

    return reliability_table(y_true, y_proba, n_bins=n_bins)


def coefficient_bootstrap(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    n_boot: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Re-fit the estimator on bootstrap samples; return per-replicate coefficients.

    Used to assess coefficient stability - anything whose 5-95% range crosses zero
    becomes a caveat.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    rows: list[dict[str, float]] = []
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        Xb = X.iloc[idx]
        yb = y.iloc[idx]
        est = estimator.__class__(**estimator.get_params())
        est.fit(Xb, yb)
        # Pull coefficients from the fitted logreg.
        if hasattr(est, "named_steps") and "logreg" in est.named_steps:
            logreg = est.named_steps["logreg"]
            coefs = np.ravel(logreg.coef_)
            try:
                names = est.named_steps["preprocessor"].get_feature_names_out()
            except Exception:
                names = [f"f{i}" for i in range(len(coefs))]
            for nm, c in zip(names, coefs, strict=False):
                rows.append({"replicate": b, "feature": nm, "coef": float(c)})
    return pd.DataFrame(rows)


__all__ = ["calibration_curve_data", "coefficient_bootstrap", "compute_vif"]
