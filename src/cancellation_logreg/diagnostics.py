"""Multicollinearity, calibration, and stability diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """Compute Variance Inflation Factor for every numeric column.

    Returns a frame sorted by VIF descending. Threshold convention: 5 = warn, 10 = drop or merge.
    """
    raise NotImplementedError(
        "Phase 2: use statsmodels.stats.outliers_influence.variance_inflation_factor"
    )


def calibration_curve_data(
    y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Reliability-diagram bins."""
    raise NotImplementedError("Phase 2")


def coefficient_bootstrap(model, X: pd.DataFrame, y: np.ndarray, n_boot: int = 200) -> pd.DataFrame:
    """Re-fit on bootstrap samples; return a frame of coefficients per bootstrap replicate."""
    raise NotImplementedError("Phase 2")


__all__ = ["calibration_curve_data", "coefficient_bootstrap", "compute_vif"]
