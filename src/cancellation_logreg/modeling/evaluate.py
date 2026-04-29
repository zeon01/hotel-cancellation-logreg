"""Evaluation: discrimination, calibration, business-cost threshold selection, plots."""

from __future__ import annotations

import numpy as np
import pandas as pd


def discrimination_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    """Return ROC-AUC and PR-AUC."""
    raise NotImplementedError("Phase 2")


def brier(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Brier score (mean squared error of probabilistic predictions)."""
    raise NotImplementedError("Phase 2")


def reliability_table(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Bin predictions and return a per-bin frame with mean predicted, mean observed, count."""
    raise NotImplementedError("Phase 2")


def expected_cost(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    cost_fp: float = 5.0,
    cost_fn: float = 50.0,
) -> float:
    """Expected operational cost at a given threshold under a toy cost matrix."""
    raise NotImplementedError("Phase 2")


def best_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_fp: float = 5.0,
    cost_fn: float = 50.0,
) -> float:
    """Return the threshold minimising expected cost over a grid in (0, 1)."""
    raise NotImplementedError("Phase 2")


def main() -> None:
    raise NotImplementedError("Phase 2")


if __name__ == "__main__":
    main()
