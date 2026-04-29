"""Inferential interpretation: statsmodels Logit, odds ratios, plain-English summaries."""

from __future__ import annotations

import pandas as pd


def fit_statsmodels_logit(X: pd.DataFrame, y: pd.Series):
    """Fit ``statsmodels.Logit`` on standardised, non-target-encoded features.

    Used to extract coefficient estimates, robust SEs, p-values, and odds ratios — not for
    prediction. The corresponding sklearn pipeline is the prediction model.
    """
    raise NotImplementedError("Phase 2")


def odds_ratio_table(result) -> pd.DataFrame:
    """Return a frame with coef, OR, 2.5 / 97.5 percentile CIs, and p-value, sorted by |OR-1|."""
    raise NotImplementedError("Phase 2")


def plain_english(result, top_k: int = 5) -> list[str]:
    """Translate the top-k coefficients into prose ('each additional 30 days of lead time...')."""
    raise NotImplementedError("Phase 2")


__all__ = ["fit_statsmodels_logit", "odds_ratio_table", "plain_english"]
