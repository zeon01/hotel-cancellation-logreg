"""Inferential interpretation: statsmodels Logit, odds ratios, plain-English summaries."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm

log = logging.getLogger(__name__)


def fit_statsmodels_logit(
    X: pd.DataFrame,
    y: pd.Series,
    cov_type: str = "HC3",
):
    """Fit ``statsmodels.Logit`` on a *standardised, dense, numeric* design matrix.

    Used to extract coefficient estimates, robust SEs, p-values, and odds ratios — not
    for prediction. The corresponding sklearn pipeline is the prediction model.
    """
    Xc = sm.add_constant(X.astype(float), has_constant="add")
    model = sm.Logit(y.astype(int), Xc)
    return model.fit(method="lbfgs", maxiter=200, disp=0, cov_type=cov_type)


def odds_ratio_table(result, sort_by: str = "abs_logodds") -> pd.DataFrame:
    """coef, OR, 2.5/97.5 % CI, p — sorted by |log-odds| (i.e. effect size)."""
    params = result.params
    conf = result.conf_int().rename(columns={0: "lo", 1: "hi"})
    pvals = result.pvalues
    out = pd.DataFrame(
        {
            "feature": params.index,
            "coef": params.to_numpy(),
            "or": np.exp(params.to_numpy()),
            "or_lo": np.exp(conf["lo"].to_numpy()),
            "or_hi": np.exp(conf["hi"].to_numpy()),
            "p": pvals.to_numpy(),
        }
    )
    out["abs_logodds"] = out["coef"].abs()
    return out.sort_values(sort_by, ascending=False).reset_index(drop=True)


def plain_english(result, top_k: int = 5) -> list[str]:
    """Translate the top-k coefficients (by |log-odds|) into one-sentence summaries."""
    table = odds_ratio_table(result).head(top_k + 1)  # +1 to skip const
    table = table[table["feature"] != "const"].head(top_k)
    sentences: list[str] = []
    for _, row in table.iterrows():
        direction = "raises" if row["coef"] > 0 else "lowers"
        sentences.append(
            f"A one-unit increase in `{row['feature']}` {direction} the odds of cancellation "
            f"by ~{abs(row['or'] - 1) * 100:.1f}% "
            f"(OR={row['or']:.2f}, 95% CI [{row['or_lo']:.2f}, {row['or_hi']:.2f}], p={row['p']:.3g})."
        )
    return sentences


__all__ = ["fit_statsmodels_logit", "odds_ratio_table", "plain_english"]
