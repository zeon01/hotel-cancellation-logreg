"""End-to-end orchestrator invoked by ``make all``.

By default: download -> clean -> features -> train -> evaluate.
With ``--report`` also renders the coefficient-forest plot used by the README.
"""

from __future__ import annotations

import argparse
import logging
import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from cancellation_logreg.config import FIGURES_DIR, MODELS_DIR, PROCESSED_DIR
from cancellation_logreg.interpret import fit_statsmodels_logit, odds_ratio_table
from cancellation_logreg.modeling.splits import split_xy, time_based_split
from cancellation_logreg.plotting import set_style

log = logging.getLogger(__name__)


def render_coefficient_forest_plot() -> None:
    """Fit statsmodels Logit on standardised numeric design matrix; save forest plot."""
    set_style()

    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    train, val, _ = time_based_split(df)
    fit_df = pd.concat([train, val], ignore_index=True)

    # Use the fitted preprocessor from the uncalibrated pipeline so feature names match.
    pipe = joblib.load(MODELS_DIR / "logreg_uncalibrated.joblib")
    pre = pipe.named_steps["preprocessor"]
    X_design = pre.transform(split_xy(fit_df)[0])

    feature_names = list(pre.get_feature_names_out())
    X_design_df = pd.DataFrame(X_design, columns=feature_names)

    y_fit = fit_df["is_canceled"]
    # Top-30 features by sklearn |coef| to keep the design matrix tractable for statsmodels.
    sk_coefs = pd.Series(pipe.named_steps["logreg"].coef_.ravel(), index=feature_names)
    top_cols = sk_coefs.abs().sort_values(ascending=False).head(30).index.tolist()
    X_top = X_design_df[top_cols]

    result = fit_statsmodels_logit(X_top, y_fit)
    table = odds_ratio_table(result)
    # Drop intercept for the plot.
    table = table[table["feature"] != "const"].head(15)
    table = table.iloc[::-1]  # plot smallest at top

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.errorbar(
        table["or"],
        range(len(table)),
        xerr=[table["or"] - table["or_lo"], table["or_hi"] - table["or"]],
        fmt="o",
        color="#205493",
        ecolor="#999",
        capsize=3,
    )
    ax.axvline(1.0, ls="--", color="gray", lw=1)
    ax.set_yticks(range(len(table)))
    ax.set_yticklabels(table["feature"])
    ax.set_xlabel("Odds ratio (95% CI, HC3)")
    ax.set_title("Top 15 cancellation drivers — calibrated logistic regression")
    fig.savefig(FIGURES_DIR / "04_top_coefficients_forest.png")
    plt.close(fig)
    table.to_csv(MODELS_DIR / "top_coefficients.csv", index=False)
    log.info("Coefficient forest plot saved.")


def main() -> int:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true", help="also render report figures")
    args = parser.parse_args()
    if args.report:
        render_coefficient_forest_plot()
    return 0


if __name__ == "__main__":
    sys.exit(main())
