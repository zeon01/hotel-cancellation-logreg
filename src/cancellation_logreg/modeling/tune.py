"""Hyperparameter tuning over C with TimeSeriesSplit CV."""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from cancellation_logreg.modeling.train import build_pipeline

log = logging.getLogger(__name__)


def tune_C(
    X: pd.DataFrame,
    y: pd.Series,
    C_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0),
    n_splits: int = 5,
    scoring: str = "average_precision",
) -> tuple[float, pd.DataFrame]:
    """Grid-search C via TimeSeriesSplit CV, optimising PR-AUC (``average_precision``).

    Returns ``(best_C, cv_results_dataframe)``. The dataframe is suitable for plotting in
    the appendix.
    """
    pipe = build_pipeline(X, C=1.0)
    cv = TimeSeriesSplit(n_splits=n_splits)

    grid = GridSearchCV(
        estimator=pipe,
        param_grid={"logreg__C": list(C_grid)},
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        refit=False,
        verbose=1,
    )
    grid.fit(X, y)

    cv_results = pd.DataFrame(grid.cv_results_)[
        ["param_logreg__C", "mean_test_score", "std_test_score", "rank_test_score"]
    ].rename(columns={"param_logreg__C": "C"})
    cv_results = cv_results.sort_values("rank_test_score")
    best_C = float(grid.best_params_["logreg__C"])

    log.info("Best C=%s (CV %s=%.4f)", best_C, scoring, grid.best_score_)
    return best_C, cv_results


def main() -> None:
    import logging as _log

    _log.basicConfig(level=_log.INFO)
    from cancellation_logreg.config import PROCESSED_DIR
    from cancellation_logreg.modeling.splits import split_xy, time_based_split

    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    train, _, _ = time_based_split(df)
    X, y = split_xy(train)
    best_C, results = tune_C(X, y)
    print(f"best_C={best_C}")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
