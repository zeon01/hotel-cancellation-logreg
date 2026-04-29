"""Fit the L2-regularised logistic-regression pipeline, calibrate, persist."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from cancellation_logreg.config import (
    HIGH_CARD_CATEGORICAL,
    LOW_CARD_CATEGORICAL,
    MODELS_DIR,
    NUMERIC_FEATURES,
    PROCESSED_DIR,
    SEED,
)

log = logging.getLogger(__name__)


# Engineered numeric / categorical features added by features.py
ENGINEERED_NUMERIC: tuple[str, ...] = (
    "total_nights",
    "total_guests",
    "adr_per_person",
    "arrival_month_sin",
    "arrival_month_cos",
    "prior_cancel_rate",
)

ENGINEERED_BINARY: tuple[str, ...] = (
    "room_was_changed",
    "is_short_lead",
    "is_long_lead",
    "has_deposit",
    "is_corporate",
    "has_company_id",
    "is_likely_group_booking",
)


def get_feature_columns(X: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Return ``(numeric, low_card_cat, high_card_cat)`` lists, restricted to columns in X."""
    numeric = [
        c for c in (*NUMERIC_FEATURES, *ENGINEERED_NUMERIC, *ENGINEERED_BINARY) if c in X.columns
    ]
    low_card = [c for c in LOW_CARD_CATEGORICAL if c in X.columns]
    high_card = [c for c in HIGH_CARD_CATEGORICAL if c in X.columns]
    return numeric, low_card, high_card


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Numeric standardisation, one-hot for low-card, target encoding for high-card.

    TargetEncoder is fit per CV fold inside any CV-based wrapper (``CalibratedClassifierCV``,
    ``GridSearchCV``), so leakage from the validation portion is avoided automatically.
    """
    numeric, low_card, high_card = get_feature_columns(X)
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=True, with_std=True), numeric),
            (
                "low_card",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=0.01,
                    sparse_output=False,
                ),
                low_card,
            ),
            (
                "high_card",
                TargetEncoder(smoothing=10.0),
                high_card,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(X: pd.DataFrame, C: float = 1.0) -> Pipeline:
    """Build ``ColumnTransformer -> LogisticRegression(penalty='l2', class_weight='balanced')``."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X)),
            (
                "logreg",
                LogisticRegression(
                    penalty="l2",
                    C=C,
                    solver="lbfgs",
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=SEED,
                ),
            ),
        ]
    )


def fit(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> Pipeline:
    return model.fit(X, y)


def fit_calibrated(
    model: Pipeline, X: pd.DataFrame, y: pd.Series, cv: int = 5
) -> CalibratedClassifierCV:
    """Wrap the pipeline in ``CalibratedClassifierCV(method='isotonic')`` and fit."""
    calibrated = CalibratedClassifierCV(model, method="isotonic", cv=cv)
    return calibrated.fit(X, y)


def save(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    log.info("Wrote %s", path)


def main() -> None:
    import logging as _log

    _log.basicConfig(level=_log.INFO)

    from cancellation_logreg.modeling.splits import split_xy, time_based_split

    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    train, val, _ = time_based_split(df)

    # Tune C on train-only, then refit + calibrate on the union (train + val).
    from cancellation_logreg.modeling.tune import tune_C

    X_train, y_train = split_xy(train)
    best_C, cv_results = tune_C(X_train, y_train)
    log.info("Best C from tuning: %s", best_C)

    fit_df = pd.concat([train, val], ignore_index=True)
    X_fit, y_fit = split_xy(fit_df)
    pipe = build_pipeline(X_fit, C=best_C)
    calibrated = fit_calibrated(pipe, X_fit, y_fit, cv=5)
    save(calibrated, MODELS_DIR / "logreg_calibrated.joblib")

    # Also persist the uncalibrated pipeline for diagnostics / coefficient inspection.
    uncal = build_pipeline(X_fit, C=best_C)
    fit(uncal, X_fit, y_fit)
    save(uncal, MODELS_DIR / "logreg_uncalibrated.joblib")

    # Persist tuning artefact for the appendix notebook.
    cv_results.to_csv(MODELS_DIR / "tune_cv_results.csv", index=False)


if __name__ == "__main__":
    main()
