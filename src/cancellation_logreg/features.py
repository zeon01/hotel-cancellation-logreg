"""Feature engineering. Pure functions of a clean frame; produces a feature frame."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from cancellation_logreg.config import PROCESSED_DIR

log = logging.getLogger(__name__)


_MONTH_ORDER: dict[str, int] = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}


def add_total_nights(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["total_nights"] = out["stays_in_weekend_nights"] + out["stays_in_week_nights"]
    return out


def add_total_guests(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["total_guests"] = (
        out["adults"].fillna(0) + out["children"].fillna(0) + out["babies"].fillna(0)
    )
    return out


def add_adr_per_person(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "total_guests" not in out.columns:
        out = add_total_guests(out)
    out["adr_per_person"] = out["adr"] / out["total_guests"].clip(lower=1)
    return out


def add_room_change_flag(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["room_was_changed"] = (out["reserved_room_type"] != out["assigned_room_type"]).astype(int)
    return out


def add_lead_time_buckets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["is_short_lead"] = (out["lead_time"] <= 7).astype(int)
    out["is_long_lead"] = (out["lead_time"] > 180).astype(int)
    return out


def add_cyclical_month(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "arrival_date" in out.columns:
        m = out["arrival_date"].dt.month
    else:
        m = out["arrival_date_month"].map(_MONTH_ORDER).astype(int)
    out["arrival_month_sin"] = np.sin(2 * np.pi * m / 12)
    out["arrival_month_cos"] = np.cos(2 * np.pi * m / 12)
    return out


def add_prior_cancel_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Beta(1,1)-smoothed previous-cancellation rate."""
    out = df.copy()
    pc = out["previous_cancellations"].astype(float)
    pn = out["previous_bookings_not_canceled"].astype(float)
    out["prior_cancel_rate"] = (pc + 1) / (pc + pn + 2)
    return out


def add_deposit_flag(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["has_deposit"] = (out["deposit_type"] != "No Deposit").astype(int)
    return out


def add_corporate_flag(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["is_corporate"] = out["market_segment"].isin({"Corporate", "Complementary"}).astype(int)
    return out


def build_feature_frame(clean: pd.DataFrame | None = None) -> pd.DataFrame:
    """Apply all feature transforms in order. If ``clean`` is None, read processed parquet."""
    if clean is None:
        clean = pd.read_parquet(PROCESSED_DIR / "clean.parquet")
    log.info("build_feature_frame: input shape=%s", clean.shape)

    out = clean
    out = add_total_nights(out)
    out = add_total_guests(out)
    out = add_adr_per_person(out)
    out = add_room_change_flag(out)
    out = add_lead_time_buckets(out)
    out = add_cyclical_month(out)
    out = add_prior_cancel_rate(out)
    out = add_deposit_flag(out)
    out = add_corporate_flag(out)

    out_path = PROCESSED_DIR / "features.parquet"
    out.to_parquet(out_path, index=False)
    log.info("Wrote %s; output shape=%s", out_path, out.shape)
    return out


__all__ = [
    "add_adr_per_person",
    "add_corporate_flag",
    "add_cyclical_month",
    "add_deposit_flag",
    "add_lead_time_buckets",
    "add_prior_cancel_rate",
    "add_room_change_flag",
    "add_total_guests",
    "add_total_nights",
    "build_feature_frame",
]
