"""Cleaning, deduplication, leakage guards. Pure-pandas; no fitting on data."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from cancellation_logreg.config import LEAKAGE_COLUMNS, PROCESSED_DIR

log = logging.getLogger(__name__)


_BOOKING_DEFINING_COLS: tuple[str, ...] = (
    "hotel",
    "lead_time",
    "arrival_date_year",
    "arrival_date_month",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "deposit_type",
    "agent",
    "customer_type",
    "adr",
)


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


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop ``reservation_status`` and ``reservation_status_date``.

    These columns encode the outcome and would leak the target. A unit test asserts they
    are absent from ``X`` before training.
    """
    present = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    if present:
        log.info("Dropping leakage columns: %s", present)
    return df.drop(columns=list(LEAKAGE_COLUMNS), errors="ignore")


def remove_invalid_bookings(df: pd.DataFrame) -> pd.DataFrame:
    """Filter rows where ``adr < 0`` (data error) or ``adults+children+babies == 0``."""
    n_before = len(df)
    guests = df["adults"].fillna(0) + df["children"].fillna(0) + df["babies"].fillna(0)
    out = df.loc[(df["adr"] >= 0) & (guests > 0)].copy()
    log.info("Dropped %d invalid bookings (adr<0 or zero-guest).", n_before - len(out))
    return out


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute ``children=0``; convert ``company`` to ``has_company_id``; encode ``agent`` and
    ``country`` missingness."""
    out = df.copy()
    out["children"] = out["children"].fillna(0).astype(int)

    out["has_company_id"] = out["company"].notna().astype(int)
    out = out.drop(columns=["company"])

    # agent is high-cardinality; we keep an explicit "missing" marker before collapsing.
    out["agent"] = out["agent"].astype("Float64").astype(str).where(out["agent"].notna(), "missing")
    out["agent"] = out["agent"].str.replace(r"\.0$", "", regex=True)

    out["country"] = out["country"].fillna("UNK")
    return out


def winsorize_adr(df: pd.DataFrame, upper_q: float = 0.995) -> pd.DataFrame:
    """Cap ``adr`` at the upper_q quantile to reduce influence of extreme errors."""
    out = df.copy()
    cap = float(np.nanquantile(out["adr"], upper_q))
    n_clipped = int((out["adr"] > cap).sum())
    out["adr"] = out["adr"].clip(upper=cap)
    log.info("Winsorized adr at q=%.3f -> cap=%.2f (%d rows clipped).", upper_q, cap, n_clipped)
    return out


def collapse_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse low-information categories.

    - ``meal``: "Undefined" -> "SC"
    - ``country``: top-20 + "OTHER"
    - ``market_segment``: top-10 + "OTHER"
    - ``agent``: top-10 + "OTHER" (or "missing" stays as its own bucket)
    """
    out = df.copy()
    out["meal"] = out["meal"].replace({"Undefined": "SC"})

    def keep_top(series: pd.Series, n: int, other: str = "OTHER") -> pd.Series:
        keep = set(series.value_counts().head(n).index)
        return series.where(series.isin(keep), other)

    out["country"] = keep_top(out["country"], 20)
    out["market_segment"] = keep_top(out["market_segment"], 10)
    out["agent"] = keep_top(out["agent"], 10)
    return out


def add_duplicate_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """Flag rows that look like part of a group / bulk booking via the booking-defining cols."""
    cols = [c for c in _BOOKING_DEFINING_COLS if c in df.columns]
    out = df.copy()
    counts = out.groupby(cols, dropna=False, observed=True).transform("size")
    out["is_likely_group_booking"] = (counts > 1).astype(int)
    n_grp = int(out["is_likely_group_booking"].sum())
    log.info("Tagged %d rows as is_likely_group_booking (~%.1f%%).", n_grp, 100 * n_grp / len(out))
    return out


def build_arrival_date(df: pd.DataFrame) -> pd.DataFrame:
    """Construct ``arrival_date`` and derive ``booking_date`` (= arrival - lead_time)."""
    out = df.copy()
    months = out["arrival_date_month"].map(_MONTH_ORDER).astype("Int64")
    out["arrival_date"] = pd.to_datetime(
        {
            "year": out["arrival_date_year"].astype(int),
            "month": months.astype(int),
            "day": out["arrival_date_day_of_month"].astype(int),
        },
        errors="coerce",
    )
    out["booking_date"] = out["arrival_date"] - pd.to_timedelta(out["lead_time"], unit="D")
    return out


def build_clean_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Orchestrator: leakage-drop -> invalid-filter -> impute -> winsorize -> collapse ->
    duplicate-indicator -> arrival_date.
    """
    log.info("build_clean_frame: input shape=%s", raw.shape)
    out = drop_leakage_columns(raw)
    out = remove_invalid_bookings(out)
    out = impute_missing(out)
    out = winsorize_adr(out)
    out = collapse_categories(out)
    out = add_duplicate_indicator(out)
    out = build_arrival_date(out)
    log.info("build_clean_frame: output shape=%s", out.shape)
    return out


def write_processed() -> None:
    """End-to-end: load_raw -> build_clean_frame -> write parquet."""
    from cancellation_logreg.data import ensure_raw_available, load_raw

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_raw(ensure_raw_available())
    clean = build_clean_frame(raw)
    out_path = PROCESSED_DIR / "clean.parquet"
    clean.to_parquet(out_path, index=False)
    log.info("Wrote %s (rows=%d).", out_path, len(clean))


__all__ = [
    "add_duplicate_indicator",
    "build_arrival_date",
    "build_clean_frame",
    "collapse_categories",
    "drop_leakage_columns",
    "impute_missing",
    "remove_invalid_bookings",
    "winsorize_adr",
    "write_processed",
]
