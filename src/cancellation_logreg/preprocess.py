"""Cleaning, deduplication, leakage guards. Pure-pandas; no fitting on data."""

from __future__ import annotations

import pandas as pd


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop ``reservation_status`` and ``reservation_status_date`` from ``df``.

    These columns encode the outcome and would leak the target if used as features.
    A unit test asserts they are absent from ``X`` before training.
    """
    raise NotImplementedError(
        "Phase 2: return df.drop(columns=list(LEAKAGE_COLUMNS), errors='ignore') with logging"
    )


def remove_invalid_bookings(df: pd.DataFrame) -> pd.DataFrame:
    """Filter rows where ``adr < 0`` (data error) or ``adults+children+babies == 0``."""
    raise NotImplementedError("Phase 2")


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute ``children=0``, ``agent=-1`` (cat 'missing'), ``has_company_id`` from company,
    ``country='UNK'`` if missing.
    """
    raise NotImplementedError("Phase 2")


def winsorize_adr(df: pd.DataFrame, upper_q: float = 0.995) -> pd.DataFrame:
    """Cap ``adr`` at the upper_q quantile."""
    raise NotImplementedError("Phase 2")


def collapse_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse low-information categories.

    - meal: "Undefined" -> "SC"
    - country: top-20 + "OTHER"
    - agent / market_segment: top-10 + "OTHER"
    """
    raise NotImplementedError("Phase 2")


def add_duplicate_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``is_likely_group_booking`` based on duplicate rows across booking-defining cols."""
    raise NotImplementedError("Phase 2")


def build_arrival_date(df: pd.DataFrame) -> pd.DataFrame:
    """Construct ``arrival_date`` from year/month/day; derive ``booking_date``."""
    raise NotImplementedError("Phase 2")


def build_clean_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Orchestrator: leakage drop -> invalid filter -> impute -> winsorize -> collapse ->
    duplicate-indicator -> arrival_date.
    """
    raise NotImplementedError("Phase 2: chain the above and return a single clean frame")


def write_processed() -> None:
    """End-to-end: load_raw -> build_clean_frame -> write parquet to data/processed/."""
    raise NotImplementedError("Phase 2: invoked by 'make data'")


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
