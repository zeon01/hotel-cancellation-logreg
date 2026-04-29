"""Feature engineering. Pure functions of a clean frame; produces a feature frame."""

from __future__ import annotations

import pandas as pd


def add_total_nights(df: pd.DataFrame) -> pd.DataFrame:
    """``total_nights = weekend_nights + week_nights``."""
    raise NotImplementedError("Phase 2")


def add_total_guests(df: pd.DataFrame) -> pd.DataFrame:
    """``total_guests = adults + children + babies``."""
    raise NotImplementedError("Phase 2")


def add_adr_per_person(df: pd.DataFrame) -> pd.DataFrame:
    """``adr_per_person = adr / max(total_guests, 1)``."""
    raise NotImplementedError("Phase 2")


def add_room_change_flag(df: pd.DataFrame) -> pd.DataFrame:
    """``room_was_changed = reserved_room_type != assigned_room_type``."""
    raise NotImplementedError("Phase 2")


def add_lead_time_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """``is_short_lead`` and ``is_long_lead`` per Supply Ops thresholds."""
    raise NotImplementedError("Phase 2")


def add_cyclical_month(df: pd.DataFrame) -> pd.DataFrame:
    """``arrival_month_sin/cos`` from arrival month."""
    raise NotImplementedError("Phase 2")


def add_prior_cancel_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Smoothed ratio: previous_cancellations / (previous_cancellations + previous_bookings_not_canceled)."""
    raise NotImplementedError("Phase 2: use Beta(1, 1) smoothing")


def add_deposit_flag(df: pd.DataFrame) -> pd.DataFrame:
    """``has_deposit = deposit_type != 'No Deposit'``."""
    raise NotImplementedError("Phase 2")


def add_corporate_flag(df: pd.DataFrame) -> pd.DataFrame:
    """``is_corporate = market_segment in {'Corporate', 'Complementary'}``."""
    raise NotImplementedError("Phase 2")


def build_feature_frame(clean: pd.DataFrame | None = None) -> pd.DataFrame:
    """Apply all feature transforms in order. If ``clean`` is None, read from ``data/processed/``."""
    raise NotImplementedError("Phase 2")


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
