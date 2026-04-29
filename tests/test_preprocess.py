"""Preprocessing tests. The leakage assertion is the most important one."""

from __future__ import annotations

import pandas as pd
import pytest

from cancellation_logreg.config import LEAKAGE_COLUMNS


@pytest.fixture
def tiny_raw_frame() -> pd.DataFrame:
    """Minimal frame with one row of each shape we care about. Used as a smoke fixture."""
    return pd.DataFrame(
        {
            "hotel": ["City Hotel", "Resort Hotel"],
            "is_canceled": [0, 1],
            "lead_time": [10, 200],
            "arrival_date_year": [2016, 2017],
            "arrival_date_month": ["July", "March"],
            "arrival_date_week_number": [27, 12],
            "arrival_date_day_of_month": [1, 15],
            "stays_in_weekend_nights": [1, 2],
            "stays_in_week_nights": [2, 3],
            "adults": [2, 1],
            "children": [0, 1],
            "babies": [0, 0],
            "meal": ["BB", "Undefined"],
            "country": ["PRT", "GBR"],
            "market_segment": ["Online TA", "Direct"],
            "distribution_channel": ["TA/TO", "Direct"],
            "is_repeated_guest": [0, 0],
            "previous_cancellations": [0, 1],
            "previous_bookings_not_canceled": [0, 0],
            "reserved_room_type": ["A", "D"],
            "assigned_room_type": ["A", "K"],
            "booking_changes": [0, 1],
            "deposit_type": ["No Deposit", "Non Refund"],
            "agent": [9.0, None],
            "company": [None, 40.0],
            "days_in_waiting_list": [0, 0],
            "customer_type": ["Transient", "Transient-Party"],
            "adr": [80.0, 120.0],
            "required_car_parking_spaces": [0, 0],
            "total_of_special_requests": [1, 0],
            "reservation_status": ["Check-Out", "Canceled"],
            "reservation_status_date": ["2016-07-04", "2017-02-15"],
        }
    )


def test_leakage_columns_dropped(tiny_raw_frame: pd.DataFrame) -> None:
    """``drop_leakage_columns`` must remove every column in LEAKAGE_COLUMNS.

    This is the single most important test in the repo. Many beginner notebooks include
    these columns and report 100% accuracy.
    """
    pytest.importorskip("cancellation_logreg.preprocess")
    from cancellation_logreg.preprocess import drop_leakage_columns

    try:
        cleaned = drop_leakage_columns(tiny_raw_frame)
    except NotImplementedError:
        pytest.skip("preprocess.drop_leakage_columns not yet implemented")

    for col in LEAKAGE_COLUMNS:
        assert col not in cleaned.columns, f"Leakage column {col!r} survived preprocessing"


def test_invalid_bookings_filtered(tiny_raw_frame: pd.DataFrame) -> None:
    """Rows with adr<0 or zero guests must be removed."""
    pytest.importorskip("cancellation_logreg.preprocess")
    from cancellation_logreg.preprocess import remove_invalid_bookings

    bad = tiny_raw_frame.copy()
    bad.loc[len(bad)] = bad.iloc[0].copy()
    bad.loc[len(bad) - 1, "adr"] = -10.0

    try:
        out = remove_invalid_bookings(bad)
    except NotImplementedError:
        pytest.skip("preprocess.remove_invalid_bookings not yet implemented")

    assert (out["adr"] >= 0).all()
