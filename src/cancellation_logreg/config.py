"""Project paths, seeds, and column constants. All paths derive from this file's location."""

from __future__ import annotations

from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"
EXTERNAL_DIR: Path = DATA_DIR / "external"

REPORTS_DIR: Path = ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
DOCS_FIGURES_DIR: Path = ROOT / "docs" / "figures"
MODELS_DIR: Path = ROOT / "models"

SEED: int = 42

RAW_FILENAME: str = "hotel_bookings.csv"
KAGGLE_DATASET: str = "jessemostipak/hotel-booking-demand"
MENDELEY_DOI: str = "10.17632/j83f5fsh6c.1"

TARGET: str = "is_canceled"
LEAKAGE_COLUMNS: tuple[str, ...] = ("reservation_status", "reservation_status_date")

EXPECTED_COLUMNS: tuple[str, ...] = (
    "hotel",
    "is_canceled",
    "lead_time",
    "arrival_date_year",
    "arrival_date_month",
    "arrival_date_week_number",
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
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "reserved_room_type",
    "assigned_room_type",
    "booking_changes",
    "deposit_type",
    "agent",
    "company",
    "days_in_waiting_list",
    "customer_type",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
    "reservation_status",
    "reservation_status_date",
)

NUMERIC_FEATURES: tuple[str, ...] = (
    "lead_time",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "days_in_waiting_list",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
)

LOW_CARD_CATEGORICAL: tuple[str, ...] = (
    "hotel",
    "meal",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "assigned_room_type",
    "deposit_type",
    "customer_type",
)

HIGH_CARD_CATEGORICAL: tuple[str, ...] = ("country", "agent")

TRAIN_END: str = "2016-12-31"
VAL_END: str = "2017-04-30"
