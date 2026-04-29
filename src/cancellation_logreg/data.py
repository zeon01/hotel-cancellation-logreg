"""Dataset acquisition, raw load, and schema validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def download_kaggle(dataset: str, dest: Path) -> Path:
    """Download a Kaggle dataset to ``dest``. Requires ``~/.kaggle/kaggle.json``.

    Returns the path to the downloaded CSV.
    """
    raise NotImplementedError("Phase 2: implement via kaggle.api.dataset_download_files")


def download_mendeley(doi: str, dest: Path) -> Path:
    """Fallback: download the original Mendeley file for the dataset DOI.

    Mirrors the Kaggle path so the rest of the pipeline is agnostic.
    """
    raise NotImplementedError("Phase 2: implement Mendeley fallback")


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Read the raw CSV into a DataFrame."""
    raise NotImplementedError("Phase 2: pd.read_csv with explicit dtypes")


def validate_schema(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` if expected columns are missing or dtypes are unexpected."""
    raise NotImplementedError("Phase 2: assert all EXPECTED_COLUMNS present, log dtype map")


def ensure_raw_available() -> Path:
    """Idempotent: ensure ``data/raw/hotel_bookings.csv`` exists; download if not."""
    raise NotImplementedError(
        "Phase 2: try kaggle, fall back to mendeley, return RAW_DIR / RAW_FILENAME"
    )


__all__ = [
    "download_kaggle",
    "download_mendeley",
    "ensure_raw_available",
    "load_raw",
    "validate_schema",
]
