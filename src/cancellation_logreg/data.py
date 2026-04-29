"""Dataset acquisition, raw load, and schema validation."""

from __future__ import annotations

import logging
import os
import zipfile
from pathlib import Path

import pandas as pd
import requests

from cancellation_logreg.config import (
    EXPECTED_COLUMNS,
    KAGGLE_DATASET,
    MENDELEY_DOI,
    RAW_DIR,
    RAW_FILENAME,
)

log = logging.getLogger(__name__)


def _ensure_kaggle_credentials() -> bool:
    """Return True if kaggle.json or KAGGLE_API_TOKEN is present."""
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    return kaggle_json.exists()


def download_kaggle(dataset: str = KAGGLE_DATASET, dest: Path = RAW_DIR) -> Path:
    """Download a Kaggle dataset to ``dest`` and return the path to ``hotel_bookings.csv``."""
    dest.mkdir(parents=True, exist_ok=True)

    target = dest / RAW_FILENAME
    if target.exists():
        log.info("Kaggle dataset already present at %s", target)
        return target

    if not _ensure_kaggle_credentials():
        raise RuntimeError(
            "No Kaggle credentials found. Provide ~/.kaggle/kaggle.json (mode 600) or "
            "set KAGGLE_API_TOKEN."
        )

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    log.info("Downloading %s to %s", dataset, dest)
    api.dataset_download_files(dataset, path=str(dest), unzip=True, quiet=False)

    if not target.exists():
        # Some Kaggle datasets land in a subdirectory; flatten if so.
        for csv in dest.rglob(RAW_FILENAME):
            csv.replace(target)
            break

    if not target.exists():
        raise FileNotFoundError(f"Kaggle download finished but {RAW_FILENAME} not found in {dest}")
    return target


def download_mendeley(doi: str = MENDELEY_DOI, dest: Path = RAW_DIR) -> Path:
    """Fallback: download from Mendeley's data.mendeley.com.

    The Mendeley dataset has a different file layout (separate H1/H2 files); this
    function consolidates them into a single ``hotel_bookings.csv`` with a ``hotel`` column
    so the rest of the pipeline is unchanged.
    """
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / RAW_FILENAME
    if target.exists():
        return target

    # Mendeley download URLs are versioned; we fetch the dataset zip and extract.
    zip_url = f"https://data.mendeley.com/api/datasets-v2/datasets/{doi}/files-archive"
    log.info("Mendeley fallback: GET %s", zip_url)

    resp = requests.get(zip_url, stream=True, timeout=120)
    resp.raise_for_status()
    zip_path = dest / "mendeley.zip"
    with zip_path.open("wb") as fh:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            fh.write(chunk)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    zip_path.unlink(missing_ok=True)

    h1 = dest / "H1.csv"
    h2 = dest / "H2.csv"
    if h1.exists() and h2.exists():
        df_h1 = pd.read_csv(h1)
        df_h2 = pd.read_csv(h2)
        df_h1["hotel"] = "Resort Hotel"
        df_h2["hotel"] = "City Hotel"
        merged = pd.concat([df_h1, df_h2], ignore_index=True)
        merged.to_csv(target, index=False)
    elif (dest / "hotel_bookings.csv").exists():
        # Some mirrors already ship the merged file
        pass
    else:
        raise FileNotFoundError(
            f"Mendeley archive extracted but expected files not found in {dest}"
        )

    return target


def ensure_raw_available() -> Path:
    """Idempotent: ensure the raw CSV exists; download via Kaggle, fall back to Mendeley."""
    target = RAW_DIR / RAW_FILENAME
    if target.exists():
        return target
    try:
        return download_kaggle()
    except Exception as exc:
        log.warning("Kaggle download failed (%s); falling back to Mendeley.", exc)
        return download_mendeley()


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Read the raw CSV, parsing ``reservation_status_date`` into datetime."""
    p = path or (RAW_DIR / RAW_FILENAME)
    if not p.exists():
        p = ensure_raw_available()
    df = pd.read_csv(
        p,
        parse_dates=["reservation_status_date"],
        dtype={"agent": "Float64", "company": "Float64"},
    )
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` if expected columns are missing.

    Logs the dtype map for any reviewer who wants to verify the load matches Antonio et
    al. (2019).
    """
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Raw frame missing expected columns: {sorted(missing)}")
    log.info("Schema valid. Shape=%s, dtypes:\n%s", df.shape, df.dtypes.to_string())


__all__ = [
    "download_kaggle",
    "download_mendeley",
    "ensure_raw_available",
    "load_raw",
    "validate_schema",
]
