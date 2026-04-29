"""Download the Antonio hotel-bookings dataset.

Tries Kaggle first (requires ``~/.kaggle/kaggle.json`` or ``KAGGLE_API_TOKEN``), then
falls back to the Mendeley mirror so a reviewer without Kaggle credentials can still run
the pipeline.
"""

from __future__ import annotations

import logging
import sys

from cancellation_logreg.data import ensure_raw_available


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    path = ensure_raw_available()
    print(f"raw available at: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
