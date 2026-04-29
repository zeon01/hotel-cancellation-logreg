"""Download the Antonio hotel-bookings dataset.

Tries Kaggle first (requires ``~/.kaggle/kaggle.json``), then falls back to the Mendeley
mirror so a reviewer without Kaggle credentials can still run the pipeline.
"""

from __future__ import annotations

import sys


def main() -> int:
    raise NotImplementedError(
        "Phase 2: try kaggle.api.dataset_download_files, fall back to Mendeley URL"
    )


if __name__ == "__main__":
    sys.exit(main())
