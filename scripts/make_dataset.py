"""End-to-end orchestrator invoked by ``make all``.

By default: load -> clean -> features -> persist. With ``--report`` also renders headline
figures into ``reports/figures/``.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true", help="also render report figures")
    args = parser.parse_args()
    raise NotImplementedError(f"Phase 2: orchestrate the full pipeline (report={args.report})")


if __name__ == "__main__":
    sys.exit(main())
