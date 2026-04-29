"""Hyperparameter tuning over C with TimeSeriesSplit CV."""

from __future__ import annotations


def tune_C(X, y, C_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)):
    """Grid-search C via 5-fold TimeSeriesSplit on PR-AUC. Return the best estimator and CV results."""
    raise NotImplementedError("Phase 2")


def main() -> None:
    raise NotImplementedError("Phase 2")


if __name__ == "__main__":
    main()
