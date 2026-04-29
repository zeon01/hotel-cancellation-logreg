"""Evaluation: discrimination, calibration, threshold selection, bootstrap CIs,
subgroup grid, cost-sensitivity surface, and figures.

The figures and tables are written to ``reports/figures`` and ``models/`` so the README
and notebooks consume the artefacts directly.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from cancellation_logreg.config import FIGURES_DIR, MODELS_DIR, PROCESSED_DIR
from cancellation_logreg.plotting import set_style
from cancellation_logreg.subgroup import subgroup_metrics
from cancellation_logreg.uncertainty import cluster_bootstrap_metrics, summarise_ci

log = logging.getLogger(__name__)


def discrimination_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
    }


def brier(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    return float(brier_score_loss(y_true, y_proba))


def reliability_table(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_proba, bins) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin": b,
                "bin_low": bins[b],
                "bin_high": bins[b + 1],
                "n": int(mask.sum()),
                "mean_predicted": float(y_proba[mask].mean()),
                "mean_observed": float(y_true[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def expected_cost(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    cost_fp: float = 5.0,
    cost_fn: float = 50.0,
) -> float:
    pred = (y_proba >= threshold).astype(int)
    fp = ((pred == 1) & (y_true == 0)).sum()
    fn = ((pred == 0) & (y_true == 1)).sum()
    return (cost_fp * fp + cost_fn * fn) / len(y_true)


def best_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_fp: float = 5.0,
    cost_fn: float = 50.0,
    grid: np.ndarray | None = None,
) -> float:
    if grid is None:
        grid = np.linspace(0.05, 0.95, 91)
    costs = np.array([expected_cost(y_true, y_proba, t, cost_fp, cost_fn) for t in grid])
    return float(grid[int(np.argmin(costs))])


def cost_sensitivity_grid(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    fp_grid: tuple[float, ...] = (1.0, 2.5, 5.0, 10.0, 20.0),
    fn_grid: tuple[float, ...] = (10.0, 25.0, 50.0, 100.0, 200.0),
) -> pd.DataFrame:
    rows = []
    for fp in fp_grid:
        for fn in fn_grid:
            t = best_threshold(y_true, y_proba, fp, fn)
            c = expected_cost(y_true, y_proba, t, fp, fn)
            rows.append(
                {
                    "cost_fp": fp,
                    "cost_fn": fn,
                    "ratio": fn / fp,
                    "best_threshold": t,
                    "expected_cost": c,
                }
            )
    return pd.DataFrame(rows)


def lift_table(y_true: np.ndarray, y_proba: np.ndarray, n_deciles: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": y_true, "p": y_proba})
    df["decile"] = pd.qcut(df["p"], n_deciles, labels=False, duplicates="drop")
    base_rate = float(df["y"].mean())
    out = (
        df.groupby("decile", observed=True)
        .agg(n=("y", "size"), positives=("y", "sum"), mean_p=("p", "mean"))
        .reset_index()
    )
    out["positive_rate"] = out["positives"] / out["n"]
    out["lift"] = out["positive_rate"] / base_rate
    return out.sort_values("decile", ascending=False).reset_index(drop=True)


# ---- figure savers ---------------------------------------------------------------


def _save_pr_curve(y_true: np.ndarray, y_proba: np.ndarray, out: Path) -> None:
    p, r, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    fig, ax = plt.subplots()
    ax.plot(r, p, lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall - PR-AUC = {ap:.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.savefig(out)
    plt.close(fig)


def _save_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, out: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, lw=2)
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC - AUC = {auc:.3f}")
    fig.savefig(out)
    plt.close(fig)


def _save_calibration(y_true: np.ndarray, y_proba: np.ndarray, out: Path) -> None:
    table = reliability_table(y_true, y_proba, n_bins=10)
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Perfect")
    ax.plot(table["mean_predicted"], table["mean_observed"], "o-", lw=2, label="Calibrated logreg")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed cancellation rate")
    ax.set_title(f"Calibration - Brier = {brier(y_true, y_proba):.4f}")
    ax.legend(loc="upper left")
    fig.savefig(out)
    plt.close(fig)


def _save_confusion(y_true: np.ndarray, y_pred: np.ndarray, out: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
        ax=ax,
    )
    ax.set_title("Confusion matrix at cost-optimal threshold")
    fig.savefig(out)
    plt.close(fig)


def _save_cost_surface(grid: pd.DataFrame, out: Path) -> None:
    pivot_t = grid.pivot_table(index="cost_fn", columns="cost_fp", values="best_threshold")
    pivot_c = grid.pivot_table(index="cost_fn", columns="cost_fp", values="expected_cost")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(pivot_t, annot=True, fmt=".2f", cmap="viridis", ax=axes[0])
    axes[0].set_title("Cost-optimal threshold")
    axes[0].invert_yaxis()
    sns.heatmap(pivot_c, annot=True, fmt=".2f", cmap="rocket_r", ax=axes[1])
    axes[1].set_title("Expected cost / row")
    axes[1].invert_yaxis()
    fig.suptitle("Cost-sensitivity surface (cost_fp x cost_fn)")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    import logging as _log

    _log.basicConfig(level=_log.INFO)
    set_style()

    from cancellation_logreg.modeling.splits import split_xy, time_based_split

    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    _, val, test = time_based_split(df)

    model = joblib.load(MODELS_DIR / "logreg_calibrated.joblib")

    X_val, y_val = split_xy(val)
    X_test, y_test = split_xy(test)

    proba_val = model.predict_proba(X_val)[:, 1]
    proba_test = model.predict_proba(X_test)[:, 1]

    threshold = best_threshold(y_val.to_numpy(), proba_val)
    log.info("Cost-optimal threshold (val): %.3f", threshold)

    metrics_test = discrimination_metrics(y_test.to_numpy(), proba_test)
    metrics_test["brier"] = brier(y_test.to_numpy(), proba_test)
    metrics_test["threshold"] = threshold
    metrics_test["expected_cost_per_row"] = expected_cost(y_test.to_numpy(), proba_test, threshold)

    pred_test = (proba_test >= threshold).astype(int)
    metrics_test["precision"] = float(precision_score(y_test, pred_test))
    metrics_test["recall"] = float(recall_score(y_test, pred_test))
    metrics_test["f1"] = float(f1_score(y_test, pred_test))

    log.info("Test metrics: %s", metrics_test)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    pd.Series(metrics_test).to_csv(MODELS_DIR / "test_metrics.csv")

    # ---- bootstrap CIs (cluster by month) -----------------------------------------
    cluster_col = (
        test["arrival_date"].dt.to_period("M").astype(str)
        if "arrival_date" in test.columns
        else pd.Series(np.zeros(len(test)))
    )
    log.info("Bootstrapping CIs (1000 replicates, cluster by month)...")
    reps = cluster_bootstrap_metrics(
        y_test.to_numpy(), proba_test, cluster_col.to_numpy(), n_boot=1000
    )
    summary = summarise_ci(reps)
    summary.to_csv(MODELS_DIR / "metric_cis.csv", index=False)
    log.info("CI summary:\n%s", summary.to_string(index=False))

    # ---- subgroup grid -------------------------------------------------------------
    sub = subgroup_metrics(test, proba_test)
    sub.to_csv(MODELS_DIR / "subgroup_metrics.csv", index=False)
    log.info("Subgroup metrics rows=%d", len(sub))

    # ---- cost-sensitivity surface --------------------------------------------------
    cost_grid = cost_sensitivity_grid(y_test.to_numpy(), proba_test)
    cost_grid.to_csv(MODELS_DIR / "cost_surface.csv", index=False)

    # ---- lift ---------------------------------------------------------------------
    lift = lift_table(y_test.to_numpy(), proba_test)
    lift.to_csv(MODELS_DIR / "test_lift.csv", index=False)
    log.info(
        "Top-decile lift: %.2fx (positive rate=%.3f)",
        lift.iloc[0]["lift"],
        lift.iloc[0]["positive_rate"],
    )

    # ---- figures ------------------------------------------------------------------
    _save_pr_curve(y_test.to_numpy(), proba_test, FIGURES_DIR / "03_pr_curve.png")
    _save_roc_curve(y_test.to_numpy(), proba_test, FIGURES_DIR / "03_roc_curve.png")
    _save_calibration(y_test.to_numpy(), proba_test, FIGURES_DIR / "03_calibration_curve.png")
    _save_confusion(y_test.to_numpy(), pred_test, FIGURES_DIR / "03_confusion_matrix.png")
    _save_cost_surface(cost_grid, FIGURES_DIR / "05_cost_surface.png")

    log.info("Figures saved to %s", FIGURES_DIR)


if __name__ == "__main__":
    main()
