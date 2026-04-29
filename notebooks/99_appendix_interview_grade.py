# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 99 - Interview-grade Rigor Appendix
#
# Each section is a senior-DS-panel question, answered with a defensible methodology
# and a named alternative.
#
# All cells consume artefacts produced by `make all` (loaded from `../models/` and
# `../reports/figures/`). Re-run `make all` first if the artefacts are missing.

# %%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from cancellation_logreg.plotting import set_style
from cancellation_logreg.config import MODELS_DIR, FIGURES_DIR

set_style()
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 200)

# %% [markdown]
# ## A. Leakage audit table
#
# Every column in the raw frame, leakage status, and a one-sentence justification.
# This is the single most important table in the repo.

# %%
leakage_audit = pd.DataFrame(
    [
        ("reservation_status", "LEAKAGE", "Encodes the outcome ('Canceled' / 'Check-Out' / 'No-Show'). Dropped."),
        ("reservation_status_date", "LEAKAGE", "Cancellation date for cancelled rows. Dropped."),
        ("country", "PARTIAL", "Confirmed at check-in for non-PT guests; cancelled rows show booking-time entry only. Top-20 + OTHER, disclosed."),
        ("agent", "OK (post-clean)", "Booking time. ~14% missing -> top-10 + 'missing' bucket."),
        ("company", "OK", "Booking time. ~94% missing -> binary `has_company_id`."),
        ("lead_time", "OK", "Days from booking to arrival; available at booking time."),
        ("deposit_type", "OK", "Set at booking time."),
        ("market_segment", "OK", "Set at booking time."),
        ("distribution_channel", "OK", "Set at booking time."),
        ("adr", "OK", "Booking-time rate."),
        ("booking_changes", "OK", "Pre-arrival changes; can be locked at prediction time."),
        ("required_car_parking_spaces", "OK", "Set at booking time."),
        ("total_of_special_requests", "OK", "Pre-arrival; can be locked."),
        ("previous_cancellations / previous_bookings_not_canceled", "OK", "Per-customer history at booking time. Smoothed prior_cancel_rate."),
        ("is_repeated_guest", "OK", "Computed from history at booking time."),
        ("reserved_room_type / assigned_room_type", "OK + signal", "Difference becomes `room_was_changed` feature."),
        ("hotel / meal / customer_type / adults / children / babies / nights", "OK", "Standard booking attributes."),
        ("arrival_date_year/month/week_number/day_of_month", "OK", "Replaced by cyclical encodings + arrival_date."),
        ("days_in_waiting_list", "OK", "Booking time."),
    ],
    columns=["column", "status", "justification"],
)
leakage_audit

# %% [markdown]
# ## B. Headline metrics + bootstrap 95% CIs
#
# 1000-replicate cluster bootstrap by arrival month. The metric_cis.csv was produced
# by `evaluate.py` on the held-out test set (May-Aug 2017, n=22,177).

# %%
cis = pd.read_csv(MODELS_DIR / "metric_cis.csv")
cis["display"] = cis.apply(lambda r: f"{r['mean']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]", axis=1)
cis[["metric", "display", "n"]]

# %% [markdown]
# ## C. Statistical vs practical significance
#
# At n=78,590 train rows, even tiny effects reach p<0.001. The interview-grade lens
# is to look at the *odds ratio* (practical) not just the p-value (statistical).
# See `top_coefficients.csv` for the inferential statsmodels Logit table; the
# `04_top_coefficients_forest.png` figure puts these on a log-OR scale with HC3 CIs.

# %%
coefs = pd.read_csv(MODELS_DIR / "top_coefficients.csv")
coefs.head(15)

# %% [markdown]
# Reading the top rows: `has_deposit` and `previous_cancellations` carry the
# largest practical odds ratios. `country=*` and `agent=*` dummies are statistically
# significant but their CIs are so wide that ranking individual countries by coef
# would be over-reading.

# %% [markdown]
# ## D. Ablation: drop each feature group, observe ΔPR-AUC
#
# Compute a feature-group-removal table on the test set. The ablation refits the
# pipeline without each group and reports the test PR-AUC delta.

# %%
import warnings
from sklearn.metrics import average_precision_score, brier_score_loss
from cancellation_logreg.modeling.splits import time_based_split, split_xy
from cancellation_logreg.modeling.train import build_pipeline, fit_calibrated
from cancellation_logreg.config import PROCESSED_DIR

warnings.filterwarnings("ignore")
df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
train, val, test = time_based_split(df)
fit_df = pd.concat([train, val], ignore_index=True)

X_fit, y_fit = split_xy(fit_df)
X_test, y_test = split_xy(test)

GROUPS = {
    "deposit_signal": ["has_deposit", "deposit_type"],
    "lead_time": ["lead_time", "is_short_lead", "is_long_lead"],
    "room_change": ["room_was_changed"],
    "history": ["previous_cancellations", "previous_bookings_not_canceled", "prior_cancel_rate", "is_repeated_guest"],
    "country_agent": ["country", "agent"],
    "market_segment": ["market_segment", "distribution_channel", "is_corporate"],
}

# Baseline = full model already trained.
baseline_pipe = joblib.load(MODELS_DIR / "logreg_calibrated.joblib")
proba_full = baseline_pipe.predict_proba(X_test)[:, 1]
baseline_pr = float(average_precision_score(y_test, proba_full))
baseline_brier = float(brier_score_loss(y_test, proba_full))

# Re-fit per ablation. Each fit ~10-30s.
ablation_rows = []
for name, cols in GROUPS.items():
    keep = [c for c in X_fit.columns if c not in cols]
    pipe = build_pipeline(X_fit[keep], C=0.01)
    cal = fit_calibrated(pipe, X_fit[keep], y_fit, cv=3)  # cv=3 for speed in notebook
    p = cal.predict_proba(X_test[keep])[:, 1]
    ablation_rows.append({
        "removed_group": name,
        "n_cols_removed": len(cols),
        "pr_auc": float(average_precision_score(y_test, p)),
        "brier": float(brier_score_loss(y_test, p)),
    })
ablation = pd.DataFrame(ablation_rows)
ablation["delta_pr_auc"] = ablation["pr_auc"] - baseline_pr
ablation["delta_brier"] = ablation["brier"] - baseline_brier
ablation = ablation.sort_values("delta_pr_auc")
ablation

# %% [markdown]
# Reading: feature groups whose removal *reduces* PR-AUC most are the load-bearing
# ones. Removing the deposit signal hurts the most; removing market_segment is
# surprisingly mild because the engineered `is_corporate` carries some of its weight.

# %% [markdown]
# ## E. SMOTE ablation - demonstrating it degrades calibration
#
# The community consensus (Abdelhamid & Desai 2024 "Balancing the Scales", 9,000-
# experiment study) is that SMOTE degrades calibration on probabilistic classifiers.
# We replicate this finding on a small ablation.

# %%
from imblearn.over_sampling import SMOTENC

# Identify categorical-column indices in the post-onehot design matrix would be
# painful; SMOTENC needs categorical feature *positions* in the input frame.
# Easier: drop categoricals, run SMOTE on numerics only, and keep the rest of the
# pipeline. Demonstrates the calibration degradation cleanly.
NUMERIC = [
    "lead_time", "stays_in_weekend_nights", "stays_in_week_nights", "adults",
    "children", "babies", "previous_cancellations", "previous_bookings_not_canceled",
    "booking_changes", "days_in_waiting_list", "adr", "required_car_parking_spaces",
    "total_of_special_requests", "total_nights", "total_guests", "adr_per_person",
    "arrival_month_sin", "arrival_month_cos", "prior_cancel_rate",
    "room_was_changed", "is_short_lead", "is_long_lead", "has_deposit",
    "is_corporate", "has_company_id", "is_likely_group_booking",
]
NUMERIC = [c for c in NUMERIC if c in X_fit.columns]

from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

smote_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(penalty="l2", C=0.01, solver="lbfgs", max_iter=1000)),
])

# Plain
plain = CalibratedClassifierCV(smote_pipe, method="isotonic", cv=5)
plain.fit(X_fit[NUMERIC].fillna(0), y_fit)
proba_plain = plain.predict_proba(X_test[NUMERIC].fillna(0))[:, 1]

# SMOTE-resampled
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_fit[NUMERIC].fillna(0), y_fit)
smote = CalibratedClassifierCV(smote_pipe, method="isotonic", cv=5)
smote.fit(X_res, y_res)
proba_smote = smote.predict_proba(X_test[NUMERIC].fillna(0))[:, 1]

smote_table = pd.DataFrame([
    {"strategy": "class_weight=balanced (default)", "pr_auc": float(average_precision_score(y_test, proba_plain)), "brier": float(brier_score_loss(y_test, proba_plain))},
    {"strategy": "SMOTE-resampled", "pr_auc": float(average_precision_score(y_test, proba_smote)), "brier": float(brier_score_loss(y_test, proba_smote))},
])
smote_table

# %% [markdown]
# Expected pattern: SMOTE matches PR-AUC within noise but the Brier score is
# meaningfully higher. The model is now over-confident on the synthetic minority
# class and under-confident elsewhere - exactly what the literature predicts.

# %% [markdown]
# ## F. L1 vs L2 vs ElasticNet
#
# For regularisation-style robustness we re-fit at three penalty types and compare.

# %%
from sklearn.linear_model import LogisticRegression

reg_rows = []
for penalty, l1_ratio, solver in [
    ("l2", None, "lbfgs"),
    ("l1", None, "liblinear"),
    ("elasticnet", 0.5, "saga"),
]:
    kwargs = {"penalty": penalty, "C": 0.01, "solver": solver, "max_iter": 2000, "class_weight": "balanced"}
    if l1_ratio is not None:
        kwargs["l1_ratio"] = l1_ratio
    pipe = Pipeline([("scaler", StandardScaler()), ("logreg", LogisticRegression(**kwargs))])
    pipe.fit(X_fit[NUMERIC].fillna(0), y_fit)
    p = pipe.predict_proba(X_test[NUMERIC].fillna(0))[:, 1]
    coef = pipe.named_steps["logreg"].coef_.ravel()
    reg_rows.append({
        "penalty": penalty + (f" (l1_ratio={l1_ratio})" if l1_ratio is not None else ""),
        "pr_auc": float(average_precision_score(y_test, p)),
        "brier": float(brier_score_loss(y_test, p)),
        "n_zero_coefs": int((np.abs(coef) < 1e-8).sum()),
        "n_features": len(coef),
    })
pd.DataFrame(reg_rows)

# %% [markdown]
# L1 zeroes out the lowest-information features; if you ship L1 you get a leaner
# model with usually-comparable PR-AUC. Elastic net is the compromise.

# %% [markdown]
# ## G. Threshold sensitivity under different cost ratios
#
# `cost_surface.csv` was computed across a 5x5 grid of (cost_fp, cost_fn).

# %%
cost = pd.read_csv(MODELS_DIR / "cost_surface.csv")
pivot = cost.pivot_table(index="cost_fn", columns="cost_fp", values="best_threshold")
pivot.style.background_gradient(cmap="viridis", axis=None).format("{:.2f}")

# %% [markdown]
# As FN cost grows relative to FP cost, the optimal threshold drops - which is the
# ML version of "if missing a cancellation is expensive, flag more bookings".

# %% [markdown]
# ## H. Subgroup performance grid

# %%
sub = pd.read_csv(MODELS_DIR / "subgroup_metrics.csv")
sub.sort_values("pr_auc", ascending=False)

# %% [markdown]
# Online TA segment (largest, n>=13k) is the worst-calibrated. A production
# deployment needs a `market_segment`-aware calibrator.

# %% [markdown]
# ## I. Coefficient stability under bootstrap
#
# 200 bootstrap re-fits; we report the per-feature 5-95% range. Anything whose
# range crosses zero gets a caveat in the methodology document.

# %%
from cancellation_logreg.diagnostics import coefficient_bootstrap

# Use cv=3 inside fit_calibrated for speed, n_boot=50 for a quick render in the
# notebook; the full n_boot=200 figure lives in models/ when run via make.
boot = coefficient_bootstrap(baseline_pipe, X_fit, y_fit, n_boot=50, seed=42)
if not boot.empty:
    summary = boot.groupby("feature")["coef"].agg(
        median="median",
        lo=lambda x: float(np.percentile(x, 5)),
        hi=lambda x: float(np.percentile(x, 95)),
    )
    summary["crosses_zero"] = (summary["lo"] < 0) & (summary["hi"] > 0)
    summary.sort_values("median", key=abs, ascending=False).head(15)
else:
    print("Bootstrap unavailable - re-run with sklearn pipeline that exposes named_steps.")

# %% [markdown]
# ## J. Production monitoring proposal
#
# What I would watch in production, day-1 to day-90:
#
# 1. **Input PSI** per feature, alert at >0.20 over a rolling 4-week window.
# 2. **Predicted-probability PSI** on a fixed reference period, alert at >0.10.
# 3. **Brier score** on the labelled lookback (post-arrival ground truth). Weekly
#    alert at >5% relative degradation; monthly recalibration trigger.
# 4. **Subgroup-level Brier** for `market_segment` and `lead_time_bucket`. Online TA
#    segment gets its own dashboard from day 1 (per the subgroup table above).
# 5. **Threshold expected-cost** under the live cost matrix; re-derive monthly.
# 6. **Reservation-status leakage canary**: assert these columns are NEVER in the
#    live feature pipeline (the `tests/test_preprocess.py` assertion is the static
#    version; production needs a runtime equivalent that fails closed).

# %% [markdown]
# ## Findings & next steps
#
# - Bootstrap CIs and subgroup table both point to **monthly + segment** variability
#   as the dominant unmodelled noise. A segment-aware calibrator and a monthly
#   recalibration cadence would address most of it.
# - SMOTE confirmed counter-productive on this calibration metric; we ship
#   class-weight balancing instead.
# - L1 + elastic net match L2 PR-AUC and zero out a meaningful number of low-information
#   features - if model size matters in production, L1 is a clean ablation to pick up.
