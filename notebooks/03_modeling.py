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
# # 03 — Modelling
#
# Fit the L2-regularised logistic regression, calibrate it, and pick a threshold.
# Inferential interpretation (statsmodels coefficients, odds ratios) lives in `04_diagnostics`.

# %%
from cancellation_logreg.plotting import set_style

set_style()

# %% [markdown]
# ## 1. Time-based split

# %%
# from cancellation_logreg.modeling.splits import time_based_split
# train, val, test = time_based_split(df)

# %% [markdown]
# ## 2. Build pipeline
# `ColumnTransformer` (StandardScaler / OneHot / TargetEncoder) -> `LogisticRegression(penalty='l2')`.

# %% [markdown]
# ## 3. Tune `C` via TimeSeriesSplit CV (PR-AUC)

# %% [markdown]
# ## 4. Calibrate (`CalibratedClassifierCV(method='isotonic')`)

# %% [markdown]
# ## 5. Threshold selection
# Minimise expected cost under cost(FP)=$5, cost(FN)=$50.

# %% [markdown]
# ## 6. Headline metrics
# PR-AUC, ROC-AUC, Brier, precision/recall/F1 at chosen threshold.

# %% [markdown]
# ## 7. Save artefact
# `models/logreg_calibrated.joblib`.

# %% [markdown]
# ## Findings & next steps
#
# - _Phase 2_
# - _Phase 2_
# - _Phase 2_
