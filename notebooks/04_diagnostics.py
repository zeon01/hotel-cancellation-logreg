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
# # 04 — Diagnostics & Inferential Interpretation
#
# Where 03 cared about prediction, 04 cares about *understanding*. Statsmodels coefficients,
# odds ratios, VIF, calibration-curve deep dive, coefficient stability under bootstrap.

# %%
from cancellation_logreg.plotting import set_style

set_style()

# %% [markdown]
# ## VIF table
# Threshold 5 (warn), 10 (drop or merge).

# %% [markdown]
# ## Statsmodels Logit
# Fit on standardised, non-target-encoded features. Extract:
# - coefficient estimates with HC3 robust SE
# - p-values
# - 95% CIs
# - odds ratios

# %% [markdown]
# ## Plain-English interpretation of the top 5 coefficients
# Each as: "each additional X of `feature` multiplies the odds of cancellation by Y".

# %% [markdown]
# ## Calibration curves
# Uncalibrated vs. Platt vs. isotonic. Brier and ECE for each.

# %% [markdown]
# ## Coefficient stability
# Re-fit on 200 bootstrap samples; box-plot the distribution per coefficient. Anything
# whose 5-95% range crosses zero gets a caveat.

# %% [markdown]
# ## Findings & next steps
#
# - _Phase 2_
# - _Phase 2_
# - _Phase 2_
