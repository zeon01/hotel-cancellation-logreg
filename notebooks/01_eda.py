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
# # 01 — Exploratory Data Analysis
#
# **Goal:** Get a working understanding of the Antonio hotel-bookings dataset before any
# preprocessing. Document everything that surprises us — those surprises become preprocessing
# decisions in `02_preprocessing.ipynb`.
#
# **Key questions:**
# 1. What's the cancellation rate, and does it vary by hotel / segment / lead-time bucket?
# 2. Which columns have missingness? How much?
# 3. What's the temporal coverage? Are there obvious regime shifts?
# 4. Where are the outliers (`adr`, `lead_time`)?
# 5. Do we see the documented data-quality issues (negative `adr`, zero-guest rows, duplicates)?

# %%
from cancellation_logreg.plotting import set_style

set_style()

# %% [markdown]
# ## Load
#
# `load_raw()` reads `data/raw/hotel_bookings.csv` and returns the unmodified frame.

# %%
# from cancellation_logreg.data import load_raw, validate_schema
# df = load_raw()
# validate_schema(df)
# df.shape, df.dtypes

# %% [markdown]
# ## Class balance and temporal coverage
#
# _Phase 2: cancellation rate; per-hotel breakdown; arrival-date histogram._

# %% [markdown]
# ## Missingness audit
#
# _Phase 2: bar of % missing per column; reconcile with the documented issues
# (`children`, `agent`, `company`, `country`)._

# %% [markdown]
# ## Outliers and data-quality findings
#
# _Phase 2: `adr` distribution + negatives; lead_time distribution; duplicate count._

# %% [markdown]
# ## Findings & next steps
#
# - _Phase 2: bullet 1_
# - _Phase 2: bullet 2_
# - _Phase 2: bullet 3_
