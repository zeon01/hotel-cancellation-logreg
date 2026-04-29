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
# # 02 — Preprocessing
#
# Implements the leakage guards and data-quality treatments documented in
# `docs/methodology.md`. Each cell calls a function from `cancellation_logreg.preprocess`
# rather than inlining logic, so the notebook stays diffable.

# %%
from cancellation_logreg.plotting import set_style

set_style()

# %% [markdown]
# ## 1. Drop leakage columns
#
# `reservation_status` and `reservation_status_date` encode the outcome and must not be
# used as features. A unit test in `tests/test_preprocess.py` enforces this.

# %%
# from cancellation_logreg.data import load_raw
# from cancellation_logreg.preprocess import drop_leakage_columns
# df = drop_leakage_columns(load_raw())

# %% [markdown]
# ## 2. Remove invalid bookings
# `adr < 0` (data error) and zero-guest rows.

# %% [markdown]
# ## 3. Impute missing
# `children=0`, `agent` -> top-K + missing indicator, `company` -> binary.

# %% [markdown]
# ## 4. Winsorize `adr` at 99.5th percentile

# %% [markdown]
# ## 5. Collapse low-information categories
# `meal: Undefined -> SC`; `country` top-20 + OTHER; `agent` / `market_segment` top-10 + OTHER.

# %% [markdown]
# ## 6. Duplicate-row indicator
# Add `is_likely_group_booking` rather than dropping duplicates — and run a "drop dups"
# ablation in the appendix.

# %% [markdown]
# ## 7. Build `arrival_date` and `booking_date`

# %% [markdown]
# ## Persist
# Write to `data/processed/clean.parquet`.

# %% [markdown]
# ## Findings & next steps
#
# - _Phase 2_
# - _Phase 2_
# - _Phase 2_
