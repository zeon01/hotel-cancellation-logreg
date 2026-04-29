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
# # 99 — Interview-grade Rigor Appendix
#
# Goes beyond the standard portfolio depth. Each section is the kind of question a senior
# DS panel would ask; the answer is here with a defensible methodology and a named
# alternative.

# %%
from cancellation_logreg.plotting import set_style

set_style()

# %% [markdown]
# ## A. Leakage audit table
# Every column, leakage status, one-sentence justification.

# %% [markdown]
# ## B. Learning-curve diagnostic
# Train vs. validation score over training-set size. Argues bias/variance.

# %% [markdown]
# ## C. Statistical vs. practical significance
# Where p-values are shown, also show effect size (odds ratio, partial R²) and discuss
# whether "significant" means "useful".

# %% [markdown]
# ## D. Ablation table
# Drop each feature group, observe ΔPR-AUC, ΔBrier.

# %% [markdown]
# ## E. SMOTE ablation
# Numeric demonstration that SMOTE degrades Brier and barely moves AUC. Reference:
# Abdelhamid & Desai (2024) "Balancing the Scales", arXiv:2409.19751.

# %% [markdown]
# ## F. L1 vs. L2 vs. ElasticNet
# Comparison table; which features L1 zeroes out as a robustness check.

# %% [markdown]
# ## G. Threshold sensitivity
# How does business value change as cost ratios FP:FN ∈ {1:5, 1:10, 1:20} change?

# %% [markdown]
# ## H. Subgroup performance
# AUC and calibration broken down by `hotel`, `market_segment`, `is_repeated_guest`,
# lead-time bucket. Discuss segments where the model is meaningfully worse.

# %% [markdown]
# ## I. Coefficient stability under bootstrap
# Re-fit on bootstrap samples; coefficient distributions. Anything not robust gets a caveat.

# %% [markdown]
# ## J. Production monitoring proposal
# PSI on inputs, PSI on predictions, monthly recalibration trigger, weekly Brier alarm.
