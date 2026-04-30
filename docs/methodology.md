# Methodology — hotel-cancellation-logreg

This document records every modelling decision, the chosen option, and the rejected alternative. Each section is written so it can be paraphrased in a technical review.

## 1. Dataset and licensing

**Source:** Antonio, N., de Almeida, A., & Nunes, L. (2019). _Hotel booking demand datasets._ **Data in Brief**, 22, 41–49. DOI 10.1016/j.dib.2018.11.126. License: CC BY 4.0.

**Variant used:** the TidyTuesday-cleaned `hotel_bookings.csv` (Kaggle: `jessemostipak/hotel-booking-demand`), ~119,390 rows, 32 columns. The original Mendeley file (DOI 10.17632/j83f5fsh6c.1) is supported as a fallback in `scripts/download_data.py`. **Document this explicitly** in the README because the cleaned variant differs in a few column names (e.g. `distribution_channel` is corrected from the original typo).

**Rejected alternatives:** the raw two-hotel split (H1/H2) from the Mendeley repo — useful for replicating Antonio et al. exactly, but the cleaned single-file variant is what the broader community models on, so reproducibility is easier.

## 2. Leakage guards

Two columns encode the outcome and **must be dropped before training**:

- `reservation_status` — literally "Canceled", "Check-Out", "No-Show".
- `reservation_status_date` — the date that status was recorded; mostly the cancellation date for cancelled rows.

A unit test in `tests/test_preprocess.py` asserts these columns are absent from `X` after `build_clean_frame`.

A third column, `country`, is **partially leaky**. Antonio et al. note that the country field is confirmed/corrected at check-in for non-Portuguese guests; for cancelled bookings, only the booking-time entry is recorded. The PMC re-analysis (Tree-Based Neural Network paper, 2024) drops `country` for this reason. **Decision for this repo:** keep `country`, bin to top-20 + "OTHER", disclose the caveat here, in the README, and in the appendix leakage-audit table. Rationale: the geographic signal is strong and the leakage direction (cancelled rows over-represent Portugal) is bounded and disclosable; dropping the column over-corrects.

## 3. Data quality treatment

| Issue | Treatment | Rejected alternatives |
| ----- | --------- | --------------------- |
| `children` has 4 NaNs | Impute with 0 | Median (also 0); no material difference |
| `agent` ~14% missing, `company` ~94% missing | `agent`: top-K + "missing" indicator; `company`: convert to binary `has_company_id` | Drop both (loses information); KNN-impute (overkill, leakage-prone) |
| `adr` has negative values and outliers > 5000 | Drop `adr < 0` (data error); winsorise at 99.5th percentile | Median imputation (these are errors, not missing); IQR-based filtering (less defensible threshold) |
| ~180 rows with zero adults+children+babies | Drop (not real bookings) | Keep (corrupts training) |
| `meal` has duplicate categories ("Undefined", "SC") | Collapse | Keep separate (introduces spurious cardinality) |
| `reserved_room_type` ≠ `assigned_room_type` mismatch | Encode as feature `room_was_changed` (operational signal) | Treat as data error; correct (loses information) |
| ~32k duplicate rows (likely group bookings) | Add `is_likely_group_booking` indicator; **do not drop**. Run "drop dups" as appendix ablation | Drop blindly (loses ~25% of data and structure); keep silently (model conflates group with single bookings) |

## 4. Class imbalance — opinionated stance

The dataset is ~37% positive class. **Not severe.** The community consensus in 2024–25 (Abdelhamid & Desai 2024 "Balancing the Scales", 9,000-experiment study; Elor & Averbuch-Elor 2022) is that for modern probabilistic classifiers:

> Decision-threshold tuning + class weights matches or beats SMOTE on AUC/PR-AUC/Brier, **and SMOTE degrades calibration**.

**Therefore:**

1. `class_weight="balanced"` in the logistic regression.
2. Threshold tuning on the validation set against expected business value (toy cost matrix).
3. Calibration via `CalibratedClassifierCV(method="isotonic")` — the dataset is large enough.

**Rejected alternatives:** SMOTE / SMOTE-NC / ADASYN (will be run as an appendix ablation specifically to demonstrate it degrades Brier); RandomUnderSampler (loses data); cost-sensitive base learner only without threshold tuning (leaves business value on the table).

## 5. Train / validation / test split — time-based

- **Train:** 2015-07 to 2016-12.
- **Validation:** 2017-01 to 2017-04.
- **Test:** 2017-05 to 2017-08.

**Rationale (write this verbatim in the README if asked):** Antonio et al.'s follow-up paper (Data Science Journal, 2019) explicitly identifies stratified random splits as inflating reported performance versus a time-based split. A cancellation model deployed in production sees future bookings — train on past, validate on near future, test on further future. This is also how a typical OTA experimentation framework conceives of generalisation.

**On the training set:** `TimeSeriesSplit` 5-fold for hyperparameter CV. Stratified `KFold` is run as an appendix ablation; the optimism gap is reported.

**Rejected alternatives:** stratified random K-fold (optimistic on time-structured data); group K-fold by `agent` or `country` (would help generalisation but wastes scarce minority groups in small folds).

## 6. Modelling — logistic regression specifics

**Pipeline:** `ColumnTransformer` →

- Numeric: `StandardScaler`.
- Low-cardinality categorical: `OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01)`.
- High-cardinality categorical (`country`, `agent`): `category_encoders.TargetEncoder` _inside CV folds only_ (else leakage).

**Regularisation:** L2 (`penalty="l2"`, `solver="lbfgs"`), `C` tuned via 5-fold time-series CV over `{0.01, 0.1, 1, 10, 100}`. L1 and elastic net are appendix ablations for feature selection robustness.

**Inferential model:** parallel `statsmodels.Logit` fit on standardised, non-target-encoded features, used only to extract coefficient estimates, robust SEs, p-values, and odds ratios. The two coefficient sets may differ slightly because sklearn's regularisation shrinks toward zero while statsmodels gives the MLE.

**Threshold selection:** maximise expected business value under a toy cost matrix (`cost(FP) = $5`, `cost(FN) = $50`); show the cost-optimal threshold differs from 0.5 and from the F1-optimal threshold. Cost-ratio sensitivity reported in appendix.

## 7. Calibration

`CalibratedClassifierCV(method="isotonic", cv=5)`. Brier score and reliability diagram (10 bins) reported. Platt scaling shown as ablation for completeness — isotonic preferred at this dataset size.

## 8. Evaluation metrics

- Discrimination: ROC-AUC, **PR-AUC** (preferred for the imbalanced business framing).
- Calibration: **Brier**, reliability diagram with predicted-probability histogram.
- At threshold: precision, recall, F1, balanced accuracy, confusion matrix.
- Business: expected cost, lift in top-decile of predicted-risk bookings.

## 9. Diagnostics

- VIF table, threshold 5 (warn) / 10 (drop or merge). Implemented in `src/cancellation_logreg/diagnostics.py`.
- Coefficient stability: bootstrap re-fits, plot coefficient distributions.
- Subgroup performance: AUC and Brier broken down by `hotel`, `market_segment`, `is_repeated_guest`, lead-time bucket.

## 10. Defensive choices for ambiguity

Where the spec gave two reasonable options, the more defensive one was chosen and noted here:

- _TODO Phase 2: log any judgement calls made during implementation that weren't fully specified._
