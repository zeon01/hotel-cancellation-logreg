# Model Card — hotel-cancellation-logreg

**Status:** Skeleton (Phase 1). Populate during Phase 2 once the pipeline runs end-to-end.

## Model details

- **Type:** L2-regularised logistic regression with isotonic calibration.
- **Inputs:** Booking-time features only — no post-booking signal. See `docs/data_dictionary.md`.
- **Output:** Calibrated probability of cancellation in [0, 1] plus a thresholded class label.
- **Owner / contact:** Saad Sharif Ahmed (portfolio).
- **Version:** 0.1.0.

## Intended use

- Risk score for booking-level Supply Operations decisions: allotment release timing, deposit-policy targeting, agent-attention prioritisation.
- Inferential coefficients for supplier-facing communication ("which booking attributes drive cancellation").

## Out-of-scope use

- Real-time pricing decisions (no price-elasticity modelling).
- Fraud detection (different label, different feature space).
- Use on lead times beyond ~700 days (dataset max).
- Use on geographies outside Portugal without re-fit and re-validation.

## Training data

Antonio et al. (2019) hotel bookings, July 2015 – August 2017, two Portuguese hotels (one resort, one city). CC BY 4.0. ~119k rows. ~37% positive class.

## Evaluation

Time-based test set: 2017-05 to 2017-08. Metrics: PR-AUC, ROC-AUC, Brier (calibrated), expected operational cost. _Phase-2 numeric values to be filled in._

## Ethical considerations

- The model is trained on Portuguese leisure travel patterns. Deploying on APAC supply, business travel, or post-COVID booking patterns without re-validation would propagate distribution shift.
- `country` is partially leaky and could bias predictions by geography; this is documented and an ablation is run without it.

## Caveats

- Performance ceiling is below tree ensembles. The sibling repo `hotel-cancellation-rf` is the higher-performance counterpart.
- Calibration is stable in the tested time window but should be re-checked monthly in any deployment.
