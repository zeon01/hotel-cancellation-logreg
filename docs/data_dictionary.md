# Data Dictionary — hotel-cancellation-logreg

## Source columns (TidyTuesday `hotel_bookings.csv`, n ≈ 119,390)

| Column | Type | Notes |
| ------ | ---- | ----- |
| `hotel` | categorical | "City Hotel" / "Resort Hotel" |
| `is_canceled` | binary | **Target.** 1 = cancelled, 0 = not. |
| `lead_time` | int | Days from booking to arrival. Right-skewed. |
| `arrival_date_year` | int | 2015..2017 |
| `arrival_date_month` | str | Month name; cycle-encoded downstream |
| `arrival_date_week_number` | int | 1..53 |
| `arrival_date_day_of_month` | int | 1..31 |
| `stays_in_weekend_nights` | int | |
| `stays_in_week_nights` | int | |
| `adults` | int | |
| `children` | int | 4 NaNs → impute 0 |
| `babies` | int | |
| `meal` | categorical | "Undefined" and "SC" collapsed |
| `country` | categorical | **Partially leaky** (confirmed at check-in for non-PT guests); top-20 + "OTHER" |
| `market_segment` | categorical | top-K + "OTHER" |
| `distribution_channel` | categorical | (TidyTuesday version corrects the original "distribution_chanel" typo) |
| `is_repeated_guest` | binary | |
| `previous_cancellations` | int | |
| `previous_bookings_not_canceled` | int | |
| `reserved_room_type` | categorical | |
| `assigned_room_type` | categorical | |
| `booking_changes` | int | |
| `deposit_type` | categorical | "No Deposit" / "Non Refund" / "Refundable" |
| `agent` | categorical | ~14% missing; high cardinality; top-K + missing |
| `company` | categorical | ~94% missing; converted to `has_company_id` |
| `days_in_waiting_list` | int | |
| `customer_type` | categorical | |
| `adr` | float | Avg Daily Rate. Drop `< 0`; winsorise at 99.5%ile |
| `required_car_parking_spaces` | int | |
| `total_of_special_requests` | int | |
| `reservation_status` | categorical | **LEAKAGE — drop** |
| `reservation_status_date` | date | **LEAKAGE — drop** |

## Engineered features

| Feature | Definition | Rationale (Supply Ops framing) |
| ------- | ---------- | ------------------------------ |
| `total_nights` | `weekend_nights + week_nights` | Trip length is a stronger signal than its parts |
| `total_guests` | `adults + children + babies` | |
| `adr_per_person` | `adr / total_guests` (ε-protected) | Normalises rate signal across party sizes |
| `room_was_changed` | `reserved_room_type != assigned_room_type` | Operational friction marker; correlates with overbooking handling |
| `is_short_lead` | `lead_time <= 7` | Short-lead bookings cancel less; common Supply rule-of-thumb |
| `is_long_lead` | `lead_time > 180` | Long-lead is the bigger cancellation risk — drives allotment-release decisions |
| `arrival_month_sin`, `arrival_month_cos` | cyclical encoding of month | Captures seasonality without ordering artefact |
| `prior_cancel_rate` | `prev_cancels / (prev_cancels + prev_not_cancels)` with Beta(α=1, β=1) smoothing | Repeat-cancellation propensity |
| `has_deposit` | `deposit_type != "No Deposit"` | High-signal cancellation predictor |
| `is_corporate` | `market_segment in {"Corporate", "Complementary"}` | Behaviour differs strongly from leisure |
| `is_likely_group_booking` | duplicate-row indicator across booking-defining cols | Distinguishes group from single-booking patterns |
| `arrival_date` | combined year/month/day datetime | For temporal split & EDA |
| `booking_date` | `arrival_date - lead_time` | For temporal split |

## Dropped from `X`

- `reservation_status`, `reservation_status_date` (leakage)
- `arrival_date_year/month/day_of_month` (replaced by cyclical encodings + `arrival_date`)
- `agent`, `company` raw cols (replaced by transformed versions)
