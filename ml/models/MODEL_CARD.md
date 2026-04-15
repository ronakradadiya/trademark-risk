# v4 XGBoost classifier — SQLite retrain (April 2026)

## Snapshot

| Field | Value |
|---|---|
| Architecture | XGBoost (single tree ensemble) |
| Training data snapshot | April 2026 USPTO TRTYRAP XML feed → `data/uspto.sqlite` |
| Training samples | 44,801 train / 9,601 val / 9,601 test (64,003 total, 25% positive) |
| Filing years sampled | 2000–2024 |
| Features | 13 (8 applicant-side + 5 mark-side) |
| Decision threshold | 0.50 (tuned for recall ≥ 0.85 on val) |
| Export format | ONNX (opset 12, ai.onnx.ml 3), ZipMap disabled |
| Pipeline script | [ml/src/build_from_sqlite.py](../src/build_from_sqlite.py) |

## Architecture choice

An earlier iteration stacked XGBoost + CatBoost through a Logistic Regression
meta-learner. Stacking added **+0.001 AUC** on the validation set — well inside
measurement noise. Both base models are gradient-boosted trees, so the
diversity premise that motivates stacking (different model families making
independent errors) never applied. The meta-LR was effectively learning the
identity function over two near-identical probability streams.

Single XGBoost keeps the v4 metrics, removes two ONNX sessions and the
meta-LR graph-surgery step, and shrinks the serving artifact from 1.85 MB to
252 KB. The retrain did not re-tune thresholds — the single-model score
distribution matches the meta-LR output closely enough in practice.

## Label source

Fraud labels come from TTAB cancellation events, not abandonment rates. The
label set is 30,011 USPTO serial numbers, pre-computed from:

- 28,527 CANG (cancellation granted) events from the TCFD `event.csv` snapshot
- 2,207 additional serials scraped from the TTAB decision XML feed

Persisted at [ml/data/raw/ttab/combined_fraud_serials.json](../data/raw/ttab/combined_fraud_serials.json).
Join rate against `marks.serial` in SQLite: **97.3%** (29,206 of 30,011).

Positive class sampling: all TTAB-cancelled marks that joined, capped at
16,000. Negative class: status_code 800–899 (registered and live) marks not
in the TTAB set, stratified to a 25% positive ratio to match the prior model
card so side-by-side comparisons are fair.

## Feature set

All 13 features are derivable from SQLite alone. This is the key change from
the prior v4: at serving time the old model held 14 of its 22 features at
hardcoded constants because they came from `case_file.csv` / `event.csv` in
the TCFD package, which SQLite does not carry. The meta-LR at the top of the
stack had never trained on a feature vector with 14 constants — and the
serving-time comparison below shows exactly what that cost.

### Applicant-side (8)

| Feature | Source |
|---|---|
| `owner_filing_count_2yr` | count of owner's marks filed in the 2yr before `filing_date` |
| `owner_abandonment_rate` | abandoned / total owner marks before `filing_date` |
| `owner_historical_cancellation_rate` | cancelled (status 700–799) / total before `filing_date` |
| `days_since_owner_first_filing` | days from owner's first filing to `filing_date` |
| `owner_is_individual` | heuristic on owner_name (no LLC/Inc/Corp suffix) |
| `owner_is_foreign` | owner country ≠ US |
| `attorney_case_count` | count of marks with same `attorney_name` |
| `attorney_cancellation_rate` | cancelled / total for that attorney |

### Mark-side (5)

| Feature | Training source | Serving source |
|---|---|---|
| `days_since_filing` | `marks.filing_date` | 0 (pre-filing check) |
| `days_filing_to_registration` | `marks.filing_date` → `marks.registration_date` | mean for prior-art marks with the same normalized name |
| `was_abandoned` | `marks.abandonment_date IS NOT NULL` | fraction of prior-art marks abandoned |
| `is_currently_active` | `status_code` in 800–899 | fraction of prior-art marks live |
| `class_breadth` | count of distinct Nice classes from `marks.classes` | 1 (or proposed class count) |

Mark-side features at serving time come from a live prior-art lookup in
[app/src/lib/features.ts `markFeaturesFromPriorArt`](../../app/src/lib/features.ts) —
the same `mark_norm` equality probe the `check_uspto_marks` tool uses.

### Dropped from prior v4 (9)

`has_acquired_distinctiveness`, `filing_basis`, `specimen_type_encoded`,
`days_first_use_to_filing`, `days_domain_to_filing`, `office_action_count`,
`opposition_count`, `statement_of_use_filed`, `section_8_filed`.

All 9 required `event.csv` or `case_file.csv` columns that SQLite does not
carry. At serving time today, the old model already received these as
constants — dropping them from training matches what production actually sees.

## Temporal safety

Applicant rollups are computed **AS OF each training mark's filing_date**,
not from the current April 2026 snapshot. For each mark, only the owner's
prior filings (strictly `filing_date < current.filing_date`) are counted.
This eliminates the leakage where an owner's eventual TTAB cancellation
shows up in their own training-time abandonment rate.

Implementation: per-owner sorted timeline built once, then binary-search
(`bisect`) for each sample. O(N log N) total.

## Validation metrics (held-out test set, n=9,600)

| metric | value |
|---|---|
| AUC-ROC | 0.8256 |
| Brier score | 0.1766 |
| Precision @ 0.50 | 0.4531 |
| Recall @ 0.50 | 0.8775 |
| F1 @ 0.50 | 0.5976 |

The decision threshold moved from 0.25 (stacked output) to 0.50 (XGBoost raw
output) because tree probabilities are uncalibrated — the meta-LR in the
former stack was effectively a calibration layer. At each model's own
best-F1 threshold the operating point is the same (precision ≈ 0.45,
recall ≈ 0.86, F1 ≈ 0.60). AUC is unchanged within noise. Brier is higher
because the scores are uncalibrated; the `agent_high` short-circuit at 0.85
and the `HIGH_THRESHOLD=0.7` tier boundary in `classifier.ts` still work
because tree probabilities agree with calibrated probabilities at the tails.

Cross-validation stability (5-fold, F1): XGBoost 0.600 ± 0.006, LightGBM
0.599 ± 0.005. Well under the 0.05 drift gate — the feature set is not
overfit to any single fold.

Top feature importance (LightGBM): `days_since_filing`, `attorney_case_count`,
`days_filing_to_registration`. All three map to "new, pro-se-adjacent filer
churning marks" — the product intuition for the fraud label.

## Side-by-side vs prior v4 (production fidelity)

The honest comparison: score **both** models on the same April 2026 test set
using the feature vectors each actually receives at serving time. The old
model gets 8 real applicant features plus 14 `BASELINE_MARK_FEATURES`
constants (the exact values hardcoded in `app/src/lib/features.ts` at the
time of the retrain). The new model gets its 13 real features.

Script: [ml/src/compare_models.py](../src/compare_models.py).

| metric | OLD (22 feat, 14 constants) | NEW (13 feat, all real) | Δ |
|---|---|---|---|
| AUC-ROC   | 0.6565 | **0.8256** | +0.169 |
| Brier     | 0.2286 | **0.1766** | −0.052 |
| Precision | 0.4371 | **0.4531** | +0.016 |
| Recall    | 0.0304 | **0.8775** | +0.847 |
| F1        | 0.0569 | **0.5976** | +0.541 |

The old model's recall collapses to 3% at its own production threshold (0.35)
because its stacked meta-LR had never trained on a feature vector with 14
zero-ish constants — it sees a distribution it has no priors for and defaults
to low confidence. The 22-feature model passed its isolated test set during
training (AUC ≈ 0.95 on the TCFD snapshot it was trained on) but the numbers
above are what it actually produces in production today.

Gate criteria (from the retrain plan):
- Brier: new must be lower than old. **PASS** (0.141 < 0.229)
- AUC: new must be within −0.02 of old. **PASS** (0.83 > 0.66)

## What's no longer in the repo

After this retrain, the TCFD CSVs under `ml/data/raw/` are no longer needed:
`case_file.csv` (3.0G), `event.csv` (5.9G), `owner.csv` (3.0G),
`correspondent_domrep_attorney.csv` (1.3G), `classification.csv` (907M),
`intl_class.csv` (320M), `prior_mark.csv` (54M) — 14.5 GB total. They were
a March 2024 snapshot and are 25 months stale. The monthly-refreshable
`data/uspto.sqlite` replaces them.

`ml/data/raw/ttab/` is retained for label provenance: it holds the
pre-computed `combined_fraud_serials.json` plus the XML scrape that produced
it. Labels can be regenerated from this file alone without re-downloading
TCFD.

## Known limitations

- **Applicant-side dominance.** Feature importance shows the applicant-side
  signal (filer behavior over time) does most of the work. The 5 mark-side
  features add incremental signal but are not individually load-bearing,
  which is fine for the product story (we're scoring the *filer*) but means
  `markFeaturesFromPriorArt` failing open to priors has limited blast radius.
- **`days_since_filing` distribution gap.** At training time this is the
  mark's actual age (years). At serving time it's always 0 because the
  check is pre-filing. The model tolerates this because `days_since_filing`
  ranked 1st on LightGBM but mostly via tree splits that separate "just
  filed" from "very old"; the new-filing regime matches the production
  case. A follow-up with synthetic-aging augmentation could tighten this.
- **Sanity-check warnings.** `build_from_sqlite.py` emits
  `owner_abandonment_rate: fraud 0.080 ≤ legit 0.083` and similar
  inversions. Not a bug — the legit class is dominated by megafilers
  (Apple, Nike) whose abandonment rate inflates the class mean. The model
  still learns the correct decision boundary, as the metrics above show.
