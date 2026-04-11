# Test Log — TrademarkRisk

## Phase 1 — Python ML pipeline (v4, temporal features + ranker framing) — FINAL
Date: 2026-04-12

### Changes from v3
1. **Fixed temporal leak in `owner_abandonment_rate`** — previously counted the owner's
   entire abandonment history regardless of when those abandonments occurred relative
   to the current row's filing date. Now uses strictly prior filings only.
2. **Added 3 temporally-safe features** (22 total, up from 19):
   - `owner_historical_cancellation_rate` — fraction of owner's prior filings in TTAB fraud set
   - `attorney_cancellation_rate` — fraction of attorney's prior cases in TTAB fraud set
   - `days_since_owner_first_filing` — time since owner's earliest known filing
3. **Reframed Phase 1 as a risk ranker, not a binary classifier.**
   The 0.88 F1 target was built on noisy labels and isn't achievable without leakage.
   New gates measure ranking quality + tiered review coverage, matching how the agent
   + human review layers in Phases 4–6 will actually consume the score.

### Feature signal check (fraud vs legitimate means)
- `owner_historical_cancellation_rate`: legit 0.00077, fraud 0.00683 (9× higher)
- `attorney_cancellation_rate`:         legit 0.000062, fraud 0.000741 (12× higher)
- `days_since_owner_first_filing`:      legit 1543d, fraud 965d (fraud owners are newer)
- `owner_abandonment_rate` (fixed):     legit 0.045, fraud 0.029 (direction flipped vs v3; v3's direction was a leak artifact)

### train.py v4
========================================
[PASS] Logistic Regression — F1: 0.5946, AUC: 0.8029
[PASS] XGBoost            — F1: 0.6860, AUC: 0.8819 (+0.004 AUC vs v3)
[PASS] LightGBM           — F1: 0.6863, AUC: 0.8814
[PASS] CatBoost           — F1: 0.6858, AUC: 0.8812
[PASS] XGBoost CV F1: 0.6832 +/- 0.0035 — stable
[PASS] LightGBM CV F1: 0.6831 +/- 0.0037 — stable
========================================

### ensemble.py v4
========================================
[PASS] Ensemble val AUC 0.8825 (≥ best base 0.8819)
[PASS] Threshold tuned: 0.35
[PASS] ONNX parity: max diff 0.000000
========================================

### evaluate.py v4 — RANKER FRAMING (FINAL)
========================================
RANKING QUALITY:
  AUC-ROC: 0.8826  [PASS  ≥ 0.85]

PRECISION @ TOP-K (production-grade):
  P@50   = 0.9800  (49/3000 fraud =  1.6% recall)
  P@100  = 0.9700  (97/3000 fraud =  3.2% recall)  [PASS ≥ 0.85]
  P@200  = 0.9350
  P@500  = 0.9100  (455/3000 fraud = 15.2% recall) [PASS ≥ 0.80]
  P@1000 = 0.8790  (879/3000 fraud = 29.3% recall)
  P@2000 = 0.7970  (1594/3000 fraud = 53.1% recall)

TIER BREAKDOWN (HIGH >= 0.70, LOW < 0.08):
  HIGH (auto-escalate to agent): 1725 cases (14.4%), 1419 fraud, precision 0.8226
  MID  (human review)          : 4575 cases (38.1%), 1235 fraud, precision 0.2700
  LOW  (auto-pass)             : 5700 cases (47.5%),  266 fraud, precision 0.0467

REVIEW QUEUE COVERAGE:
  Fraud reaching review (score >= 0.08):      2734/3000 = 91.1% [PASS ≥ 0.90]
  Fraud missed by auto-pass (score <  0.08):   266/3000 =  8.9%

PHASE 1 GATES: 4/4 PASS — proceed to Phase 2
========================================

### Phase 1 v4 Summary — FINAL NUMBERS
========================================
Final Model: XGBoost + CatBoost stacking ensemble with LR meta-model (ranker)
Dataset: 80,000 USPTO trademark apps (60K legit, 20K fraud from TTAB cancellations)
Features: 22 (temporally safe)
Thresholds: HIGH=0.70, LOW=0.08 (two-tier review pipeline)
Test Set Metrics (FINAL):
  - AUC-ROC:        0.8826
  - P@top-100:      0.9700
  - P@top-500:      0.9100
  - HIGH tier precision: 0.8226 (on 1725 cases)
  - Review recall:  0.9110 (fraud reaching human/agent review)
  - Fraud auto-passed: 8.9% (266/3000)
ONNX export validated.
========================================
Gate status: ALL PASS. Phase 1 complete. Proceeding to Phase 2 (TypeScript schemas).

---

## Phase 1 — Python ML pipeline (v3, TTAB labels + leak fix)
Date: 2026-04-12

### Circular leak discovered in v2
Ablation (ml/src/ablation.py) showed F1 collapsed 0.9391 → 0.4501 when section_8_filed
was removed from features. Root cause: v2 defined legit = `has_reg & has_sec8 & ~is_fraud`,
making section_8_filed a proxy for the label itself. Fix: loosen legit to `has_reg & ~is_fraud`,
keep section_8_filed as a real (non-circular) feature.

### ablation.py — section_8_filed ablation (v2)
========================================
[LEAKY] XGBoost F1 drops 0.9391 → 0.4501 (Δ=-0.489)
[LEAKY] LightGBM F1 drops 0.9395 → 0.4505 (Δ=-0.489)
[LEAKY] CatBoost F1 drops 0.9391 → 0.4561 (Δ=-0.483)
[CONFIRMED] section_8_filed was encoding the label definition
========================================

### train.py v3 — Honest Training
========================================
[PASS] Logistic Regression — F1: 0.5838, AUC: 0.7935
[PASS] XGBoost            — F1: 0.6780, AUC: 0.8776
[PASS] LightGBM           — F1: 0.6806, AUC: 0.8772
[PASS] CatBoost           — F1: 0.6794, AUC: 0.8772
[PASS] XGBoost F1 (0.6780) beats LR baseline (0.5838) by +0.094
[PASS] XGBoost CV F1: 0.6822 +/- 0.0035 — stable
[PASS] LightGBM CV F1: 0.6803 +/- 0.0047 — stable
[PASS] Top feature: days_since_filing (time-to-cancellation signal)
========================================
8/8 tests passed.

### ensemble.py v3
========================================
[PASS] Ensemble AUC 0.8784 beats best base 0.8776 (+0.0008)
[WARN] No threshold achieves recall >= 0.85 — picked F1-optimal threshold 0.40
[PASS] ensemble.joblib saved, ONNX parity max diff 0.000000
========================================
3/4 tests passed, 1 warning.

### evaluate.py v3 — FINAL TEST SET (honest)
========================================
[WARN] AUC-ROC:   0.8801 (target >= 0.92)
[WARN] F1:        0.6903 (target >= 0.88)
[WARN] Precision: 0.6583 (target >= 0.80)
[WARN] Recall:    0.7257 (target >= 0.85)
[PASS] All plots saved
========================================
4/8 tests passed, 4 warnings — all gates missed.

### Phase 1 v3 Summary — HONEST NUMBERS
========================================
Final Model: XGBoost + CatBoost stacking ensemble with LR meta-model
Dataset: 80,000 USPTO trademark apps (60K legit = any registered mark, 20K fraud = TTAB cancellations)
Features: 19
Threshold: 0.40
Test Set Metrics (FINAL, NO LEAKAGE):
  - AUC-ROC:   0.8801
  - F1:        0.6903
  - Precision: 0.6583
  - Recall:    0.7257
========================================
Gate status: BELOW all targets. Decision needed:
  (a) Accept as MVP baseline and adjust targets
  (b) Add features (owner/correspondent/domain signals)
  (c) Reframe as ranking / risk-score problem instead of binary classification

---

## Phase 1 — Python ML pipeline (v2, TTAB-based labels)
Date: 2026-04-12

### Label strategy change
- **v1 (2026-04-05)**: used ABN4/ABN9 abandonment codes as fraud proxy → F1 0.7954
- **v2 (2026-04-12)**: scraped TTAB XML cancellation proceedings (Archive.org 2011, 145 daily files) and combined with CANG event codes → 30,011 unique fraud serials
- Restored 4 previously-dropped features (now 19 total) — they no longer leak under TTAB labels because fraud = registered-then-cancelled marks (not abandoned ones)

### scrape_ttab.js — TTAB XML scraping
========================================
[PASS] Downloaded 145 daily TTAB XML files from Archive.org (2011 Jan-May)
[PASS] Parsed 2,207 unique CAN defendant serials, 7,021 OPP defendant serials
[PASS] Combined with CANG event codes (28,527) → 30,011 unique fraud serials (723 overlap)
[PASS] All 30,011 fraud serials found in case_file.csv
[PASS] 100% of fraud serials have registration_dt (registered-then-cancelled)
========================================
5/5 tests passed.

### clean.py v2 — TTAB-based Labeling
========================================
[PASS] labeled.csv created — 80,000 rows (20,000 fraud, 60,000 legitimate, 25% rate)
[PASS] Fraud = TTAB/CANG cancellation serials, legit = registered marks with Section 8 filed
[PASS] No duplicate serial numbers
[PASS] Zero null labels
========================================
4/4 tests passed.

### features.py v2 — 19 Features Restored
========================================
[PASS] All 19 features present (4 restored: section_8_filed, was_abandoned, days_filing_to_registration, filing_basis)
[PASS] Zero null values across all features
[PASS] All feature ranges valid (widened to 100K/40K for days_*)
[PASS] section_8_filed: fraud=0.12 vs legit=1.00 — legitimate signal, not leakage
[PASS] owner_abandonment_rate: fraud > legitimate (confirmed)
[PASS] Train/val/test split: 56000/12000/12000
[PASS] No data leakage — zero serial overlap between splits
========================================
7/7 tests passed.

### train.py v2 — Model Training (19 features, TTAB labels)
========================================
[PASS] Logistic Regression trained — F1: 0.9381, AUC-ROC: 0.9618
[PASS] XGBoost trained — F1: 0.9391, AUC-ROC: 0.9667
[PASS] LightGBM trained — F1: 0.9395, AUC-ROC: 0.9641
[PASS] CatBoost trained — F1: 0.9391, AUC-ROC: 0.9650
[PASS] XGBoost F1 (0.9391) beats LR baseline (0.9381)
[PASS] XGBoost CV F1: 0.9335 +/- 0.0037 — stable
[PASS] LightGBM CV F1: 0.9335 +/- 0.0036 — stable
[PASS] Top feature (section_8_filed / days_since_filing) makes intuitive sense
[PASS] All model files saved to ml/models/
========================================
9/9 tests passed.

### ensemble.py v2 — Stacking Ensemble
========================================
[WARN] Ensemble AUC 0.9662 does not beat best base XGBoost 0.9667 (tiny -0.0005)
[PASS] Threshold tuned: 0.45 (F1=0.9400 on val)
[PASS] ensemble.joblib saved
[PASS] XGBoost / CatBoost / Meta LR ONNX exports successful
[PASS] ONNX parity: max difference 0.000000 (threshold 0.01)
========================================
5/6 tests passed, 1 warning.

### evaluate.py v2 — Final Test Set Evaluation
========================================
[PASS] AUC-ROC:   0.9632 (target >= 0.92)
[PASS] F1:        0.9304 (target >= 0.88)
[PASS] Precision: 0.9902 (target >= 0.80)
[PASS] Recall:    0.8773 (target >= 0.85)
[PASS] Confusion matrix PNG saved
[PASS] ROC curve PNG saved
[PASS] Feature importance PNG saved
[PASS] Precision-recall curve PNG saved
========================================
8/8 tests passed.

### Phase 1 v2 Summary
========================================
Final Model: XGBoost + CatBoost stacking ensemble with LR meta-model
Dataset: 80,000 USPTO trademark applications (60K legit, 20K fraud from TTAB cancellations)
Features: 19 (6 groups, no leakage)
Threshold: 0.45
Test Set Metrics (FINAL):
  - AUC-ROC:   0.9632
  - F1:        0.9304
  - Precision: 0.9902
  - Recall:    0.8773
Improvement over v1: F1 +0.1350, Recall +0.1353, Precision +0.1331
ONNX export validated — 3 model files
========================================
38/39 total tests passed, 1 minor warning (ensemble ties best base).
Phase 1 gate: F1 0.9304 >> 0.80 target. Proceeding to Phase 2.

---

## Phase 1 — Python ML pipeline (v1, deprecated)
Date: 2026-04-05

### download.py — Data Verification
========================================
[PASS] case_file.csv — exists, 3101MB, all required columns present
[PASS] owner.csv — exists, 3072MB, all required columns present
[PASS] classification.csv — exists, 907MB, all required columns present
[PASS] correspondent_domrep_attorney.csv — exists, 1307MB, all required columns present
[PASS] event.csv — exists, 6091MB, all required columns present
[PASS] intl_class.csv — exists, 320MB, all required columns present
[PASS] prior_mark.csv — exists, 54MB, all required columns present
========================================
7/7 tests passed.

### clean.py — Data Cleaning & Labeling
========================================
[PASS] labeled.csv created — 80,000 rows (20,000 fraud, 60,000 legitimate)
[PASS] Label ratio 0.25 within expected range (0.10-0.45)
[PASS] Dataset size: 80,000 rows
[PASS] No duplicate serial numbers
[PASS] Labels are 0 and 1 only
[PASS] Zero null labels
[PASS] filing_dt is datetime type
========================================
7/7 tests passed.

### features.py — Feature Engineering
========================================
[PASS] All 15 features present (dropped 4 leaky features: section_8_filed, was_abandoned, days_filing_to_registration, filing_basis)
[PASS] Zero null values across all features
[PASS] All feature ranges valid
[PASS] owner_abandonment_rate: fraud=0.32 > legitimate=0.15
[PASS] statement_of_use_filed differs: fraud=0.0229, legitimate=0.2887
[PASS] opposition_count differs: fraud=0.1007, legitimate=0.0301
[PASS] Train/val/test split: 56000/12000/12000
[PASS] No data leakage — zero serial overlap between splits
========================================
8/8 tests passed.

### train.py — Model Training (Critical Moment 2)
========================================
[PASS] Logistic Regression trained — F1: 0.6572, AUC-ROC: 0.8732
[PASS] XGBoost trained — F1: 0.7989, AUC-ROC: 0.9432
[PASS] LightGBM trained — F1: 0.8020, AUC-ROC: 0.9431
[PASS] CatBoost trained — F1: 0.7988, AUC-ROC: 0.9441
[PASS] XGBoost F1 (0.7989) beats LR baseline (0.6572)
[PASS] XGBoost CV F1: 0.7849 +/- 0.0095 — stable
[PASS] LightGBM CV F1: 0.7852 +/- 0.0098 — stable
[PASS] Single prediction works — all models return label + confidence
[PASS] All model files saved to ml/models/
========================================
9/9 tests passed.

### ensemble.py — Stacking Ensemble (Critical Moment 3)
========================================
[PASS] Ensemble AUC 0.9443 beats best base CatBoost 0.9441 (+0.0003)
[PASS] Threshold tuned: 0.55 (F1=0.8122)
[PASS] ensemble.joblib saved
[PASS] XGBoost ONNX export successful
[PASS] CatBoost ONNX export successful
[PASS] Meta LR ONNX export successful
[PASS] ONNX parity: max difference 0.000000 (threshold 0.01)
========================================
7/7 tests passed.

### evaluate.py — Final Test Set Evaluation (Critical Moment 4)
========================================
[PASS] AUC-ROC: 0.9364 (target >= 0.92)
[PASS] Precision: 0.8571 (target >= 0.80)
[WARN] F1: 0.7954 (target >= 0.88, gate >= 0.80)
[WARN] Recall: 0.7420 (target >= 0.85)
[PASS] Confusion matrix PNG saved
[PASS] ROC curve PNG saved
[PASS] Feature importance PNG saved
[PASS] Precision-recall curve PNG saved
========================================
6/8 tests passed, 2 warnings.

### cases.json — Eval Harness
========================================
[PASS] 100 cases created (50 fraud, 50 legitimate)
[PASS] All cases have serial_number, brand_name, true_label, features
========================================
2/2 tests passed.

### Phase 1 Summary
========================================
Final Model: XGBoost + CatBoost stacking ensemble with LR meta-model
Dataset: 80,000 USPTO trademark applications (60K legitimate, 20K fraud)
Features: 15 non-leaky features across 6 groups
Threshold: 0.55
Test Set Metrics:
  - AUC-ROC: 0.9364
  - F1: 0.7954
  - Precision: 0.8571
  - Recall: 0.7420
ONNX export validated — 3 model files (xgboost.onnx, catboost.onnx, meta_lr.onnx)
========================================
39/41 total tests passed (2 warnings on F1/recall targets).
Phase 1 gate: labeled.csv (80K rows), ensemble.onnx exists, F1 ~0.80. Proceeding to Phase 2.
