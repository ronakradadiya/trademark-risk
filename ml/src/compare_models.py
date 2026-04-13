"""
Side-by-side gate: score OLD v4 (22 features) vs NEW v4 (13 features) on the
common held-out test set.

Fairness: the OLD model in production receives 14 BASELINE constants from
app/src/lib/features.ts for all mark-side features (it never had the real ones
at serving time). We score OLD the same way here — 8 real applicant features
plus 14 constants — because that's what it would return for a live request.
NEW gets its real 13 features.

Gate (from plan verification step 5):
  PASS if: new Brier <= old Brier  AND  new AUC >= old AUC - 0.02
  FAIL otherwise — pause and report before wiring anything downstream.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, precision_score, recall_score, f1_score,
)

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
OLD_DIR = os.path.join(os.path.dirname(__file__), '..', 'models_v4_old')
NEW_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

# These match app/src/lib/features.ts BASELINE_MARK_FEATURES exactly — the
# constants the OLD model receives for every live request today.
BASELINE_MARK_FEATURES = {
    'days_domain_to_filing': 180,
    'days_since_filing': 0,
    'days_first_use_to_filing': 90,
    'days_filing_to_registration': 180,
    'office_action_count': 0,
    'opposition_count': 0,
    'statement_of_use_filed': 0,
    'section_8_filed': 0,
    'has_acquired_distinctiveness': 0,
    'is_currently_active': 1,
    'was_abandoned': 0,
    'filing_basis': 1,
    'class_breadth': 1,
    'specimen_type_encoded': 1,
}

NEW_FEATURE_COLS = [
    'owner_filing_count_2yr', 'owner_abandonment_rate',
    'owner_historical_cancellation_rate', 'days_since_owner_first_filing',
    'owner_is_individual', 'owner_is_foreign',
    'attorney_case_count', 'attorney_cancellation_rate',
    'days_since_filing', 'days_filing_to_registration',
    'was_abandoned', 'is_currently_active', 'class_breadth',
]


def ensemble_score(bundle, X):
    xgb_p = bundle['xgb'].predict_proba(X)[:, 1]
    cat_p = bundle['cat'].predict_proba(X)[:, 1]
    meta_in = np.column_stack([xgb_p, cat_p])
    return bundle['meta_lr'].predict_proba(meta_in)[:, 1]


def metrics_block(name, y_true, y_prob, threshold):
    auc = roc_auc_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"\n  {name}")
    print(f"    AUC-ROC:   {auc:.4f}")
    print(f"    Brier:     {brier:.4f}  (lower = better calibration)")
    print(f"    Threshold: {threshold}")
    print(f"    Precision: {p:.4f}")
    print(f"    Recall:    {r:.4f}")
    print(f"    F1:        {f1:.4f}")
    return {'auc': auc, 'brier': brier, 'precision': p, 'recall': r, 'f1': f1}


def main():
    print("=== Loading test set ===")
    test = pd.read_csv(os.path.join(PROCESSED_DIR, 'features_test.csv'))
    y = test['label'].values
    n_pos = int(y.sum())
    print(f"  rows: {len(test)}  positives: {n_pos}  ({n_pos / len(test) * 100:.1f}%)")

    print("\n=== Loading bundles ===")
    old = joblib.load(os.path.join(OLD_DIR, 'ensemble.joblib'))
    new = joblib.load(os.path.join(NEW_DIR, 'ensemble.joblib'))
    old_cols = old['feature_cols']
    old_thr = old['threshold']
    new_cols = new['feature_cols']
    new_thr = new['threshold']
    print(f"  OLD: {len(old_cols)} features, threshold={old_thr}")
    print(f"  NEW: {len(new_cols)} features, threshold={new_thr}")
    assert new_cols == NEW_FEATURE_COLS, "new model feature order mismatch"

    print("\n=== Building OLD feature matrix (prod-fidelity) ===")
    # Applicant features (8): real from test set.
    # Mark features (14): BASELINE constants — what OLD sees at serving time.
    applicant_cols = [
        'owner_filing_count_2yr', 'owner_abandonment_rate',
        'owner_historical_cancellation_rate', 'days_since_owner_first_filing',
        'owner_is_individual', 'attorney_case_count',
        'attorney_cancellation_rate', 'owner_is_foreign',
    ]
    old_rows = []
    for col in old_cols:
        if col in applicant_cols:
            old_rows.append(test[col].values)
        elif col in BASELINE_MARK_FEATURES:
            old_rows.append(np.full(len(test), BASELINE_MARK_FEATURES[col], dtype=float))
        else:
            raise RuntimeError(f"unknown old feature: {col}")
    X_old = np.column_stack(old_rows).astype(float)
    print(f"  shape: {X_old.shape}")

    print("\n=== Building NEW feature matrix ===")
    X_new = test[new_cols].values.astype(float)
    print(f"  shape: {X_new.shape}")

    print("\n=== Scoring ===")
    p_old = ensemble_score(old, X_old)
    p_new = ensemble_score(new, X_new)

    print("\n" + "=" * 60)
    print("SIDE-BY-SIDE (common test set, prod-fidelity)")
    print("=" * 60)
    old_m = metrics_block("OLD (22 features, 14 constants at serving)", y, p_old, old_thr)
    new_m = metrics_block("NEW (13 features, all real at serving)", y, p_new, new_thr)

    print("\n" + "=" * 60)
    print("DELTAS (new - old)")
    print("=" * 60)
    for k in ['auc', 'brier', 'precision', 'recall', 'f1']:
        d = new_m[k] - old_m[k]
        arrow = "↓" if d < 0 else "↑"
        print(f"  {k:<10} {d:+.4f} {arrow}")

    print("\n" + "=" * 60)
    print("GATE")
    print("=" * 60)
    brier_win = new_m['brier'] <= old_m['brier']
    auc_ok = new_m['auc'] >= old_m['auc'] - 0.02
    print(f"  Brier: new {new_m['brier']:.4f} vs old {old_m['brier']:.4f}  "
          f"{'PASS' if brier_win else 'FAIL'} (new must be lower)")
    print(f"  AUC:   new {new_m['auc']:.4f} vs old {old_m['auc']:.4f}  "
          f"{'PASS' if auc_ok else 'FAIL'} (new must be within -0.02)")

    if brier_win and auc_ok:
        print("\n  [GATE PASS] ship the new model")
        return 0
    else:
        print("\n  [GATE FAIL] do not wire new model — investigate")
        return 1


if __name__ == '__main__':
    sys.exit(main())
