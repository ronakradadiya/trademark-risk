"""
Ablation: retrain XGBoost/LightGBM/CatBoost without section_8_filed.
If F1 collapses, section_8_filed is a leaky proxy for the label definition.
If F1 stays high, the other 18 features carry real signal.
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')

FEATURE_COLS_FULL = [
    'owner_filing_count_2yr', 'owner_abandonment_rate', 'owner_is_individual',
    'attorney_case_count', 'owner_is_foreign',
    'days_domain_to_filing', 'days_since_filing', 'days_first_use_to_filing',
    'days_filing_to_registration',
    'office_action_count', 'opposition_count', 'statement_of_use_filed',
    'section_8_filed', 'has_acquired_distinctiveness', 'is_currently_active',
    'was_abandoned', 'filing_basis',
    'class_breadth', 'specimen_type_encoded'
]
FEATURE_COLS_ABLATED = [f for f in FEATURE_COLS_FULL if f != 'section_8_filed']


def run(feature_cols, label):
    train = pd.read_csv(os.path.join(PROCESSED_DIR, 'features_train.csv'))
    val = pd.read_csv(os.path.join(PROCESSED_DIR, 'features_val.csv'))

    X_train = train[feature_cols].values
    y_train = train['label'].values
    X_val = val[feature_cols].values
    y_val = val['label'].values

    print(f"\n{'='*60}\n{label} ({len(feature_cols)} features)\n{'='*60}")

    results = []
    models = [
        ("XGBoost", XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                                  scale_pos_weight=3, eval_metric='aucpr',
                                  early_stopping_rounds=50, random_state=42, verbosity=0)),
        ("LightGBM", LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                                    class_weight='balanced', random_state=42, verbose=-1)),
        ("CatBoost", CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05,
                                        auto_class_weights='Balanced', random_seed=42, verbose=0)),
    ]
    for name, model in models:
        if name == "XGBoost":
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        elif name == "LightGBM":
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        else:
            model.fit(X_train, y_train, eval_set=(X_val, y_val))
        prob = model.predict_proba(X_val)[:, 1]
        pred = (prob >= 0.5).astype(int)
        r = {
            'model': name,
            'auc': roc_auc_score(y_val, prob),
            'f1': f1_score(y_val, pred),
            'precision': precision_score(y_val, pred),
            'recall': recall_score(y_val, pred),
        }
        results.append(r)
        print(f"  {name:10s}  AUC={r['auc']:.4f}  F1={r['f1']:.4f}  "
              f"P={r['precision']:.4f}  R={r['recall']:.4f}")
    return results


if __name__ == '__main__':
    print("Running ablation — comparing full vs. without section_8_filed\n")
    full = run(FEATURE_COLS_FULL, "FULL (19 features)")
    ablated = run(FEATURE_COLS_ABLATED, "ABLATED (no section_8_filed)")

    print(f"\n{'='*60}\nDELTA (ablated - full)\n{'='*60}")
    print(f"{'Model':<10} {'ΔAUC':>10} {'ΔF1':>10} {'ΔPrec':>10} {'ΔRec':>10}")
    for f, a in zip(full, ablated):
        print(f"{f['model']:<10} "
              f"{a['auc']-f['auc']:>+10.4f} "
              f"{a['f1']-f['f1']:>+10.4f} "
              f"{a['precision']-f['precision']:>+10.4f} "
              f"{a['recall']-f['recall']:>+10.4f}")

    print(f"\n{'='*60}")
    worst_drop = min(a['f1'] - f['f1'] for f, a in zip(full, ablated))
    if worst_drop < -0.10:
        print(f"[LEAKY]   F1 drops by {abs(worst_drop):.3f} — section_8_filed was doing the work")
    elif worst_drop < -0.03:
        print(f"[PARTIAL] F1 drops by {abs(worst_drop):.3f} — feature matters but isn't the whole story")
    else:
        print(f"[OK]      F1 drops by {abs(worst_drop):.3f} — other features carry real signal")
