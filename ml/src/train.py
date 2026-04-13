"""
Trains Logistic Regression baseline, XGBoost, LightGBM, and CatBoost.
Prints full model reports (Critical Moment 2) and feature importance.
Saves trained models to ml/models/.

Input: ml/data/processed/features_train.csv, features_val.csv
Output: ml/models/{logistic_regression,xgboost,lightgbm,catboost}.joblib
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURE_COLS = [
    'owner_filing_count_2yr', 'owner_abandonment_rate',
    'owner_historical_cancellation_rate', 'days_since_owner_first_filing',
    'owner_is_individual', 'owner_is_foreign',
    'attorney_case_count', 'attorney_cancellation_rate',
    'days_since_filing', 'days_filing_to_registration',
    'was_abandoned', 'is_currently_active', 'class_breadth',
]


def load_data():
    train = pd.read_csv(os.path.join(PROCESSED_DIR, 'features_train.csv'))
    val = pd.read_csv(os.path.join(PROCESSED_DIR, 'features_val.csv'))

    X_train = train[FEATURE_COLS].values
    y_train = train['label'].values
    X_val = val[FEATURE_COLS].values
    y_val = val['label'].values

    print(f"Train: {len(X_train)} rows ({y_train.mean()*100:.1f}% fraud)")
    print(f"Val:   {len(X_val)} rows ({y_val.mean()*100:.1f}% fraud)")
    return X_train, y_train, X_val, y_val


def full_model_report(model_name, y_true, y_pred, y_prob):
    """Critical Moment 2 — full metrics for each model."""
    print(f"\n{'='*50}")
    print(f"MODEL: {model_name}")
    print(f"{'='*50}")

    print(classification_report(y_true, y_pred, target_names=['legitimate', 'fraud']))

    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion matrix:")
    print(f"  True Negative  (caught legitimate correctly): {cm[0][0]}")
    print(f"  False Positive (flagged legitimate as fraud): {cm[0][1]}")
    print(f"  False Negative (missed actual fraud):         {cm[1][0]}")
    print(f"  True Positive  (caught fraud correctly):      {cm[1][1]}")

    auc = roc_auc_score(y_true, y_prob)
    print(f"AUC-ROC: {auc:.4f}")

    fraud_precision = cm[1][1] / (cm[0][1] + cm[1][1]) if (cm[0][1] + cm[1][1]) > 0 else 0
    fraud_recall = cm[1][1] / (cm[1][0] + cm[1][1]) if (cm[1][0] + cm[1][1]) > 0 else 0
    f1 = f1_score(y_true, y_pred)
    print(f"Fraud precision: {fraud_precision:.4f}")
    print(f"Fraud recall:    {fraud_recall:.4f}")
    print(f"F1 score:        {f1:.4f}")

    if fraud_recall < 0.85:
        print(f"[WARN] Recall {fraud_recall:.2f} below target 0.85")
    if fraud_precision < 0.80:
        print(f"[WARN] Precision {fraud_precision:.2f} below target 0.80")

    return {'model': model_name, 'auc': auc, 'recall': fraud_recall,
            'precision': fraud_precision, 'f1': f1}


def check_feature_importance(model, model_name, feature_names):
    importance = pd.Series(model.feature_importances_, index=feature_names)
    importance = importance.sort_values(ascending=False)
    max_imp = importance.max() if importance.max() > 0 else 1
    print(f"\n=== FEATURE IMPORTANCE: {model_name} ===")
    for feat, score in importance.items():
        bar = '█' * int((score / max_imp) * 30)
        print(f"  {feat:35s}: {score:.4f} {bar}")

    top_feature = importance.index[0]
    print(f"\nTop feature: {top_feature}")
    expected_top = ['owner_filing_count_2yr', 'section_8_filed', 'days_filing_to_registration',
                    'opposition_count', 'was_abandoned', 'filing_basis',
                    'owner_abandonment_rate', 'statement_of_use_filed',
                    'days_since_filing', 'days_first_use_to_filing']
    if top_feature in expected_top:
        print(f"[PASS] Top feature makes intuitive sense for fraud detection")
    else:
        print(f"[WARN] Unexpected top feature '{top_feature}' — review feature engineering")


def train_all():
    X_train, y_train, X_val, y_val = load_data()
    results = []

    # --- 1. Logistic Regression baseline ---
    print("\n\n### Training Logistic Regression (baseline)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr.fit(X_train_scaled, y_train)

    lr_pred = lr.predict(X_val_scaled)
    lr_prob = lr.predict_proba(X_val_scaled)[:, 1]
    lr_result = full_model_report("Logistic Regression", y_val, lr_pred, lr_prob)
    results.append(lr_result)

    joblib.dump({'model': lr, 'scaler': scaler}, os.path.join(MODELS_DIR, 'logistic_regression.joblib'))

    # --- 2. XGBoost ---
    print("\n\n### Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=3,
        eval_metric='aucpr',
        early_stopping_rounds=50,
        random_state=42,
        verbosity=0
    )
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    xgb_pred = xgb.predict(X_val)
    xgb_prob = xgb.predict_proba(X_val)[:, 1]
    xgb_result = full_model_report("XGBoost", y_val, xgb_pred, xgb_prob)
    results.append(xgb_result)
    check_feature_importance(xgb, "XGBoost", FEATURE_COLS)

    joblib.dump(xgb, os.path.join(MODELS_DIR, 'xgboost.joblib'))

    # --- 3. LightGBM ---
    print("\n\n### Training LightGBM...")
    lgbm = LGBMClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        class_weight='balanced',
        metric='average_precision',
        random_state=42,
        verbose=-1
    )
    lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    lgbm_pred = lgbm.predict(X_val)
    lgbm_prob = lgbm.predict_proba(X_val)[:, 1]
    lgbm_result = full_model_report("LightGBM", y_val, lgbm_pred, lgbm_prob)
    results.append(lgbm_result)
    check_feature_importance(lgbm, "LightGBM", FEATURE_COLS)

    joblib.dump(lgbm, os.path.join(MODELS_DIR, 'lightgbm.joblib'))

    # --- 4. CatBoost ---
    print("\n\n### Training CatBoost...")
    cat = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        auto_class_weights='Balanced',
        random_seed=42,
        verbose=0
    )
    cat.fit(X_train, y_train, eval_set=(X_val, y_val))

    cat_pred = cat.predict(X_val).astype(int)
    cat_prob = cat.predict_proba(X_val)[:, 1]
    cat_result = full_model_report("CatBoost", y_val, cat_pred, cat_prob)
    results.append(cat_result)
    check_feature_importance(cat, "CatBoost", FEATURE_COLS)

    joblib.dump(cat, os.path.join(MODELS_DIR, 'catboost.joblib'))

    # --- Cross-validation stability ---
    print(f"\n\n{'='*50}")
    print("CROSS-VALIDATION STABILITY CHECK")
    print(f"{'='*50}")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Use fresh models without early_stopping for CV (no eval set available)
    xgb_cv = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                           scale_pos_weight=3, random_state=42, verbosity=0)
    lgbm_cv = LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                             class_weight='balanced', random_state=42, verbose=-1)
    for name, model in [("XGBoost", xgb_cv), ("LightGBM", lgbm_cv)]:
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
        print(f"\n{name} CV F1: {scores.mean():.4f} +/- {scores.std():.4f}")
        print(f"  Fold scores: {[f'{s:.4f}' for s in scores]}")
        if scores.std() > 0.05:
            print(f"  [WARN] High variance ({scores.std():.4f}) — model unstable")
        else:
            print(f"  [PASS] Stable across folds (std={scores.std():.4f} < 0.05)")

    # --- Model comparison (Critical Moment 3 prep) ---
    print(f"\n\n{'='*70}")
    print(f"{'MODEL COMPARISON TABLE':^70}")
    print(f"{'='*70}")
    print(f"{'Model':<25} {'AUC-ROC':>10} {'Recall':>10} {'Precision':>10} {'F1':>10}")
    print(f"{'-'*70}")
    for r in sorted(results, key=lambda x: x['auc'], reverse=True):
        print(f"{r['model']:<25} {r['auc']:>10.4f} {r['recall']:>10.4f} "
              f"{r['precision']:>10.4f} {r['f1']:>10.4f}")
    print(f"{'='*70}")

    best = max(results, key=lambda x: x['auc'])
    print(f"\nBest single model: {best['model']} (AUC-ROC: {best['auc']:.4f})")

    # Verify XGBoost beats LR baseline
    lr_f1 = results[0]['f1']
    xgb_f1 = results[1]['f1']
    if xgb_f1 > lr_f1:
        print(f"[PASS] XGBoost F1 ({xgb_f1:.4f}) beats LR baseline ({lr_f1:.4f})")
    else:
        print(f"[WARN] XGBoost F1 ({xgb_f1:.4f}) does NOT beat LR baseline ({lr_f1:.4f})")

    # Single prediction test
    print(f"\n=== SINGLE PREDICTION TEST ===")
    test_row = X_val[:1]
    for name, model in [("XGBoost", xgb), ("LightGBM", lgbm), ("CatBoost", cat)]:
        pred = model.predict(test_row)[0]
        prob = model.predict_proba(test_row)[0]
        print(f"  {name}: label={pred}, confidence=[{prob[0]:.4f}, {prob[1]:.4f}]")

    return results


if __name__ == '__main__':
    results = train_all()
    sys.exit(0)
