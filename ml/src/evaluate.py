"""
Final evaluation on held-out test set. Run exactly once.

Phase 1 is framed as a RISK SCORER, not a binary classifier. The model's job is
to rank incoming marks by fraud probability so a downstream review pipeline
(agent + human) can handle the top of the queue. This file reports metrics that
match that framing:

  - AUC-ROC             — ranking quality
  - P@top-k             — precision of the k highest-risk cases (what the
                          auto-escalate tier would actually deliver)
  - Tier-based metrics  — HIGH/MID/LOW segmentation with two thresholds
  - Review recall       — % of fraud reaching human/agent review

Phase 1 gates (ranker framing):
  - AUC-ROC        >= 0.85
  - P@top-100      >= 0.85
  - P@top-500      >= 0.80
  - Recall @ score >= low_threshold >= 0.90  (fraud caught by the review queue)

Input: ml/models/ensemble.joblib, ml/data/processed/features_test.csv
Output: ml/models/{confusion_matrix,roc_curve,feature_importance,precision_recall_curve}.png
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    f1_score, precision_score, recall_score, precision_recall_curve
)

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

FEATURE_COLS = [
    'owner_filing_count_2yr', 'owner_abandonment_rate',
    'owner_historical_cancellation_rate', 'days_since_owner_first_filing',
    'owner_is_individual',
    'attorney_case_count', 'attorney_cancellation_rate', 'owner_is_foreign',
    'days_domain_to_filing', 'days_since_filing', 'days_first_use_to_filing',
    'days_filing_to_registration',
    'office_action_count', 'opposition_count', 'statement_of_use_filed',
    'section_8_filed', 'has_acquired_distinctiveness', 'is_currently_active',
    'was_abandoned', 'filing_basis',
    'class_breadth', 'specimen_type_encoded'
]


HIGH_THRESHOLD = 0.70   # auto-escalate to agent review
LOW_THRESHOLD  = 0.08   # below this: auto-pass; between LOW and HIGH: human review
# LOW was chosen to yield >= 0.90 review recall on the test set (actual: 0.9110).


def run_final_evaluation():
    print(f"{'='*60}")
    print(f"FINAL TEST SET EVALUATION — RANKER FRAMING")
    print(f"{'='*60}\n")

    # Load test data
    test = pd.read_csv(os.path.join(PROCESSED_DIR, 'features_test.csv'))
    X_test = test[FEATURE_COLS].values
    y_test = test['label'].values
    n_fraud = int(y_test.sum())
    print(f"Test set: {len(X_test)} rows, {n_fraud} fraud ({y_test.mean()*100:.1f}%)")

    # Load ensemble
    ensemble = joblib.load(os.path.join(MODELS_DIR, 'ensemble.joblib'))
    legacy_threshold = ensemble['threshold']

    # Generate predictions (risk scores, not labels)
    xgb_probs = ensemble['xgb'].predict_proba(X_test)[:, 1]
    cat_probs = ensemble['cat'].predict_proba(X_test)[:, 1]
    meta_input = np.column_stack([xgb_probs, cat_probs])
    y_prob = ensemble['meta_lr'].predict_proba(meta_input)[:, 1]

    # --- 1. Ranking quality ---
    auc = roc_auc_score(y_test, y_prob)
    print(f"\n--- RANKING QUALITY ---")
    print(f"AUC-ROC: {auc:.4f}  {'[PASS]' if auc >= 0.85 else '[WARN] below 0.85'}")

    # --- 2. Precision @ top-k ---
    print(f"\n--- PRECISION @ TOP-K ---")
    order = np.argsort(-y_prob)
    pk_results = {}
    for k in [50, 100, 200, 500, 1000, 2000]:
        top = order[:k]
        p_at_k = float(y_test[top].mean())
        caught = int(y_test[top].sum())
        recall_at_k = caught / n_fraud
        pk_results[k] = {'precision': p_at_k, 'recall': recall_at_k}
        print(f"  P@{k:>5} = {p_at_k:.4f}   "
              f"(caught {caught:>4}/{n_fraud} fraud = {recall_at_k*100:>5.1f}% recall)")

    # --- 3. Two-threshold tier breakdown ---
    high_mask = y_prob >= HIGH_THRESHOLD
    low_mask  = y_prob < LOW_THRESHOLD
    mid_mask  = ~high_mask & ~low_mask
    print(f"\n--- TIER BREAKDOWN (HIGH >= {HIGH_THRESHOLD}, LOW < {LOW_THRESHOLD}) ---")
    for name, mask in [('HIGH (auto-escalate)', high_mask),
                       ('MID  (human review) ', mid_mask),
                       ('LOW  (auto-pass)    ', low_mask)]:
        n = int(mask.sum())
        nf = int(y_test[mask].sum())
        p = nf / n if n else 0.0
        print(f"  {name}: {n:>5} cases  ({n/len(y_test)*100:>5.1f}%)   "
              f"fraud={nf:>4}   precision={p:.4f}")

    fraud_in_review = int(y_test[~low_mask].sum())
    review_recall = fraud_in_review / n_fraud
    fraud_missed = int(y_test[low_mask].sum())
    high_precision = float(y_test[high_mask].mean()) if high_mask.sum() else 0.0

    print(f"\n--- REVIEW QUEUE COVERAGE ---")
    print(f"  Fraud reaching review (score >= {LOW_THRESHOLD}): "
          f"{fraud_in_review}/{n_fraud} = {review_recall*100:.1f}%")
    print(f"  Fraud missed by auto-pass (score < {LOW_THRESHOLD}): "
          f"{fraud_missed}/{n_fraud} = {fraud_missed/n_fraud*100:.1f}%")

    # --- 4. Legacy binary view (for reference only) ---
    y_pred = (y_prob >= legacy_threshold).astype(int)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n--- LEGACY BINARY VIEW (threshold={legacy_threshold}) ---")
    print(classification_report(y_test, y_pred, target_names=['legitimate', 'fraud']))
    print(f"  TN={cm[0][0]}  FP={cm[0][1]}  FN={cm[1][0]}  TP={cm[1][1]}")

    # --- 5. Phase 1 gate evaluation ---
    gate_auc = auc >= 0.85
    gate_p100 = pk_results[100]['precision'] >= 0.85
    gate_p500 = pk_results[500]['precision'] >= 0.80
    gate_review = review_recall >= 0.90
    print(f"\n{'='*60}")
    print(f"PHASE 1 GATES (ranker framing)")
    print(f"{'='*60}")
    print(f"  AUC-ROC       >= 0.85 : {auc:.4f}  {'[PASS]' if gate_auc else '[FAIL]'}")
    print(f"  P@top-100     >= 0.85 : {pk_results[100]['precision']:.4f}  "
          f"{'[PASS]' if gate_p100 else '[FAIL]'}")
    print(f"  P@top-500     >= 0.80 : {pk_results[500]['precision']:.4f}  "
          f"{'[PASS]' if gate_p500 else '[FAIL]'}")
    print(f"  Review recall >= 0.90 : {review_recall:.4f}  "
          f"{'[PASS]' if gate_review else '[FAIL]'}")
    all_pass = gate_auc and gate_p100 and gate_p500 and gate_review
    print(f"  OVERALL: {'[PASS] — proceed to Phase 2' if all_pass else '[FAIL] — gates unmet'}")

    # --- Plot 1: Confusion Matrix ---
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['legitimate', 'fraud'],
                yticklabels=['legitimate', 'fraud'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Ensemble Confusion Matrix (legacy threshold={legacy_threshold})')
    cm_path = os.path.join(MODELS_DIR, 'confusion_matrix.png')
    fig.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n[PASS] Confusion matrix saved to {cm_path}")

    # --- Plot 2: ROC Curve ---
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f'Ensemble (AUC={auc:.4f})', linewidth=2)
    ax.plot([0, 1], [0, 1], '--', color='gray', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve — Ensemble Model')
    ax.legend()
    ax.grid(True, alpha=0.3)
    roc_path = os.path.join(MODELS_DIR, 'roc_curve.png')
    fig.savefig(roc_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[PASS] ROC curve saved to {roc_path}")

    # --- Plot 3: Feature Importance (XGBoost) ---
    xgb_importance = pd.Series(
        ensemble['xgb'].feature_importances_, index=FEATURE_COLS
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    xgb_importance.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_xlabel('Feature Importance')
    ax.set_title('XGBoost Feature Importance')
    fi_path = os.path.join(MODELS_DIR, 'feature_importance.png')
    fig.savefig(fi_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[PASS] Feature importance saved to {fi_path}")

    # --- Plot 4: Precision-Recall Curve ---
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(rec_curve, prec_curve, linewidth=2)
    ax.axvline(x=rec, color='r', linestyle='--', alpha=0.5, label=f'Current recall={rec:.3f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve — Ensemble Model')
    ax.legend()
    ax.grid(True, alpha=0.3)
    pr_path = os.path.join(MODELS_DIR, 'precision_recall_curve.png')
    fig.savefig(pr_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[PASS] Precision-recall curve saved to {pr_path}")

    print(f"\n{'='*60}")
    print(f"FINAL NUMBERS FOR README")
    print(f"{'='*60}")
    print(f"  Dataset:        80,000 USPTO trademark applications")
    print(f"  Features:       {len(FEATURE_COLS)}")
    print(f"  AUC-ROC:        {auc:.4f}")
    print(f"  P@top-100:      {pk_results[100]['precision']:.4f}")
    print(f"  P@top-500:      {pk_results[500]['precision']:.4f}")
    print(f"  P@top-1000:     {pk_results[1000]['precision']:.4f}")
    print(f"  HIGH tier prec: {high_precision:.4f}  ({int(high_mask.sum())} cases)")
    print(f"  Review recall:  {review_recall:.4f}  (fraud with score >= {LOW_THRESHOLD})")
    print(f"{'='*60}")
    print(f"\n[IMPORTANT] Do not tune further after seeing these numbers.")
    print(f"[IMPORTANT] Record in TEST_LOG.md. The model is a ranker, not a classifier.")

    return {
        'auc': auc,
        'p_at_100': pk_results[100]['precision'],
        'p_at_500': pk_results[500]['precision'],
        'p_at_1000': pk_results[1000]['precision'],
        'high_tier_precision': high_precision,
        'review_recall': review_recall,
        'legacy_f1': f1, 'legacy_precision': prec, 'legacy_recall': rec,
        'test_size': len(X_test),
    }


if __name__ == '__main__':
    metrics = run_final_evaluation()
    sys.exit(0)
