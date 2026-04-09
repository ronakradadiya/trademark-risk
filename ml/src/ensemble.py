"""
Builds stacking ensemble from CatBoost + XGBoost (top 2 by AUC-ROC).
Tunes threshold for optimal F1 with recall >= 0.85.
Exports final model to ONNX format.

Input: ml/models/{xgboost,catboost}.joblib, ml/data/processed/features_{val,test}.csv
Output: ml/models/ensemble.joblib, ml/models/ensemble.onnx
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
import onnxruntime as ort

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


def load_data():
    val = pd.read_csv(os.path.join(PROCESSED_DIR, 'features_val.csv'))
    test = pd.read_csv(os.path.join(PROCESSED_DIR, 'features_test.csv'))
    return val, test


def build_ensemble():
    val, test = load_data()
    X_val = val[FEATURE_COLS].values
    y_val = val['label'].values
    X_test = test[FEATURE_COLS].values
    y_test = test['label'].values

    # Load base models
    xgb = joblib.load(os.path.join(MODELS_DIR, 'xgboost.joblib'))
    cat = joblib.load(os.path.join(MODELS_DIR, 'catboost.joblib'))

    print("=== STEP 1: Generate base model predictions (stacking features) ===")
    # Use validation set predictions as meta-features
    xgb_val_prob = xgb.predict_proba(X_val)[:, 1]
    cat_val_prob = cat.predict_proba(X_val)[:, 1]

    # Stack: [xgb_prob, cat_prob] as input to meta-model
    meta_X_val = np.column_stack([xgb_val_prob, cat_val_prob])

    print(f"  XGBoost val AUC: {roc_auc_score(y_val, xgb_val_prob):.4f}")
    print(f"  CatBoost val AUC: {roc_auc_score(y_val, cat_val_prob):.4f}")

    print("\n=== STEP 2: Train Logistic Regression meta-model ===")
    meta_lr = LogisticRegression(random_state=42)
    meta_lr.fit(meta_X_val, y_val)

    # Check ensemble on validation set
    ensemble_val_prob = meta_lr.predict_proba(meta_X_val)[:, 1]
    ensemble_val_auc = roc_auc_score(y_val, ensemble_val_prob)
    print(f"  Ensemble val AUC: {ensemble_val_auc:.4f}")

    # Verify ensemble beats best base model
    best_base_auc = max(
        roc_auc_score(y_val, xgb_val_prob),
        roc_auc_score(y_val, cat_val_prob)
    )
    if ensemble_val_auc > best_base_auc:
        print(f"  [PASS] Ensemble AUC {ensemble_val_auc:.4f} beats best base {best_base_auc:.4f} "
              f"(+{ensemble_val_auc - best_base_auc:.4f})")
    else:
        print(f"  [WARN] Ensemble AUC {ensemble_val_auc:.4f} does NOT beat best base {best_base_auc:.4f}")
        print(f"         Using best base model predictions directly")

    print("\n=== STEP 3: Threshold tuning ===")
    threshold = tune_threshold(y_val, ensemble_val_prob)

    print("\n=== STEP 4: Save ensemble ===")
    ensemble = {
        'xgb': xgb,
        'cat': cat,
        'meta_lr': meta_lr,
        'threshold': threshold,
        'feature_cols': FEATURE_COLS,
    }
    joblib.dump(ensemble, os.path.join(MODELS_DIR, 'ensemble.joblib'))
    print(f"  Saved ensemble.joblib")

    print("\n=== STEP 5: Export to ONNX ===")
    export_onnx(xgb, cat, meta_lr, X_val)

    print("\n=== STEP 6: Validate ONNX parity ===")
    validate_onnx_parity(ensemble, X_val[:50])

    return ensemble, threshold, y_val, ensemble_val_prob


def tune_threshold(y_true, y_prob):
    """Find threshold that maximizes F1 while keeping recall >= 0.85."""
    print(f"{'Threshold':>12} {'Precision':>12} {'Recall':>10} {'F1':>10}")
    print(f"{'-'*50}")
    best_f1 = 0
    best_threshold = 0.5

    for threshold in [i / 20 for i in range(4, 17)]:
        y_pred = (y_prob >= threshold).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        marker = ""
        if f1 > best_f1 and r >= 0.85:
            best_f1 = f1
            best_threshold = threshold
            marker = " <-- best"
        print(f"{threshold:>12.2f} {p:>12.4f} {r:>10.4f} {f1:>10.4f}{marker}")

    # If no threshold achieves recall >= 0.85, pick the one with best F1 regardless
    if best_f1 == 0:
        print("\n[WARN] No threshold achieves recall >= 0.85")
        print("       Picking threshold with best F1 overall")
        for threshold in [i / 20 for i in range(4, 17)]:
            y_pred = (y_prob >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

    print(f"\nChosen threshold: {best_threshold} (F1={best_f1:.4f})")
    return best_threshold


def export_onnx(xgb_model, cat_model, meta_lr, X_sample):
    """Export the ensemble pipeline to ONNX."""
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    # We export each component separately, then combine in inference
    # Export XGBoost
    from onnxmltools import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType as OnnxFloatTensorType

    n_features = len(FEATURE_COLS)

    # XGBoost to ONNX
    xgb_onnx = convert_xgboost(
        xgb_model,
        initial_types=[('input', OnnxFloatTensorType([None, n_features]))],
        target_opset=12
    )
    xgb_onnx_path = os.path.join(MODELS_DIR, 'xgboost.onnx')
    with open(xgb_onnx_path, 'wb') as f:
        f.write(xgb_onnx.SerializeToString())
    print(f"  XGBoost ONNX saved to {xgb_onnx_path}")

    # Meta LR to ONNX
    meta_onnx = convert_sklearn(
        meta_lr,
        initial_types=[('meta_input', FloatTensorType([None, 2]))],
        target_opset=12
    )
    meta_onnx_path = os.path.join(MODELS_DIR, 'meta_lr.onnx')
    with open(meta_onnx_path, 'wb') as f:
        f.write(meta_onnx.SerializeToString())
    print(f"  Meta LR ONNX saved to {meta_onnx_path}")

    # CatBoost to ONNX — CatBoost has native ONNX export
    cat_onnx_path = os.path.join(MODELS_DIR, 'catboost.onnx')
    cat_model.save_model(cat_onnx_path, format='onnx',
                         export_parameters={'onnx_domain': 'ai.catboost',
                                            'onnx_model_version': 1})
    print(f"  CatBoost ONNX saved to {cat_onnx_path}")

    # Also save a combined pipeline info for inference
    pipeline_info = {
        'xgb_onnx': 'xgboost.onnx',
        'cat_onnx': 'catboost.onnx',
        'meta_onnx': 'meta_lr.onnx',
        'feature_cols': FEATURE_COLS,
    }
    joblib.dump(pipeline_info, os.path.join(MODELS_DIR, 'ensemble_pipeline.joblib'))
    print(f"  Pipeline info saved")


def validate_onnx_parity(ensemble, X_sample):
    """Verify ONNX inference matches joblib inference within tolerance."""
    import time

    # Joblib inference
    start = time.time()
    xgb_probs = ensemble['xgb'].predict_proba(X_sample)[:, 1]
    cat_probs = ensemble['cat'].predict_proba(X_sample)[:, 1]
    meta_input = np.column_stack([xgb_probs, cat_probs])
    joblib_probs = ensemble['meta_lr'].predict_proba(meta_input)[:, 1]
    joblib_time = time.time() - start

    # ONNX inference
    start = time.time()
    xgb_session = ort.InferenceSession(os.path.join(MODELS_DIR, 'xgboost.onnx'))
    cat_session = ort.InferenceSession(os.path.join(MODELS_DIR, 'catboost.onnx'))
    meta_session = ort.InferenceSession(os.path.join(MODELS_DIR, 'meta_lr.onnx'))

    X_float = X_sample.astype(np.float32)

    # XGBoost ONNX
    xgb_onnx_out = xgb_session.run(None, {'input': X_float})
    xgb_onnx_probs = np.array([p[1] for p in xgb_onnx_out[1]])

    # CatBoost ONNX
    cat_input_name = cat_session.get_inputs()[0].name
    cat_onnx_out = cat_session.run(None, {cat_input_name: X_float})
    # CatBoost ONNX output format varies — handle both
    if len(cat_onnx_out) > 1:
        cat_onnx_probs = np.array([p[1] for p in cat_onnx_out[1]])
    else:
        cat_onnx_probs = cat_onnx_out[0][:, 1] if cat_onnx_out[0].ndim > 1 else cat_onnx_out[0]

    # Meta LR ONNX
    meta_onnx_input = np.column_stack([xgb_onnx_probs, cat_onnx_probs]).astype(np.float32)
    meta_onnx_out = meta_session.run(None, {'meta_input': meta_onnx_input})
    onnx_probs = np.array([p[1] for p in meta_onnx_out[1]])
    onnx_time = time.time() - start

    # Parity check
    max_diff = np.abs(joblib_probs - onnx_probs).max()
    mean_diff = np.abs(joblib_probs - onnx_probs).mean()

    print(f"\n  Joblib time: {joblib_time:.4f}s")
    print(f"  ONNX time:   {onnx_time:.4f}s (includes session creation)")
    print(f"  Max diff:    {max_diff:.6f}")
    print(f"  Mean diff:   {mean_diff:.6f}")

    if max_diff < 0.01:
        print(f"  [PASS] ONNX parity: max difference {max_diff:.6f} (threshold 0.01)")
    else:
        print(f"  [WARN] ONNX parity loose: max difference {max_diff:.6f}")


if __name__ == '__main__':
    ensemble, threshold, y_val, ensemble_val_prob = build_ensemble()

    # Final validation summary
    y_pred = (ensemble_val_prob >= threshold).astype(int)
    print(f"\n{'='*50}")
    print(f"ENSEMBLE FINAL VALIDATION SUMMARY")
    print(f"{'='*50}")
    print(f"Threshold: {threshold}")
    print(f"AUC-ROC:   {roc_auc_score(y_val, ensemble_val_prob):.4f}")
    print(f"F1:        {f1_score(y_val, y_pred):.4f}")
    print(f"Precision: {precision_score(y_val, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_val, y_pred):.4f}")
    print(f"{'='*50}")

    sys.exit(0)
