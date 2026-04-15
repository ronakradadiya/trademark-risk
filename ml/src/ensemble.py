"""
Tunes the decision threshold for the XGBoost classifier and exports to ONNX.

The v4 model was originally a stacking ensemble (XGBoost + CatBoost → LR
meta-learner). The stacking step added ~0.001 AUC over XGBoost alone — not
worth the extra inference complexity for two tree models. This script is the
cleaned-up single-model replacement.

Input:  ml/models/xgboost.joblib, ml/data/processed/features_{val,test}.csv
Output: ml/models/xgboost.onnx (zipmap stripped for onnxruntime-node)
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
)
import onnxruntime as ort

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

FEATURE_COLS = [
    'owner_filing_count_2yr', 'owner_abandonment_rate',
    'owner_historical_cancellation_rate', 'days_since_owner_first_filing',
    'owner_is_individual', 'owner_is_foreign',
    'attorney_case_count', 'attorney_cancellation_rate',
    'days_since_filing', 'days_filing_to_registration',
    'was_abandoned', 'is_currently_active', 'class_breadth',
]


def load_data():
    val = pd.read_csv(os.path.join(PROCESSED_DIR, 'features_val.csv'))
    test = pd.read_csv(os.path.join(PROCESSED_DIR, 'features_test.csv'))
    return val, test


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

    if best_f1 == 0:
        print("\n[WARN] No threshold achieves recall >= 0.85; picking best F1 overall")
        for threshold in [i / 20 for i in range(4, 17)]:
            y_pred = (y_prob >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

    print(f"\nChosen threshold: {best_threshold} (F1={best_f1:.4f})")
    return best_threshold


def export_onnx(xgb_model):
    """Export XGBoost to ONNX with ZipMap disabled.

    onnxruntime-node (used by the TypeScript serving layer) does not support
    ZipMap / sequence-of-map outputs. skl2onnx via onnxmltools can emit a
    plain [N, 2] probability tensor when we set zipmap=False.
    """
    from skl2onnx import convert_sklearn, update_registered_converter
    from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
    from skl2onnx.common.data_types import FloatTensorType
    from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost as xgb_converter
    from xgboost import XGBClassifier

    n_features = len(FEATURE_COLS)
    target_opset = {'': 12, 'ai.onnx.ml': 3}

    update_registered_converter(
        XGBClassifier, 'XGBoostXGBClassifier',
        calculate_linear_classifier_output_shapes,
        xgb_converter,
        options={'nocl': [True, False], 'zipmap': [True, False, 'columns']},
    )

    xgb_onnx = convert_sklearn(
        xgb_model,
        initial_types=[('input', FloatTensorType([None, n_features]))],
        target_opset=target_opset,
        options={id(xgb_model): {'zipmap': False}},
    )
    xgb_onnx_path = os.path.join(MODELS_DIR, 'xgboost.onnx')
    with open(xgb_onnx_path, 'wb') as f:
        f.write(xgb_onnx.SerializeToString())
    print(f"  XGBoost ONNX saved to {xgb_onnx_path} (zipmap disabled)")


def validate_onnx_parity(xgb_model, X_sample):
    """Verify ONNX inference matches joblib inference within tolerance."""
    joblib_probs = xgb_model.predict_proba(X_sample)[:, 1]

    session = ort.InferenceSession(os.path.join(MODELS_DIR, 'xgboost.onnx'))
    input_name = session.get_inputs()[0].name
    outs = session.run(None, {input_name: X_sample.astype(np.float32)})

    onnx_probs = None
    for t in outs:
        if hasattr(t, 'shape') and len(t.shape) == 2 and t.shape[1] == 2:
            onnx_probs = t[:, 1]
            break
    if onnx_probs is None:
        raise RuntimeError(f'no [N,2] tensor in outputs: {[type(o).__name__ for o in outs]}')

    max_diff = np.abs(joblib_probs - onnx_probs).max()
    mean_diff = np.abs(joblib_probs - onnx_probs).mean()
    print(f"  Max diff:  {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")
    if max_diff < 0.01:
        print(f"  [PASS] ONNX parity: max difference {max_diff:.6f}")
    else:
        print(f"  [WARN] ONNX parity loose: max difference {max_diff:.6f}")


def build():
    val, test = load_data()
    X_val = val[FEATURE_COLS].values
    y_val = val['label'].values

    xgb = joblib.load(os.path.join(MODELS_DIR, 'xgboost.joblib'))

    print("=== STEP 1: XGBoost validation probabilities ===")
    val_prob = xgb.predict_proba(X_val)[:, 1]
    print(f"  Val AUC: {roc_auc_score(y_val, val_prob):.4f}")

    print("\n=== STEP 2: Threshold tuning ===")
    threshold = tune_threshold(y_val, val_prob)

    print("\n=== STEP 3: Export to ONNX ===")
    export_onnx(xgb)

    print("\n=== STEP 4: Validate ONNX parity ===")
    validate_onnx_parity(xgb, X_val[:50])

    return xgb, threshold, y_val, val_prob


if __name__ == '__main__':
    _, threshold, y_val, val_prob = build()
    y_pred = (val_prob >= threshold).astype(int)
    print(f"\n{'='*50}")
    print(f"FINAL VALIDATION SUMMARY")
    print(f"{'='*50}")
    print(f"Threshold: {threshold}")
    print(f"AUC-ROC:   {roc_auc_score(y_val, val_prob):.4f}")
    print(f"F1:        {f1_score(y_val, y_pred):.4f}")
    print(f"Precision: {precision_score(y_val, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_val, y_pred):.4f}")
    print(f"{'='*50}")
    sys.exit(0)
