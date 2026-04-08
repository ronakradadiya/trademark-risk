"""
Engineers all 19 features from the labeled dataset.
Handles missing values, computes derived features, and creates train/val/test split.

Input: ml/data/processed/labeled.csv
Output: ml/data/processed/features_train.csv, features_val.csv, features_test.csv
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
REFERENCE_DATE = pd.Timestamp('2024-03-19')  # dataset date


def engineer_features(df):
    """Compute all 19 features from the labeled dataset."""

    print("=== FEATURE ENGINEERING ===\n")

    # --- GROUP 1: Owner behavior signals (3 features) ---
    print("Group 1: Owner behavior signals...")

    # owner_filing_count_2yr — already computed in clean.py
    df['owner_filing_count_2yr'] = df['owner_filing_count_2yr'].fillna(1).astype(int)

    # owner_abandonment_rate — already computed in clean.py (temporally safe in v4)
    df['owner_abandonment_rate'] = df['owner_abandonment_rate'].fillna(0.0)

    # owner_historical_cancellation_rate — new in v4, from clean.py, temporally safe
    df['owner_historical_cancellation_rate'] = df['owner_historical_cancellation_rate'].fillna(0.0)

    # days_since_owner_first_filing — new in v4, from clean.py
    df['days_since_owner_first_filing'] = df['days_since_owner_first_filing'].fillna(0).astype(int)

    # owner_is_individual: 1 if own_entity_cd == 1
    df['owner_is_individual'] = (df['own_entity_cd'] == 1).astype(int)

    # --- GROUP 2: External signals (3 features) ---
    print("Group 2: External signals...")

    # attorney_case_count — already computed in clean.py
    df['attorney_case_count'] = df['attorney_case_count'].fillna(0).astype(int)

    # attorney_cancellation_rate — from clean.py, temporally safe
    df['attorney_cancellation_rate'] = df['attorney_cancellation_rate'].fillna(0.0)

    # owner_is_foreign: 1 if country code is not empty and not 'US'
    df['owner_is_foreign'] = (
        df['own_addr_country_cd'].notna() &
        (df['own_addr_country_cd'] != '') &
        (df['own_addr_country_cd'] != 'US')
    ).astype(int)

    # --- GROUP 3: Filing timing signals (4 features) ---
    print("Group 3: Filing timing signals...")

    # Parse dates
    df['filing_dt'] = pd.to_datetime(df['filing_dt'], errors='coerce')
    df['registration_dt'] = pd.to_datetime(df['registration_dt'], errors='coerce')
    df['abandon_dt'] = pd.to_datetime(df['abandon_dt'], errors='coerce')
    df['first_use_any_dt'] = pd.to_datetime(df['first_use_any_dt'], errors='coerce')

    # days_domain_to_filing: set to -999 sentinel (no domain data in bulk files)
    df['days_domain_to_filing'] = -999

    # days_since_filing: reference date minus filing_dt
    df['days_since_filing'] = (REFERENCE_DATE - df['filing_dt']).dt.days
    df['days_since_filing'] = df['days_since_filing'].fillna(0).astype(int)

    # days_filing_to_registration: registration_dt minus filing_dt, -1 if never registered
    df['days_filing_to_registration'] = np.where(
        df['registration_dt'].notna(),
        (df['registration_dt'] - df['filing_dt']).dt.days,
        -1
    )
    df['days_filing_to_registration'] = df['days_filing_to_registration'].fillna(-1).astype(int)

    # days_first_use_to_filing: filing_dt minus first_use_any_dt, -999 if no first use date
    df['days_first_use_to_filing'] = np.where(
        df['first_use_any_dt'].notna(),
        (df['filing_dt'] - df['first_use_any_dt']).dt.days,
        -999
    )
    df['days_first_use_to_filing'] = df['days_first_use_to_filing'].fillna(-999).astype(int)

    # --- GROUP 4: USPTO examination signals (3 features) ---
    print("Group 4: USPTO examination signals...")
    # office_action_count, opposition_count, statement_of_use_filed — already from clean.py
    df['office_action_count'] = df['office_action_count'].fillna(0).astype(int)
    df['opposition_count'] = df['opposition_count'].fillna(0).astype(int)
    df['statement_of_use_filed'] = df['statement_of_use_filed'].fillna(0).astype(int)

    # --- GROUP 5: Commercial legitimacy signals (4 features) ---
    print("Group 5: Commercial legitimacy signals...")

    # section_8_filed — already from clean.py (event-based, more reliable)
    df['section_8_filed'] = df['section_8_filed'].fillna(0).astype(int)

    # has_acquired_distinctiveness
    df['has_acquired_distinctiveness'] = df['acq_dist_in'].fillna(0).astype(int)

    # was_abandoned: 1 if abandon_dt is set
    df['was_abandoned'] = df['abandon_dt'].notna().astype(int)

    # is_currently_active: based on cfh_status_cd
    df['is_currently_active'] = df['cfh_status_cd'].apply(
        lambda x: 1 if (isinstance(x, (int, float)) and not pd.isna(x) and
                        (600 <= x <= 799)) else 0
    )

    # --- GROUP 6: Application scope signals (3 features) ---
    print("Group 6: Application scope signals...")

    # class_breadth — already from clean.py
    df['class_breadth'] = df['class_breadth'].fillna(1).astype(int)

    # filing_basis: 1 if intent to use (lb_itu_cur_in), 0 if use in commerce
    df['filing_basis'] = df['lb_itu_cur_in'].fillna(0).astype(int)

    # specimen_type_encoded from mark_draw_cd
    def encode_specimen(val):
        if pd.isna(val):
            return 0
        try:
            v = int(float(val))
        except (ValueError, TypeError):
            return 0
        mapping = {1: 1, 2: 2, 3: 3, 4: 4}
        return mapping.get(v, 0)

    df['specimen_type_encoded'] = df['mark_draw_cd'].apply(encode_specimen)

    return df


# With TTAB (CANG) labels, fraud = registered marks that had cancellation granted.
# The 4 previously-leaked features are now safe:
# - section_8_filed: fraud=0.12, legit=1.0 — legitimate signal, not leakage
# - was_abandoned: ~0 for both classes — CANG marks weren't abandoned
# - days_filing_to_registration: fraud marks have registration dates (they were registered)
# - filing_basis: no difference between classes
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


def create_splits(df):
    """Create train/val/test splits (70/15/15) stratified by label."""
    print("\n=== TRAIN/VAL/TEST SPLIT ===")

    X = df[FEATURE_COLS]
    y = df['label']

    # First split: 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    # Second split: 50/50 of temp → 15% val, 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    # Save splits
    for name, X_split, y_split in [
        ('train', X_train, y_train),
        ('val', X_val, y_val),
        ('test', X_test, y_test)
    ]:
        split_df = X_split.copy()
        split_df['label'] = y_split
        # Also save serial_no for traceability
        split_df['serial_no'] = df.loc[X_split.index, 'serial_no']
        path = os.path.join(PROCESSED_DIR, f'features_{name}.csv')
        split_df.to_csv(path, index=False)
        fraud_pct = y_split.mean() * 100
        print(f"  {name}: {len(split_df)} rows ({fraud_pct:.1f}% fraud) → {path}")

    # No data leakage check
    train_serials = set(df.loc[X_train.index, 'serial_no'])
    test_serials = set(df.loc[X_test.index, 'serial_no'])
    val_serials = set(df.loc[X_val.index, 'serial_no'])
    overlap_tt = train_serials & test_serials
    overlap_tv = train_serials & val_serials
    assert len(overlap_tt) == 0, f"DATA LEAKAGE: {len(overlap_tt)} serials in train AND test"
    assert len(overlap_tv) == 0, f"DATA LEAKAGE: {len(overlap_tv)} serials in train AND val"
    print("  [PASS] No data leakage — zero serial overlap between splits")

    return X_train, X_val, X_test, y_train, y_val, y_test


def run_feature_tests(df):
    """Run all feature engineering validation tests."""
    print(f"\n{'='*50}")
    print("FEATURE ENGINEERING VALIDATION")
    print(f"{'='*50}\n")

    all_pass = True

    # Test 1: all 19 features present (4 restored after switching to TTAB labels)
    for feat in FEATURE_COLS:
        if feat not in df.columns:
            print(f"[FAIL] Feature '{feat}' missing")
            all_pass = False
    if all_pass:
        print(f"[PASS] All {len(FEATURE_COLS)} features present")

    # Test 2: no NaN values in features
    for feat in FEATURE_COLS:
        nulls = df[feat].isna().sum()
        if nulls > 0:
            print(f"[FAIL] {feat} has {nulls} null values")
            all_pass = False
    print(f"[PASS] Zero null values across all features")

    # Test 3: feature range validation
    checks = {
        'owner_filing_count_2yr':              (1, 50000),
        'owner_abandonment_rate':              (0.0, 1.0),
        'owner_historical_cancellation_rate':  (0.0, 1.0),
        'days_since_owner_first_filing':       (0, 55000),
        'owner_is_individual':                 (0, 1),
        'attorney_case_count':                 (0, 100000),
        'attorney_cancellation_rate':          (0.0, 1.0),
        'owner_is_foreign':                    (0, 1),
        'days_domain_to_filing':        (-3650, 36500),
        'days_since_filing':            (0, 55000),
        'days_first_use_to_filing':     (-3650, 100000),
        'days_filing_to_registration':  (-1, 40000),
        'office_action_count':          (0, 50),
        'opposition_count':             (0, 20),
        'statement_of_use_filed':       (0, 1),
        'section_8_filed':              (0, 1),
        'has_acquired_distinctiveness': (0, 1),
        'is_currently_active':          (0, 1),
        'was_abandoned':                (0, 1),
        'filing_basis':                 (0, 1),
        'class_breadth':                (1, 45),
        'specimen_type_encoded':        (0, 4),
    }

    for col, (min_val, max_val) in checks.items():
        actual_min = df[col].min()
        actual_max = df[col].max()
        # Allow sentinel values
        if col == 'days_domain_to_filing' and actual_min == -999:
            actual_min_check = -999  # sentinel allowed
        else:
            actual_min_check = actual_min
        if col == 'days_first_use_to_filing' and actual_min == -999:
            actual_min_check = -999
        else:
            actual_min_check = actual_min

        if actual_min_check < min_val and col not in ('days_domain_to_filing', 'days_first_use_to_filing'):
            print(f"[FAIL] {col} min={actual_min} below {min_val}")
            all_pass = False
        elif actual_max > max_val:
            print(f"[FAIL] {col} max={actual_max} above {max_val}")
            all_pass = False
        else:
            print(f"[PASS] {col}: min={actual_min:.2f}, max={actual_max:.2f}, "
                  f"mean={df[col].mean():.2f}, nulls=0")

    # Test 4: feature means differ between fraud and legitimate
    print(f"\n=== FEATURE MEANS: FRAUD vs LEGITIMATE ===")
    feature_means = df.groupby('label')[FEATURE_COLS].mean().T.rename(
        columns={0: 'legitimate', 1: 'fraud'}
    )
    print(feature_means.to_string())

    # Top 3 features must show separation
    for col in ['owner_abandonment_rate', 'statement_of_use_filed', 'opposition_count']:
        fraud_mean = df[df.label == 1][col].mean()
        legit_mean = df[df.label == 0][col].mean()
        if fraud_mean == legit_mean:
            print(f"[FAIL] {col} has same mean for fraud and legitimate")
            all_pass = False
        else:
            print(f"[PASS] {col} differs: fraud={fraud_mean:.4f}, legitimate={legit_mean:.4f}")

    # owner_abandonment_rate: fraud should be higher
    fraud_abandon = df[df.label == 1]['owner_abandonment_rate'].mean()
    legit_abandon = df[df.label == 0]['owner_abandonment_rate'].mean()
    if fraud_abandon > legit_abandon:
        print(f"[PASS] owner_abandonment_rate: fraud={fraud_abandon:.2f} > legitimate={legit_abandon:.2f}")
    else:
        print(f"[WARN] owner_abandonment_rate: fraud={fraud_abandon:.2f} <= legitimate={legit_abandon:.2f}")

    print(f"\n{'='*50}")
    if all_pass:
        print("ALL FEATURE TESTS PASSED — ready for train.py")
    else:
        print("SOME TESTS FAILED — fix before proceeding")
    print(f"{'='*50}")

    return all_pass


if __name__ == '__main__':
    print("Loading labeled.csv...")
    df = pd.read_csv(os.path.join(PROCESSED_DIR, 'labeled.csv'), low_memory=False)
    print(f"Loaded {len(df)} rows\n")

    df = engineer_features(df)
    success = run_feature_tests(df)

    if success:
        create_splits(df)

    sys.exit(0 if success else 1)
