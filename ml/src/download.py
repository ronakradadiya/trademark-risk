"""
Verifies that all required USPTO dataset files are present and valid.
The raw data should already be downloaded from:
https://www.uspto.gov/ip-policy/economic-research/research-datasets/trademark-case-files-dataset
"""

import os
import sys
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')

EXPECTED_FILES = {
    'case_file.csv': {
        'min_size_mb': 100,
        'required_cols': ['serial_no', 'filing_dt', 'registration_dt', 'abandon_dt',
                          'acq_dist_in', 'use_afdv_acc_in', 'opposit_pend_in',
                          'cfh_status_cd', 'cfh_status_dt', 'mark_draw_cd',
                          'lb_itu_cur_in', 'lb_use_cur_in', 'renewal_file_in',
                          'publication_dt', 'registration_no'],
    },
    'owner.csv': {
        'min_size_mb': 100,
        'required_cols': ['serial_no', 'own_id', 'own_name', 'own_entity_cd',
                          'own_type_cd', 'own_addr_country_cd', 'own_addr_state_cd'],
    },
    'classification.csv': {
        'min_size_mb': 50,
        'required_cols': ['serial_no', 'class_primary_cd', 'class_intl_count',
                          'class_status_cd', 'first_use_any_dt', 'first_use_com_dt', 'class_id'],
    },
    'correspondent_domrep_attorney.csv': {
        'min_size_mb': 50,
        'required_cols': ['serial_no', 'attorney_no', 'attorney_name', 'domestic_rep_name'],
    },
    'event.csv': {
        'min_size_mb': 500,
        'required_cols': ['serial_no', 'event_cd', 'event_dt', 'event_seq', 'event_type_cd'],
    },
    'intl_class.csv': {
        'min_size_mb': 10,
        'required_cols': ['serial_no', 'intl_class_cd', 'class_id'],
    },
    'prior_mark.csv': {
        'min_size_mb': 5,
        'required_cols': ['serial_no', 'prior_no', 'prior_type_cd', 'rec_error'],
    },
}


def verify_data():
    results = []
    all_pass = True

    for filename, spec in EXPECTED_FILES.items():
        filepath = os.path.join(RAW_DIR, filename)

        # Check file exists
        if not os.path.exists(filepath):
            print(f"[FAIL] {filename} — file not found at {filepath}")
            all_pass = False
            continue

        # Check file size
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        if size_mb < spec['min_size_mb']:
            print(f"[FAIL] {filename} — size {size_mb:.0f}MB below minimum {spec['min_size_mb']}MB")
            all_pass = False
            continue
        print(f"[PASS] {filename} — exists, {size_mb:.0f}MB")

        # Check columns by reading header only
        df_head = pd.read_csv(filepath, nrows=5)
        missing_cols = [c for c in spec['required_cols'] if c not in df_head.columns]
        if missing_cols:
            print(f"[FAIL] {filename} — missing columns: {missing_cols}")
            all_pass = False
        else:
            print(f"[PASS] {filename} — all required columns present")

        # Check can read first 100 rows
        df_sample = pd.read_csv(filepath, nrows=100)
        print(f"[PASS] {filename} — loaded {len(df_sample)} sample rows, {len(df_sample.columns)} columns")

        results.append({
            'file': filename,
            'size_mb': size_mb,
            'columns': len(df_head.columns),
            'sample_rows': len(df_sample),
        })

    print(f"\n{'='*50}")
    if all_pass:
        print("ALL FILES VERIFIED — ready for clean.py")
    else:
        print("SOME FILES MISSING OR INVALID — fix before proceeding")
    print(f"{'='*50}")

    return all_pass


if __name__ == '__main__':
    success = verify_data()
    sys.exit(0 if success else 1)
