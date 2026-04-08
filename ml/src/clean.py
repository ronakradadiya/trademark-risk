"""
Cleans and joins USPTO datasets, creates fraud/legitimate labels.

Labeling strategy (v2 — TTAB-based):
  - Legitimate (label=0): Marks with registration_dt set AND section 8 affidavit
    events in event.csv — these proved continued commercial use over 5+ years.
    Excludes any marks that appear in the fraud set.
  - Fraud (label=1): Marks with CANG (cancellation granted) events in event.csv
    OR cancellation proceedings in TTAB XML data — these had a TTAB petition
    filed against them and the cancellation was GRANTED. Much stronger signal
    than the previous proxy (abandoned = fraud).

Output: ml/data/processed/labeled.csv
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime

RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
TTAB_DIR = os.path.join(RAW_DIR, 'ttab')
os.makedirs(PROCESSED_DIR, exist_ok=True)

RANDOM_SEED = 42
TARGET_LEGIT = 60000
TARGET_FRAUD = 20000


def get_section8_serials():
    """Scan event.csv in chunks to find serial numbers with section 8 events."""
    sec8_codes = ['C8..', 'C8.T', 'C8P.', '8.OK', '8.AF', '8.PR', '8AFT', '8OKT', '8PRT']
    sec8_serials = set()

    print("Scanning event.csv for section 8 events (chunked)...")
    for i, chunk in enumerate(pd.read_csv(
        os.path.join(RAW_DIR, 'event.csv'), chunksize=5_000_000
    )):
        s8 = chunk[chunk.event_cd.isin(sec8_codes)]
        sec8_serials.update(s8.serial_no.unique())
        if (i + 1) % 8 == 0:
            print(f"  Processed {(i+1)*5}M rows... section 8 serials: {len(sec8_serials)}")

    print(f"  Total section 8 serials: {len(sec8_serials)}")
    return sec8_serials


def get_event_features():
    """Scan event.csv in chunks to extract per-serial event counts."""
    # Actual event codes from the USPTO data
    oa_codes = ['NOAM']  # Non-final office action mailed
    opp_codes = ['OP.I', 'OP.D', 'OP.S', 'OP.T', 'OPPF']  # Opposition events
    sou_codes = ['SUPC']  # Statement of use processed/accepted

    oa_counts = {}
    opp_counts = {}
    sou_filed = {}

    print("Scanning event.csv for office actions, oppositions, SOU (chunked)...")
    for i, chunk in enumerate(pd.read_csv(
        os.path.join(RAW_DIR, 'event.csv'), chunksize=5_000_000
    )):
        # Office actions
        oa = chunk[chunk.event_cd.isin(oa_codes)]
        for serial, count in oa.groupby('serial_no').size().items():
            oa_counts[serial] = oa_counts.get(serial, 0) + count

        # Oppositions
        opp = chunk[chunk.event_cd.isin(opp_codes)]
        for serial, count in opp.groupby('serial_no').size().items():
            opp_counts[serial] = opp_counts.get(serial, 0) + count

        # Statement of use
        sou = chunk[chunk.event_cd.isin(sou_codes)]
        for serial in sou.serial_no.unique():
            sou_filed[serial] = 1

        if (i + 1) % 8 == 0:
            print(f"  Processed {(i+1)*5}M rows... OA: {len(oa_counts)}, "
                  f"Opp: {len(opp_counts)}, SOU: {len(sou_filed)}")

    print(f"  Final: OA serials: {len(oa_counts)}, Opposition serials: {len(opp_counts)}, "
          f"SOU serials: {len(sou_filed)}")
    return oa_counts, opp_counts, sou_filed


def build_labeled_dataset():
    """Build the labeled dataset by joining case_file with labels."""
    print("\n=== STEP 1: Load case_file.csv ===")
    cf = pd.read_csv(
        os.path.join(RAW_DIR, 'case_file.csv'),
        usecols=[
            'serial_no', 'filing_dt', 'registration_dt', 'abandon_dt',
            'acq_dist_in', 'use_afdv_acc_in', 'cfh_status_cd', 'cfh_status_dt',
            'mark_draw_cd', 'lb_itu_cur_in', 'lb_use_cur_in', 'publication_dt',
        ],
        low_memory=False
    )
    print(f"  Loaded {len(cf)} rows from case_file.csv")

    print("\n=== STEP 2: Get section 8 serials from event.csv ===")
    sec8_serials = get_section8_serials()

    print("\n=== STEP 3: Load TTAB fraud serials ===")
    # Load combined CANG + TTAB XML cancellation serials
    ttab_path = os.path.join(TTAB_DIR, 'combined_fraud_serials.json')
    with open(ttab_path) as f:
        ttab_data = json.load(f)
    fraud_serials = set(ttab_data['serials'])
    print(f"  Loaded {len(fraud_serials)} TTAB fraud serials (CANG events + TTAB XML)")

    print("\n=== STEP 4: Create labels ===")
    # Fraud: marks with CANG (cancellation granted) — TTAB ruled against the mark owner
    is_fraud = cf.serial_no.isin(fraud_serials)
    fraud_df = cf[is_fraud].copy()
    print(f"  Fraud candidates (TTAB cancellation granted): {len(fraud_df)}")

    # Legitimate: any registered mark NOT in the TTAB fraud set.
    # Do NOT require section_8_filed here — that would make section_8_filed a
    # definition of the label rather than a feature, causing circular leakage
    # (ablation showed F1 collapses 0.94 → 0.45 when section_8_filed is removed).
    has_reg = cf.registration_dt.notna()
    legit_df = cf[has_reg & ~is_fraud].copy()
    print(f"  Legitimate candidates (registered, no TTAB cancellation): {len(legit_df)}")

    # Sample to target sizes
    np.random.seed(RANDOM_SEED)
    if len(legit_df) > TARGET_LEGIT:
        legit_df = legit_df.sample(n=TARGET_LEGIT, random_state=RANDOM_SEED)
    if len(fraud_df) > TARGET_FRAUD:
        fraud_df = fraud_df.sample(n=TARGET_FRAUD, random_state=RANDOM_SEED)

    legit_df['label'] = 0
    fraud_df['label'] = 1

    df = pd.concat([legit_df, fraud_df], ignore_index=True)
    print(f"  Combined dataset: {len(df)} rows "
          f"({len(legit_df)} legit, {len(fraud_df)} fraud)")

    print("\n=== STEP 5: Parse date columns ===")
    date_cols = ['filing_dt', 'registration_dt', 'abandon_dt', 'cfh_status_dt', 'publication_dt']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    print("  Date columns parsed")

    print("\n=== STEP 6: Get event-based features ===")
    oa_counts, opp_counts, sou_filed = get_event_features()

    df['office_action_count'] = df.serial_no.map(oa_counts).fillna(0).astype(int)
    df['opposition_count'] = df.serial_no.map(opp_counts).fillna(0).astype(int)
    df['statement_of_use_filed'] = df.serial_no.map(sou_filed).fillna(0).astype(int)

    # Also store section 8 from events (more reliable than use_afdv_acc_in column)
    df['section_8_filed'] = df.serial_no.isin(sec8_serials).astype(int)

    print(f"  Event features mapped to {len(df)} rows")

    print("\n=== STEP 7: Join owner.csv ===")
    owner = pd.read_csv(
        os.path.join(RAW_DIR, 'owner.csv'),
        usecols=['serial_no', 'own_id', 'own_name', 'own_entity_cd',
                 'own_type_cd', 'own_addr_country_cd']
    )
    # Keep first owner per serial (primary owner)
    owner_dedup = owner.sort_values('own_type_cd').drop_duplicates(subset='serial_no', keep='first')
    df = df.merge(owner_dedup[['serial_no', 'own_id', 'own_name', 'own_entity_cd',
                                'own_addr_country_cd']],
                  on='serial_no', how='left')
    print(f"  Joined owner data. Rows: {len(df)}")

    print("\n=== STEP 8: Join classification.csv ===")
    classification = pd.read_csv(
        os.path.join(RAW_DIR, 'classification.csv'),
        usecols=['serial_no', 'class_primary_cd', 'first_use_any_dt', 'first_use_com_dt']
    )
    # Class breadth = count of distinct classes per serial
    class_breadth = classification.groupby('serial_no')['class_primary_cd'].nunique().reset_index()
    class_breadth.columns = ['serial_no', 'class_breadth']

    # First use date — take the earliest per serial
    classification['first_use_any_dt'] = pd.to_datetime(
        classification['first_use_any_dt'], errors='coerce'
    )
    first_use = classification.groupby('serial_no')['first_use_any_dt'].min().reset_index()

    df = df.merge(class_breadth, on='serial_no', how='left')
    df = df.merge(first_use, on='serial_no', how='left')
    df['class_breadth'] = df['class_breadth'].fillna(1).astype(int)
    print(f"  Joined classification data. Rows: {len(df)}")

    print("\n=== STEP 9: Join attorney data ===")
    attorney = pd.read_csv(
        os.path.join(RAW_DIR, 'correspondent_domrep_attorney.csv'),
        usecols=['serial_no', 'attorney_no', 'attorney_name']
    )
    df = df.merge(attorney[['serial_no', 'attorney_no']], on='serial_no', how='left')
    print(f"  Joined attorney data. Rows: {len(df)}")

    print("\n=== STEP 10: Compute attorney case counts ===")
    # Count total cases per attorney across the full attorney file
    atty_counts = attorney.groupby('attorney_no').size().reset_index(name='attorney_case_count')
    df = df.merge(atty_counts, on='attorney_no', how='left')
    df['attorney_case_count'] = df['attorney_case_count'].fillna(0).astype(int)
    print(f"  Attorney case counts computed")

    print("\n=== STEP 11: Compute owner+attorney features (temporally aware) ===")
    # All aggregations below use only prior filings (filing_dt < current row's
    # filing_dt) to avoid temporal leakage. The previous version used
    # whole-history aggregates which included filings AFTER each row's filing
    # date, leaking the future into the past.
    import bisect

    cf_full = pd.read_csv(
        os.path.join(RAW_DIR, 'case_file.csv'),
        usecols=['serial_no', 'filing_dt', 'abandon_dt'],
        low_memory=False
    )
    cf_full['filing_dt'] = pd.to_datetime(cf_full['filing_dt'], errors='coerce')
    cf_full['abandon_dt'] = pd.to_datetime(cf_full['abandon_dt'], errors='coerce')
    cf_full['is_fraud'] = cf_full['serial_no'].isin(fraud_serials)
    cf_full = cf_full.merge(
        owner_dedup[['serial_no', 'own_name']], on='serial_no', how='left'
    )

    # Also merge attorney_no so we can build attorney-level history
    cf_full = cf_full.merge(
        attorney[['serial_no', 'attorney_no']], on='serial_no', how='left'
    )

    print("  Building per-owner sorted filing index...")
    owner_records = {}
    cf_o = cf_full.dropna(subset=['own_name', 'filing_dt']).sort_values('filing_dt')
    for oname, grp in cf_o.groupby('own_name'):
        owner_records[oname] = {
            'filing_dts': grp['filing_dt'].tolist(),
            'abandon_dts': grp['abandon_dt'].tolist(),
            'is_fraud': grp['is_fraud'].tolist(),
        }

    print("  Building per-attorney sorted filing index...")
    attorney_records = {}
    cf_a = cf_full.dropna(subset=['attorney_no', 'filing_dt']).sort_values('filing_dt')
    for aid, grp in cf_a.groupby('attorney_no'):
        attorney_records[aid] = {
            'filing_dts': grp['filing_dt'].tolist(),
            'is_fraud': grp['is_fraud'].tolist(),
        }

    print("  Computing per-row temporal features "
          "(abandonment rate, cancellation rate, 2yr count, days since first, attorney rate)...")

    abandon_rates = []
    cancel_rates = []
    counts_2yr = []
    days_since_first = []
    attorney_cancel_rates = []

    MIN_PRIOR = 3  # need at least 3 prior filings for a reliable rate

    for _, row in df.iterrows():
        oname = row.get('own_name')
        fdt = row.get('filing_dt')
        aid = row.get('attorney_no')

        # --- Owner-level features ---
        if pd.isna(oname) or pd.isna(fdt) or oname not in owner_records:
            abandon_rates.append(0.0)
            cancel_rates.append(0.0)
            counts_2yr.append(1)
            days_since_first.append(0)
        else:
            od = owner_records[oname]
            # Strictly prior filings: filing_dt < current fdt
            prior_idx = bisect.bisect_left(od['filing_dts'], fdt)
            n_prior = prior_idx

            if n_prior >= MIN_PRIOR:
                prior_ab = od['abandon_dts'][:prior_idx]
                abandoned = sum(1 for d in prior_ab if pd.notna(d) and d < fdt)
                abandon_rates.append(abandoned / n_prior)
                cancelled = sum(od['is_fraud'][:prior_idx])
                cancel_rates.append(cancelled / n_prior)
            else:
                abandon_rates.append(0.0)
                cancel_rates.append(0.0)

            # 2yr rolling count: filings in [fdt - 730d, fdt] inclusive
            two_years_ago = fdt - pd.Timedelta(days=730)
            lo = bisect.bisect_left(od['filing_dts'], two_years_ago)
            hi = bisect.bisect_right(od['filing_dts'], fdt)
            counts_2yr.append(max(hi - lo, 1))

            # Days since owner's first filing
            first_dt = od['filing_dts'][0]
            days_since_first.append(max((fdt - first_dt).days, 0))

        # --- Attorney-level feature ---
        if pd.isna(aid) or pd.isna(fdt) or aid not in attorney_records:
            attorney_cancel_rates.append(0.0)
        else:
            ar = attorney_records[aid]
            prior_idx = bisect.bisect_left(ar['filing_dts'], fdt)
            n_prior = prior_idx
            if n_prior >= MIN_PRIOR:
                cancelled = sum(ar['is_fraud'][:prior_idx])
                attorney_cancel_rates.append(cancelled / n_prior)
            else:
                attorney_cancel_rates.append(0.0)

    df['owner_abandonment_rate'] = abandon_rates
    df['owner_historical_cancellation_rate'] = cancel_rates
    df['owner_filing_count_2yr'] = counts_2yr
    df['days_since_owner_first_filing'] = days_since_first
    df['attorney_cancellation_rate'] = attorney_cancel_rates
    print(f"  Owner + attorney features computed (temporally safe)")

    print("\n=== STEP 12: Save labeled dataset ===")
    output_path = os.path.join(PROCESSED_DIR, 'labeled.csv')
    df.to_csv(output_path, index=False)
    print(f"  Saved to {output_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    return df


def run_sanity_checks(df):
    """Critical Moment 1 — sanity checks on labeled data."""
    print(f"\n{'='*50}")
    print("CRITICAL MOMENT 1 — DATA CLEANING VALIDATION")
    print(f"{'='*50}")

    all_pass = True

    # Label sanity check
    total = len(df)
    fraud = df.label.sum()
    legit = (df.label == 0).sum()
    fraud_rate = df.label.mean()
    print(f"\nTotal rows: {total}")
    print(f"Fraud (label=1): {fraud} ({fraud_rate*100:.1f}%)")
    print(f"Legitimate (label=0): {legit} ({(1-fraud_rate)*100:.1f}%)")

    if not (0.10 <= fraud_rate <= 0.45):
        print(f"[FAIL] Label ratio {fraud_rate:.2f} outside expected range 0.10-0.45")
        all_pass = False
    else:
        print(f"[PASS] Label ratio {fraud_rate:.2f} within expected range")

    if total < 10000:
        print(f"[FAIL] Dataset too small: {total} rows (need >= 10000)")
        all_pass = False
    else:
        print(f"[PASS] Dataset size: {total} rows")

    # No duplicate serial numbers
    n_unique = df.serial_no.nunique()
    if n_unique != total:
        print(f"[FAIL] {total - n_unique} duplicate serial numbers")
        all_pass = False
    else:
        print(f"[PASS] No duplicate serial numbers")

    # Label column only has 0 and 1
    unique_labels = sorted(df.label.unique())
    if unique_labels != [0, 1]:
        print(f"[FAIL] Unexpected label values: {unique_labels}")
        all_pass = False
    else:
        print(f"[PASS] Labels are 0 and 1 only")

    # No null labels
    null_labels = df.label.isna().sum()
    if null_labels > 0:
        print(f"[FAIL] {null_labels} null labels")
        all_pass = False
    else:
        print(f"[PASS] Zero null labels")

    # Date column is datetime
    if pd.api.types.is_datetime64_any_dtype(df.filing_dt):
        print(f"[PASS] filing_dt is datetime type")
    else:
        print(f"[FAIL] filing_dt is not datetime type")
        all_pass = False

    # Manual spot check
    print(f"\n=== 10 RANDOM FRAUD CASES ===")
    fraud_sample = df[df.label == 1].sample(min(10, fraud), random_state=42)
    print(fraud_sample[['serial_no', 'own_name', 'filing_dt', 'class_breadth',
                         'owner_filing_count_2yr', 'owner_abandonment_rate']].to_string())

    print(f"\n=== 10 RANDOM LEGITIMATE CASES ===")
    legit_sample = df[df.label == 0].sample(min(10, legit), random_state=42)
    print(legit_sample[['serial_no', 'own_name', 'filing_dt', 'section_8_filed',
                          'class_breadth']].to_string())

    print(f"\n{'='*50}")
    if all_pass:
        print("ALL SANITY CHECKS PASSED — ready for features.py")
    else:
        print("SOME CHECKS FAILED — fix before proceeding")
    print(f"{'='*50}")

    return all_pass


if __name__ == '__main__':
    df = build_labeled_dataset()
    success = run_sanity_checks(df)
    sys.exit(0 if success else 1)
