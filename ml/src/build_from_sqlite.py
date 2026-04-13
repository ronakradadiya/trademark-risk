"""
Builds the labeled training dataset directly from data/uspto.sqlite (April 2026
TRTYRAP snapshot), replacing the old clean.py + features.py CSV-ingest path.

Why: training previously read TCFD CSVs from March 2024 while serving reads
the April 2026 SQLite. That mismatch left 14 of 22 features pinned at
constants in app/src/lib/features.ts at inference time. Single source of truth
fixes both drift and the constants gap in one shot.

Labels: TTAB fraud serials from ml/data/raw/ttab/combined_fraud_serials.json
joined against marks.serial. 30,011 input serials, ~97% join rate to SQLite.

Features (13): five mark-side + eight applicant-side. All applicant rollups
are recomputed temporally-safely (only marks with filing_date < current
mark's filing_date are aggregated) so the model does not learn
"this filer eventually became prolific" — only what was visible at filing time.

Output: ml/data/processed/{labeled,features_train,features_val,features_test}.csv
"""

from __future__ import annotations

import bisect
import json
import os
import random
import sqlite3
import sys
from datetime import date, datetime
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SQLITE_PATH = os.path.join(REPO_ROOT, 'data', 'uspto.sqlite')
TTAB_PATH = os.path.join(REPO_ROOT, 'ml', 'data', 'raw', 'ttab', 'combined_fraud_serials.json')
PROCESSED_DIR = os.path.join(REPO_ROOT, 'ml', 'data', 'processed')
os.makedirs(PROCESSED_DIR, exist_ok=True)

RANDOM_SEED = 42
TARGET_FRAUD = 16_000
TARGET_LEGIT = 48_000
FILING_DATE_MIN = '20000101'
REFERENCE_DATE = date(2026, 4, 13)
MIN_PRIOR = 3

# Column order MUST match the FEATURE_COLS list emitted to
# ml/src/features.py / train.py / ensemble.py and consumed by
# app/src/lib/features.ts. Any change here must be mirrored on both sides
# of the train/serve boundary.
FEATURE_COLS = [
    # Applicant-side (8) — temporally-safe rollups
    'owner_filing_count_2yr',
    'owner_abandonment_rate',
    'owner_historical_cancellation_rate',
    'days_since_owner_first_filing',
    'owner_is_individual',
    'owner_is_foreign',
    'attorney_case_count',
    'attorney_cancellation_rate',
    # Mark-side (5) — at training time computed from the mark's own row
    'days_since_filing',
    'days_filing_to_registration',
    'was_abandoned',
    'is_currently_active',
    'class_breadth',
]


def parse_yyyymmdd(s: str | None) -> date | None:
    if not s or len(s) != 8:
        return None
    try:
        return date(int(s[0:4]), int(s[4:6]), int(s[6:8]))
    except ValueError:
        return None


def days_between(a: date | None, b: date | None) -> int | None:
    if a is None or b is None:
        return None
    return (a - b).days


def open_db() -> sqlite3.Connection:
    if not os.path.exists(SQLITE_PATH):
        raise FileNotFoundError(f'SQLite database not found: {SQLITE_PATH}')
    db = sqlite3.connect(f'file:{SQLITE_PATH}?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    return db


def load_fraud_serials() -> set[str]:
    with open(TTAB_PATH) as f:
        ttab = json.load(f)
    serials = {str(s) for s in ttab['serials']}
    print(f'[labels] loaded {len(serials):,} TTAB fraud serials from combined_fraud_serials.json')
    return serials


def verify_join_rate(db: sqlite3.Connection, fraud_serials: set[str]) -> int:
    """First-thing gate: stop if TTAB serials don't join cleanly to SQLite."""
    db.execute('CREATE TEMP TABLE _ttab_serials (s TEXT PRIMARY KEY)')
    db.executemany('INSERT OR IGNORE INTO _ttab_serials VALUES (?)', [(s,) for s in fraud_serials])
    n_input = db.execute('SELECT COUNT(*) FROM _ttab_serials').fetchone()[0]
    n_joined = db.execute('SELECT COUNT(*) FROM _ttab_serials t JOIN marks m ON m.serial = t.s').fetchone()[0]
    rate = n_joined / n_input if n_input else 0
    print(f'[labels] TTAB serial join rate: {n_joined:,}/{n_input:,} = {rate*100:.1f}%')
    if rate < 0.90:
        raise RuntimeError(
            f'TTAB join rate {rate:.2%} below 0.90 gate. Investigate serial format mismatch '
            f'before proceeding.'
        )
    return n_joined


def sample_marks(db: sqlite3.Connection, fraud_serials: set[str]) -> pd.DataFrame:
    """Sample fraud + legit marks from SQLite within the recent filing-year window."""
    print(f'[sample] filing year cutoff: {FILING_DATE_MIN[:4]} '
          f'(targets: {TARGET_FRAUD:,} fraud + {TARGET_LEGIT:,} legit = '
          f'{TARGET_FRAUD + TARGET_LEGIT:,} total)')

    fraud_rows = db.execute(
        """
        SELECT m.serial, m.filing_date, m.registration_date, m.abandonment_date,
               m.status_code, m.classes, m.owner_name, m.owner_name_norm,
               m.owner_country, m.owner_legal_entity, m.attorney_name
          FROM _ttab_serials t JOIN marks m ON m.serial = t.s
         WHERE m.filing_date >= ?
        """,
        (FILING_DATE_MIN,),
    ).fetchall()
    print(f'[sample] fraud candidates after year filter: {len(fraud_rows):,}')
    rng = random.Random(RANDOM_SEED)
    if len(fraud_rows) > TARGET_FRAUD:
        fraud_rows = rng.sample(fraud_rows, TARGET_FRAUD)

    # SQLite's RANDOM() is fine for unbiased sampling on millions of rows; the
    # NOT EXISTS subquery against the temp ttab table excludes positives.
    legit_rows = db.execute(
        """
        SELECT m.serial, m.filing_date, m.registration_date, m.abandonment_date,
               m.status_code, m.classes, m.owner_name, m.owner_name_norm,
               m.owner_country, m.owner_legal_entity, m.attorney_name
          FROM marks m
         WHERE m.registration_date IS NOT NULL
           AND m.filing_date >= ?
           AND NOT EXISTS (SELECT 1 FROM _ttab_serials t WHERE t.s = m.serial)
         ORDER BY RANDOM()
         LIMIT ?
        """,
        (FILING_DATE_MIN, TARGET_LEGIT),
    ).fetchall()
    print(f'[sample] legit drawn: {len(legit_rows):,}')

    rows = [dict(r) | {'label': 1} for r in fraud_rows] + [dict(r) | {'label': 0} for r in legit_rows]
    rng.shuffle(rows)
    df = pd.DataFrame(rows)
    print(f'[sample] combined shape: {df.shape}, fraud rate: {df.label.mean()*100:.1f}%')
    return df


def build_owner_history_index(db: sqlite3.Connection, owner_norms: Iterable[str]) -> dict:
    """Per-owner sorted timeline of filings, used for temporally-safe rollups.

    For each unique owner_name_norm in the sampled set, fetch every one of
    their marks (sorted by filing_date) along with abandonment / cancellation
    flags. Lookups during feature computation are O(log K) per mark via bisect.
    """
    unique = sorted({o for o in owner_norms if o})
    print(f'[history] building owner timelines for {len(unique):,} unique owners...')
    cache: dict[str, dict] = {}
    cursor = db.cursor()
    for i, owner in enumerate(unique):
        rows = cursor.execute(
            """
            SELECT filing_date, status_code, abandonment_date
              FROM marks
             WHERE owner_name_norm = ?
             ORDER BY filing_date
            """,
            (owner,),
        ).fetchall()
        filing_dates = []
        is_cancelled = []  # status_code in [700, 800)
        is_abandoned = []  # status_code in [600, 700) OR abandonment_date set
        for r in rows:
            fd = parse_yyyymmdd(r['filing_date'])
            if fd is None:
                continue
            filing_dates.append(fd)
            sc = r['status_code'] or ''
            is_cancelled.append(1 if (sc >= '700' and sc < '800') else 0)
            is_abandoned.append(1 if ((sc >= '600' and sc < '700') or r['abandonment_date']) else 0)
        cache[owner] = {
            'filing_dates': filing_dates,
            'is_cancelled': is_cancelled,
            'is_abandoned': is_abandoned,
        }
        if (i + 1) % 5000 == 0:
            print(f'  built {i+1:,}/{len(unique):,} owner timelines')
    print(f'[history] owner index complete')
    return cache


def build_attorney_history_index(db: sqlite3.Connection, attorney_names: Iterable[str]) -> dict:
    unique = sorted({a for a in attorney_names if a})
    print(f'[history] building attorney timelines for {len(unique):,} unique attorneys...')
    cache: dict[str, dict] = {}
    cursor = db.cursor()
    for i, atty in enumerate(unique):
        rows = cursor.execute(
            """
            SELECT filing_date, status_code
              FROM marks
             WHERE attorney_name = ?
             ORDER BY filing_date
            """,
            (atty,),
        ).fetchall()
        filing_dates = []
        is_cancelled = []
        for r in rows:
            fd = parse_yyyymmdd(r['filing_date'])
            if fd is None:
                continue
            filing_dates.append(fd)
            sc = r['status_code'] or ''
            is_cancelled.append(1 if (sc >= '700' and sc < '800') else 0)
        cache[atty] = {
            'filing_dates': filing_dates,
            'is_cancelled': is_cancelled,
        }
        if (i + 1) % 5000 == 0:
            print(f'  built {i+1:,}/{len(unique):,} attorney timelines')
    print(f'[history] attorney index complete')
    return cache


def compute_owner_features(history: dict, filing_dt: date) -> dict:
    """Apply the same definitions as the serving-side `applicants` rollup,
    but restricted to filings strictly before `filing_dt` (temporally safe).

    Matches:
      app/scripts/ingest_uspto.ts lines 388-393, 412 (status code definitions)
    """
    fds = history['filing_dates']
    n_total = bisect.bisect_left(fds, filing_dt)
    if n_total == 0:
        return {
            'owner_filing_count_2yr': 0,
            'owner_abandonment_rate': 0.0,
            'owner_historical_cancellation_rate': 0.0,
            'days_since_owner_first_filing': 0,
        }

    two_years_ago = date(filing_dt.year - 2, filing_dt.month, filing_dt.day) \
        if not (filing_dt.month == 2 and filing_dt.day == 29) \
        else date(filing_dt.year - 2, 2, 28)
    n_2yr = n_total - bisect.bisect_left(fds, two_years_ago)

    if n_total >= MIN_PRIOR:
        abandoned = sum(history['is_abandoned'][:n_total])
        cancelled = sum(history['is_cancelled'][:n_total])
        abandon_rate = abandoned / n_total
        cancel_rate = cancelled / n_total
    else:
        abandon_rate = 0.0
        cancel_rate = 0.0

    days_since_first = (filing_dt - fds[0]).days

    return {
        'owner_filing_count_2yr': n_2yr,
        'owner_abandonment_rate': abandon_rate,
        'owner_historical_cancellation_rate': cancel_rate,
        'days_since_owner_first_filing': max(days_since_first, 0),
    }


def compute_attorney_features(history: dict, filing_dt: date) -> dict:
    fds = history['filing_dates']
    n_total = bisect.bisect_left(fds, filing_dt)
    if n_total == 0:
        return {'attorney_case_count': 0, 'attorney_cancellation_rate': 0.0}
    if n_total >= MIN_PRIOR:
        cancelled = sum(history['is_cancelled'][:n_total])
        cancel_rate = cancelled / n_total
    else:
        cancel_rate = 0.0
    return {'attorney_case_count': n_total, 'attorney_cancellation_rate': cancel_rate}


def compute_mark_features(row: dict) -> dict:
    fd = parse_yyyymmdd(row.get('filing_date'))
    rd = parse_yyyymmdd(row.get('registration_date'))
    sc = row.get('status_code') or ''
    classes = row.get('classes') or ''

    days_since_filing = (REFERENCE_DATE - fd).days if fd else 0
    days_filing_to_reg = (rd - fd).days if (fd and rd) else -1
    was_abandoned = 1 if row.get('abandonment_date') else 0
    is_active = 1 if (sc >= '800' and sc < '900') else 0
    class_count = len([c for c in classes.split(',') if c.strip()]) if classes else 1

    return {
        'days_since_filing': max(days_since_filing, 0),
        'days_filing_to_registration': days_filing_to_reg,
        'was_abandoned': was_abandoned,
        'is_currently_active': is_active,
        'class_breadth': max(class_count, 1),
    }


def compute_owner_invariant_flags(row: dict) -> dict:
    """is_individual / is_foreign — invariant per owner, derived directly from
    the mark's row to match the serving-side ingest_uspto.ts definition."""
    return {
        'owner_is_individual': 1 if row.get('owner_legal_entity') == '01' else 0,
        'owner_is_foreign': 1 if (row.get('owner_country') and row['owner_country'] != 'US') else 0,
    }


def build_features(df: pd.DataFrame, owner_idx: dict, atty_idx: dict) -> pd.DataFrame:
    print(f'[features] computing 13-feature rows for {len(df):,} marks...')
    out_rows = []
    for i, row in enumerate(df.to_dict('records')):
        fd = parse_yyyymmdd(row.get('filing_date'))
        if fd is None:
            continue

        feats: dict = {}
        feats.update(compute_mark_features(row))
        feats.update(compute_owner_invariant_flags(row))

        owner_norm = row.get('owner_name_norm')
        if owner_norm and owner_norm in owner_idx:
            feats.update(compute_owner_features(owner_idx[owner_norm], fd))
        else:
            feats.update({
                'owner_filing_count_2yr': 0,
                'owner_abandonment_rate': 0.0,
                'owner_historical_cancellation_rate': 0.0,
                'days_since_owner_first_filing': 0,
            })

        atty = row.get('attorney_name')
        if atty and atty in atty_idx:
            feats.update(compute_attorney_features(atty_idx[atty], fd))
        else:
            feats.update({'attorney_case_count': 0, 'attorney_cancellation_rate': 0.0})

        feats['serial'] = row['serial']
        feats['label'] = row['label']
        out_rows.append(feats)

        if (i + 1) % 10000 == 0:
            print(f'  computed {i+1:,}/{len(df):,}')

    out = pd.DataFrame(out_rows)
    out = out[FEATURE_COLS + ['serial', 'label']]
    print(f'[features] complete: shape={out.shape}, fraud rate={out.label.mean()*100:.1f}%')
    return out


def sanity_checks(df: pd.DataFrame) -> bool:
    print('\n=== sanity checks ===')
    ok = True
    fraud_rate = df.label.mean()
    if not (0.10 <= fraud_rate <= 0.45):
        print(f'  [FAIL] label ratio {fraud_rate:.2f} outside 0.10-0.45')
        ok = False
    else:
        print(f'  [PASS] label ratio {fraud_rate:.2f}')

    nulls = df[FEATURE_COLS].isna().sum().sum()
    if nulls > 0:
        print(f'  [FAIL] {nulls} null values in feature columns')
        ok = False
    else:
        print(f'  [PASS] no nulls')

    print('\n=== feature means: fraud vs legit ===')
    means = df.groupby('label')[FEATURE_COLS].mean().T.rename(columns={0: 'legit', 1: 'fraud'})
    means['delta'] = means['fraud'] - means['legit']
    print(means.to_string())

    # Sanity: known-direction features
    fraud_ab = df[df.label == 1].owner_abandonment_rate.mean()
    legit_ab = df[df.label == 0].owner_abandonment_rate.mean()
    if fraud_ab > legit_ab:
        print(f'\n  [PASS] owner_abandonment_rate: fraud {fraud_ab:.3f} > legit {legit_ab:.3f}')
    else:
        print(f'\n  [WARN] owner_abandonment_rate: fraud {fraud_ab:.3f} <= legit {legit_ab:.3f}')

    return ok


def create_splits(df: pd.DataFrame) -> None:
    print('\n=== train/val/test split (70/15/15) ===')
    X = df[FEATURE_COLS]
    y = df['label']
    serial = df['serial']

    X_train, X_temp, y_train, y_temp, s_train, s_temp = train_test_split(
        X, y, serial, test_size=0.30, random_state=RANDOM_SEED, stratify=y
    )
    X_val, X_test, y_val, y_test, s_val, s_test = train_test_split(
        X_temp, y_temp, s_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_temp
    )

    for name, X_split, y_split, s_split in [
        ('train', X_train, y_train, s_train),
        ('val', X_val, y_val, s_val),
        ('test', X_test, y_test, s_test),
    ]:
        out = X_split.copy()
        out['label'] = y_split.values
        out['serial'] = s_split.values
        path = os.path.join(PROCESSED_DIR, f'features_{name}.csv')
        out.to_csv(path, index=False)
        print(f'  {name}: {len(out):,} rows ({y_split.mean()*100:.1f}% fraud) -> {path}')

    overlap_tt = set(s_train) & set(s_test)
    overlap_tv = set(s_train) & set(s_val)
    assert not overlap_tt, f'leak: {len(overlap_tt)} serials in train AND test'
    assert not overlap_tv, f'leak: {len(overlap_tv)} serials in train AND val'
    print('  [PASS] no serial overlap between splits')


def main() -> int:
    db = open_db()
    fraud_serials = load_fraud_serials()
    verify_join_rate(db, fraud_serials)

    sampled = sample_marks(db, fraud_serials)

    owner_idx = build_owner_history_index(db, sampled.owner_name_norm.tolist())
    atty_idx = build_attorney_history_index(db, sampled.attorney_name.tolist())

    features = build_features(sampled, owner_idx, atty_idx)

    labeled_path = os.path.join(PROCESSED_DIR, 'labeled.csv')
    features.to_csv(labeled_path, index=False)
    print(f'\n[output] wrote {labeled_path}')

    if not sanity_checks(features):
        print('\nFAILED sanity checks — not creating splits')
        return 1

    create_splits(features)
    print('\n[done] build_from_sqlite complete')
    return 0


if __name__ == '__main__':
    sys.exit(main())
