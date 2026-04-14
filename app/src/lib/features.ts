import type { FeatureVector } from './classifier.js';
import type { ApplicantHistory } from '../schemas/index.js';
import { getUsptoDb, normalizeOwnerName } from './uspto_db.js';

/**
 * Neutral mark-side priors used when the prior-art lookup finds nothing for
 * this brand name. A pre-filing check has no filing_date of its own, so
 * `days_since_filing` is always 0 regardless.
 */
export const UNKNOWN_MARK_FEATURES = {
  days_since_filing: 0,
  days_filing_to_registration: 365,
  was_abandoned: 0.1,
  is_currently_active: 0.8,
  class_breadth: 1,
} as const;

/**
 * Neutral applicant features used when no history is available (unknown filer,
 * network failure). These mirror a small, low-risk first-time filer so the
 * classifier doesn't hallucinate a troll signal from missing data.
 */
export const UNKNOWN_APPLICANT_FEATURES = {
  owner_filing_count_2yr: 1,
  owner_abandonment_rate: 0.05,
  owner_historical_cancellation_rate: 0.02,
  days_since_owner_first_filing: 365,
  owner_is_individual: 0,
  attorney_case_count: 50,
  attorney_cancellation_rate: 0.03,
  owner_is_foreign: 0,
} as const;

export const DEFAULT_PRE_FILING_FEATURES: FeatureVector = {
  ...UNKNOWN_MARK_FEATURES,
  ...UNKNOWN_APPLICANT_FEATURES,
};

/**
 * Map an ApplicantHistory (from lookup_applicant_history) to the 8 applicant
 * feature columns the v4 ranker consumes. Unknown / not-found applicants fall
 * back to the neutral priors.
 */
export function applicantToFeatures(
  history: ApplicantHistory | null,
  now: Date = new Date()
): Partial<FeatureVector> {
  if (!history || !history.found) return { ...UNKNOWN_APPLICANT_FEATURES };

  let daysSinceFirst = 365;
  if (history.first_filing_date) {
    const t = Date.parse(history.first_filing_date);
    if (Number.isFinite(t)) {
      daysSinceFirst = Math.max(0, Math.round((now.getTime() - t) / (24 * 60 * 60 * 1000)));
    }
  }

  return {
    owner_filing_count_2yr: history.filing_count_2yr,
    owner_abandonment_rate: history.abandonment_rate,
    owner_historical_cancellation_rate: history.cancellation_rate,
    days_since_owner_first_filing: daysSinceFirst,
    owner_is_individual: history.is_individual ? 1 : 0,
    attorney_case_count: history.attorney_case_count,
    attorney_cancellation_rate: history.attorney_cancellation_rate,
    owner_is_foreign: history.is_foreign ? 1 : 0,
  };
}

type MarkFeatureFields =
  | 'days_since_filing'
  | 'days_filing_to_registration'
  | 'was_abandoned'
  | 'is_currently_active'
  | 'class_breadth';

interface PriorArtRow {
  filing_date: string | null;
  registration_date: string | null;
  abandonment_date: string | null;
  status_code: string | null;
}

/**
 * Compute the 5 mark-side features for a pre-filing check by aggregating over
 * historical USPTO marks with the same normalized brand name. This replaces
 * the old BASELINE_MARK_FEATURES constants so the ML model sees a real signal
 * instead of neutral placeholders.
 *
 * Shape contract:
 * - days_since_filing: always 0 (pre-filing check, no filing date yet)
 * - days_filing_to_registration: mean of filing→registration for priors that
 *   reached registration; UNKNOWN prior if none
 * - was_abandoned: fraction of priors with an abandonment_date
 * - is_currently_active: fraction of priors with status_code in 800..899
 * - class_breadth: count of distinct classes in the proposed filing (defaults
 *   to 1 when class_code is not given)
 */
export function markFeaturesFromPriorArt(
  brand_name: string,
  class_code: number | undefined
): Pick<FeatureVector, MarkFeatureFields> {
  const class_breadth = class_code !== undefined ? 1 : 1;

  let rows: PriorArtRow[] = [];
  try {
    const db = getUsptoDb();
    const normalized = normalizeOwnerName(brand_name);
    if (normalized.length === 0) {
      return { ...UNKNOWN_MARK_FEATURES, class_breadth };
    }
    rows = db
      .prepare(
        `SELECT filing_date, registration_date, abandonment_date, status_code
         FROM marks
         WHERE mark_norm = ?
         LIMIT 200`
      )
      .all(normalized) as PriorArtRow[];
  } catch {
    return { ...UNKNOWN_MARK_FEATURES, class_breadth };
  }

  if (rows.length === 0) {
    return { ...UNKNOWN_MARK_FEATURES, class_breadth };
  }

  const parse8 = (s: string | null): number | null => {
    if (!s || !/^\d{8}$/.test(s)) return null;
    const y = Number(s.slice(0, 4));
    const m = Number(s.slice(4, 6));
    const d = Number(s.slice(6, 8));
    const t = Date.UTC(y, m - 1, d);
    return Number.isFinite(t) ? t : null;
  };

  let abandonedCount = 0;
  let activeCount = 0;
  const regDeltas: number[] = [];
  for (const r of rows) {
    if (r.abandonment_date) abandonedCount++;
    if (r.status_code && r.status_code >= '800' && r.status_code < '900') {
      activeCount++;
    }
    const f = parse8(r.filing_date);
    const reg = parse8(r.registration_date);
    if (f !== null && reg !== null && reg >= f) {
      regDeltas.push(Math.round((reg - f) / (24 * 60 * 60 * 1000)));
    }
  }

  const was_abandoned = abandonedCount / rows.length;
  const is_currently_active = activeCount / rows.length;
  const days_filing_to_registration =
    regDeltas.length > 0
      ? regDeltas.reduce((a, b) => a + b, 0) / regDeltas.length
      : UNKNOWN_MARK_FEATURES.days_filing_to_registration;

  return {
    days_since_filing: 0,
    days_filing_to_registration,
    was_abandoned,
    is_currently_active,
    class_breadth,
  };
}

export function buildFeatures(overrides: Partial<FeatureVector> = {}): FeatureVector {
  return { ...DEFAULT_PRE_FILING_FEATURES, ...overrides };
}
