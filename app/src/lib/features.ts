import type { FeatureVector } from './classifier.js';
import type { ApplicantHistory } from '../schemas/index.js';

/**
 * Baseline prosecution/mark features for a pre-filing check.
 *
 * These 14 columns describe the mark itself (office actions, opposition,
 * statement of use, class breadth, etc.). At pre-filing time none of them
 * exist yet, so we hold them at neutral priors. The 8 applicant-side columns
 * are populated per-request from ApplicantHistory via applicantToFeatures().
 */
export const BASELINE_MARK_FEATURES: Omit<
  FeatureVector,
  | 'owner_filing_count_2yr'
  | 'owner_abandonment_rate'
  | 'owner_historical_cancellation_rate'
  | 'days_since_owner_first_filing'
  | 'owner_is_individual'
  | 'attorney_case_count'
  | 'attorney_cancellation_rate'
  | 'owner_is_foreign'
> = {
  days_domain_to_filing: 180,
  days_since_filing: 0,
  days_first_use_to_filing: 90,
  days_filing_to_registration: 180,
  office_action_count: 0,
  opposition_count: 0,
  statement_of_use_filed: 0,
  section_8_filed: 0,
  has_acquired_distinctiveness: 0,
  is_currently_active: 1,
  was_abandoned: 0,
  filing_basis: 1,
  class_breadth: 1,
  specimen_type_encoded: 1,
};

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
  ...BASELINE_MARK_FEATURES,
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

export function buildFeatures(overrides: Partial<FeatureVector> = {}): FeatureVector {
  return { ...DEFAULT_PRE_FILING_FEATURES, ...overrides };
}
