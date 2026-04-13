import type { FeatureVector } from './classifier.js';

/**
 * Default feature vector for a pre-filing risk check.
 *
 * For a brand that has not been filed yet we have no applicant history,
 * attorney history, or prosecution signals. We start from a neutral-to-
 * slightly-legitimate baseline so the ML score reflects "no priors" rather
 * than fabricating troll signals. The agent layer does the real investigation.
 */
export const DEFAULT_PRE_FILING_FEATURES: FeatureVector = {
  owner_filing_count_2yr: 1,
  owner_abandonment_rate: 0.05,
  owner_historical_cancellation_rate: 0.02,
  days_since_owner_first_filing: 365,
  owner_is_individual: 0,
  attorney_case_count: 50,
  attorney_cancellation_rate: 0.03,
  owner_is_foreign: 0,
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

export function buildFeatures(overrides: Partial<FeatureVector> = {}): FeatureVector {
  return { ...DEFAULT_PRE_FILING_FEATURES, ...overrides };
}
