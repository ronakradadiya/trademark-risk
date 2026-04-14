import type { ApplicantHistory, PolicyBundle, PolicyResult } from '../schemas/index.js';

export interface HardRuleHit {
  id: string;
  reason: string;
  /** Which policy this rule maps to, so the PolicyBundle can be pre-populated. */
  policy: 'P1' | 'P2' | 'P3' | 'P4' | 'P5';
}

/**
 * Deterministic pre-agent rule overlay. These exist because certain filer
 * patterns are so unambiguously bad that neither the ML nor the LLM should
 * get to "disagree" with them — a shell LLC with 50+ filings in 24 months
 * and a 70% abandonment rate is a troll by any reasonable definition, and
 * we don't want to burn an OpenAI call to confirm it.
 *
 * Rules are evaluated against the applicant_history populated by the USPTO
 * SQLite lookup, so they fire on *real* filing data, not model inference.
 */
export function evaluateHardRules(history: ApplicantHistory): HardRuleHit[] {
  const hits: HardRuleHit[] = [];
  if (!history.found) return hits;

  // R1: Prolific pro-se filer. Legitimate high-volume filers retain counsel;
  // when someone files >20 marks in 24 months with no attorney of record,
  // it's almost always a serial troll, IP broker, or shell-mill operation.
  if (
    history.attorney_of_record === null &&
    history.filing_count_2yr > 20
  ) {
    hits.push({
      id: 'R1',
      policy: 'P4',
      reason: `Pro se filer with ${history.filing_count_2yr} filings in last 24 months. Legitimate high-volume filers retain counsel.`,
    });
  }

  // R2: High-volume filer with majority-abandonment track record.
  // This is the classic "file to block, never intend to use" pattern.
  if (history.filing_count_total >= 20 && history.abandonment_rate >= 0.5) {
    hits.push({
      id: 'R2',
      policy: 'P2',
      reason: `${history.filing_count_total} total filings with ${(history.abandonment_rate * 100).toFixed(0)}% abandonment rate — pattern consistent with non-use / intent-to-block filings.`,
    });
  }

  // R3: Burst-filing shell LLC. filing_count_2yr == filing_count_total AND
  // both are large means the entity was created just to run a filing spree —
  // no historical legitimate business.
  if (
    history.filing_count_total >= 15 &&
    history.filing_count_2yr === history.filing_count_total &&
    !history.is_individual
  ) {
    hits.push({
      id: 'R3',
      policy: 'P1',
      reason: `Entity's entire ${history.filing_count_total}-filing history is concentrated in the last 24 months with no prior track record — burst-filing shell pattern.`,
    });
  }

  // Note: an "attorney_cancellation_rate" rule is deliberately omitted. The
  // USPTO status-code 700–799 bucket the rollup uses as "cancelled" also
  // contains routine non-renewal expirations, so blue-chip attorneys (Apple's
  // Kimberly Eckhart, Nike's Jennifer Reynolds) would wrongly trip it. Let
  // the agent evaluate P4 nuance from the raw fields instead of hard-coding
  // a threshold on a noisy signal.

  return hits;
}

/**
 * Turn hard-rule hits into a PolicyBundle that mirrors the shape the agent
 * would have produced, so downstream code can treat a rule-driven verdict
 * the same way it treats an agent verdict.
 */
export function hitsToPolicyBundle(hits: HardRuleHit[]): PolicyBundle {
  const neutral: PolicyResult = {
    triggered: false,
    confidence: 0,
    reason: 'Not evaluated — verdict decided by deterministic rule overlay.',
  };
  const bundle: PolicyBundle = {
    P1: { ...neutral },
    P2: { ...neutral },
    P3: { ...neutral },
    P4: { ...neutral },
    P5: { ...neutral },
  };
  for (const h of hits) {
    bundle[h.policy] = { triggered: true, confidence: 0.95, reason: `[${h.id}] ${h.reason}` };
  }
  return bundle;
}
