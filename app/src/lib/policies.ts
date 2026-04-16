import type { PolicyKey, ToolName } from '../schemas/index.js';

export interface PolicyDefinition {
  key: PolicyKey;
  name: string;
  description: string;
  tools: ToolName[];
}

export const POLICIES: Record<PolicyKey, PolicyDefinition> = {
  P1: {
    key: 'P1',
    name: 'Confusingly similar mark exists',
    description:
      'Another registered or pending mark has the same or similar name in the same or related international class. Similarity includes phonetic (sounds alike), visual (looks alike), and meaning (translates to the same thing). Evaluate using check_uspto_marks results.',
    tools: ['check_uspto_marks'],
  },
  P2: {
    key: 'P2',
    name: 'Filer shows trademark troll / shell-filer pattern',
    description: [
      'Judge the filer holistically from applicant_history. Trigger P2 if you see ANY of the following patterns (these are illustrative, not a checklist — weigh them together):',
      '',
      '(a) Spam-and-abandon: filing_count_total >= 20 AND abandonment_rate >= 0.5. Classic "file to block, never intend to use" pattern — the filer keeps filing marks they never commercialize. A 2024 example: an LLC with 46 filings and 93% abandonment across 420-themed marks.',
      '',
      '(b) Burst-filing shell: the entity\'s entire filing history is concentrated in the last 24 months (filing_count_2yr == filing_count_total) AND volume is high (>= 15) AND is_individual == false. The LLC was created to run a filing spree — no prior operating history. Numeric-corporation names ("1660929 Ontario Limited") are a strong tell.',
      '',
      '(c) Sub-threshold spam with class concentration: even if filings < 20, if the filer concentrates most marks in one class AND has abandonment_rate >= 0.4, treat it as a trademark mill. Call web_search to check for news of settlement-demand letters.',
      '',
      '(d) Known troll signals from web_search: news articles or legal records of them demanding money from businesses, or a documented history of filing-then-settling patterns.',
      '',
      'Do NOT trigger P2 for blue-chip filers (Apple, Nike, Microsoft) even if abandonment_rate is moderate — megafilers have elevated absolute abandonment simply from volume. Read the is_foreign + attorney_of_record + first_filing_date fields together: a 20-year-old filer with counsel and 10% abandonment is NOT a troll.',
    ].join('\n'),
    tools: ['web_search'],
  },
  P3: {
    key: 'P3',
    name: 'Filing timing suspicious',
    description: [
      'Trigger P3 ONLY when the check_domain_age tool returns a concrete negative signal:',
      '',
      '(a) exists=true AND registered_at is within 30 days of (or after) the filing date — the filer registered a domain around the same time they filed the trademark, suggesting no organic business buildup.',
      '',
      '(b) exists=false AND the filer does not have other strong evidence of a real operating business from web_search (no storefront, no reviews, no press).',
      '',
      'Do NOT trigger P3 if:',
      '- No domain_name was provided in the request (nothing to evaluate — leave triggered=false, low confidence).',
      '- The check_domain_age tool failed with a network/timeout error (tool failure is not a fraud signal — leave triggered=false and note the unavailable lookup).',
      '- The domain exists and was registered well before the filing (normal case, safe).',
      '',
      'When the tool is unavailable, lean on web_search results for commercial presence instead of defaulting to triggered.',
    ].join('\n'),
    tools: ['check_domain_age'],
  },
  P4: {
    key: 'P4',
    name: 'Attorney of record is a red flag',
    description: [
      'Evaluate from applicant_history alone (attorney_of_record, attorney_case_count, attorney_cancellation_rate, filing_count_2yr). Do NOT call a tool for this policy.',
      '',
      'Trigger P4 if ANY of these hold:',
      '',
      '(a) Pro-se high volume: attorney_of_record is null AND the filer has an unusually high 24-month filing count. Small businesses typically file 1-3 marks in their lifetime; a pro-se filer filing 10+ marks in 24 months is acting like a legitimate mid-to-large filer but has not retained counsel, which is the shell-mill / IP-broker signature. Use judgment on borderline counts (5-10): a handful of related marks for one real business is normal, but a burst of unrelated marks in different industries is not.',
      '',
      '(b) Elevated attorney cancellation: attorney_cancellation_rate is materially above the USPTO ~8% baseline AND attorney_case_count is large enough to be meaningful (not noise from a few cases). A blue-chip attorney serving megabrands can run 10-15% from routine non-renewals — that is NOT a red flag. An attorney in the 20%+ range with a reasonable caseload is.',
      '',
      'Do NOT trigger P4 for:',
      '- Pro-se filings with low 24-month volume — that is just a small business that never hired an attorney.',
      '- Attorneys with high case counts but cancellation within baseline range (roughly 10-15%) — those are high-volume firms serving megabrands (Apple, Nike), not trolls.',
    ].join('\n'),
    tools: [],
  },
  P5: {
    key: 'P5',
    name: 'No genuine commercial use evidence',
    description:
      'There is no real website with actual products or services, no customer reviews, no social media presence with real activity, no press coverage or business listings, and no evidence the brand name is associated with a real operating business. Evaluate using web_search results.',
    tools: ['web_search'],
  },
};

export const POLICY_LIST: PolicyDefinition[] = [
  POLICIES.P1,
  POLICIES.P2,
  POLICIES.P3,
  POLICIES.P4,
  POLICIES.P5,
];
