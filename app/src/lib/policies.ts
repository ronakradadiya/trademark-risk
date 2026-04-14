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
    name: 'Filer shows trademark troll pattern',
    description:
      'The filer has filed many marks across unrelated industries, there is no evidence of a real operating business, there are news articles or legal records of them demanding money from businesses, or they have a history of filing and abandoning marks after settlement. Evaluate using web_search results and the ML owner_filing_count feature.',
    tools: ['web_search'],
  },
  P3: {
    key: 'P3',
    name: 'Filing timing suspicious',
    description:
      'The associated domain was registered fewer than 30 days before the trademark filing, the domain does not exist, or the domain was registered after the filing. This suggests the filer is not a real business that grew organically into needing a trademark. Evaluate using check_domain_age results.',
    tools: ['check_domain_age'],
  },
  P4: {
    key: 'P4',
    name: 'Attorney of record is a red flag',
    description:
      'The applicant filed pro se (no attorney of record) despite high filing volume, or their attorney of record has an elevated cancellation rate relative to the USPTO baseline (~8%). Pro se filing with more than five filings in the last 24 months is itself a strong signal: legitimate high-volume filers retain counsel. Evaluate using the applicant_history fields already provided in the user message (attorney_of_record, attorney_case_count, attorney_cancellation_rate, filing_count_2yr). Do NOT call a tool for this policy.',
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
