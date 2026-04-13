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
    name: 'Attorney credentials suspicious',
    description:
      'The attorney is not found in the USPTO OED directory, their bar status is inactive or suspended, they have disciplinary history or sanctions, or they are known to work exclusively with high-volume troll filers. Evaluate using check_attorney results.',
    tools: ['check_attorney'],
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
