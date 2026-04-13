import { z } from 'zod';

export const POLICY_KEYS = ['P1', 'P2', 'P3', 'P4', 'P5'] as const;
export type PolicyKey = (typeof POLICY_KEYS)[number];

export const RISK_TIERS = ['high', 'mid', 'low'] as const;
export const VERDICTS = ['safe', 'review', 'high_risk'] as const;
export const VERDICT_SOURCES = ['ml_only', 'ml_and_agent'] as const;

export const CheckRequestSchema = z.object({
  brand_name: z.string().min(1).max(200),
  applicant_name: z.string().min(1).max(200),
  domain_name: z.string().max(253).optional(),
  attorney_name: z.string().max(200).optional(),
  attorney_bar_number: z.string().max(50).optional(),
  class_code: z.number().int().min(1).max(45).optional(),
});
export type CheckRequest = z.infer<typeof CheckRequestSchema>;

export const ApplicantHistorySchema = z.object({
  applicant_name: z.string().min(1),
  found: z.boolean(),
  filing_count_total: z.number().int().min(0),
  filing_count_2yr: z.number().int().min(0),
  abandonment_rate: z.number().min(0).max(1),
  cancellation_rate: z.number().min(0).max(1),
  first_filing_date: z.string().nullable(),
  is_individual: z.boolean(),
  is_foreign: z.boolean(),
  attorney_of_record: z.string().nullable(),
  attorney_case_count: z.number().int().min(0),
  attorney_cancellation_rate: z.number().min(0).max(1),
  source: z.enum(['fixture', 'live', 'unknown']),
});
export type ApplicantHistory = z.infer<typeof ApplicantHistorySchema>;

export const LookupApplicantInputSchema = z.object({
  applicant_name: z.string().min(1).max(200),
});
export const LookupApplicantOutputSchema = ApplicantHistorySchema;

export const MLPredictionSchema = z.object({
  score: z.number().min(0).max(1),
  tier: z.enum(RISK_TIERS),
  high_threshold: z.number().min(0).max(1),
  low_threshold: z.number().min(0).max(1),
  model_version: z.string(),
});
export type MLPrediction = z.infer<typeof MLPredictionSchema>;

export const PolicyResultSchema = z.object({
  triggered: z.boolean(),
  confidence: z.number().min(0).max(1),
  reason: z.string().min(1).max(1000),
});
export type PolicyResult = z.infer<typeof PolicyResultSchema>;

export const PolicyBundleSchema = z.object({
  P1: PolicyResultSchema,
  P2: PolicyResultSchema,
  P3: PolicyResultSchema,
  P4: PolicyResultSchema,
  P5: PolicyResultSchema,
});
export type PolicyBundle = z.infer<typeof PolicyBundleSchema>;

export const USPTOMarkSchema = z.object({
  serial_number: z.string(),
  mark_name: z.string(),
  owner_name: z.string().optional(),
  filing_date: z.string().optional(),
  status: z.string().optional(),
  class_codes: z.array(z.number().int()).default([]),
});
export type USPTOMark = z.infer<typeof USPTOMarkSchema>;

export const CheckUsptoMarksInputSchema = z.object({
  brand_name: z.string().min(1),
  class_code: z.number().int().min(1).max(45).optional(),
});
export const CheckUsptoMarksOutputSchema = z.object({
  query: z.string(),
  total: z.number().int().min(0),
  results: z.array(USPTOMarkSchema),
});

export const CheckDomainAgeInputSchema = z.object({
  domain_name: z.string().min(1).max(253),
});
export const CheckDomainAgeOutputSchema = z.object({
  domain_name: z.string(),
  exists: z.boolean(),
  registered_at: z.string().nullable(),
  age_days: z.number().int().nullable(),
});

export const WebSearchInputSchema = z.object({
  query: z.string().min(1).max(500),
  max_results: z.number().int().min(1).max(20).default(5),
});
export const WebSearchResultSchema = z.object({
  title: z.string(),
  url: z.string().url(),
  snippet: z.string(),
});
export const WebSearchOutputSchema = z.object({
  query: z.string(),
  results: z.array(WebSearchResultSchema),
});

export const CheckAttorneyInputSchema = z.object({
  attorney_name: z.string().min(1),
  bar_number: z.string().optional(),
});
export const CheckAttorneyOutputSchema = z.object({
  found: z.boolean(),
  name: z.string().nullable(),
  bar_status: z.enum(['active', 'inactive', 'suspended', 'unknown']),
  disciplinary_history: z.boolean(),
});

export const TOOL_NAMES = [
  'lookup_applicant_history',
  'check_uspto_marks',
  'check_domain_age',
  'web_search',
  'check_attorney',
] as const;
export type ToolName = (typeof TOOL_NAMES)[number];

export const AgentStepSchema = z.object({
  step: z.number().int().min(0),
  tool: z.enum(TOOL_NAMES),
  input: z.unknown(),
  output: z.unknown(),
  error: z.string().nullable(),
  duration_ms: z.number().int().min(0),
});
export type AgentStep = z.infer<typeof AgentStepSchema>;

export const VerdictSchema = z.object({
  brand: z.string().min(1),
  applicant: z.string().min(1),
  verdict: z.enum(VERDICTS),
  overall_confidence: z.number().min(0).max(1),
  ml: MLPredictionSchema,
  applicant_history: ApplicantHistorySchema.nullable(),
  source: z.enum(VERDICT_SOURCES),
  policies: PolicyBundleSchema,
  tools_used: z.array(z.enum(TOOL_NAMES)),
  trace: z.array(AgentStepSchema).default([]),
  summary: z.string().min(1).max(500),
  checked_at: z.string().datetime(),
});
export type Verdict = z.infer<typeof VerdictSchema>;

export const AuditRecordSchema = VerdictSchema.extend({
  id: z.string().uuid(),
  ttl: z.number().int().positive(),
});
export type AuditRecord = z.infer<typeof AuditRecordSchema>;
