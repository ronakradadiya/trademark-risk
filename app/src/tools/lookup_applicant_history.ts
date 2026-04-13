import {
  ApplicantHistorySchema,
  LookupApplicantInputSchema,
  type ApplicantHistory,
} from '../schemas/index.js';
import { fetchJson, runTool, ToolError, type FetchImpl, type ToolResult } from '../lib/http.js';
import type { z } from 'zod';

type Input = z.infer<typeof LookupApplicantInputSchema>;
type Output = ApplicantHistory;

const DEFAULT_BASE = 'https://tmsearch.uspto.gov/api/search/tm';

/**
 * Demo fixtures keyed by normalized applicant name. These give the CEO demo
 * deterministic, dramatically-scripted answers for well-known filer archetypes
 * (blue-chip, famous troll, shell LLC, foreign filer). Live lookup is still
 * attempted for anything not in this table.
 */
const FIXTURES: Record<string, Omit<ApplicantHistory, 'applicant_name' | 'source'>> = {
  'apple inc': {
    found: true,
    filing_count_total: 15420,
    filing_count_2yr: 612,
    abandonment_rate: 0.02,
    cancellation_rate: 0.004,
    first_filing_date: '1977-03-15',
    is_individual: false,
    is_foreign: false,
    attorney_of_record: 'Thomas R. La Perle',
    attorney_case_count: 4800,
    attorney_cancellation_rate: 0.006,
  },
  'nike inc': {
    found: true,
    filing_count_total: 9200,
    filing_count_2yr: 310,
    abandonment_rate: 0.03,
    cancellation_rate: 0.008,
    first_filing_date: '1971-06-18',
    is_individual: false,
    is_foreign: false,
    attorney_of_record: 'Helen Hill Minsker',
    attorney_case_count: 2100,
    attorney_cancellation_rate: 0.009,
  },
  'leo stoller': {
    found: true,
    filing_count_total: 1834,
    filing_count_2yr: 47,
    abandonment_rate: 0.74,
    cancellation_rate: 0.31,
    first_filing_date: '1985-11-02',
    is_individual: true,
    is_foreign: false,
    attorney_of_record: null,
    attorney_case_count: 0,
    attorney_cancellation_rate: 0,
  },
  'meridian labs llc': {
    found: true,
    filing_count_total: 47,
    filing_count_2yr: 47,
    abandonment_rate: 0.61,
    cancellation_rate: 0.14,
    first_filing_date: '2024-02-11',
    is_individual: false,
    is_foreign: false,
    attorney_of_record: null,
    attorney_case_count: 0,
    attorney_cancellation_rate: 0,
  },
};

function normalizeName(name: string): string {
  return name.trim().toLowerCase().replace(/[.,]/g, '').replace(/\s+/g, ' ');
}

function fromFixture(rawName: string): ApplicantHistory | null {
  const key = normalizeName(rawName);
  const base = FIXTURES[key];
  if (!base) return null;
  return { applicant_name: rawName, source: 'fixture', ...base };
}

interface UsptoOwnerResponse {
  total?: number;
  results?: Array<{
    serialNumber?: string;
    filingDate?: string;
    status?: string;
    abandoned?: boolean;
    cancelled?: boolean;
    ownerIsIndividual?: boolean;
    ownerCountry?: string;
    attorneyName?: string;
  }>;
}

function deriveFromLive(
  applicantName: string,
  response: UsptoOwnerResponse
): ApplicantHistory {
  const rows = response.results ?? [];
  const total = typeof response.total === 'number' ? response.total : rows.length;

  const now = Date.now();
  const twoYearsAgo = now - 2 * 365 * 24 * 60 * 60 * 1000;
  const filing2yr = rows.filter((r) => {
    if (!r.filingDate) return false;
    const t = Date.parse(r.filingDate);
    return Number.isFinite(t) && t >= twoYearsAgo;
  }).length;

  const abandoned = rows.filter((r) => r.abandoned === true).length;
  const cancelled = rows.filter((r) => r.cancelled === true).length;
  const denom = rows.length || 1;

  const earliest = rows
    .map((r) => (r.filingDate ? Date.parse(r.filingDate) : NaN))
    .filter((t) => Number.isFinite(t))
    .sort((a, b) => a - b)[0];

  const firstRow = rows[0];
  const isIndividual = firstRow?.ownerIsIndividual === true;
  const country = firstRow?.ownerCountry ?? 'US';
  const attorney = firstRow?.attorneyName ?? null;

  return {
    applicant_name: applicantName,
    found: total > 0,
    filing_count_total: total,
    filing_count_2yr: filing2yr,
    abandonment_rate: abandoned / denom,
    cancellation_rate: cancelled / denom,
    first_filing_date: earliest ? new Date(earliest).toISOString().slice(0, 10) : null,
    is_individual: isIndividual,
    is_foreign: country !== 'US',
    attorney_of_record: attorney,
    attorney_case_count: attorney ? rows.filter((r) => r.attorneyName === attorney).length : 0,
    attorney_cancellation_rate: 0,
    source: 'live',
  };
}

export function unknownApplicant(applicantName: string): ApplicantHistory {
  return {
    applicant_name: applicantName,
    found: false,
    filing_count_total: 0,
    filing_count_2yr: 0,
    abandonment_rate: 0,
    cancellation_rate: 0,
    first_filing_date: null,
    is_individual: false,
    is_foreign: false,
    attorney_of_record: null,
    attorney_case_count: 0,
    attorney_cancellation_rate: 0,
    source: 'unknown',
  };
}

export async function lookupApplicantHistory(
  input: Input,
  opts: {
    fetchImpl?: FetchImpl;
    baseUrl?: string;
    apiKey?: string;
    timeoutMs?: number;
    disableLive?: boolean;
  } = {}
): Promise<ToolResult<Output>> {
  const parsed = LookupApplicantInputSchema.parse(input);

  return runTool(async () => {
    const fixture = fromFixture(parsed.applicant_name);
    if (fixture) return fixture;

    if (opts.disableLive) return unknownApplicant(parsed.applicant_name);

    const base = opts.baseUrl ?? process.env.USPTO_OWNER_SEARCH_URL ?? DEFAULT_BASE;
    const apiKey = opts.apiKey ?? process.env.USPTO_API_KEY;

    try {
      const url = new URL(base);
      url.searchParams.set('owner', parsed.applicant_name);

      const response = await fetchJson<UsptoOwnerResponse>(url.toString(), {
        headers: apiKey ? { 'x-api-key': apiKey } : {},
        timeoutMs: opts.timeoutMs ?? 5000,
        ...(opts.fetchImpl ? { fetchImpl: opts.fetchImpl } : {}),
      });

      const history = deriveFromLive(parsed.applicant_name, response);
      const validated = ApplicantHistorySchema.safeParse(history);
      if (!validated.success) {
        throw new ToolError(
          `response failed schema validation: ${validated.error.message}`,
          'parse'
        );
      }
      return validated.data;
    } catch (e) {
      if (e instanceof ToolError && e.code !== 'parse') {
        return unknownApplicant(parsed.applicant_name);
      }
      throw e;
    }
  });
}
