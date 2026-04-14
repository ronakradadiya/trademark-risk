import {
  ApplicantHistorySchema,
  LookupApplicantInputSchema,
  type ApplicantHistory,
} from '../schemas/index.js';
import { runTool, ToolError, type ToolResult } from '../lib/http.js';
import { getUsptoDb, normalizeOwnerName } from '../lib/uspto_db.js';
import type { z } from 'zod';
import type Database from 'better-sqlite3';

type Input = z.infer<typeof LookupApplicantInputSchema>;
type Output = ApplicantHistory;

/**
 * Demo fixtures keyed by normalized applicant name. These give the CEO demo
 * deterministic, dramatically-scripted answers for well-known filer archetypes
 * (blue-chip, famous troll, shell LLC). Only used when the local SQLite lookup
 * returns no rows — real USPTO data always wins.
 */
const FIXTURES: Record<string, Omit<ApplicantHistory, 'applicant_name' | 'source'>> = {
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

interface ApplicantRow {
  display_name: string;
  filing_count_total: number;
  filing_count_2yr: number;
  abandonment_rate: number;
  cancellation_rate: number;
  first_filing_date: string | null;
  is_individual: number;
  is_foreign: number;
  attorney_of_record: string | null;
  attorney_case_count: number;
  attorney_cancellation_rate: number;
}

function formatDate(d: string | null): string | null {
  if (!d) return null;
  if (/^\d{8}$/.test(d)) return `${d.slice(0, 4)}-${d.slice(4, 6)}-${d.slice(6, 8)}`;
  return d;
}

function rowToHistory(applicantName: string, row: ApplicantRow): ApplicantHistory {
  return {
    applicant_name: applicantName,
    found: true,
    filing_count_total: row.filing_count_total,
    filing_count_2yr: row.filing_count_2yr,
    abandonment_rate: row.abandonment_rate,
    cancellation_rate: row.cancellation_rate,
    first_filing_date: formatDate(row.first_filing_date),
    is_individual: row.is_individual === 1,
    is_foreign: row.is_foreign === 1,
    attorney_of_record: row.attorney_of_record,
    attorney_case_count: row.attorney_case_count,
    attorney_cancellation_rate: row.attorney_cancellation_rate,
    source: 'live',
  };
}

function queryApplicant(
  db: Database.Database,
  applicantName: string
): ApplicantHistory | null {
  const normalized = normalizeOwnerName(applicantName);
  if (!normalized) return null;

  const sql = `
    SELECT display_name, filing_count_total, filing_count_2yr,
           abandonment_rate, cancellation_rate, first_filing_date,
           is_individual, is_foreign, attorney_of_record,
           attorney_case_count, attorney_cancellation_rate
    FROM applicants
    WHERE owner_name_norm = ?
    LIMIT 1
  `;
  const row = db.prepare(sql).get(normalized) as ApplicantRow | undefined;
  if (!row) return null;
  return rowToHistory(applicantName, row);
}

function fromFixture(rawName: string): ApplicantHistory | null {
  const key = normalizeOwnerName(rawName);
  const base = FIXTURES[key];
  if (!base) return null;
  return { applicant_name: rawName, source: 'fixture', ...base };
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
  _opts: Record<string, never> = {}
): Promise<ToolResult<Output>> {
  const parsed = LookupApplicantInputSchema.parse(input);

  return runTool(async () => {
    let db: Database.Database | null = null;
    try {
      db = getUsptoDb();
    } catch {
      // DB missing — fall through to fixtures / unknown so the tool still
      // returns a valid ApplicantHistory instead of a hard config error.
    }

    if (db) {
      const hit = queryApplicant(db, parsed.applicant_name);
      if (hit) {
        const validated = ApplicantHistorySchema.safeParse(hit);
        if (!validated.success) {
          throw new ToolError(
            `response failed schema validation: ${validated.error.message}`,
            'parse'
          );
        }
        return validated.data;
      }
    }

    const fixture = fromFixture(parsed.applicant_name);
    if (fixture) return fixture;

    return unknownApplicant(parsed.applicant_name);
  });
}
