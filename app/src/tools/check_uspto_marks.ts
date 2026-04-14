import {
  CheckUsptoMarksInputSchema,
  CheckUsptoMarksOutputSchema,
  type USPTOMark,
} from '../schemas/index.js';
import { runTool, ToolError, type ToolResult } from '../lib/http.js';
import { getUsptoDb, toFtsQuery, normalizeOwnerName } from '../lib/uspto_db.js';
import type { z } from 'zod';
import type Database from 'better-sqlite3';

type Input = z.infer<typeof CheckUsptoMarksInputSchema>;
type Output = z.infer<typeof CheckUsptoMarksOutputSchema>;

/**
 * Live USPTO marks are status < 600 (examination/publication) or 800-899
 * (registered). Dead marks (600-799) are abandoned/cancelled and should not
 * trigger confusion concerns.
 */
const LIVE_STATUS_SQL = `(status_code IS NULL OR status_code < '600' OR (status_code >= '800' AND status_code < '900'))`;

const RESULT_LIMIT = 15;

interface MarkRow {
  serial: string;
  mark: string;
  owner_name: string | null;
  filing_date: string | null;
  status_code: string | null;
  classes: string;
}

function rowToMark(row: MarkRow): USPTOMark {
  const classNums = (row.classes || '')
    .split(',')
    .map((s) => Number(s))
    .filter((n) => Number.isInteger(n) && n >= 1 && n <= 45);

  const mark: USPTOMark = {
    serial_number: row.serial,
    mark_name: row.mark,
    class_codes: classNums,
  };
  if (row.owner_name) mark.owner_name = row.owner_name;
  if (row.filing_date) {
    // Stored as YYYYMMDD; emit as YYYY-MM-DD for downstream consumers.
    const d = row.filing_date;
    if (/^\d{8}$/.test(d)) {
      mark.filing_date = `${d.slice(0, 4)}-${d.slice(4, 6)}-${d.slice(6, 8)}`;
    } else {
      mark.filing_date = d;
    }
  }
  if (row.status_code) mark.status = row.status_code;
  return mark;
}

/**
 * Search strategy, in descending precedence:
 *   1. Exact normalized match on mark_norm
 *   2. FTS5 phrase match (all query tokens must appear)
 * Results are deduplicated by serial, live marks only, class-filtered if
 * the caller supplied a Nice class, then capped at RESULT_LIMIT.
 */
function searchMarks(
  db: Database.Database,
  brand: string,
  classCode: number | undefined
): MarkRow[] {
  const normalized = normalizeOwnerName(brand); // reuse the same normalization
  const classPattern = classCode !== undefined ? `%${String(classCode).padStart(3, '0')}%` : null;

  const seen = new Set<string>();
  const out: MarkRow[] = [];

  const pushRow = (row: MarkRow) => {
    if (seen.has(row.serial)) return;
    seen.add(row.serial);
    out.push(row);
  };

  // Stage 1: exact normalized match.
  const exactSql = `
    SELECT serial, mark, owner_name, filing_date, status_code, classes
    FROM marks
    WHERE mark_norm = ? AND ${LIVE_STATUS_SQL}
    ${classPattern ? 'AND classes LIKE ?' : ''}
    LIMIT ${RESULT_LIMIT}
  `;
  const exactParams: unknown[] = [normalized];
  if (classPattern) exactParams.push(classPattern);
  const exactRows = db.prepare(exactSql).all(...exactParams) as MarkRow[];
  for (const r of exactRows) pushRow(r);

  if (out.length >= RESULT_LIMIT) return out;

  // Stage 2: FTS5 phrase search.
  const ftsQuery = toFtsQuery(brand);
  if (ftsQuery.length > 0) {
    const ftsSql = `
      SELECT m.serial, m.mark, m.owner_name, m.filing_date, m.status_code, m.classes
      FROM marks_fts f
      JOIN marks m ON f.rowid = m.rowid
      WHERE marks_fts MATCH ?
        AND ${LIVE_STATUS_SQL.replace(/status_code/g, 'm.status_code')}
        ${classPattern ? 'AND m.classes LIKE ?' : ''}
      ORDER BY bm25(marks_fts)
      LIMIT ${RESULT_LIMIT * 2}
    `;
    const ftsParams: unknown[] = [ftsQuery];
    if (classPattern) ftsParams.push(classPattern);
    try {
      const ftsRows = db.prepare(ftsSql).all(...ftsParams) as MarkRow[];
      for (const r of ftsRows) {
        if (out.length >= RESULT_LIMIT) break;
        pushRow(r);
      }
    } catch {
      // FTS5 can throw on empty or malformed queries — fall through silently.
    }
  }

  return out;
}

/**
 * Count the full live-match total (before LIMIT) so the verdict surfaces
 * "we looked across N conflicting marks" rather than only the capped page.
 */
function countLiveMatches(
  db: Database.Database,
  brand: string,
  classCode: number | undefined
): number {
  const normalized = normalizeOwnerName(brand);
  const classPattern = classCode !== undefined ? `%${String(classCode).padStart(3, '0')}%` : null;

  // Exact match count
  const exactSql = `
    SELECT COUNT(*) AS c
    FROM marks
    WHERE mark_norm = ? AND ${LIVE_STATUS_SQL}
    ${classPattern ? 'AND classes LIKE ?' : ''}
  `;
  const exactParams: unknown[] = [normalized];
  if (classPattern) exactParams.push(classPattern);
  const exact = (db.prepare(exactSql).get(...exactParams) as { c: number }).c;

  // FTS5 count
  let fts = 0;
  const ftsQuery = toFtsQuery(brand);
  if (ftsQuery.length > 0) {
    try {
      const ftsSql = `
        SELECT COUNT(*) AS c
        FROM marks_fts f
        JOIN marks m ON f.rowid = m.rowid
        WHERE marks_fts MATCH ?
          AND ${LIVE_STATUS_SQL.replace(/status_code/g, 'm.status_code')}
          ${classPattern ? 'AND m.classes LIKE ?' : ''}
      `;
      const ftsParams: unknown[] = [ftsQuery];
      if (classPattern) ftsParams.push(classPattern);
      fts = (db.prepare(ftsSql).get(...ftsParams) as { c: number }).c;
    } catch {
      fts = 0;
    }
  }

  // Exact is a subset of FTS when FTS matches all tokens; return the larger.
  return Math.max(exact, fts);
}

export async function checkUsptoMarks(
  input: Input,
  _opts: Record<string, never> = {}
): Promise<ToolResult<Output>> {
  const parsed = CheckUsptoMarksInputSchema.parse(input);

  return runTool(async () => {
    let db: Database.Database;
    try {
      db = getUsptoDb();
    } catch (e) {
      throw new ToolError(
        e instanceof Error ? e.message : String(e),
        'config'
      );
    }

    const rows = searchMarks(db, parsed.brand_name, parsed.class_code);
    const total = countLiveMatches(db, parsed.brand_name, parsed.class_code);
    const results = rows.map(rowToMark).filter((m) => m.serial_number && m.mark_name);

    const output: Output = {
      query: parsed.brand_name,
      total,
      results,
    };

    const parseResult = CheckUsptoMarksOutputSchema.safeParse(output);
    if (!parseResult.success) {
      throw new ToolError(`response failed schema validation: ${parseResult.error.message}`, 'parse');
    }
    return parseResult.data;
  });
}
