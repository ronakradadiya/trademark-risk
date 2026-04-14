import Database from 'better-sqlite3';
import { existsSync } from 'node:fs';
import { resolve } from 'node:path';

let cached: Database.Database | null = null;

/**
 * Resolve the USPTO SQLite database path. Checks the USPTO_DB env var first,
 * then falls back to `<cwd>/../data/uspto.sqlite` (i.e. repo-root/data from the
 * app dir). Throws if the file does not exist — callers that want graceful
 * degradation should catch and emit a tool-level error.
 */
export function resolveDbPath(): string {
  const override = process.env.USPTO_DB;
  if (override) return resolve(override);
  return resolve(process.cwd(), '..', 'data', 'uspto.sqlite');
}

export function getUsptoDb(): Database.Database {
  if (cached) return cached;
  const path = resolveDbPath();
  if (!existsSync(path)) {
    throw new Error(
      `uspto db missing at ${path}. Run \`npx tsx scripts/ingest_uspto.ts\` to build it.`
    );
  }
  cached = new Database(path, { readonly: true, fileMustExist: true });
  cached.pragma('journal_mode = OFF');
  cached.pragma('query_only = ON');
  return cached;
}

/**
 * Test-only hook: inject a pre-opened DB instance (e.g. an in-memory test
 * fixture) so tools can be exercised without touching disk.
 */
export function setUsptoDbForTest(db: Database.Database | null): void {
  cached = db;
}

/**
 * Normalize a free-form search string for FTS5 MATCH. FTS5 treats punctuation
 * as separators; we quote each token so users entering "AirTag Pro" get a
 * phrase-match rather than a syntax error when the input contains operators.
 */
export function toFtsQuery(input: string): string {
  const tokens = input
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter((t) => t.length >= 2);
  if (tokens.length === 0) return '';
  return tokens.map((t) => `"${t}"`).join(' ');
}

/**
 * Build a case-insensitive LIKE pattern for owner-name lookups. We strip
 * punctuation and collapse whitespace so "Apple, Inc." matches "Apple Inc".
 */
export function normalizeOwnerName(name: string): string {
  return name
    .toLowerCase()
    .replace(/[.,]/g, '')
    .replace(/\s+/g, ' ')
    .trim();
}
