#!/usr/bin/env tsx
/**
 * Ingest TRTYRAP trademark bulk XML into a local SQLite database.
 *
 * Usage:
 *   tsx scripts/ingest_uspto.ts [--zips ../data/trtyrap/zips] [--db ../data/uspto.sqlite]
 *
 * Pipeline per zip:
 *   unzip -p <file> → sax stream parse → batched transactional insert into `marks`.
 * After all zips are processed, rolls up `applicants` from `marks` via SQL aggregates.
 *
 * The parser is deliberately forgiving: records missing mark-identification are dropped,
 * everything else is kept (status filtering happens at query time in the tool layer).
 */

import { spawn } from 'node:child_process';
import { readdirSync, statSync, existsSync, mkdirSync, unlinkSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import sax from 'sax';
import Database from 'better-sqlite3';

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(__dirname, '..', '..');

function parseArgs(): { zipsDir: string; dbPath: string; limit?: number } {
  const args = process.argv.slice(2);
  let zipsDir = resolve(REPO_ROOT, 'data', 'trtyrap', 'zips');
  let dbPath = resolve(REPO_ROOT, 'data', 'uspto.sqlite');
  let limit: number | undefined;
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === '--zips') zipsDir = resolve(args[++i]!);
    else if (a === '--db') dbPath = resolve(args[++i]!);
    else if (a === '--limit') limit = Number(args[++i]);
  }
  return { zipsDir, dbPath, ...(limit !== undefined ? { limit } : {}) };
}

interface MarkRecord {
  serial: string;
  mark: string;
  mark_norm: string;
  filing_date: string | null;
  status_code: string | null;
  status_date: string | null;
  registration_date: string | null;
  abandonment_date: string | null;
  attorney_name: string | null;
  classes: string; // comma-separated int codes, e.g. "009,025"
  owner_name: string | null;
  owner_name_norm: string | null;
  owner_country: string | null;
  owner_legal_entity: string | null;
}

function normalizeText(s: string): string {
  return s
    .toLowerCase()
    .replace(/[.,]/g, '')
    .replace(/\s+/g, ' ')
    .trim();
}

function openDb(dbPath: string): Database.Database {
  mkdirSync(dirname(dbPath), { recursive: true });
  // Fresh DB — blow away any previous ingest.
  if (existsSync(dbPath)) unlinkSync(dbPath);
  const db = new Database(dbPath);
  db.pragma('journal_mode = OFF');
  db.pragma('synchronous = OFF');
  db.pragma('temp_store = MEMORY');
  db.pragma('cache_size = -65536'); // 64 MB
  db.exec(`
    CREATE TABLE marks (
      serial TEXT PRIMARY KEY,
      mark TEXT NOT NULL,
      mark_norm TEXT NOT NULL,
      filing_date TEXT,
      status_code TEXT,
      status_date TEXT,
      registration_date TEXT,
      abandonment_date TEXT,
      attorney_name TEXT,
      classes TEXT NOT NULL,
      owner_name TEXT,
      owner_name_norm TEXT,
      owner_country TEXT,
      owner_legal_entity TEXT
    );
  `);
  return db;
}

/**
 * Streaming XML parser for one TRTYRAP file. Emits one MarkRecord per <case-file>.
 * Uses a lean state machine instead of holding records in memory.
 */
function makeParser(onRecord: (r: MarkRecord) => void): {
  write(chunk: string): void;
  end(): void;
} {
  const parser = sax.parser(true, { trim: false, normalize: false });

  let inCaseFile = false;
  let inCaseFileHeader = false;
  let inClassification = false;
  let inCaseFileOwner = false;
  let inNationality = false;

  let currentText = '';
  let cur: Partial<MarkRecord> & { _classes?: string[]; _primaryOwner?: Record<string, string | null> } = {};

  const resetCurrent = () => {
    cur = { _classes: [], _primaryOwner: null };
  };

  parser.onopentag = (tag) => {
    const name = tag.name;
    currentText = '';
    switch (name) {
      case 'case-file':
        inCaseFile = true;
        resetCurrent();
        break;
      case 'case-file-header':
        inCaseFileHeader = true;
        break;
      case 'classification':
        inClassification = true;
        break;
      case 'case-file-owner':
        inCaseFileOwner = true;
        break;
      case 'nationality':
        inNationality = true;
        break;
    }
  };

  parser.ontext = (text) => {
    currentText += text;
  };
  parser.oncdata = (text) => {
    currentText += text;
  };

  parser.onclosetag = (name) => {
    if (!inCaseFile) {
      currentText = '';
      return;
    }
    const text = currentText.trim();
    currentText = '';

    if (name === 'case-file') {
      inCaseFile = false;
      // Emit only if we have the essentials.
      if (cur.serial && cur.mark) {
        const classes = (cur._classes ?? []).join(',');
        const owner = cur._primaryOwner ?? {};
        const ownerName = (owner['party-name'] as string | null) ?? null;
        onRecord({
          serial: String(cur.serial),
          mark: String(cur.mark),
          mark_norm: normalizeText(String(cur.mark)),
          filing_date: cur.filing_date ?? null,
          status_code: cur.status_code ?? null,
          status_date: cur.status_date ?? null,
          registration_date: cur.registration_date ?? null,
          abandonment_date: cur.abandonment_date ?? null,
          attorney_name: cur.attorney_name ?? null,
          classes,
          owner_name: ownerName,
          owner_name_norm: ownerName ? normalizeText(ownerName) : null,
          owner_country: (owner['country'] as string | null) ?? null,
          owner_legal_entity: (owner['legal-entity-type-code'] as string | null) ?? null,
        });
      }
      resetCurrent();
      return;
    }
    if (name === 'case-file-header') {
      inCaseFileHeader = false;
      return;
    }
    if (name === 'classification') {
      inClassification = false;
      return;
    }
    if (name === 'case-file-owner') {
      inCaseFileOwner = false;
      return;
    }
    if (name === 'nationality') {
      inNationality = false;
      return;
    }

    // Capture serial at case-file level (not in header).
    if (name === 'serial-number' && !inCaseFileHeader && !inClassification && !inCaseFileOwner) {
      cur.serial = text;
      return;
    }

    if (inCaseFileHeader) {
      switch (name) {
        case 'filing-date':
          cur.filing_date = text;
          break;
        case 'status-code':
          cur.status_code = text;
          break;
        case 'status-date':
          cur.status_date = text;
          break;
        case 'registration-date':
          cur.registration_date = text;
          break;
        case 'abandonment-date':
          cur.abandonment_date = text;
          break;
        case 'mark-identification':
          cur.mark = text;
          break;
        case 'attorney-name':
          cur.attorney_name = text;
          break;
      }
      return;
    }

    if (inClassification) {
      if (name === 'international-code') {
        if (/^\d{1,3}$/.test(text)) {
          cur._classes!.push(text.padStart(3, '0'));
        }
      }
      return;
    }

    if (inCaseFileOwner) {
      // Prefer the first owner encountered (usually current owner).
      if (!cur._primaryOwner) cur._primaryOwner = {};
      if (name === 'party-name' && cur._primaryOwner['party-name'] == null) {
        cur._primaryOwner['party-name'] = text;
      } else if (name === 'legal-entity-type-code' && cur._primaryOwner['legal-entity-type-code'] == null) {
        cur._primaryOwner['legal-entity-type-code'] = text;
      } else if (name === 'country' && inNationality && cur._primaryOwner['country'] == null) {
        cur._primaryOwner['country'] = text;
      }
      return;
    }
  };

  return {
    write(chunk: string) {
      parser.write(chunk);
    },
    end() {
      parser.close();
    },
  };
}

async function ingestZip(
  zipPath: string,
  db: Database.Database,
  stats: { total: number; inserted: number; filesDone: number }
): Promise<void> {
  return new Promise((resolveP, rejectP) => {
    const insert = db.prepare(`
      INSERT OR IGNORE INTO marks
        (serial, mark, mark_norm, filing_date, status_code, status_date, registration_date,
         abandonment_date, attorney_name, classes, owner_name, owner_name_norm, owner_country, owner_legal_entity)
      VALUES
        (@serial, @mark, @mark_norm, @filing_date, @status_code, @status_date, @registration_date,
         @abandonment_date, @attorney_name, @classes, @owner_name, @owner_name_norm, @owner_country, @owner_legal_entity)
    `);

    const BATCH = 5000;
    let batch: MarkRecord[] = [];
    const flush = () => {
      if (batch.length === 0) return;
      const tx = db.transaction((rows: MarkRecord[]) => {
        for (const r of rows) insert.run(r);
      });
      tx(batch);
      stats.inserted += batch.length;
      batch = [];
    };

    const parser = makeParser((r) => {
      stats.total++;
      if (!r.mark) return; // skip empty-mark records
      batch.push(r);
      if (batch.length >= BATCH) flush();
    });

    const child = spawn('unzip', ['-p', zipPath], { stdio: ['ignore', 'pipe', 'pipe'] });
    let stderr = '';
    child.stderr.on('data', (d) => {
      stderr += d.toString();
    });
    child.stdout.setEncoding('utf8');
    child.stdout.on('data', (chunk: string) => {
      try {
        parser.write(chunk);
      } catch (e) {
        child.kill();
        rejectP(e);
      }
    });
    child.stdout.on('end', () => {
      try {
        parser.end();
        flush();
        stats.filesDone++;
        resolveP();
      } catch (e) {
        rejectP(e);
      }
    });
    child.on('error', rejectP);
    child.on('exit', (code) => {
      if (code !== 0 && code !== null) {
        rejectP(new Error(`unzip exited ${code}: ${stderr}`));
      }
    });
  });
}

function buildIndexesAndRollups(db: Database.Database): void {
  console.log('[ingest] building indexes...');
  db.exec(`
    CREATE INDEX idx_marks_owner_norm ON marks(owner_name_norm);
    CREATE INDEX idx_marks_mark_norm ON marks(mark_norm);
  `);

  console.log('[ingest] building FTS5 index...');
  db.exec(`
    CREATE VIRTUAL TABLE marks_fts USING fts5(
      mark,
      content='marks',
      content_rowid='rowid',
      tokenize='unicode61 remove_diacritics 2'
    );
    INSERT INTO marks_fts(rowid, mark) SELECT rowid, mark FROM marks WHERE mark IS NOT NULL;
  `);

  console.log('[ingest] rolling up applicants...');
  db.exec(`
    CREATE TABLE applicants (
      owner_name_norm TEXT PRIMARY KEY,
      display_name TEXT NOT NULL,
      filing_count_total INTEGER NOT NULL,
      filing_count_2yr INTEGER NOT NULL,
      abandonment_rate REAL NOT NULL,
      cancellation_rate REAL NOT NULL,
      first_filing_date TEXT,
      last_filing_date TEXT,
      is_individual INTEGER NOT NULL,
      is_foreign INTEGER NOT NULL,
      attorney_of_record TEXT,
      attorney_case_count INTEGER NOT NULL,
      attorney_cancellation_rate REAL NOT NULL
    );
  `);

  // filing_date is stored as YYYYMMDD; compare lexically.
  const now = new Date();
  const twoYearsAgo = new Date(now.getTime() - 2 * 365 * 24 * 60 * 60 * 1000);
  const twoYearsAgoStr =
    twoYearsAgo.getFullYear().toString() +
    String(twoYearsAgo.getMonth() + 1).padStart(2, '0') +
    String(twoYearsAgo.getDate()).padStart(2, '0');

  db.exec(`
    INSERT INTO applicants
      (owner_name_norm, display_name, filing_count_total, filing_count_2yr, abandonment_rate, cancellation_rate,
       first_filing_date, last_filing_date, is_individual, is_foreign,
       attorney_of_record, attorney_case_count, attorney_cancellation_rate)
    SELECT
      owner_name_norm,
      MIN(owner_name) AS display_name,
      COUNT(*) AS filing_count_total,
      SUM(CASE WHEN filing_date >= '${twoYearsAgoStr}' THEN 1 ELSE 0 END) AS filing_count_2yr,
      CAST(SUM(CASE WHEN status_code >= '600' AND status_code < '700' THEN 1 ELSE 0 END) AS REAL) / COUNT(*) AS abandonment_rate,
      CAST(SUM(CASE WHEN status_code >= '700' AND status_code < '800' THEN 1 ELSE 0 END) AS REAL) / COUNT(*) AS cancellation_rate,
      MIN(filing_date) AS first_filing_date,
      MAX(filing_date) AS last_filing_date,
      CASE WHEN MAX(CASE WHEN owner_legal_entity = '01' THEN 1 ELSE 0 END) = 1 THEN 1 ELSE 0 END AS is_individual,
      CASE WHEN MIN(CASE WHEN owner_country = 'US' OR owner_country IS NULL THEN 1 ELSE 0 END) = 0 THEN 1 ELSE 0 END AS is_foreign,
      (SELECT attorney_name FROM marks m2
        WHERE m2.owner_name_norm = m1.owner_name_norm
          AND m2.attorney_name IS NOT NULL
        ORDER BY m2.filing_date DESC LIMIT 1) AS attorney_of_record,
      0 AS attorney_case_count,
      0.0 AS attorney_cancellation_rate
    FROM marks m1
    WHERE owner_name_norm IS NOT NULL
    GROUP BY owner_name_norm;
  `);

  console.log('[ingest] computing attorney rollups...');
  db.exec(`
    CREATE INDEX idx_marks_attorney ON marks(attorney_name);
  `);
  const attyStats = db.prepare(`
    SELECT
      COUNT(*) AS case_count,
      CAST(SUM(CASE WHEN status_code >= '700' AND status_code < '800' THEN 1 ELSE 0 END) AS REAL) / COUNT(*) AS cancel_rate
    FROM marks WHERE attorney_name = ?
  `);
  const updateApplicant = db.prepare(`
    UPDATE applicants SET attorney_case_count = ?, attorney_cancellation_rate = ? WHERE owner_name_norm = ?
  `);
  const applicantsWithAtty = db
    .prepare(`SELECT owner_name_norm, attorney_of_record FROM applicants WHERE attorney_of_record IS NOT NULL`)
    .all() as Array<{ owner_name_norm: string; attorney_of_record: string }>;

  const attyCache = new Map<string, { case_count: number; cancel_rate: number }>();
  const updateTx = db.transaction(() => {
    for (const row of applicantsWithAtty) {
      let s = attyCache.get(row.attorney_of_record);
      if (!s) {
        s = attyStats.get(row.attorney_of_record) as { case_count: number; cancel_rate: number };
        attyCache.set(row.attorney_of_record, s);
      }
      updateApplicant.run(s.case_count, s.cancel_rate ?? 0, row.owner_name_norm);
    }
  });
  updateTx();
}

async function main(): Promise<void> {
  const { zipsDir, dbPath, limit } = parseArgs();
  console.log(`[ingest] zipsDir=${zipsDir}`);
  console.log(`[ingest] dbPath=${dbPath}`);
  if (!existsSync(zipsDir)) {
    console.error(`[ingest] missing zips dir: ${zipsDir}`);
    process.exit(1);
  }
  const zips = readdirSync(zipsDir)
    .filter((f) => f.endsWith('.zip'))
    .map((f) => join(zipsDir, f))
    .sort();
  console.log(`[ingest] found ${zips.length} zip files`);

  const db = openDb(dbPath);
  const stats = { total: 0, inserted: 0, filesDone: 0 };
  const target = limit !== undefined ? Math.min(limit, zips.length) : zips.length;

  const globalStart = Date.now();
  for (let i = 0; i < target; i++) {
    const zip = zips[i]!;
    const zipSize = statSync(zip).size;
    const start = Date.now();
    try {
      await ingestZip(zip, db, stats);
    } catch (e) {
      console.error(`[ingest] FAIL ${zip}: ${e instanceof Error ? e.message : String(e)}`);
      continue;
    }
    const elapsed = ((Date.now() - start) / 1000).toFixed(1);
    console.log(
      `[ingest] ${i + 1}/${target} ${zip.split('/').pop()} zipSize=${(zipSize / 1e6).toFixed(1)}MB elapsed=${elapsed}s inserted=${stats.inserted}`
    );
  }

  buildIndexesAndRollups(db);

  const markCount = db.prepare('SELECT COUNT(*) AS c FROM marks').get() as { c: number };
  const applicantCount = db.prepare('SELECT COUNT(*) AS c FROM applicants').get() as { c: number };
  const totalSeconds = ((Date.now() - globalStart) / 1000).toFixed(1);
  console.log(`\n[ingest] DONE in ${totalSeconds}s — ${markCount.c} marks, ${applicantCount.c} applicants`);
  db.close();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
