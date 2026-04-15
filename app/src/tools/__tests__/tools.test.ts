import { strict as assert } from 'node:assert';
import Database from 'better-sqlite3';
import { checkUsptoMarks } from '../check_uspto_marks.js';
import { checkDomainAge } from '../check_domain_age.js';
import { webSearch } from '../web_search.js';
import { lookupApplicantHistory } from '../lookup_applicant_history.js';
import { setUsptoDbForTest } from '../../lib/uspto_db.js';
import {
  CheckUsptoMarksOutputSchema,
  CheckDomainAgeOutputSchema,
  WebSearchOutputSchema,
  ApplicantHistorySchema,
} from '../../schemas/index.js';

/**
 * Build an in-memory USPTO SQLite fixture mirroring the production schema so
 * check_uspto_marks and lookup_applicant_history can be exercised without
 * touching the 12GB real DB.
 */
function buildTestDb(): Database.Database {
  const db = new Database(':memory:');
  db.pragma('journal_mode = OFF');
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
    CREATE VIRTUAL TABLE marks_fts USING fts5(mark, content='marks', content_rowid='rowid', tokenize='unicode61 remove_diacritics 2');
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

  const insertMark = db.prepare(`
    INSERT INTO marks (serial, mark, mark_norm, filing_date, status_code, classes, owner_name, owner_name_norm)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
  `);
  insertMark.run('12345678', 'NIKE', 'nike', '20010101', '800', '025,028', 'Nike Inc', 'nike inc');
  insertMark.run('22345678', 'NIKE PRO', 'nike pro', '20200101', '800', '025', 'Nike Inc', 'nike inc');
  insertMark.run('33345678', 'DEADMARK', 'deadmark', '19950101', '710', '025', 'Old Co', 'old co');
  db.exec(`INSERT INTO marks_fts(rowid, mark) SELECT rowid, mark FROM marks`);

  const insertApplicant = db.prepare(`
    INSERT INTO applicants (owner_name_norm, display_name, filing_count_total, filing_count_2yr,
      abandonment_rate, cancellation_rate, first_filing_date, last_filing_date,
      is_individual, is_foreign, attorney_of_record, attorney_case_count, attorney_cancellation_rate)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `);
  insertApplicant.run('nike inc', 'Nike Inc', 9200, 310, 0.03, 0.008, '19710618', '20250101', 0, 0, 'Helen Hill Minsker', 2100, 0.009);
  insertApplicant.run('hardkoo wu', 'Hardkoo Wu', 222, 180, 1.0, 0.0, '20220101', '20250101', 1, 0, null, 0, 0);
  return db;
}

let passed = 0;
let failed = 0;
async function test(name: string, fn: () => Promise<void> | void): Promise<void> {
  try {
    await fn();
    passed++;
    console.log(`  PASS  ${name}`);
  } catch (e) {
    failed++;
    console.log(`  FAIL  ${name}\n        ${e instanceof Error ? e.message : String(e)}`);
  }
}

type FetchInput = Parameters<typeof fetch>[0];
type FetchInit = Parameters<typeof fetch>[1];

function mockFetch(
  responder: (url: string, init?: FetchInit) => { status: number; body: string }
): typeof fetch {
  return (async (input: FetchInput, init?: FetchInit) => {
    const url = typeof input === 'string' ? input : input.toString();
    if (init?.signal?.aborted) throw new DOMException('aborted', 'AbortError');
    const { status, body } = responder(url, init);
    return new Response(body, { status, headers: { 'content-type': 'application/json' } });
  }) as typeof fetch;
}

function timeoutFetch(): typeof fetch {
  return ((_input: FetchInput, init?: FetchInit) =>
    new Promise((_resolve, reject) => {
      init?.signal?.addEventListener('abort', () =>
        reject(new DOMException('aborted', 'AbortError'))
      );
    })) as typeof fetch;
}

function erroringFetch(msg: string): typeof fetch {
  return (async () => {
    throw new TypeError(msg);
  }) as typeof fetch;
}

async function main() {
  const testDb = buildTestDb();
  setUsptoDbForTest(testDb);

  console.log('check_uspto_marks');

  await test('returns exact-normalized match', async () => {
    const r = await checkUsptoMarks({ brand_name: 'Nike', class_code: 25 });
    assert.equal(r.ok, true);
    if (!r.ok) throw new Error('unreachable');
    CheckUsptoMarksOutputSchema.parse(r.data);
    const serials = r.data.results.map((m) => m.serial_number);
    assert.ok(serials.includes('12345678'));
    const nike = r.data.results.find((m) => m.serial_number === '12345678')!;
    assert.equal(nike.mark_name, 'NIKE');
    assert.deepEqual(nike.class_codes, [25, 28]);
    assert.equal(nike.filing_date, '2001-01-01');
  });

  await test('excludes dead marks (status 700-799) from results', async () => {
    const r = await checkUsptoMarks({ brand_name: 'deadmark' });
    assert.equal(r.ok, true);
    if (!r.ok) throw new Error('unreachable');
    assert.equal(r.data.total, 0);
    assert.equal(r.data.results.length, 0);
  });

  await test('returns empty array for nonsense mark', async () => {
    const r = await checkUsptoMarks({ brand_name: 'xyzzy99999abc' });
    assert.equal(r.ok, true);
    if (!r.ok) throw new Error('unreachable');
    assert.equal(r.data.results.length, 0);
    assert.equal(r.data.total, 0);
  });

  await test('filters by nice class when requested', async () => {
    const r = await checkUsptoMarks({ brand_name: 'Nike', class_code: 42 });
    assert.equal(r.ok, true);
    if (!r.ok) throw new Error('unreachable');
    // no nike rows in class 042 → zero results (FTS may still match but class filter excludes)
    assert.equal(r.data.results.length, 0);
  });

  await test('FTS5 phrase match finds multi-word marks', async () => {
    const r = await checkUsptoMarks({ brand_name: 'nike pro' });
    assert.equal(r.ok, true);
    if (!r.ok) throw new Error('unreachable');
    assert.ok(r.data.results.some((m) => m.serial_number === '22345678'));
  });

  await test('throws config error when db is missing', async () => {
    setUsptoDbForTest(null);
    const saved = process.env.USPTO_DB;
    process.env.USPTO_DB = '/nonexistent/path/uspto.sqlite';
    const r = await checkUsptoMarks({ brand_name: 'Nike' });
    if (saved === undefined) delete process.env.USPTO_DB;
    else process.env.USPTO_DB = saved;
    setUsptoDbForTest(testDb);
    assert.equal(r.ok, false);
    if (r.ok) throw new Error('unreachable');
    assert.equal(r.error.code, 'config');
  });

  console.log('\ncheck_domain_age');

  await test('computes age_days from registration event', async () => {
    const tenYearsAgo = new Date(Date.now() - 10 * 365 * 86_400_000).toISOString();
    const fetchImpl = mockFetch(() => ({
      status: 200,
      body: JSON.stringify({
        events: [{ eventAction: 'registration', eventDate: tenYearsAgo }],
      }),
    }));
    const r = await checkDomainAge(
      { domain_name: 'google.com' },
      { fetchImpl, baseUrl: 'https://example.test/rdap/' }
    );
    assert.equal(r.ok, true);
    if (!r.ok) throw new Error('unreachable');
    CheckDomainAgeOutputSchema.parse(r.data);
    assert.equal(r.data.exists, true);
    assert.ok(r.data.age_days !== null && r.data.age_days > 3000);
  });

  await test('returns exists=false on 404', async () => {
    const fetchImpl = mockFetch(() => ({ status: 404, body: JSON.stringify({}) }));
    const r = await checkDomainAge(
      { domain_name: 'nonexistent.xyz' },
      { fetchImpl, baseUrl: 'https://example.test/rdap/' }
    );
    assert.equal(r.ok, true);
    if (!r.ok) throw new Error('unreachable');
    assert.equal(r.data.exists, false);
    assert.equal(r.data.registered_at, null);
    assert.equal(r.data.age_days, null);
  });

  await test('handles invalid date as parse error', async () => {
    const fetchImpl = mockFetch(() => ({
      status: 200,
      body: JSON.stringify({ events: [{ eventAction: 'registration', eventDate: 'not-a-date' }] }),
    }));
    const r = await checkDomainAge(
      { domain_name: 'weird.com' },
      { fetchImpl, baseUrl: 'https://example.test/rdap/' }
    );
    assert.equal(r.ok, false);
    if (r.ok) throw new Error('unreachable');
    assert.equal(r.error.code, 'parse');
  });

  await test('handles network error without throwing', async () => {
    const r = await checkDomainAge(
      { domain_name: 'google.com' },
      { fetchImpl: erroringFetch('dns fail'), baseUrl: 'https://example.test/rdap/' }
    );
    assert.equal(r.ok, false);
  });

  console.log('\nweb_search');

  await test('returns mapped serper results', async () => {
    const fetchImpl = mockFetch(() => ({
      status: 200,
      body: JSON.stringify({
        organic: [
          { title: 'Nike', link: 'https://nike.com', snippet: 'Just do it' },
          { title: 'Nike Shoes', link: 'https://nike.com/shoes', snippet: 'Shop shoes' },
          { title: 'Nike Wiki', link: 'https://en.wikipedia.org/wiki/Nike', snippet: 'Company' },
        ],
      }),
    }));
    const r = await webSearch(
      { query: 'Nike shoes', max_results: 5 },
      { fetchImpl, url: 'https://example.test/search', apiKey: 'k' }
    );
    assert.equal(r.ok, true);
    if (!r.ok) throw new Error('unreachable');
    WebSearchOutputSchema.parse(r.data);
    assert.equal(r.data.results.length, 3);
    for (const res of r.data.results) {
      assert.ok(res.title && res.url && res.snippet);
    }
  });

  await test('returns empty results gracefully', async () => {
    const fetchImpl = mockFetch(() => ({ status: 200, body: JSON.stringify({ organic: [] }) }));
    const r = await webSearch(
      { query: 'xyzzy' },
      { fetchImpl, url: 'https://example.test/search', apiKey: 'k' }
    );
    assert.equal(r.ok, true);
    if (!r.ok) throw new Error('unreachable');
    assert.equal(r.data.results.length, 0);
  });

  await test('returns config error when API key missing', async () => {
    const prev = process.env.SERPER_API_KEY;
    delete process.env.SERPER_API_KEY;
    try {
      const r = await webSearch(
        { query: 'Nike' },
        { fetchImpl: mockFetch(() => ({ status: 200, body: '{}' })), url: 'https://example.test/search' }
      );
      assert.equal(r.ok, false);
      if (r.ok) throw new Error('unreachable');
      assert.equal(r.error.code, 'config');
    } finally {
      if (prev !== undefined) process.env.SERPER_API_KEY = prev;
    }
  });

  await test('handles HTTP 401 (bad key) without throwing', async () => {
    const fetchImpl = mockFetch(() => ({ status: 401, body: 'unauthorized' }));
    const r = await webSearch(
      { query: 'Nike' },
      { fetchImpl, url: 'https://example.test/search', apiKey: 'bad' }
    );
    assert.equal(r.ok, false);
    if (r.ok) throw new Error('unreachable');
    assert.equal(r.error.status, 401);
  });

  console.log('\nlookup_applicant_history');

  await test('returns live SQLite row for known blue-chip applicant', async () => {
    const r = await lookupApplicantHistory({ applicant_name: 'Nike Inc' });
    assert.equal(r.ok, true);
    if (!r.ok) throw new Error('unreachable');
    ApplicantHistorySchema.parse(r.data);
    assert.equal(r.data.source, 'live');
    assert.equal(r.data.found, true);
    assert.equal(r.data.filing_count_total, 9200);
    assert.ok(r.data.abandonment_rate < 0.1);
    assert.equal(r.data.attorney_of_record, 'Helen Hill Minsker');
    assert.equal(r.data.first_filing_date, '1971-06-18');
  });

  await test('returns live SQLite row for troll-style individual filer', async () => {
    const r = await lookupApplicantHistory({ applicant_name: 'hardkoo wu' });
    assert.equal(r.ok, true);
    if (!r.ok) throw new Error('unreachable');
    assert.equal(r.data.source, 'live');
    assert.equal(r.data.is_individual, true);
    assert.equal(r.data.abandonment_rate, 1.0);
    assert.equal(r.data.attorney_of_record, null);
  });

  await test('normalizes punctuation when matching owner_name_norm', async () => {
    const r = await lookupApplicantHistory({ applicant_name: 'NIKE, INC' });
    assert.equal(r.ok, true);
    if (!r.ok) throw new Error('unreachable');
    assert.equal(r.data.source, 'live');
    assert.ok(r.data.found);
  });

  await test('returns unknown-applicant shape for arbitrary unseen name', async () => {
    const r = await lookupApplicantHistory({ applicant_name: 'Nonexistent Holdings ZZZ' });
    assert.equal(r.ok, true);
    if (!r.ok) throw new Error('unreachable');
    assert.equal(r.data.source, 'unknown');
    assert.equal(r.data.found, false);
    assert.equal(r.data.filing_count_total, 0);
  });

  setUsptoDbForTest(null);

  console.log(`\n${passed} passed, ${failed} failed`);
  if (failed > 0) process.exit(1);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
