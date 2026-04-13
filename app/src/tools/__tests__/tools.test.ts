import { strict as assert } from 'node:assert';
import { checkUsptoMarks } from '../check_uspto_marks.js';
import { checkDomainAge } from '../check_domain_age.js';
import { webSearch } from '../web_search.js';
import { checkAttorney } from '../check_attorney.js';
import { lookupApplicantHistory } from '../lookup_applicant_history.js';
import {
  CheckUsptoMarksOutputSchema,
  CheckDomainAgeOutputSchema,
  WebSearchOutputSchema,
  CheckAttorneyOutputSchema,
  ApplicantHistorySchema,
} from '../../schemas/index.js';

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
  console.log('check_uspto_marks');

  await test('returns parsed results on happy path', async () => {
    const fetchImpl = mockFetch(() => ({
      status: 200,
      body: JSON.stringify({
        total: 1,
        results: [
          {
            serialNumber: '12345678',
            markLiteralElement: 'NIKE',
            ownerName: 'Nike Inc',
            filingDate: '2001-01-01',
            statusDescription: 'Registered',
            internationalClass: [25, 28],
          },
        ],
      }),
    }));
    const r = await checkUsptoMarks(
      { brand_name: 'Nike', class_code: 25 },
      { fetchImpl, baseUrl: 'https://example.test/search' }
    );
    assert.equal(r.ok, true);
    if (!r.ok) throw new Error('unreachable');
    CheckUsptoMarksOutputSchema.parse(r.data);
    assert.equal(r.data.results.length, 1);
    assert.equal(r.data.results[0]!.mark_name, 'NIKE');
    assert.deepEqual(r.data.results[0]!.class_codes, [25, 28]);
  });

  await test('returns empty array for nonsense mark', async () => {
    const fetchImpl = mockFetch(() => ({
      status: 200,
      body: JSON.stringify({ total: 0, results: [] }),
    }));
    const r = await checkUsptoMarks(
      { brand_name: 'xyzzy99999abc' },
      { fetchImpl, baseUrl: 'https://example.test/search' }
    );
    assert.equal(r.ok, true);
    if (!r.ok) throw new Error('unreachable');
    assert.equal(r.data.results.length, 0);
    assert.equal(r.data.total, 0);
  });

  await test('handles timeout without throwing', async () => {
    const r = await checkUsptoMarks(
      { brand_name: 'Nike' },
      { fetchImpl: timeoutFetch(), baseUrl: 'https://example.test/search', timeoutMs: 50 }
    );
    assert.equal(r.ok, false);
    if (r.ok) throw new Error('unreachable');
    assert.equal(r.error.code, 'timeout');
  });

  await test('handles network failure without throwing', async () => {
    const r = await checkUsptoMarks(
      { brand_name: 'Nike' },
      { fetchImpl: erroringFetch('ECONNRESET'), baseUrl: 'https://example.test/search' }
    );
    assert.equal(r.ok, false);
    if (r.ok) throw new Error('unreachable');
    assert.equal(r.error.code, 'network');
  });

  await test('handles HTTP 500 without throwing', async () => {
    const fetchImpl = mockFetch(() => ({ status: 500, body: 'boom' }));
    const r = await checkUsptoMarks(
      { brand_name: 'Nike' },
      { fetchImpl, baseUrl: 'https://example.test/search' }
    );
    assert.equal(r.ok, false);
    if (r.ok) throw new Error('unreachable');
    assert.equal(r.error.code, 'http');
    assert.equal(r.error.status, 500);
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

  console.log('\ncheck_attorney');

  await test('detects active status and no discipline', async () => {
    const html = '<html><body>Name: Jane Doe. Status: Active, in good standing.</body></html>';
    const fetchImpl = (async () =>
      new Response(html, { status: 200, headers: { 'content-type': 'text/html' } })) as typeof fetch;
    const r = await checkAttorney(
      { attorney_name: 'Jane Doe', bar_number: '12345' },
      { fetchImpl, url: 'https://example.test/oed' }
    );
    assert.equal(r.ok, true);
    if (!r.ok) throw new Error('unreachable');
    CheckAttorneyOutputSchema.parse(r.data);
    assert.equal(r.data.found, true);
    assert.equal(r.data.bar_status, 'active');
    assert.equal(r.data.disciplinary_history, false);
  });

  await test('detects not_found', async () => {
    const html = '<html><body>No results found for your query.</body></html>';
    const fetchImpl = (async () =>
      new Response(html, { status: 200, headers: { 'content-type': 'text/html' } })) as typeof fetch;
    const r = await checkAttorney(
      { attorney_name: 'Fake Name 99999' },
      { fetchImpl, url: 'https://example.test/oed' }
    );
    assert.equal(r.ok, true);
    if (!r.ok) throw new Error('unreachable');
    assert.equal(r.data.found, false);
    assert.equal(r.data.bar_status, 'unknown');
  });

  await test('detects suspended + disciplinary history', async () => {
    const html =
      '<html>Practitioner: Suspended on 2020-01-01 after disciplinary action (censure).</html>';
    const fetchImpl = (async () =>
      new Response(html, { status: 200, headers: { 'content-type': 'text/html' } })) as typeof fetch;
    const r = await checkAttorney(
      { attorney_name: 'Bad Lawyer' },
      { fetchImpl, url: 'https://example.test/oed' }
    );
    assert.equal(r.ok, true);
    if (!r.ok) throw new Error('unreachable');
    assert.equal(r.data.bar_status, 'suspended');
    assert.equal(r.data.disciplinary_history, true);
  });

  await test('handles scraping failure without throwing', async () => {
    const r = await checkAttorney(
      { attorney_name: 'Jane' },
      { fetchImpl: erroringFetch('dns fail'), url: 'https://example.test/oed' }
    );
    assert.equal(r.ok, false);
  });

  console.log('\nlookup_applicant_history');

  await test('returns fixture for known blue-chip applicant', async () => {
    const r = await lookupApplicantHistory(
      { applicant_name: 'Apple Inc.' },
      { disableLive: true }
    );
    assert.equal(r.ok, true);
    if (!r.ok) throw new Error('unreachable');
    ApplicantHistorySchema.parse(r.data);
    assert.equal(r.data.source, 'fixture');
    assert.equal(r.data.found, true);
    assert.ok(r.data.filing_count_total > 10000);
    assert.ok(r.data.abandonment_rate < 0.1);
  });

  await test('returns fixture for known troll applicant', async () => {
    const r = await lookupApplicantHistory(
      { applicant_name: 'leo stoller' },
      { disableLive: true }
    );
    assert.equal(r.ok, true);
    if (!r.ok) throw new Error('unreachable');
    assert.equal(r.data.source, 'fixture');
    assert.ok(r.data.abandonment_rate > 0.5);
    assert.equal(r.data.attorney_of_record, null);
  });

  await test('normalizes punctuation when matching fixtures', async () => {
    const r = await lookupApplicantHistory(
      { applicant_name: 'NIKE, INC' },
      { disableLive: true }
    );
    assert.equal(r.ok, true);
    if (!r.ok) throw new Error('unreachable');
    assert.equal(r.data.source, 'fixture');
    assert.ok(r.data.found);
  });

  await test('returns unknown-applicant shape when live disabled and no fixture', async () => {
    const r = await lookupApplicantHistory(
      { applicant_name: 'Nonexistent Holdings ZZZ' },
      { disableLive: true }
    );
    assert.equal(r.ok, true);
    if (!r.ok) throw new Error('unreachable');
    assert.equal(r.data.source, 'unknown');
    assert.equal(r.data.found, false);
    assert.equal(r.data.filing_count_total, 0);
  });

  await test('derives stats from live response when no fixture', async () => {
    const fetchImpl = mockFetch(() => ({
      status: 200,
      body: JSON.stringify({
        total: 4,
        results: [
          { serialNumber: '1', filingDate: '2025-01-01', abandoned: true },
          { serialNumber: '2', filingDate: '2025-06-10', abandoned: false },
          { serialNumber: '3', filingDate: '2024-03-01', cancelled: true },
          { serialNumber: '4', filingDate: '2022-11-15', abandoned: false },
        ],
      }),
    }));
    const r = await lookupApplicantHistory(
      { applicant_name: 'Somename Holdings' },
      { fetchImpl, baseUrl: 'https://example.test/owner' }
    );
    assert.equal(r.ok, true);
    if (!r.ok) throw new Error('unreachable');
    ApplicantHistorySchema.parse(r.data);
    assert.equal(r.data.source, 'live');
    assert.equal(r.data.filing_count_total, 4);
    assert.ok(r.data.abandonment_rate > 0 && r.data.abandonment_rate < 1);
  });

  await test('falls back to unknown-applicant on network error for non-fixture', async () => {
    const r = await lookupApplicantHistory(
      { applicant_name: 'Unknown Co' },
      { fetchImpl: erroringFetch('dns fail'), baseUrl: 'https://example.test/owner' }
    );
    assert.equal(r.ok, true);
    if (!r.ok) throw new Error('unreachable');
    assert.equal(r.data.source, 'unknown');
  });

  console.log(`\n${passed} passed, ${failed} failed`);
  if (failed > 0) process.exit(1);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
