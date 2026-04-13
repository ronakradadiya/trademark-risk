import { strict as assert } from 'node:assert';
import { ZodError } from 'zod';
import {
  CheckRequestSchema,
  VerdictSchema,
  MLPredictionSchema,
  PolicyResultSchema,
  AuditRecordSchema,
  CheckUsptoMarksOutputSchema,
  CheckDomainAgeOutputSchema,
  CheckAttorneyOutputSchema,
  WebSearchOutputSchema,
  type CheckRequest,
  type Verdict,
  type AuditRecord,
} from '../index.js';

let passed = 0;
let failed = 0;

function test(name: string, fn: () => void): void {
  try {
    fn();
    passed++;
    console.log(`  PASS  ${name}`);
  } catch (e) {
    failed++;
    const msg = e instanceof Error ? e.message : String(e);
    console.log(`  FAIL  ${name}\n        ${msg}`);
  }
}

function expectThrows(fn: () => unknown, schemaName: string): void {
  try {
    fn();
  } catch (e) {
    assert.ok(e instanceof ZodError, `${schemaName} should throw ZodError`);
    return;
  }
  throw new Error(`${schemaName} did not throw`);
}

console.log('CheckRequestSchema');
test('accepts minimal valid input', () => {
  const r: CheckRequest = CheckRequestSchema.parse({ brand_name: 'Nike' });
  assert.equal(r.brand_name, 'Nike');
});
test('accepts full valid input', () => {
  CheckRequestSchema.parse({
    brand_name: 'BrewBox Coffee',
    domain_name: 'brewbox.com',
    attorney_name: 'Jane Doe',
    attorney_bar_number: 'CA-12345',
    class_code: 30,
  });
});
test('rejects empty brand_name', () => {
  expectThrows(() => CheckRequestSchema.parse({ brand_name: '' }), 'CheckRequest');
});
test('rejects brand_name over 200 chars', () => {
  expectThrows(
    () => CheckRequestSchema.parse({ brand_name: 'x'.repeat(201) }),
    'CheckRequest'
  );
});
test('rejects class_code out of range', () => {
  expectThrows(() => CheckRequestSchema.parse({ brand_name: 'A', class_code: 99 }), 'CheckRequest');
});

console.log('\nMLPredictionSchema');
test('accepts valid prediction', () => {
  MLPredictionSchema.parse({
    score: 0.42,
    tier: 'mid',
    high_threshold: 0.7,
    low_threshold: 0.08,
    model_version: 'v4',
  });
});
test('rejects score > 1', () => {
  expectThrows(
    () =>
      MLPredictionSchema.parse({
        score: 1.5,
        tier: 'high',
        high_threshold: 0.7,
        low_threshold: 0.08,
        model_version: 'v4',
      }),
    'MLPrediction'
  );
});
test('rejects unknown tier', () => {
  expectThrows(
    () =>
      MLPredictionSchema.parse({
        score: 0.5,
        tier: 'medium',
        high_threshold: 0.7,
        low_threshold: 0.08,
        model_version: 'v4',
      }),
    'MLPrediction'
  );
});

console.log('\nPolicyResultSchema');
test('accepts valid policy result', () => {
  PolicyResultSchema.parse({ triggered: true, confidence: 0.9, reason: 'troll pattern' });
});
test('rejects confidence outside 0-1', () => {
  expectThrows(
    () => PolicyResultSchema.parse({ triggered: true, confidence: 1.2, reason: 'x' }),
    'PolicyResult'
  );
});

const validVerdict: Verdict = {
  brand: 'BrewBox Coffee',
  verdict: 'review',
  overall_confidence: 0.55,
  ml: {
    score: 0.42,
    tier: 'mid',
    high_threshold: 0.7,
    low_threshold: 0.08,
    model_version: 'v4',
  },
  source: 'ml_and_agent',
  policies: {
    P1: { triggered: false, confidence: 0.1, reason: 'no similar marks' },
    P2: { triggered: false, confidence: 0.2, reason: 'single filing history' },
    P3: { triggered: true, confidence: 0.6, reason: 'domain 10 days old' },
    P4: { triggered: false, confidence: 0.05, reason: 'attorney active' },
    P5: { triggered: false, confidence: 0.3, reason: 'light web presence' },
  },
  tools_used: ['check_uspto_marks', 'check_domain_age'],
  trace: [],
  summary: 'Review suggested due to recent domain registration.',
  checked_at: '2026-04-12T10:00:00.000Z',
};

console.log('\nVerdictSchema');
test('accepts valid verdict', () => {
  VerdictSchema.parse(validVerdict);
});
test('rejects confidence > 1', () => {
  expectThrows(
    () => VerdictSchema.parse({ ...validVerdict, overall_confidence: 1.5 }),
    'Verdict'
  );
});
test('rejects unknown verdict value', () => {
  expectThrows(
    () => VerdictSchema.parse({ ...validVerdict, verdict: 'probably_bad' }),
    'Verdict'
  );
});
test('rejects missing policy', () => {
  const { P1: _p1, ...rest } = validVerdict.policies;
  expectThrows(
    () => VerdictSchema.parse({ ...validVerdict, policies: rest }),
    'Verdict'
  );
});
test('rejects non-ISO checked_at', () => {
  expectThrows(
    () => VerdictSchema.parse({ ...validVerdict, checked_at: 'yesterday' }),
    'Verdict'
  );
});

console.log('\nAuditRecordSchema');
test('accepts valid audit record', () => {
  const record: AuditRecord = AuditRecordSchema.parse({
    ...validVerdict,
    id: '11111111-1111-4111-8111-111111111111',
    ttl: 1_800_000_000,
  });
  assert.equal(record.brand, 'BrewBox Coffee');
});
test('rejects non-uuid id', () => {
  expectThrows(
    () => AuditRecordSchema.parse({ ...validVerdict, id: 'not-a-uuid', ttl: 123 }),
    'AuditRecord'
  );
});

console.log('\nTool I/O schemas');
test('CheckUsptoMarksOutput accepts valid', () => {
  CheckUsptoMarksOutputSchema.parse({
    query: 'Nike',
    total: 1,
    results: [{ serial_number: '12345', mark_name: 'NIKE', class_codes: [25] }],
  });
});
test('CheckDomainAgeOutput accepts null age', () => {
  CheckDomainAgeOutputSchema.parse({
    domain_name: 'nonexistent.xyz',
    exists: false,
    registered_at: null,
    age_days: null,
  });
});
test('CheckAttorneyOutput accepts valid', () => {
  CheckAttorneyOutputSchema.parse({
    found: true,
    name: 'Jane Doe',
    bar_status: 'active',
    disciplinary_history: false,
  });
});
test('CheckAttorneyOutput rejects unknown bar_status', () => {
  expectThrows(
    () =>
      CheckAttorneyOutputSchema.parse({
        found: true,
        name: 'X',
        bar_status: 'retired',
        disciplinary_history: false,
      }),
    'CheckAttorneyOutput'
  );
});
test('WebSearchOutput rejects invalid url', () => {
  expectThrows(
    () =>
      WebSearchOutputSchema.parse({
        query: 'x',
        results: [{ title: 't', url: 'not-a-url', snippet: 's' }],
      }),
    'WebSearchOutput'
  );
});

console.log(`\n${passed} passed, ${failed} failed`);
if (failed > 0) process.exit(1);
