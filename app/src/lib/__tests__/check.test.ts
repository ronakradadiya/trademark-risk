import { strict as assert } from 'node:assert';
import { handleCheck } from '../check.js';
import {
  VerdictSchema,
  type MLPrediction,
  type ApplicantHistory,
} from '../../schemas/index.js';
import type { FeatureVector } from '../classifier.js';
import type { AuditStore } from '../dynamo.js';

let passed = 0;
let failed = 0;
async function test(name: string, fn: () => Promise<void> | void): Promise<void> {
  try {
    await fn();
    passed++;
    console.log(`  PASS  ${name}`);
  } catch (e) {
    failed++;
    console.log(`  FAIL  ${name}\n        ${e instanceof Error ? e.stack ?? e.message : String(e)}`);
  }
}

function stubClassifier(score: number): { predict: (f: FeatureVector) => Promise<MLPrediction> } {
  return {
    predict: async () => ({
      score,
      tier: score >= 0.7 ? 'high' : score < 0.08 ? 'low' : 'mid',
      high_threshold: 0.7,
      low_threshold: 0.08,
      model_version: 'v4-stub',
    }),
  };
}

function makeFakeOpenAI(finalJson: object): object {
  let turn = 0;
  return {
    chat: {
      completions: {
        create: async () => {
          turn++;
          if (turn === 1) {
            return {
              choices: [
                {
                  message: {
                    role: 'assistant',
                    content: null,
                    tool_calls: [
                      {
                        id: 'c1',
                        type: 'function',
                        function: {
                          name: 'check_uspto_marks',
                          arguments: JSON.stringify({ brand_name: 'test', class_code: 30 }),
                        },
                      },
                    ],
                  },
                },
              ],
            };
          }
          return {
            choices: [{ message: { role: 'assistant', content: JSON.stringify(finalJson) } }],
          };
        },
      },
    },
  };
}

const fakeVerdict = {
  verdict: 'review',
  overall_confidence: 0.55,
  policies: {
    P1: { triggered: false, confidence: 0.1, reason: 'no conflicts' },
    P2: { triggered: false, confidence: 0.1, reason: 'single filing' },
    P3: { triggered: false, confidence: 0.1, reason: 'n/a' },
    P4: { triggered: false, confidence: 0.1, reason: 'n/a' },
    P5: { triggered: false, confidence: 0.3, reason: 'light web presence' },
  },
  summary: 'review suggested',
};

const stubTool = (async () => ({
  ok: true,
  data: { query: 'x', total: 0, results: [] },
  latency_ms: 1,
})) as unknown as typeof import('../../tools/check_uspto_marks.js').checkUsptoMarks;

const fakeHistory: ApplicantHistory = {
  applicant_name: 'Acme Co',
  found: true,
  filing_count_total: 5,
  filing_count_2yr: 2,
  abandonment_rate: 0.1,
  cancellation_rate: 0.0,
  first_filing_date: '2022-01-01',
  is_individual: false,
  is_foreign: false,
  attorney_of_record: 'Jane Doe',
  attorney_case_count: 200,
  attorney_cancellation_rate: 0.02,
  source: 'live',
};

function baseDeps() {
  return {
    classifier: stubClassifier(0.5),
    applicantHistory: fakeHistory,
    openai: makeFakeOpenAI(fakeVerdict) as unknown as import('openai').default,
    tools: { check_uspto_marks: stubTool },
  };
}

function req(brand: string, applicant = 'Acme Co'): {
  brand_name: string;
  applicant_name: string;
} {
  return { brand_name: brand, applicant_name: applicant };
}

async function main() {
  console.log('handleCheck');

  await test('valid body returns 200 with parsable verdict', async () => {
    const res = await handleCheck(req('BrewBox Coffee'), baseDeps());
    assert.equal(res.status, 200);
    if (res.status !== 200) throw new Error('unreachable');
    VerdictSchema.parse(res.verdict);
    assert.equal(res.verdict.brand, 'BrewBox Coffee');
    assert.equal(res.verdict.applicant, 'Acme Co');
    assert.ok(['safe', 'review', 'high_risk'].includes(res.verdict.verdict));
  });

  await test('response includes all 5 policies', async () => {
    const res = await handleCheck(req('BrewBox Coffee'), baseDeps());
    if (res.status !== 200) throw new Error('unreachable');
    for (const k of ['P1', 'P2', 'P3', 'P4', 'P5'] as const) {
      assert.ok(res.verdict.policies[k]);
    }
  });

  await test('empty brand_name returns 400 (not 500)', async () => {
    const res = await handleCheck({ brand_name: '', applicant_name: 'Acme' }, baseDeps());
    assert.equal(res.status, 400);
  });

  await test('missing brand_name returns 400', async () => {
    const res = await handleCheck({ applicant_name: 'Acme' }, baseDeps());
    assert.equal(res.status, 400);
  });

  await test('missing applicant_name returns 400', async () => {
    const res = await handleCheck({ brand_name: 'BrewBox' }, baseDeps());
    assert.equal(res.status, 400);
  });

  await test('brand_name over 200 chars returns 400', async () => {
    const res = await handleCheck(req('x'.repeat(201)), baseDeps());
    assert.equal(res.status, 400);
  });

  await test('non-object body returns 400', async () => {
    const res = await handleCheck(null, baseDeps());
    assert.equal(res.status, 400);
  });

  await test('audit store is called on success', async () => {
    const calls: Array<{ brand: string }> = [];
    const auditStore: AuditStore = {
      put: async (v) => {
        calls.push({ brand: v.brand });
        return {
          ok: true,
          record: { ...v, id: '11111111-1111-4111-8111-111111111111', ttl: 123456 },
        };
      },
      get: async () => null,
      listRecent: async () => [],
    };
    const res = await handleCheck(req('BrewBox Coffee'), { ...baseDeps(), auditStore });
    assert.equal(res.status, 200);
    assert.equal(calls.length, 1);
    assert.equal(calls[0]!.brand, 'BrewBox Coffee');
  });

  await test('audit failure does not fail the response', async () => {
    const auditStore: AuditStore = {
      put: async () => ({ ok: false, error: 'dynamo down' }),
      get: async () => null,
      listRecent: async () => [],
    };
    const res = await handleCheck(req('BrewBox Coffee'), { ...baseDeps(), auditStore });
    assert.equal(res.status, 200);
  });

  await test('classifier crash returns 500 (not thrown)', async () => {
    const crashingClassifier = {
      predict: async () => {
        throw new Error('onnx blew up');
      },
    };
    const res = await handleCheck(req('BrewBox Coffee'), {
      ...baseDeps(),
      classifier: crashingClassifier,
    });
    assert.equal(res.status, 500);
  });

  await test('concurrent requests do not interfere', async () => {
    const deps = baseDeps();
    const results = await Promise.all([
      handleCheck(req('Brand A'), deps),
      handleCheck(req('Brand B'), deps),
      handleCheck(req('Brand C'), deps),
    ]);
    for (const r of results) {
      assert.equal(r.status, 200);
      if (r.status !== 200) throw new Error('unreachable');
      VerdictSchema.parse(r.verdict);
    }
    assert.equal(results[0]!.status === 200 && results[0].verdict.brand, 'Brand A');
    assert.equal(results[1]!.status === 200 && results[1].verdict.brand, 'Brand B');
    assert.equal(results[2]!.status === 200 && results[2].verdict.brand, 'Brand C');
  });

  console.log(`\n${passed} passed, ${failed} failed`);
  if (failed > 0) process.exit(1);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
