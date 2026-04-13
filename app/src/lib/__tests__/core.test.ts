import { strict as assert } from 'node:assert';
import * as path from 'node:path';
import { POLICIES, POLICY_LIST } from '../policies.js';
import {
  loadClassifier,
  FEATURE_COLS,
  HIGH_THRESHOLD,
  LOW_THRESHOLD,
  type FeatureVector,
  ClassifierError,
} from '../classifier.js';
import { createAuditStore } from '../dynamo.js';
import { runCheck } from '../agent.js';
import {
  VerdictSchema,
  AuditRecordSchema,
  type Verdict,
  type MLPrediction,
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
    console.log(`  FAIL  ${name}\n        ${e instanceof Error ? e.stack ?? e.message : String(e)}`);
  }
}

function baseFeatures(overrides: Partial<FeatureVector> = {}): FeatureVector {
  const base: FeatureVector = {
    owner_filing_count_2yr: 1,
    owner_abandonment_rate: 0.05,
    owner_historical_cancellation_rate: 0.02,
    days_since_owner_first_filing: 3000,
    owner_is_individual: 0,
    attorney_case_count: 50,
    attorney_cancellation_rate: 0.03,
    owner_is_foreign: 0,
    days_domain_to_filing: 1500,
    days_since_filing: 4000,
    days_first_use_to_filing: 180,
    days_filing_to_registration: 300,
    office_action_count: 0,
    opposition_count: 0,
    statement_of_use_filed: 1,
    section_8_filed: 1,
    has_acquired_distinctiveness: 0,
    is_currently_active: 1,
    was_abandoned: 0,
    filing_basis: 1,
    class_breadth: 1,
    specimen_type_encoded: 1,
  };
  return { ...base, ...overrides };
}

async function main() {
  console.log('policies');

  await test('all 5 policies exported', () => {
    for (const key of ['P1', 'P2', 'P3', 'P4', 'P5'] as const) {
      assert.ok(POLICIES[key], `missing ${key}`);
    }
    assert.equal(POLICY_LIST.length, 5);
  });

  await test('each policy has name, description, tools', () => {
    for (const p of POLICY_LIST) {
      assert.ok(p.name && typeof p.name === 'string');
      assert.ok(p.description && typeof p.description === 'string');
      assert.ok(Array.isArray(p.tools) && p.tools.length >= 1);
    }
  });

  await test('descriptions are substantive (> 50 chars)', () => {
    for (const p of POLICY_LIST) {
      assert.ok(p.description.length > 50, `${p.key} description too short`);
    }
  });

  console.log('\nclassifier');

  const MODELS_DIR = path.resolve(process.cwd(), '..', 'ml', 'models');
  let classifier: Awaited<ReturnType<typeof loadClassifier>> | null = null;

  await test('ONNX session loads without error', async () => {
    classifier = await loadClassifier(MODELS_DIR);
    assert.ok(classifier);
  });

  await test('inference runs on valid input and returns valid MLPrediction', async () => {
    if (!classifier) throw new Error('classifier not loaded');
    const pred = await classifier.predict(baseFeatures());
    assert.ok(pred.score >= 0 && pred.score <= 1);
    assert.ok(['high', 'mid', 'low'].includes(pred.tier));
    assert.equal(pred.high_threshold, HIGH_THRESHOLD);
    assert.equal(pred.low_threshold, LOW_THRESHOLD);
  });

  // v4 is a RANKER not a classifier. The honest test is that a troll-shaped
  // feature vector scores strictly higher than a legit-shaped one. Absolute
  // thresholds would re-introduce the classifier framing we moved away from.
  let legitScore = 0;
  await test('safe prediction for obviously legitimate mark', async () => {
    if (!classifier) throw new Error('classifier not loaded');
    const pred = await classifier.predict(
      baseFeatures({
        owner_filing_count_2yr: 1,
        section_8_filed: 1,
        opposition_count: 0,
        owner_abandonment_rate: 0,
        was_abandoned: 0,
      })
    );
    legitScore = pred.score;
    assert.ok(pred.score < 0.4, `expected < 0.4, got ${pred.score}`);
  });

  await test('troll fixture ranks strictly above legit fixture', async () => {
    if (!classifier) throw new Error('classifier not loaded');
    const pred = await classifier.predict(
      baseFeatures({
        owner_filing_count_2yr: 300,
        section_8_filed: 0,
        opposition_count: 5,
        owner_abandonment_rate: 0.9,
        was_abandoned: 1,
        is_currently_active: 0,
        days_since_owner_first_filing: 60,
        days_domain_to_filing: 5,
      })
    );
    assert.ok(
      pred.score > legitScore,
      `troll ${pred.score} should rank above legit ${legitScore}`
    );
  });

  await test('missing feature returns error, not crash', async () => {
    if (!classifier) throw new Error('classifier not loaded');
    const broken = baseFeatures();
    delete (broken as Partial<FeatureVector>).section_8_filed;
    await assert.rejects(
      () => classifier!.predict(broken as FeatureVector),
      (e) => e instanceof ClassifierError && e.code === 'missing_feature'
    );
  });

  console.log('\ndynamo');

  interface FakeSend {
    calls: Array<{ name: string; input: unknown }>;
    store: Map<string, unknown>;
    failNext: boolean;
    send: (cmd: { constructor: { name: string }; input: Record<string, unknown> }) => Promise<unknown>;
  }
  function fakeDoc(): FakeSend {
    const calls: FakeSend['calls'] = [];
    const store = new Map<string, unknown>();
    const f: FakeSend = {
      calls,
      store,
      failNext: false,
      send: async (cmd) => {
        if (f.failNext) {
          f.failNext = false;
          throw new Error('dynamo unavailable');
        }
        const name = cmd.constructor.name;
        calls.push({ name, input: cmd.input });
        if (name === 'PutCommand') {
          const item = (cmd.input as { Item: Record<string, unknown> }).Item;
          store.set(`${item.id as string}|${item.checked_at as string}`, item);
          return {};
        }
        if (name === 'GetCommand') {
          const key = cmd.input as { Key: { id: string; checked_at: string } };
          const item = store.get(`${key.Key.id}|${key.Key.checked_at}`);
          return { Item: item };
        }
        if (name === 'ScanCommand') {
          return { Items: Array.from(store.values()) };
        }
        return {};
      },
    };
    return f;
  }

  const sampleVerdict: Verdict = VerdictSchema.parse({
    brand: 'BrewBox Coffee',
    verdict: 'review',
    overall_confidence: 0.5,
    ml: { score: 0.4, tier: 'mid', high_threshold: 0.7, low_threshold: 0.08, model_version: 'v4' },
    source: 'ml_and_agent',
    policies: {
      P1: { triggered: false, confidence: 0.1, reason: 'clean' },
      P2: { triggered: false, confidence: 0.1, reason: 'clean' },
      P3: { triggered: false, confidence: 0.1, reason: 'clean' },
      P4: { triggered: false, confidence: 0.1, reason: 'clean' },
      P5: { triggered: false, confidence: 0.1, reason: 'clean' },
    },
    tools_used: [],
    trace: [],
    summary: 'looks ok',
    checked_at: new Date().toISOString(),
  });

  await test('write then read round-trips', async () => {
    const fake = fakeDoc();
    // Cast — lib-dynamodb's DynamoDBDocumentClient is too strict for a fake.
    const store = createAuditStore(fake as unknown as import('@aws-sdk/lib-dynamodb').DynamoDBDocumentClient, 'T');
    const putRes = await store.put(sampleVerdict);
    assert.equal(putRes.ok, true);
    if (!putRes.ok) throw new Error('unreachable');
    const record = AuditRecordSchema.parse(putRes.record);
    const got = await store.get(record.id, record.checked_at);
    assert.ok(got);
    assert.equal(got!.brand, sampleVerdict.brand);
  });

  await test('TTL is ~90 days in the future', async () => {
    const fake = fakeDoc();
    const store = createAuditStore(fake as unknown as import('@aws-sdk/lib-dynamodb').DynamoDBDocumentClient, 'T');
    const putRes = await store.put(sampleVerdict);
    if (!putRes.ok) throw new Error('unreachable');
    const nowSec = Math.floor(Date.now() / 1000);
    const expected = nowSec + 90 * 86400;
    assert.ok(
      Math.abs(putRes.record.ttl - expected) < 10,
      `ttl ${putRes.record.ttl} not close to ${expected}`
    );
  });

  await test('listRecent returns array', async () => {
    const fake = fakeDoc();
    const store = createAuditStore(fake as unknown as import('@aws-sdk/lib-dynamodb').DynamoDBDocumentClient, 'T');
    await store.put(sampleVerdict);
    await store.put(sampleVerdict);
    const list = await store.listRecent();
    assert.equal(list.length, 2);
  });

  await test('handles DynamoDB unavailable without throwing', async () => {
    const fake = fakeDoc();
    fake.failNext = true;
    const store = createAuditStore(fake as unknown as import('@aws-sdk/lib-dynamodb').DynamoDBDocumentClient, 'T');
    const putRes = await store.put(sampleVerdict);
    assert.equal(putRes.ok, false);
  });

  console.log('\nagent');

  // Stub classifier — forces a given score so we can drive both short-circuit and
  // agent-loop branches deterministically without ONNX inference variance.
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

  await test('short-circuits safe when ML score < 0.3', async () => {
    const verdict = await runCheck(
      { brand_name: 'Obvious Legit Inc' },
      baseFeatures(),
      { classifier: stubClassifier(0.05) }
    );
    VerdictSchema.parse(verdict);
    assert.equal(verdict.verdict, 'safe');
    assert.equal(verdict.source, 'ml_only');
    assert.equal(verdict.tools_used.length, 0);
  });

  await test('short-circuits high_risk when ML score > 0.85', async () => {
    const verdict = await runCheck(
      { brand_name: 'Troll Shop LLC' },
      baseFeatures(),
      { classifier: stubClassifier(0.95) }
    );
    VerdictSchema.parse(verdict);
    assert.equal(verdict.verdict, 'high_risk');
    assert.equal(verdict.source, 'ml_only');
  });

  // Fake OpenAI client — two-turn: tool call, then final JSON verdict.
  function makeFakeOpenAI(opts: { finalJson: object; toolName?: string }): object {
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
                          id: 'call_1',
                          type: 'function',
                          function: {
                            name: opts.toolName ?? 'check_uspto_marks',
                            arguments: JSON.stringify({ brand_name: 'BrewBox Coffee', class_code: 30 }),
                          },
                        },
                      ],
                    },
                  },
                ],
              };
            }
            return {
              choices: [
                {
                  message: {
                    role: 'assistant',
                    content: JSON.stringify(opts.finalJson),
                  },
                },
              ],
            };
          },
        },
      },
    };
  }

  const fakeFinalVerdict = {
    verdict: 'review',
    overall_confidence: 0.55,
    policies: {
      P1: { triggered: false, confidence: 0.2, reason: 'no conflicts found' },
      P2: { triggered: false, confidence: 0.15, reason: 'single filing' },
      P3: { triggered: false, confidence: 0.1, reason: 'not applicable' },
      P4: { triggered: false, confidence: 0.1, reason: 'no attorney given' },
      P5: { triggered: false, confidence: 0.4, reason: 'light web presence' },
    },
    summary: 'Moderate risk, recommend human review.',
  };

  await test('full agent run returns valid verdict (mocked OpenAI)', async () => {
    const stubTool = (async () => ({
      ok: true,
      data: { query: 'BrewBox Coffee', total: 0, results: [] },
      latency_ms: 5,
    })) as unknown as typeof import('../../tools/check_uspto_marks.js').checkUsptoMarks;

    const verdict = await runCheck(
      { brand_name: 'BrewBox Coffee', class_code: 30 },
      baseFeatures(),
      {
        classifier: stubClassifier(0.5),
        openai: makeFakeOpenAI({ finalJson: fakeFinalVerdict }) as unknown as import('openai').default,
        tools: { check_uspto_marks: stubTool },
      }
    );
    VerdictSchema.parse(verdict);
    assert.equal(verdict.source, 'ml_and_agent');
    assert.equal(verdict.verdict, 'review');
    for (const k of ['P1', 'P2', 'P3', 'P4', 'P5'] as const) {
      assert.ok(verdict.policies[k]);
      assert.equal(typeof verdict.policies[k].triggered, 'boolean');
      assert.equal(typeof verdict.policies[k].confidence, 'number');
      assert.equal(typeof verdict.policies[k].reason, 'string');
    }
    assert.ok(verdict.tools_used.includes('check_uspto_marks'));
    assert.equal(verdict.trace.length, 1);
  });

  await test('agent continues when a tool fails mid-run', async () => {
    const failingTool = (async () => ({
      ok: false,
      error: { code: 'network' as const, message: 'rdap down' },
      latency_ms: 10,
    })) as unknown as typeof import('../../tools/check_domain_age.js').checkDomainAge;

    const verdict = await runCheck(
      { brand_name: 'BrewBox Coffee', domain_name: 'brewbox.com' },
      baseFeatures(),
      {
        classifier: stubClassifier(0.5),
        openai: makeFakeOpenAI({
          finalJson: fakeFinalVerdict,
          toolName: 'check_domain_age',
        }) as unknown as import('openai').default,
        tools: { check_domain_age: failingTool },
      }
    );
    VerdictSchema.parse(verdict);
    assert.equal(verdict.trace.length, 1);
    assert.ok(verdict.trace[0]!.error !== null);
    assert.equal(verdict.verdict, 'review');
  });

  console.log(`\n${passed} passed, ${failed} failed`);
  if (failed > 0) process.exit(1);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
