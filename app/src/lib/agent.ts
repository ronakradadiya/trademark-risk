import OpenAI from 'openai';
import {
  CheckRequestSchema,
  VerdictSchema,
  PolicyBundleSchema,
  type CheckRequest,
  type Verdict,
  type MLPrediction,
  type PolicyBundle,
  type PolicyResult,
  type AgentStep,
  type ToolName,
  POLICY_KEYS,
} from '../schemas/index.js';
import { POLICIES, POLICY_LIST } from './policies.js';
import type { Classifier, FeatureVector } from './classifier.js';
import { checkUsptoMarks } from '../tools/check_uspto_marks.js';
import { checkDomainAge } from '../tools/check_domain_age.js';
import { webSearch } from '../tools/web_search.js';
import { checkAttorney } from '../tools/check_attorney.js';

const AGENT_LOW = 0.3;   // ml_confidence < AGENT_LOW → short-circuit safe
const AGENT_HIGH = 0.85; // ml_confidence > AGENT_HIGH → short-circuit high_risk
const MAX_AGENT_STEPS = 10;
const DEFAULT_MODEL = process.env.OPENAI_MODEL ?? 'gpt-4o';

export interface AgentDeps {
  classifier: Classifier;
  openai?: OpenAI;
  model?: string;
  now?: () => Date;
  tools?: Partial<{
    check_uspto_marks: typeof checkUsptoMarks;
    check_domain_age: typeof checkDomainAge;
    web_search: typeof webSearch;
    check_attorney: typeof checkAttorney;
  }>;
}

function verdictFromScore(score: number): Verdict['verdict'] {
  if (score < AGENT_LOW) return 'safe';
  if (score > AGENT_HIGH) return 'high_risk';
  return 'review';
}

function emptyPolicies(reason: string): PolicyBundle {
  const r: PolicyResult = { triggered: false, confidence: 0, reason };
  return { P1: r, P2: r, P3: r, P4: r, P5: r };
}

function toolSpecs(): OpenAI.Chat.Completions.ChatCompletionTool[] {
  return [
    {
      type: 'function',
      function: {
        name: 'check_uspto_marks',
        description: 'Search USPTO for similar trademarks by brand name and optional class code (1-45).',
        parameters: {
          type: 'object',
          properties: {
            brand_name: { type: 'string' },
            class_code: { type: 'number' },
          },
          required: ['brand_name'],
        },
      },
    },
    {
      type: 'function',
      function: {
        name: 'check_domain_age',
        description: 'Return registration date and age in days for a domain.',
        parameters: {
          type: 'object',
          properties: { domain_name: { type: 'string' } },
          required: ['domain_name'],
        },
      },
    },
    {
      type: 'function',
      function: {
        name: 'web_search',
        description: 'Google web search for commercial activity, news, and reputation.',
        parameters: {
          type: 'object',
          properties: {
            query: { type: 'string' },
            max_results: { type: 'number' },
          },
          required: ['query'],
        },
      },
    },
    {
      type: 'function',
      function: {
        name: 'check_attorney',
        description: 'Look up USPTO OED attorney bar status and disciplinary history.',
        parameters: {
          type: 'object',
          properties: {
            attorney_name: { type: 'string' },
            bar_number: { type: 'string' },
          },
          required: ['attorney_name'],
        },
      },
    },
  ];
}

function systemPrompt(): string {
  const policies = POLICY_LIST.map((p) => `${p.key} — ${p.name}: ${p.description}`).join('\n\n');
  return `You are a trademark risk investigator helping small businesses determine if a brand name is safe to trademark before spending money on branding and legal fees.

You have 4 tools: check_uspto_marks, check_domain_age, web_search, check_attorney.

Investigate the brand name thoroughly. Call tools in whatever order makes sense based on what you find. You may call the same tool multiple times with different queries. Do NOT ask the user questions — act autonomously.

Then evaluate each of the 5 policies below and return a structured verdict.

Policies:
${policies}

Return JSON ONLY in this exact shape (no prose outside the JSON):
{
  "verdict": "safe" | "review" | "high_risk",
  "overall_confidence": number between 0 and 1,
  "policies": {
    "P1": { "triggered": boolean, "confidence": number, "reason": string },
    "P2": { "triggered": boolean, "confidence": number, "reason": string },
    "P3": { "triggered": boolean, "confidence": number, "reason": string },
    "P4": { "triggered": boolean, "confidence": number, "reason": string },
    "P5": { "triggered": boolean, "confidence": number, "reason": string }
  },
  "summary": "one sentence plain English"
}`;
}

interface ParsedAgentJson {
  verdict: Verdict['verdict'];
  overall_confidence: number;
  policies: PolicyBundle;
  summary: string;
}

function parseAgentJson(raw: string): ParsedAgentJson {
  const trimmed = raw.trim().replace(/^```json\s*/i, '').replace(/```\s*$/, '');
  const obj = JSON.parse(trimmed) as unknown;
  if (!obj || typeof obj !== 'object') throw new Error('agent output is not an object');
  const o = obj as Record<string, unknown>;
  const policies = PolicyBundleSchema.parse(o.policies);
  const verdict = String(o.verdict);
  if (verdict !== 'safe' && verdict !== 'review' && verdict !== 'high_risk') {
    throw new Error(`invalid verdict: ${verdict}`);
  }
  const oc = Number(o.overall_confidence);
  if (!(oc >= 0 && oc <= 1)) throw new Error(`invalid overall_confidence: ${o.overall_confidence}`);
  return {
    verdict,
    overall_confidence: oc,
    policies,
    summary: typeof o.summary === 'string' ? o.summary : '',
  };
}

export async function runCheck(
  request: CheckRequest,
  features: FeatureVector,
  deps: AgentDeps
): Promise<Verdict> {
  const req = CheckRequestSchema.parse(request);
  const now = (deps.now ?? (() => new Date()))();

  const ml: MLPrediction = await deps.classifier.predict(features);

  if (ml.score < AGENT_LOW || ml.score > AGENT_HIGH) {
    const verdict = verdictFromScore(ml.score);
    const reason =
      verdict === 'safe'
        ? `ML score ${ml.score.toFixed(3)} is below short-circuit threshold ${AGENT_LOW}.`
        : `ML score ${ml.score.toFixed(3)} exceeds short-circuit threshold ${AGENT_HIGH}.`;
    return VerdictSchema.parse({
      brand: req.brand_name,
      verdict,
      overall_confidence: ml.score,
      ml,
      source: 'ml_only',
      policies: emptyPolicies(reason),
      tools_used: [],
      trace: [],
      summary: reason,
      checked_at: now.toISOString(),
    });
  }

  const agentResult = await runAgentLoop(req, ml, deps);

  const finalConfidence = ml.score * 0.35 + agentResult.parsed.overall_confidence * 0.65;
  return VerdictSchema.parse({
    brand: req.brand_name,
    verdict: agentResult.parsed.verdict,
    overall_confidence: finalConfidence,
    ml,
    source: 'ml_and_agent',
    policies: agentResult.parsed.policies,
    tools_used: [...new Set(agentResult.toolsUsed)],
    trace: agentResult.trace,
    summary: agentResult.parsed.summary || `Agent verdict: ${agentResult.parsed.verdict}.`,
    checked_at: now.toISOString(),
  });
}

interface AgentRunResult {
  parsed: ParsedAgentJson;
  trace: AgentStep[];
  toolsUsed: ToolName[];
}

async function runAgentLoop(
  req: CheckRequest,
  ml: MLPrediction,
  deps: AgentDeps
): Promise<AgentRunResult> {
  const client = deps.openai ?? new OpenAI();
  const model = deps.model ?? DEFAULT_MODEL;

  const userMessage = [
    `Brand: ${req.brand_name}`,
    req.class_code !== undefined ? `Class: ${req.class_code}` : null,
    req.domain_name ? `Domain: ${req.domain_name}` : null,
    req.attorney_name ? `Attorney: ${req.attorney_name}` : null,
    `ML risk score: ${ml.score.toFixed(3)} (tier: ${ml.tier})`,
  ]
    .filter(Boolean)
    .join('\n');

  const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
    { role: 'system', content: systemPrompt() },
    { role: 'user', content: userMessage },
  ];

  const trace: AgentStep[] = [];
  const toolsUsed: ToolName[] = [];

  for (let step = 0; step < MAX_AGENT_STEPS; step++) {
    const response = await client.chat.completions.create({
      model,
      messages,
      tools: toolSpecs(),
      tool_choice: 'auto',
      response_format: { type: 'json_object' },
    });

    const msg = response.choices[0]?.message;
    if (!msg) throw new Error('empty completion');

    if (msg.tool_calls && msg.tool_calls.length > 0) {
      messages.push(msg);
      for (const call of msg.tool_calls) {
        if (call.type !== 'function') continue;
        const name = call.function.name as ToolName;
        const args = safeJsonParse(call.function.arguments);
        const started = Date.now();
        const result = await executeTool(name, args, deps);
        trace.push({
          step,
          tool: name,
          input: args,
          output: result.ok ? result.data : null,
          error: result.ok ? null : result.error.message,
          duration_ms: Date.now() - started,
        });
        toolsUsed.push(name);
        messages.push({
          role: 'tool',
          tool_call_id: call.id,
          content: JSON.stringify(result.ok ? { ok: true, data: result.data } : { ok: false, error: result.error }),
        });
      }
      continue;
    }

    const content = msg.content ?? '';
    const parsed = parseAgentJson(content);
    return { parsed, trace, toolsUsed };
  }

  throw new Error(`agent did not return a verdict within ${MAX_AGENT_STEPS} steps`);
}

function safeJsonParse(s: string): unknown {
  try {
    return JSON.parse(s);
  } catch {
    return {};
  }
}

async function executeTool(
  name: ToolName,
  args: unknown,
  deps: AgentDeps
): Promise<
  | { ok: true; data: unknown }
  | { ok: false; error: { code: string; message: string; status?: number } }
> {
  const a = (args ?? {}) as Record<string, unknown>;
  const tools = deps.tools ?? {};
  try {
    switch (name) {
      case 'check_uspto_marks': {
        const fn = tools.check_uspto_marks ?? checkUsptoMarks;
        const r = await fn({
          brand_name: String(a.brand_name ?? ''),
          ...(typeof a.class_code === 'number' ? { class_code: a.class_code } : {}),
        });
        return r.ok ? { ok: true, data: r.data } : { ok: false, error: r.error };
      }
      case 'check_domain_age': {
        const fn = tools.check_domain_age ?? checkDomainAge;
        const r = await fn({ domain_name: String(a.domain_name ?? '') });
        return r.ok ? { ok: true, data: r.data } : { ok: false, error: r.error };
      }
      case 'web_search': {
        const fn = tools.web_search ?? webSearch;
        const r = await fn({
          query: String(a.query ?? ''),
          ...(typeof a.max_results === 'number' ? { max_results: a.max_results } : {}),
        });
        return r.ok ? { ok: true, data: r.data } : { ok: false, error: r.error };
      }
      case 'check_attorney': {
        const fn = tools.check_attorney ?? checkAttorney;
        const r = await fn({
          attorney_name: String(a.attorney_name ?? ''),
          ...(typeof a.bar_number === 'string' ? { bar_number: a.bar_number } : {}),
        });
        return r.ok ? { ok: true, data: r.data } : { ok: false, error: r.error };
      }
      default: {
        const _exhaust: never = name;
        return { ok: false, error: { code: 'unknown_tool', message: `unknown tool: ${_exhaust}` } };
      }
    }
  } catch (e) {
    return {
      ok: false,
      error: { code: 'exception', message: e instanceof Error ? e.message : String(e) },
    };
  }
}

export { POLICY_KEYS, POLICIES };
