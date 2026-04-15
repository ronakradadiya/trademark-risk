# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

Two-part monorepo:

- `ml/` — Python. Data pipeline, feature engineering, and training for the **v4 stacking ensemble** (xgboost + catboost + LR meta-learner). Produces `.onnx` + `.joblib` artifacts in `ml/models/`. Not imported by the app at runtime — only the exported `.onnx` files are.
- `app/` — TypeScript / Next.js 14 app router. Contains the serving runtime (agent, classifier, tools, route handler, UI). **All day-to-day development happens here.**

There is no top-level `package.json` workspace config — `app/` and `ml/` are installed independently.

## Commands (run from `app/`)

```bash
npm run dev          # Next dev server
npm run build        # next build (also runs type checking)
npm run typecheck    # tsc --noEmit, no emit
npm test             # schemas → tools → core → check, in order
npm run test:schemas # Zod schema suite
npm run test:tools   # tool wrappers + fetch mocks
npm run test:core    # agent loop + classifier orchestration
npm run test:check   # handleCheck request/response layer
```

Tests are plain `tsx` scripts (no Jest / Vitest). Each file is a standalone program that runs `test(name, fn)` calls and exits non-zero if anything fails — **to run a single case, edit the file and comment out other `await test(...)` calls**, then `npx tsx src/lib/__tests__/core.test.ts`.

Feature-level smoke test against the dev server:

```bash
curl -sS -X POST http://localhost:3000/api/check \
  -H 'content-type: application/json' \
  -d '{"brand_name":"NovaPay","applicant_name":"Meridian Labs LLC","class_code":36}'
```

## Architecture (the parts that require reading multiple files)

### Two-layer verdict pipeline

A request flows through `api/check/route.ts → lib/check.ts → lib/agent.ts` and passes through two layers whose confidences are **blended, not short-circuited**:

1. **ML filer-risk layer** (`lib/classifier.ts`) — v4 ONNX ensemble scoring the *filer*, not the mark. Inputs: 22 `FEATURE_COLS` built by `lib/features.ts`. Only a `score > 0.85` short-circuits to `high_risk` with `source: 'ml_only'`. There is no low short-circuit — every non-high request runs the agent.
2. **Agent layer** (`runAgentLoop` in `lib/agent.ts`) — GPT-4o ReAct loop with 4 tools and 5 policies. Returns `PolicyBundle` + verdict + summary.

Final blend (line in `lib/agent.ts`): `overall_confidence = ml.score * 0.35 + agent.overall_confidence * 0.65`.

### Applicant history is the load-bearing feature source

Before either layer runs, `resolveApplicantHistory()` calls the `lookup_applicant_history` tool (fixtures + optional live USPTO fallback). Its `ApplicantHistory` output is:
- Mapped by `applicantToFeatures()` into **8 of the 22 ML feature columns** (the "applicant-side" features). The remaining 14 are pre-filing mark features held at neutral priors in `BASELINE_MARK_FEATURES`.
- Injected as a one-line `historyLine` into the agent's user message so the LLM sees the same signal the ML does.
- Emitted as a synthetic **pre-trace step 0** so the UI trace shows the lookup even though it's not an agent-driven tool call.

This means: **when you add or change a field on `ApplicantHistory`, you must update three places** — `schemas/index.ts`, `lib/features.ts` (mapping), and the `historyLine` template in `lib/agent.ts`.

### Schemas are the contract, not the UI types

Every shape crossing a boundary (HTTP, tool call, LLM JSON output, Dynamo record) is defined once in `app/src/schemas/index.ts` as a Zod schema; TS types are inferred via `z.infer<>`. The agent's final reply is `JSON.parse`d and validated with `VerdictSchema.parse()` — if you add a field to a verdict, the agent's system prompt must also be updated to emit it, or the parse fails and the request returns 500.

`TOOL_NAMES` in `schemas/index.ts` is the canonical tool list. It is used as an exhaustive-switch discriminator in `lib/agent.ts#executeTool`, so adding a tool requires: (1) entry in `TOOL_NAMES`, (2) new case in `executeTool`, (3) tool spec in `toolSpecs()`, (4) injection point in `AgentDeps.tools`.

### Dependency injection for tests

`AgentDeps` in `lib/agent.ts` exposes overrides for `classifier`, `openai`, `tools`, and `applicantHistory`. Tests in `lib/__tests__/core.test.ts` and `check.test.ts` use these to run the full agent loop with:
- a stubbed classifier returning a fixed score,
- a fake OpenAI client that replays a pre-canned tool call → final-JSON sequence (`makeFakeOpenAI`),
- `applicantHistory` set directly so the pre-step doesn't need a real tool call.

When you write new tests that need "the agent ran", follow this pattern — don't mock at the HTTP layer.

### Classifier model loading

`loadClassifier()` resolves `DEFAULT_MODELS_DIR = ../ml/models` relative to `process.cwd()`. This only works when the Next dev server is started from `app/`. The loader expects three files: `xgboost.onnx`, `catboost.onnx`, `meta_lr.onnx`. If any are missing, the classifier throws `model_load` at first call, not at startup.

### Audit store is optional

`DYNAMO_AUDIT_DISABLED=1` in env skips Dynamo initialization entirely. When enabled, `put()` failures are logged but never fail the response — audit is best-effort, not part of the request contract.

### Tool wrappers and graceful degradation

Every tool in `app/src/tools/` returns a `ToolResult<T>` (`{ ok, data, latency_ms } | { ok: false, error, latency_ms }`) via `runTool()` in `lib/http.ts`. Tools should not throw — they should return `ok: false` with a `ToolError` code (`timeout`, `network`, `http`, `parse`, `config`). `check_domain_age` additionally swallows HTTP 404 into `ok: true, exists: false` because a missing domain is a valid signal, not an error. Follow this pattern when a tool's upstream returns "not found" — don't surface it as an error.

## Known environment quirks

- The dev server must be started from `app/` so the classifier's relative model path resolves.
- `.env.local` goes in `app/`, not the repo root. Expected keys: `OPENAI_API_KEY`, `SERPER_API_KEY`, `DYNAMO_AUDIT_DISABLED=1` (for local), plus optional USPTO/RDAP URL overrides.
- `next.config.mjs` sets `serverComponentsExternalPackages: ['onnxruntime-node']` — don't remove this, it's what makes the native ONNX bindings survive Next's bundler.
