# TrademarkRisk — Plan & Status

Last updated: 2026-04-13

## What this is

A trademark-fraud due-diligence system built as a SafetyKit-style two-layer
pipeline. Given `(brand_name, applicant_name, class_code?, domain_name?)` it
returns a structured verdict (`safe` / `review` / `high_risk`) with per-policy
reasoning, an ML filer-risk score, the agent's tool trace, and an audit
record.

The product is scoped as "should I file this trademark, or is my applicant
behaving like a known troll." It scores the *filer*, not the mark — the mark
side is handled by prior-art lookup feeding both the ML layer and the agent.

## Repository layout

Two-part monorepo. No root `package.json`. Each subtree is installed
independently.

```
trademark-risk/
├── app/                          TypeScript / Next.js 14 — serving runtime
│   ├── src/
│   │   ├── app/api/check/        POST /api/check route
│   │   ├── lib/
│   │   │   ├── agent.ts          ReAct loop + verdict blending
│   │   │   ├── classifier.ts     ONNX session + 13-feature inference
│   │   │   ├── features.ts       applicantToFeatures + markFeaturesFromPriorArt
│   │   │   ├── rules.ts          deterministic hard-rule overlay (R1–R4)
│   │   │   ├── policies.ts       the 5 policy definitions
│   │   │   ├── check.ts          HTTP→agent orchestration
│   │   │   ├── uspto_db.ts       better-sqlite3 read-only DB handle
│   │   │   └── dynamo.ts         optional audit store
│   │   ├── tools/                4 agent tools (uspto_marks, domain_age, web_search, applicant_history)
│   │   └── schemas/              Zod contracts — single source of truth
│   ├── scripts/
│   │   ├── ingest_uspto.ts       builds data/uspto.sqlite from TRTYRAP XML
│   │   └── smoke_tools.ts        tool-level smoke harness
│   └── src/**/__tests__/         47 tests across schemas/tools/core/check
├── ml/                           Python — training only, not imported at runtime
│   ├── src/
│   │   ├── build_from_sqlite.py  replaces TCFD ingest; writes processed features
│   │   ├── train.py              LR/XGB/LGBM/CatBoost individual training
│   │   ├── ensemble.py           v4 stack + ONNX export + parity check
│   │   └── compare_models.py     production-fidelity gate (old vs new)
│   ├── data/
│   │   ├── raw/ttab/             TTAB fraud label provenance (kept)
│   │   └── processed/            features_train/val/test.csv + labeled.csv
│   ├── models/                   ONNX + joblib artifacts + MODEL_CARD.md
│   └── models_v4_old/            pre-retrain backup, used by compare_models.py
└── data/uspto.sqlite             12.5M marks, 5M applicants — TRTYRAP April 2026
```

## Current state — what is built and working

### Data pipeline

- **`data/uspto.sqlite`** — built by `app/scripts/ingest_uspto.ts` from
  USPTO's TRTYRAP XML feed. 12.5M marks, 4.99M applicants, FTS5 index on
  `marks_fts`, rollups in `applicants` table (filing_count_2yr,
  abandonment_rate, cancellation_rate, is_individual, is_foreign,
  attorney_case_count, attorney_cancellation_rate, first_filing_date).
  Monthly-refreshable.
- **`ml/data/raw/ttab/combined_fraud_serials.json`** — 30,011 cached fraud
  serial numbers (28,527 CANG events + 2,207 TTAB XML scrape). Portable
  label set — regenerable without re-downloading any CSVs.
- TCFD CSVs deleted after labels were baked. 14.5 GB freed.

### ML model — v4 stacking ensemble, 13 features

Architecture: `xgboost` + `catboost` → Logistic Regression meta-learner.
Unchanged from the prior v4 design; what changed is the feature set and the
training data.

**Training data:** 64,003 rows from April 2026 SQLite, 25% positive class,
stratified across filing years 2000–2024. Labels join TTAB serials against
`marks.serial` at 97.3% rate.

**Applicant-side (8):** `owner_filing_count_2yr`, `owner_abandonment_rate`,
`owner_historical_cancellation_rate`, `days_since_owner_first_filing`,
`owner_is_individual`, `owner_is_foreign`, `attorney_case_count`,
`attorney_cancellation_rate`. All computed temporally-safe at training
(`filing_date < current.filing_date` only) via per-owner sorted timelines
with `bisect` — O(N log N) total, not O(N × K).

**Mark-side (5):** `days_since_filing`, `days_filing_to_registration`,
`was_abandoned`, `is_currently_active`, `class_breadth`. At training time
computed from the mark's own row; at serving time computed by
`markFeaturesFromPriorArt()` via a `mark_norm` lookup against SQLite.

**Training metrics (held-out test, n=9,601):**
| metric | value |
|---|---|
| AUC-ROC | 0.8275 |
| Brier | 0.1407 |
| F1 @ 0.25 | 0.6017 |
| Recall @ 0.25 | 0.8546 |
| Precision @ 0.25 | 0.4643 |

5-fold CV F1 is stable at 0.600 ± 0.006 — the feature set is not overfit to
any single fold.

**Gate comparison (production fidelity, old vs new on same test set):**
| metric | OLD (22 feat, 14 constants at serving) | NEW (13 feat, all real) | Δ |
|---|---|---|---|
| AUC | 0.6565 | **0.8275** | +0.171 |
| Brier | 0.2286 | **0.1407** | −0.088 |
| Recall | 0.0304 | **0.8546** | +0.824 |
| F1 | 0.0569 | **0.6017** | +0.545 |

The old model's recall collapses to 3% in production because its meta-LR
was scoring a 22-element vector with 14 hardcoded `BASELINE_MARK_FEATURES`
constants it had never trained on. The retrain dropped those features
entirely. Full writeup in [ml/models/MODEL_CARD.md](ml/models/MODEL_CARD.md).

**ONNX export:** opset 12, ai.onnx.ml 3, ZipMap stripped (required by
`onnxruntime-node`). Joblib↔ONNX parity max diff = 0.000000.

### Serving runtime

Every request flows `api/check/route.ts → lib/check.ts → lib/agent.ts` and
passes through four layers whose outputs are combined, not short-circuited
(except at the extremes):

1. **Pre-resolve applicant history.** `resolveApplicantHistory()` runs the
   `lookup_applicant_history` tool first. Its output feeds both the ML layer
   (via `applicantToFeatures`) and the agent (via a one-line `historyLine`
   in the user message). Emitted as synthetic trace step 0.

2. **Deterministic rule overlay.** `evaluateHardRules()` in
   [app/src/lib/rules.ts](app/src/lib/rules.ts) checks 4 hard-coded filer
   patterns against `ApplicantHistory`. If any rule fires, the request
   short-circuits to `high_risk` with `source: 'rule_override'`. This runs
   *before* the ML and agent layers because some patterns are beyond
   reasonable doubt and we don't want the LLM to second-guess them.

3. **ML filer-risk layer.** `lib/classifier.ts` runs the 13-feature v4 ONNX
   stack. `ml.score > 0.85` short-circuits to `high_risk` with
   `source: 'ml_only'`. There is no low short-circuit — every non-high
   request runs the agent.

4. **Agent layer.** GPT-4o ReAct loop with 4 tools (`check_uspto_marks`,
   `check_domain_age`, `web_search`, `lookup_applicant_history`) and 5
   policies. Returns `PolicyBundle` + verdict + summary.

**Final blend:** `overall_confidence = ml.score * 0.35 + agent.overall_confidence * 0.65`.

### Schema contract

Every shape crossing a boundary (HTTP, tool call, LLM JSON output, Dynamo
record) is defined once in [app/src/schemas/index.ts](app/src/schemas/index.ts)
as a Zod schema with TS types inferred via `z.infer<>`. The agent's final
reply is parsed and validated with `VerdictSchema.parse()` — if the LLM
omits a field, the request fails 500 rather than silently shipping bad data.

`TOOL_NAMES` is the canonical tool list, used as an exhaustive-switch
discriminator in `executeTool`. Adding a tool requires: (1) entry in
`TOOL_NAMES`, (2) new case in `executeTool`, (3) tool spec in `toolSpecs()`,
(4) injection point in `AgentDeps.tools`.

### Tests

47 tests in 4 suites, all passing:
- `test:schemas` — Zod contracts
- `test:tools` — tool wrappers + fetch mocks
- `test:core` — classifier loading + agent loop with fake OpenAI/classifier
- `test:check` — HTTP layer end-to-end

Dependency injection via `AgentDeps` makes the "agent actually ran" path
testable without real OpenAI or real ONNX. Tests are plain `tsx` scripts;
no Jest / Vitest.

### End-to-end verified

- `curl /api/check NovaPay/Meridian Labs` → `rule_override` fires correctly
  (R1+R2+R3 — pro se burst filer pattern). ml.score = 0.024.
- `curl /api/check BrewBox/BrewBox Holdings` → full 4-tool agent loop runs,
  returns `review`, ml.score = 0.023, blended overall_confidence = 0.50.

## Known limitations

These are honest gaps, not nice-to-haves:

- **Applicant-side dominance.** Feature importance is lopsided toward the
  filer signal. The 5 mark-side features add incremental lift but none is
  individually load-bearing. The agent layer does most of the mark-specific
  reasoning.
- **`days_since_filing` train/serve gap.** At training time this is the
  mark's real age (years); at serving time it's always 0 for a pre-filing
  check. Tree models tolerate this, but it's a real distribution shift.
- **No mark-name text similarity in the ML layer.** Conflict detection
  against prior art currently happens only in the `check_uspto_marks` tool
  via FTS5 phrase match. There's no embedding-based "fuzzy brand name"
  signal feeding the classifier.
- **Sanity-check warnings at build time.** `build_from_sqlite.py` reports
  `owner_abandonment_rate: fraud ≤ legit` and similar univariate
  inversions. Explainable (megafiler dominance in the legit class) but it
  means the model learns the decision boundary in interaction, not in
  isolation.
- **LR baseline ~4 points behind the stack** (0.787 vs 0.828 AUC). Modest
  lift from the full xgb+cat+LR architecture, which suggests the feature
  set is close to linearly separable and the stacking complexity is not
  fully earning its keep.
- **Single static artifact swap, no canary / shadow deployment.**
- **No drift monitoring.** If TRTYRAP schema changes or the TTAB join rate
  drops over time, nothing alerts.

## Future scope

Roughly ordered by impact-per-effort.

### Tier 0 — the product-framing gap

The current system answers "is this **filer** legitimate." It does not answer
"is this **mark** fraud." The TTAB labels we train on are per-serial, but the
features are ~70% filer-shaped, ~30% weak mark-side aggregates. Two tracks
close this gap, and they should be built together:

- **Mark fraud classifier (separate model, not a feature add-on).** A second
  ONNX head that scores `(brand_name, goods_description, class_code,
  applicant_name)` on the same TTAB labels but with mark-centric features:
    - **Mark-name text features** — character n-grams, sentence-transformer
      embedding, gibberish-vs-real-word signal, brand-squatting patterns
      (contains a known brand token, one-character variants of famous marks).
    - **Fuzzy prior-art similarity** — nearest-neighbor embedding distance to
      the top-k live registered marks in the same Nice class. Captures "is
      this a variant of something already out there."
    - **Goods-description features** — length, specificity, generic-vs-concrete,
      copy-paste fingerprint against a corpus of known fraud filings. Requires
      scraping `identification_of_goods` from the TRTYRAP XML (currently
      dropped during ingest).
    - **Class coherence** — does the mark name plausibly belong in the Nice
      class it's filed under. Score via a pretrained LM over `(mark_name, class
      description)` pairs.
    - **Applicant↔mark coherence** — does the applicant's name match the
      brand they're trying to register? Squatters file under shell LLCs whose
      names have zero relation to the mark.
  This is a genuinely different model, not "add 5 more features to the
  existing stack." It's the answer to the question the product name actually
  implies.
- **Two-endpoint API split.** Today `/api/check` returns both filer-risk and
  prior-art reasoning in one blob. Split into:
    - `POST /api/mark/prior-art` — filer-facing. Consumes `(brand_name,
      class_code, domain_name?)`, runs the agent layer + mark fraud
      classifier, returns "will this filing fly" reasoning. This is what a
      LegalZoom-style customer actually pays for.
    - `POST /api/applicant/risk` — firm-facing. Consumes `(applicant_name,
      attorney_name?)`, runs the filer-risk classifier + rule overlay, returns
      "is this applicant safe to take as a client." This is what a trademark
      law firm or filing platform pays for.
  The current `/api/check` stays as a convenience wrapper that calls both
  and blends, but the two primary endpoints match the two distinct buyer
  personas (see "Who is this for" discussion in CLAUDE.md).

### Tier 1 — would materially improve the existing filer-risk model

- **Synthetic-aging augmentation for `days_since_filing`.** At training time,
  randomly re-age a fraction of rows to `days_since_filing = 0` (simulating
  "this mark was just filed") with the rest of the row unchanged. Closes the
  train/serve distribution gap.
- **Richer class_breadth at serving time.** Currently hardcoded to 1 unless
  a class is supplied. If the request carries multiple classes, count them.
  Nice-classification breadth is a known fraud signal at USPTO.
- **Drift monitor.** Weekly job that re-runs `compare_models.py` against a
  fresh SQLite snapshot and posts the AUC/Brier delta. Alert if either
  drops more than 0.02 versus the shipped artifact.
- **Extend rule overlay with R5 (high attorney cancellation rate).** Current
  rules (R1–R4) fire on pro-se burst patterns. The TTAB data shows clustered
  cancellations concentrated under a handful of repeat attorneys who don't
  trigger R1 because they have counsel of record. Worth encoding.

### Tier 2 — would improve the product story

- **Confidence calibration on the blended score.** The 0.35/0.65 ML↔agent
  blend is a heuristic. Fit a logistic regression over `(ml.score,
  agent.confidence)` against a small labeled eval set to learn the right
  weights empirically.
- **Agent tool budget / cost telemetry.** Currently the ReAct loop can run
  up to 10 steps with no per-request cost ceiling. Add latency and token
  accounting to the trace so `/api/check` responses carry real cost data
  back to the caller.
- **Shadow-deploy mode.** Allow the new model to run alongside the old one
  and log both verdicts to the audit store without affecting the response.
  Gives a few weeks of real-traffic data before committing to the swap.
- **Richer `lookup_applicant_history`.** Today this returns 8 aggregated
  fields. A "top 5 most recent cancellations with serial + date" attachment
  would give the agent more to cite in its `summary`.

### Tier 3 — infrastructure and operations

- **Monthly SQLite refresh pipeline.** Right now the ingest is a manual
  `npx tsx scripts/ingest_uspto.ts`. Wrap it in a cron + S3 upload so the
  serving database auto-refreshes.
- **Retrain cadence.** Automate the full `build_from_sqlite.py → train.py
  → ensemble.py → compare_models.py` chain behind a single CI job gated on
  the comparison script's exit code. Only promotes artifacts if the gate
  passes.
- **CDK deployment.** The app runs locally against `.env.local`. Wire up
  the Next build to a Lambda + CloudFront or ECS Fargate target with the
  ONNX files baked into the deployment artifact.
- **DynamoDB audit index.** The audit store writes work but there's no
  secondary index for "list recent verdicts by applicant." Add a GSI on
  `applicant_name_norm` so the admin UI can pull a filer's history across
  past checks.
- **Observability.** Structured logs for each request (request_id, ml
  score, rules fired, agent tool latencies, total latency). Ship to
  CloudWatch or similar. Nothing like this exists today.

### Tier 4 — research-y ideas worth exploring

- **LTV-style filer scoring.** Instead of a binary TTAB label, predict the
  probability that a filer's *next* mark will be cancelled given their
  prior-N history. Closer to the true product question.
- **Two-stage classifier.** Stage 1: cheap logistic regression on the 8
  applicant features only, used as a pre-filter. Stage 2: the full stack
  only on borderline cases. Would cut ONNX latency for obviously-clean
  filers.
- **Active learning loop.** When the agent and ML layers disagree sharply
  (high ml.score, `safe` agent verdict, or vice versa), log the case for
  human review and feed confirmed labels back into the next retrain.
- **USPTO prosecution event feed.** If we ever get access to live event
  data (office actions, oppositions, statements of use), we can restore
  the 9 dropped features and get back the richer mark-side signal the
  prior v4 was designed around. This is a data-access problem, not a
  modeling problem.
