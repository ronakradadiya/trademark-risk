# TrademarkRisk

Given a new trademark application — `(brand_name, applicant_name)` — this
system says whether the filing looks like a **troll** or a **legitimate
business**. Output is a `safe` / `review` / `high_risk` verdict with
per-policy reasoning, a calibrated ML filer-risk score, and the full agent
trace that produced the answer.

**Live:** https://trademark-risk.raradadi.workers.dev

## Why this problem

USPTO gets 600k+ trademark applications a year. A growing share are filed by
"trademark trolls" — individuals or shell LLCs who file marks with no intent
to use them, then extort real businesses for settlement. Leo Stoller is the
archetype; I420 LLC and similar numeric-corporation names appear regularly
in cancellation filings.

Two kinds of troll look different in data, and the system catches both:

- **Behavioral trolls** — high filing volume, high abandonment, no attorney.
  The ML layer sees these from USPTO history.
- **Reputational trolls** — small footprint in USPTO, famous in court
  records. The LLM agent catches these via `web_search`.

## How it works, one paragraph

A fast statistical model does the cheap triage, an LLM agent does the
expensive investigation, and the final verdict label is **derived from
structured signals — not from the LLM's free-form opinion**. The LLM is
responsible for evidence (per-policy `{triggered, confidence, reason}`); the
label is arithmetic over those.

## Architecture

```
                           ┌──────────────────┐
  POST /api/check  ───────▶│ lib/check.ts     │
  { brand, applicant }     └────────┬─────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │ Pre-resolve applicant history │  SQLite: 12.5M marks,
                    │ (lookup_applicant_history)    │  4.99M applicants
                    └────────────────┬──────────────┘
                                     ▼
              ┌──────────────────────────────┐      score > 0.85
              │ ML filer-risk (XGBoost ONNX) │ ─────────────────▶ high_risk
              │ 13 features, <50 ms          │      (ml_only)
              └────────────────┬─────────────┘
                               │ otherwise
                               ▼
              ┌──────────────────────────────────────┐
              │ Agent loop (GPT-4o, ReAct)           │
              │ Tools: check_uspto_marks,            │
              │        check_domain_age,             │
              │        web_search                    │
              │ 5 policies → PolicyBundle            │
              └────────────────┬─────────────────────┘
                               ▼
            ┌──────────────────────────────────────────┐
            │ Deterministic verdict from policies:     │
            │ score = Σ (2 if conf ≥ 0.8 else 1)       │
            │       over triggered policies            │
            │ score ≥ 4 → high_risk                    │
            │ score > 0 → review                       │
            │ else      → safe                         │
            └──────────────────────────────────────────┘
```

## The three design decisions that shape the system

**1. ML and LLM are complementary, not redundant.** The XGBoost model
(<100 ms, never hallucinates) catches trolls visible in their filing
pattern. The LLM catches the ones who aren't: reputational trolls with one
shell-company filing, mark-name collisions with famous brands, fresh
domains registered the week of filing. Each layer covers the other's blind
spots.

**2. The verdict label is derived, not LLM-chosen.** An earlier version let
the LLM pick `safe`/`review`/`high_risk` directly. It would show all 5
policies green and still return `review` — an inconsistency the UI made
obvious. The current verdict is a pure function of the policy bundle the
LLM produces. The LLM owns per-policy `{triggered, confidence, reason}`;
the label is arithmetic over those. The confidence threshold (0.8) turns
"this flag is hedgy" into "this flag counts half" — so two confident
triggers escalate, but five lukewarm maybes don't.

**3. Calibration beats marginal AUC.** An earlier model stacked XGBoost +
CatBoost through a Logistic Regression meta-learner and scored **0.001 AUC
higher** on validation. It was dropped: both base models are gradient-boosted
trees (no diversity premise), the meta-LR introduced a failure mode (14 of
22 features held constant at serving because they came from CSVs not in the
live feed), and the serving artifact was 7× larger. The current single
XGBoost hits **0.828 AUC, 0.141 Brier, 0.85 recall** on held-out test —
see [ml/models/MODEL_CARD.md](ml/models/MODEL_CARD.md) for the honest
side-by-side vs. the prior stack.

## What's real about the data

- **12.5M USPTO marks, 4.99M applicants** in a local SQLite built from
  USPTO's TRTYRAP XML feed (April 2026 snapshot). Monthly-refreshable.
  FTS5 index powers the agent's `check_uspto_marks` tool.
- **Temporally-safe applicant rollups at training time** — for each
  training mark, applicant features (`filing_count_2yr`,
  `abandonment_rate`, `attorney_case_count`, …) are computed from filings
  with `filing_date < current.filing_date` only. No label leakage.
- **Fraud labels from 30,011 TTAB cancellations** — 28,527 CANG
  (cancellation-granted) events plus 2,207 TTAB XML-scraped serials.
  Real adversarial supervision, not abandonment proxies.
- **Join rate** of TTAB serials against the live SQLite: **97.3%**.

## Demo scenarios

Built-in presets in the UI, designed to exercise each path:

| Applicant + Brand         | Signal                              | Expected           |
| ------------------------- | ----------------------------------- | ------------------ |
| Apple Inc. + AirTag Pro   | blue-chip filer, clean              | `safe`             |
| Nike Inc. + Nike Runners  | same owner, same class              | `safe`             |
| Safetykit Inc. + XBox     | clean filer, famous-mark collision  | `review` / `high_risk` (P1 fires) |
| Leo Stoller + EZ Profits  | famous reputational troll           | `high_risk` (P2 + P5 strong) |
| I420 LLC + LA420          | real shell-LLC filer                | `high_risk` (ML short-circuit) |

## Infrastructure

- **Backend:** Docker container on a `t4g.small` EC2 (Arm64), image pulled
  from ECR. Provisioned by the CDK stack in [infra/](infra/). Restart
  automated via SSM `send-command`.
- **HTTPS front:** Cloudflare Worker in [worker/](worker/) proxies every
  request to the EC2 via `nip.io` magic DNS (sidesteps CF error 1003 on raw
  IPs). Free tier, 100k requests/day. No domain purchased, no ACM cert
  managed.
- **Secrets:** `OPENAI_API_KEY`, `SERPER_API_KEY`, `OPENAI_MODEL` in SSM
  Parameter Store; the restart script pulls them at container start and
  passes via `-e` inheritance — no keys in Dockerfile layers or env files.

## Local development

```bash
cd app
npm install

# app/.env.local
#   OPENAI_API_KEY=sk-...
#   SERPER_API_KEY=...
#   DYNAMO_AUDIT_DISABLED=1

npm run dev     # → http://localhost:3000
npm test        # 70 tests across 4 suites
```

Tests are plain `tsx` scripts — no Jest / Vitest. The agent layer is
dependency-injected (`AgentDeps`) so `npm run test:core` runs the full
ReAct loop against a fake OpenAI client and a stub classifier, deterministic
and offline.

Building `data/uspto.sqlite` locally requires the TRTYRAP XML dumps
(~15 GB, `data/trtyrap/download.sh`) and `npx tsx scripts/ingest_uspto.ts`
(~40 min on an M-series laptop). The deployed image ships with a prebuilt
copy.

## Honest limitations

- **The ML can't see famous trolls with only one filing.** Leo Stoller has
  filed 2,000+ marks in his career, but spread across shell companies
  (Rentamark, Central Mfg, Stealth Industries…). Under the name "Leo
  Stoller" alone, USPTO shows one filing from 2006. So the ML scores him
  0.01 — it has no idea who he is. The LLM picks him up from news and
  court records instead. This isn't a bug; the two layers were designed to
  cover for each other.
- **One ML feature doesn't match between training and serving.** The feature
  `days_since_filing` is the age of the mark. In training data it's years;
  for a brand-new application being checked, it's always 0. The tree model
  handles this gracefully, but it's a mismatch that a future retrain
  should close.
- **The ML doesn't compare brand names.** It looks at the applicant's
  history, not whether "XBox" is already taken. That check happens in the
  agent's `check_uspto_marks` tool, and it uses exact text matching — not
  fuzzy similarity. A misspelling like "XBoxx" would slip past.
- **No safety net on model rollout.** When I ship a new model it replaces
  the old one immediately. There's no gradual rollout, no automatic alarm
  if accuracy drops, no shadow run to compare old vs. new on live traffic.

## Key files to read

- [app/src/lib/agent.ts](app/src/lib/agent.ts) — ReAct loop, tool
  orchestration, deterministic verdict derivation.
- [app/src/lib/policies.ts](app/src/lib/policies.ts) — the 5 policy
  definitions the LLM reasons over (tone, signals, do-nots).
- [app/src/schemas/index.ts](app/src/schemas/index.ts) — Zod contracts,
  single source of truth for every HTTP / tool / LLM boundary.
- [ml/src/build_from_sqlite.py](ml/src/build_from_sqlite.py) — training
  pipeline with temporally-safe rollups.
- [ml/models/MODEL_CARD.md](ml/models/MODEL_CARD.md) — honest side-by-side
  evaluation vs. the prior v4 stacking ensemble.
