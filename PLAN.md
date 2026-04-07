# TrademarkRisk — Full Build Plan for Claude Code

## What we are building

A production-grade trademark risk intelligence platform that mirrors SafetyKit's exact three-layer architecture:
- **Real-time ML** (XGBoost + LightGBM ensemble) — fast pre-classifier
- **Agentic AI** (GPT-4o with 4 tools, 5 policies) — deep investigation
- **Human intelligence** (audit dashboard) — review and validate

Target user: A small business owner who types "BrewBox Coffee" and wants to know if it is safe to trademark before spending money on branding.

---

## Monorepo structure

```
trademark-risk/
├── ml/                          # Python — data pipeline, training, eval
│   ├── data/
│   │   ├── raw/                 # downloaded USPTO files go here
│   │   └── processed/           # cleaned, labeled, feature-engineered CSVs
│   ├── src/
│   │   ├── download.py          # downloads USPTO datasets
│   │   ├── clean.py             # cleans and joins datasets, creates labels
│   │   ├── features.py          # engineers all features from raw columns
│   │   ├── train.py             # trains XGBoost + LightGBM, exports ONNX
│   │   ├── ensemble.py          # stacking ensemble with LR meta-model
│   │   └── evaluate.py          # precision, recall, F1, AUC-ROC report
│   ├── models/                  # saved model files (.onnx, .joblib)
│   ├── evals/
│   │   └── cases.json           # 100 hand-labeled real USPTO cases
│   ├── notebooks/               # exploratory analysis (optional)
│   ├── requirements.txt
│   └── README.md
│
├── app/                         # TypeScript — agent, frontend, infra
│   ├── src/
│   │   ├── app/                 # Next.js 14 app router
│   │   │   ├── page.tsx         # main UI — three panel layout
│   │   │   ├── layout.tsx
│   │   │   └── api/
│   │   │       └── check/
│   │   │           └── route.ts # POST handler — validates, calls agent, stores result
│   │   ├── components/
│   │   │   ├── CheckerForm.tsx  # brand name input + policy pills
│   │   │   ├── AgentTrace.tsx   # live tool execution log
│   │   │   ├── VerdictCard.tsx  # safe/review/high-risk output + confidence bars
│   │   │   └── AuditLog.tsx     # recent checks table
│   │   ├── lib/
│   │   │   ├── agent.ts         # GPT-4o agent orchestrator (ReAct loop)
│   │   │   ├── policies.ts      # P1-P5 policy definitions in plain English
│   │   │   ├── classifier.ts    # loads ONNX model, runs ML inference
│   │   │   └── dynamo.ts        # DynamoDB read/write helpers
│   │   ├── tools/
│   │   │   ├── check_uspto_marks.ts      # USPTO TESS API → P1
│   │   │   ├── check_domain_age.ts       # WhoisJSON API → P3
│   │   │   ├── web_search.ts             # Serper API → P2, P5
│   │   │   └── check_attorney.ts         # USPTO OED search → P4
│   │   └── schemas/
│   │       └── index.ts         # Zod schemas shared across everything
│   ├── infra/
│   │   ├── stack.ts             # AWS CDK — Lambda, DynamoDB, API Gateway, IAM
│   │   └── cdk.json
│   ├── evals/
│   │   └── run_evals.ts         # runs agent on test cases, reports accuracy
│   ├── .github/
│   │   └── workflows/
│   │       └── deploy.yml       # test → build → CDK deploy on push to main
│   ├── package.json
│   ├── tsconfig.json
│   ├── tailwind.config.ts
│   └── next.config.ts
│
└── README.md                    # SafetyKit-framed write-up with live demo link
```

---

## Tech stack

### Python (ML layer)
- Python 3.10+
- pandas — data cleaning and feature engineering
- scikit-learn — Logistic Regression baseline, preprocessing, metrics
- xgboost — primary classifier
- lightgbm — secondary classifier
- skl2onnx — export trained model to ONNX format
- onnxruntime — validate ONNX export works correctly
- matplotlib + seaborn — confusion matrix, ROC curve plots
- requests — downloading USPTO data
- jupyter — exploratory notebooks (optional)

### TypeScript (everything else)
- Next.js 14 (app router)
- TailwindCSS + shadcn/ui
- Zod — shared schemas frontend to backend to infra
- OpenAI SDK — GPT-4o agent with function calling
- onnxruntime-node — run ML model in Lambda
- AWS CDK v2 — infrastructure as code
- AWS SDK v3 — DynamoDB client
- GitHub Actions — CI/CD

### APIs
- USPTO TESS API — trademark search (free, no key)
- Serper API — Google web search (free tier, 2500 req/mo)
- WhoisJSON API — domain age lookup (free tier, 500 req/mo)
- USPTO OED — attorney bar directory (free, public)

### Infrastructure
- AWS Lambda — runs the GPT-4o agent
- DynamoDB — audit log of every check
- API Gateway — HTTP endpoint
- Vercel — Next.js frontend deployment
- GitHub Actions — CI/CD pipeline

---

## Data sources

### For ML training (downloaded once)

**Dataset 1 — USPTO Trademark Case Files Dataset**
- URL: https://www.uspto.gov/ip-policy/economic-research/research-datasets/trademark-case-files-dataset
- What it is: 12.7 million trademark applications from 1870 to March 2024
- Used for: source of all features + negative class (legitimate marks)
- How to get legitimate labels: filter to marks with status = "Registered" AND section_8_filed = true

**Dataset 2 — TTAB Show-Cause Orders**
- URL: https://ttabvue.uspto.gov
- What it is: USPTO proceedings against bad-faith filers and trademark trolls
- Used for: positive class (fraud labels)
- How to use: extract serial numbers from show-cause proceedings, join with Dataset 1

**The join operation:**
```
fraud_serials = serial numbers from TTAB show-cause orders → label = 1
legit_serials = registered marks with section_8_filed = true → label = 0
combined = join both against Dataset 1 on serial_number
```

### For live agent calls (per check, real-time)
- USPTO TESS: similar mark search → P1
- Serper: Google search for commercial activity → P2, P5
- WhoisJSON: domain registration date → P3
- USPTO OED: attorney bar status → P4

---

## ML model — full detail

### Raw files and their exact column headers (verified from actual downloaded data)

```
case_file.csv
  Key columns: serial_no, filing_dt, registration_dt, abandon_dt, acq_dist_in,
               use_afdv_acc_in, opposit_pend_in, cfh_status_cd, cfh_status_dt,
               mark_draw_cd, lb_itu_cur_in, lb_use_cur_in, renewal_file_in,
               publication_dt, registration_no

owner.csv
  Key columns: serial_no, own_id, own_name, own_entity_cd, own_entity_desc,
               own_type_cd, own_seq, own_addr_country_cd, own_addr_state_cd

classification.csv
  Key columns: serial_no, class_primary_cd, class_intl_count, class_status_cd,
               class_status_dt, first_use_any_dt, first_use_com_dt, class_id

correspondent_domrep_attorney.csv
  Key columns: serial_no, attorney_no, attorney_name, domestic_rep_name

event.csv
  Key columns: serial_no, event_cd, event_dt, event_seq, event_type_cd

intl_class.csv
  Key columns: serial_no, intl_class_cd, class_id

prior_mark.csv
  Key columns: serial_no, prior_no, prior_type_cd, rec_error
```

### Row counts (verified from actual downloaded data)

```
case_file.csv                  12,691,942 rows  — one row per trademark application
owner.csv                      28,273,548 rows  — multiple owners per application possible
event.csv                     209,031,162 rows  — every USPTO action ever taken, MUST use chunking
classification.csv             14,908,552 rows  — multiple classes per application possible
correspondent_domrep.csv       12,691,942 rows  — one row per application
prior_mark.csv                  2,570,211 rows  — only applications citing prior marks
```

### CRITICAL: event.csv requires chunked processing

event.csv has 209 million rows. Loading it entirely crashes a laptop (needs ~20GB RAM).
Always process in chunks of 1 million rows:

```python
office_action_codes = ['FOAN', 'FOAS', 'OAEX', 'NFOA', 'ROAN']
opposition_codes = ['NOPI', 'OPIN']
sou_codes = ['SOUA', 'SOUF']

oa_counts = {}
opp_counts = {}
sou_filed = {}

for chunk in pd.read_csv('ml/data/raw/event.csv', chunksize=1_000_000):
    # office actions
    oa = chunk[chunk.event_cd.isin(office_action_codes)]
    for serial, count in oa.groupby('serial_no').size().items():
        oa_counts[serial] = oa_counts.get(serial, 0) + count
    # oppositions
    opp = chunk[chunk.event_cd.isin(opposition_codes)]
    for serial, count in opp.groupby('serial_no').size().items():
        opp_counts[serial] = opp_counts.get(serial, 0) + count
    # statement of use
    sou = chunk[chunk.event_cd.isin(sou_codes)]
    for serial in sou.serial_no.unique():
        sou_filed[serial] = 1
```

### Features (what the model trains on) — final 19 features

**Why 19 and not the earlier numbers discussed:**
- Started with 12, grew to 18 with additional columns from existing files
- Added owner_abandonment_rate (19) — measures quality of owner's filing history
- Dropped owner_lifetime_filing_count — redundant with owner_filing_count_2yr for detecting burst behavior
- Dropped owner_section8_rate — too correlated with owner_abandonment_rate and section_8_filed
- Final count: 19 clean, non-redundant features each measuring something different

**Why owner_filing_count_2yr is valid despite being 1 for first filings:**
- Computed for every row in the dataset, not just first filings
- For a troll's 50th filing in a burst, count = 49. For their 200th, count = 180.
- Measures filing velocity at the exact moment of each application
- Sparse for first-time filers but very informative for established trolls
- Fraud cases in TTAB data are disproportionately later filings by high-volume owners
- Works in combination with owner_abandonment_rate which catches new trolls on first filing

**Why owner_abandonment_rate was added:**
- Directly answers "does this owner actually use their marks?"
- Nike: 2000+ filings, 2% abandoned → legitimate
- Troll: 500 filings, 90% abandoned → fraud
- Works regardless of filing volume — ratio normalizes for company size
- Catches new trolls after even 5-10 filings once abandonment pattern emerges

---

**GROUP 1 — Owner behavior signals (3 features)**
These detect the behavioral fingerprint of trademark trolls.

| Feature | Source file | Exact column(s) | Computation | Fraud signal when |
|---|---|---|---|---|
| owner_filing_count_2yr | owner.csv + case_file.csv | own_id, filing_dt | for each row, count other rows by same own_id where filing_dt is within 2 years BEFORE this row's filing_dt. Computed per row — not just first filing. | very high (50+) indicating burst behavior |
| owner_abandonment_rate | owner.csv + case_file.csv | own_id, abandon_dt | group all rows by own_id, compute: count(abandon_dt not null) / count(all rows). Rate of abandonment across owner's entire filing history. 0 if only 1 filing (unreliable sample). | very high (0.7+) — owner abandons most marks |
| owner_is_individual | owner.csv | own_entity_cd | 1 if own_entity_cd == 1 (individual), else 0. Codes: 1=individual, 2=corp, 3=LLC, 4=partnership, 5=JV, 6=sole proprietor, 7=association, 9=other | 1 combined with high filing count and abandonment rate |

---

**GROUP 2 — External signals (2 features)**
Attorney behavior and geographic signals.

| Feature | Source file | Exact column(s) | Computation | Fraud signal when |
|---|---|---|---|---|
| attorney_case_count | correspondent_domrep_attorney.csv | attorney_no | group by attorney_no, count total rows across entire file. This is attorney's lifetime case volume — not rolling window. | very high (5000+) indicating troll mill attorney |
| owner_is_foreign | owner.csv | own_addr_country_cd | 1 if own_addr_country_cd is not empty and not 'US', else 0 | 1 combined with other signals |

---

**GROUP 3 — Filing timing signals (4 features)**
These detect the rushed, reactive behavior of trolls who spot a trending name and race to file.

| Feature | Source file | Exact column(s) | Computation | Fraud signal when |
|---|---|---|---|---|
| days_domain_to_filing | case_file.csv + WhoisJSON API | filing_dt + API response | filing_dt minus domain_creation_date in days. Negative = domain registered AFTER trademark filing — strong fraud signal | less than 30 days or negative |
| days_since_filing | case_file.csv | filing_dt | today minus filing_dt in days | context only — needed to correctly interpret section_8_filed for recent marks |
| days_filing_to_registration | case_file.csv | filing_dt, registration_dt | registration_dt minus filing_dt in days. Use -1 if registration_dt is empty (never registered) | -1 (never registered) or very long (1500+ days) |
| days_first_use_to_filing | classification.csv + case_file.csv | first_use_any_dt, filing_dt | filing_dt minus first_use_any_dt in days. Use -999 sentinel if first_use_any_dt empty | zero or very small (claimed use on same day or days before filing) |

---

**GROUP 4 — USPTO examination signals (3 features)**
These capture how USPTO itself reacted. When USPTO pushes back repeatedly, something is wrong.

| Feature | Source file | Exact column(s) | Computation | Fraud signal when |
|---|---|---|---|---|
| office_action_count | event.csv | event_cd, serial_no | count rows per serial_no where event_cd in ['FOAN','FOAS','OAEX','NFOA','ROAN']. MUST use chunked processing — 209M rows | high (4+) |
| opposition_count | event.csv | event_cd, serial_no | count rows per serial_no where event_cd in ['NOPI','OPIN']. MUST use chunked processing | any opposition (1+) is significant |
| statement_of_use_filed | event.csv | event_cd, serial_no | 1 if any row for serial_no has event_cd in ['SOUA','SOUF'], else 0. MUST use chunked processing | 0 combined with filing_basis=1 (filed intent to use but never proved use) |

---

**GROUP 5 — Commercial legitimacy signals (4 features)**
These detect whether a real business exists behind the mark.

| Feature | Source file | Exact column(s) | Computation | Fraud signal when |
|---|---|---|---|---|
| section_8_filed | case_file.csv | use_afdv_acc_in | read directly — already 0 or 1 in dataset | 0 for marks older than 5 years (should have filed by now) |
| has_acquired_distinctiveness | case_file.csv | acq_dist_in | read directly — already 0 or 1 | 0 is neutral. 1 is legitimacy signal (claimed long commercial history) |
| was_abandoned | case_file.csv | abandon_dt | 1 if abandon_dt is not null/empty, else 0 | 1 combined with other signals |
| is_currently_active | case_file.csv | cfh_status_cd | 1 if cfh_status_cd in 600-699 (registered) or 700-799 (pending), 0 if 800-899 (abandoned) or 900-999 (cancelled) | 0 — mark is no longer active |

---

**GROUP 6 — Application scope signals (3 features)**
These detect overreach — trolls file as broadly as possible to maximize leverage.

| Feature | Source file | Exact column(s) | Computation | Fraud signal when |
|---|---|---|---|---|
| class_breadth | classification.csv | serial_no, class_primary_cd | count distinct class_primary_cd values per serial_no. Count rows not the class_intl_count column. | very high (6+) — filing in many unrelated industries |
| filing_basis | case_file.csv | lb_itu_cur_in | read lb_itu_cur_in directly — 1 if intent to use, 0 if use in commerce | 1 combined with statement_of_use_filed=0 |
| specimen_type_encoded | case_file.csv | mark_draw_cd | encode: 0=unknown, 1=typed drawing, 2=design mark, 3=stylized, 4=standard chars | weakest feature — typed only (1) marginally suspicious |

### own_entity_cd lookup table (from owner.csv)

```
1 = individual person
2 = corporation
3 = limited liability company (LLC)
4 = partnership
5 = joint venture
6 = sole proprietorship
7 = association
9 = other
```

### cfh_status_cd ranges (from case_file.csv)

```
600-699 = registered marks        → is_currently_active = 1
700-799 = pending marks           → is_currently_active = 1
800-899 = abandoned marks         → is_currently_active = 0
900-999 = cancelled marks         → is_currently_active = 0
Example seen in data: 626 = registered
```

### mark_draw_cd values (from case_file.csv)

```
1 = typed drawing (plain text, no design)
2 = design mark (has a logo or image element)
3 = stylized or special form (stylized text)
4 = standard characters
0 or empty = unknown
```

### days_domain_to_filing — special handling

This feature requires a live WhoisJSON API call per application. For 12 million rows this is impractical. Strategy:
- Only compute for your training sample (~80,000 rows)
- Derive domain name from brand name: lowercase, remove spaces and punctuation, append .com
- Call WhoisJSON API: GET https://whoisjson.com/api/v1/whois?domain=brewboxcoffee.com
- Extract creation_date from response
- Compute filing_dt minus creation_date in days
- If domain not found or API error: set to null, handle as missing value in model

### Missing value strategy per feature

```
GROUP 1 — Owner behavior
owner_filing_count_2yr    → 1 if own_id not found (treat as first-time filer)
owner_abandonment_rate    → 0 if owner has only 1 filing (unreliable sample size)
                            compute only when owner has 3+ filings, else 0
owner_is_individual       → 0 if own_entity_cd null (default to non-individual)

GROUP 2 — External signals
attorney_case_count       → 0 if attorney_no empty (many old marks have no attorney)
owner_is_foreign          → 0 if country code empty (default to domestic)

GROUP 3 — Timing signals
days_domain_to_filing     → -999 sentinel value (model learns this means domain unknown)
days_since_filing         → always computable — filing_dt always present in case_file
days_filing_to_registration → -1 if registration_dt empty (never registered)
days_first_use_to_filing  → -999 sentinel value if first_use_any_dt empty

GROUP 4 — USPTO examination
office_action_count       → 0 if serial_no not in event counts dict
opposition_count          → 0 if serial_no not in event counts dict
statement_of_use_filed    → 0 if serial_no not in sou dict

GROUP 5 — Commercial legitimacy
section_8_filed           → already 0 if not filed (no nulls expected)
has_acquired_distinctiveness → already 0 if not claimed (no nulls expected)
was_abandoned             → 0 if abandon_dt empty (not abandoned)
is_currently_active       → 0 if cfh_status_cd null or unrecognized code

GROUP 6 — Application scope
class_breadth             → 1 if serial_no not in classification (assume 1 class minimum)
filing_basis              → 0 if both lb_itu_cur_in and lb_use_cur_in are 0 (old marks)
specimen_type_encoded     → 0 if mark_draw_cd empty or unrecognized
```

### Training process

**Step 1 — Baseline**
Train Logistic Regression. Record precision, recall, F1, AUC-ROC on validation set. This is your floor.

**Step 2 — XGBoost**
```python
XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=3,  # handle class imbalance (3:1 ratio)
    eval_metric='aucpr',
    early_stopping_rounds=50
)
```

**Step 3 — LightGBM**
```python
LGBMClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    class_weight='balanced',
    metric='average_precision'
)
```

**Step 4 — CatBoost**
```python
CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    auto_class_weights='Balanced'
)
```

**Step 5 — Stacking ensemble**
Use best two models from steps 2-4 as base models. Train Logistic Regression as meta-model on their validation predictions. This is your final model.

**Step 6 — Threshold tuning**
Plot precision-recall curve. Find threshold that maximizes F1 while keeping recall above 0.85 (catching fraud is more important than avoiding false alarms).

**Step 7 — Export**
Export final ensemble to ONNX. Validate with onnxruntime. Copy to app/src/models/.

### Target metrics
- F1 score: 0.88+
- AUC-ROC: 0.92+
- Recall: 0.85+ (must catch most fraud)
- Precision: 0.80+ (avoid too many false alarms)

---

## GPT-4o agent — full detail

### How it works (ReAct loop)

```
1. Receive brand name + ML confidence score
2. Reason: which tool gives me the most information first?
3. Act: call that tool
4. Observe: read the result
5. Reason: what does this tell me? what do I need next?
6. Repeat until all relevant policies can be evaluated
7. Apply P1-P5 against all gathered evidence
8. Return structured JSON verdict
```

### The 5 policies

**P1 — Confusingly similar mark exists**
A confusingly similar mark exists if: another registered or pending mark has the same or similar name in the same or related international class. Similarity includes phonetic similarity (sounds the same), visual similarity (looks the same), and meaning similarity (translates to the same thing). Evaluate using check_uspto_marks results.

**P2 — Filer shows trademark troll pattern**
A filer is a trademark troll if: they have filed many trademarks across unrelated industries, there is no evidence of a real operating business under any of their marks, there are news articles or legal records of them demanding money from businesses, or they have a history of filing and abandoning marks after receiving settlement payments. Evaluate using web_search results and ML owner_filing_count feature.

**P3 — Filing timing suspicious**
Filing timing is suspicious if: the associated domain was registered less than 30 days before the trademark filing, or the domain does not exist at all, or the domain was registered after the trademark filing. This suggests the filer is not a real business that grew organically into needing a trademark. Evaluate using check_domain_age results.

**P4 — Attorney credentials suspicious**
Attorney credentials are suspicious if: the attorney is not found in the USPTO OED directory, their bar status is inactive or suspended, they have disciplinary history or sanctions, or they are known to work exclusively with high-volume troll filers. Evaluate using check_attorney results.

**P5 — No genuine commercial use evidence**
No genuine commercial use exists if: there is no real website with actual products or services, no customer reviews on any platform, no social media presence with real followers and activity, no press coverage or business listings, and no evidence the brand name is associated with a real operating business. Evaluate using web_search results.

### Agent system prompt

```
You are a trademark risk investigator helping small businesses 
determine if a brand name is safe to trademark before spending 
money on branding and legal fees.

You have 4 tools:
- check_uspto_marks(brand_name, class_code)
- check_domain_age(domain_name)
- web_search(query)
- check_attorney(attorney_name, bar_number)

Investigate the brand name thoroughly. Call tools in whatever 
order makes sense based on what you find. You may call the same 
tool multiple times with different queries.

Then evaluate each of the 5 policies below and return a 
structured verdict.

Policies:
[P1-P5 definitions as above]

Return JSON only. No prose outside the JSON structure.

Schema:
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
  "tools_used": string[],
  "summary": string (one sentence plain English)
}
```

### How ML and agent combine

```typescript
const ml_confidence = await runMLClassifier(features)

if (ml_confidence < 0.3) {
  return { verdict: 'safe', source: 'ml_only', confidence: ml_confidence }
}

if (ml_confidence > 0.85) {
  return { verdict: 'high_risk', source: 'ml_only', confidence: ml_confidence }
}

// uncertain — run full agent
const agent_result = await runAgent(brand_name, ml_confidence)

// combine scores (agent weighted higher — has richer information)
const final_confidence = (ml_confidence * 0.35) + (agent_result.overall_confidence * 0.65)

return { ...agent_result, final_confidence, source: 'ml_and_agent' }
```

---

## Zod schemas (shared contract)

```typescript
// schemas/index.ts — single source of truth for all types

export const CheckRequestSchema = z.object({
  brand_name: z.string().min(1).max(200),
  domain_name: z.string().optional(),
  attorney_name: z.string().optional(),
  class_code: z.number().optional()
})

export const PolicyResultSchema = z.object({
  triggered: z.boolean(),
  confidence: z.number().min(0).max(1),
  reason: z.string()
})

export const VerdictSchema = z.object({
  brand: z.string(),
  verdict: z.enum(['safe', 'review', 'high_risk']),
  overall_confidence: z.number().min(0).max(1),
  ml_confidence: z.number().min(0).max(1),
  source: z.enum(['ml_only', 'ml_and_agent']),
  policies: z.object({
    P1: PolicyResultSchema,
    P2: PolicyResultSchema,
    P3: PolicyResultSchema,
    P4: PolicyResultSchema,
    P5: PolicyResultSchema
  }),
  tools_used: z.array(z.string()),
  summary: z.string(),
  checked_at: z.string()
})

export const AuditRecordSchema = VerdictSchema.extend({
  id: z.string(),
  ttl: z.number()
})

export type CheckRequest = z.infer<typeof CheckRequestSchema>
export type Verdict = z.infer<typeof VerdictSchema>
export type AuditRecord = z.infer<typeof AuditRecordSchema>
```

---

## DynamoDB schema

```
Table: trademark-risk-checks
Partition key: id (string) — uuid
Sort key: checked_at (string) — ISO timestamp

Attributes:
- brand (string)
- verdict (string)
- overall_confidence (number)
- ml_confidence (number)
- source (string)
- policies (map)
- tools_used (list)
- summary (string)
- ttl (number) — unix timestamp 90 days from now (auto-expire)

GSI: brand-index
- Partition key: brand
- Sort key: checked_at
- Used for: "show me all checks for BrewBox Coffee"
```

---

## AWS CDK stack

```typescript
// infra/stack.ts

// DynamoDB table
const table = new Table(this, 'ChecksTable', {
  partitionKey: { name: 'id', type: AttributeType.STRING },
  sortKey: { name: 'checked_at', type: AttributeType.STRING },
  timeToLiveAttribute: 'ttl',
  billingMode: BillingMode.PAY_PER_REQUEST
})

// Lambda function
const agentLambda = new Function(this, 'AgentLambda', {
  runtime: Runtime.NODEJS_18_X,
  handler: 'index.handler',
  timeout: Duration.seconds(60), // agent needs time for tool calls
  memorySize: 1024,              // ONNX model needs memory
  environment: {
    OPENAI_API_KEY: process.env.OPENAI_API_KEY!,
    SERPER_API_KEY: process.env.SERPER_API_KEY!,
    WHOIS_API_KEY: process.env.WHOIS_API_KEY!,
    DYNAMODB_TABLE: table.tableName
  }
})

// Grant Lambda read/write to DynamoDB
table.grantReadWriteData(agentLambda)
```

---

## GitHub Actions CI/CD

```yaml
# .github/workflows/deploy.yml

name: Deploy

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'

      - name: Install dependencies
        run: cd app && npm install

      - name: Type check
        run: cd app && npm run typecheck

      - name: Run tests
        run: cd app && npm test

      - name: Deploy CDK
        run: cd app && npm run cdk:deploy
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          SERPER_API_KEY: ${{ secrets.SERPER_API_KEY }}
          WHOIS_API_KEY: ${{ secrets.WHOIS_API_KEY }}
```

---

## Evaluation harness

### ML eval (Python)
Run trained model against 100 hand-labeled USPTO cases in `ml/evals/cases.json`. Report:
- Accuracy
- Precision
- Recall
- F1 score
- AUC-ROC
- Confusion matrix (saved as PNG)
- Feature importance chart (saved as PNG)

### Agent eval (TypeScript)
Run full agent pipeline against same 100 cases. Report:
- Per-policy accuracy (how often did each policy trigger correctly)
- Overall verdict accuracy
- Average tool calls per check
- Average latency per check
- Average cost per check (OpenAI tokens)

### Target numbers for README
- ML F1: 0.88+
- Agent verdict accuracy: 85%+
- Average latency: under 8 seconds
- Average cost per check: under $0.02

---

## Build order for Claude Code

### Phase 1 — Python ML pipeline
Start here. Get data, clean it, train model, export ONNX. Everything else depends on this working.

```
1. ml/requirements.txt
2. ml/src/download.py        — downloads USPTO Case Files Dataset
3. ml/src/clean.py           — cleans data, joins datasets, creates labels
4. ml/src/features.py        — engineers all 19 features
5. ml/src/train.py           — trains XGBoost + LightGBM
6. ml/src/ensemble.py        — stacking ensemble
7. ml/src/evaluate.py        — full metrics report
8. ml/evals/cases.json       — 100 labeled test cases
```

### Phase 2 — TypeScript schemas and types
Define all shared contracts before writing any logic.

```
9.  app/src/schemas/index.ts  — all Zod schemas
10. app/tsconfig.json
11. app/package.json
```

### Phase 3 — Agent tools
Build each tool independently and test it before wiring into agent.

```
12. app/src/tools/check_uspto_marks.ts
13. app/src/tools/check_domain_age.ts
14. app/src/tools/web_search.ts
15. app/src/tools/check_attorney.ts
```

### Phase 4 — Agent core
```
16. app/src/lib/policies.ts   — P1-P5 definitions
17. app/src/lib/classifier.ts — ONNX model loader + inference
18. app/src/lib/agent.ts      — GPT-4o ReAct orchestrator
19. app/src/lib/dynamo.ts     — DynamoDB helpers
```

### Phase 5 — API route
```
20. app/src/app/api/check/route.ts
```

### Phase 6 — Frontend
```
21. app/src/components/CheckerForm.tsx
22. app/src/components/AgentTrace.tsx
23. app/src/components/VerdictCard.tsx
24. app/src/components/AuditLog.tsx
25. app/src/app/page.tsx
26. app/src/app/layout.tsx
```

### Phase 7 — Infrastructure
```
27. app/infra/stack.ts
28. app/infra/cdk.json
29. .github/workflows/deploy.yml
```

### Phase 8 — Evaluation
```
30. app/evals/run_evals.ts
31. ml/src/evaluate.py (final run with frozen model)
```

### Phase 9 — README
```
32. README.md (after everything works)
```

---

## Environment variables

### Python (.env in ml/)
```
# No API keys needed for data download
# USPTO data is fully public
```

### TypeScript (.env.local in app/)
```
OPENAI_API_KEY=sk-...
SERPER_API_KEY=...
WHOIS_API_KEY=...
AWS_REGION=us-east-1
DYNAMODB_TABLE=trademark-risk-checks
```

---

## Key decisions and why

**Why XGBoost + LightGBM ensemble over single model**
Each model makes different mistakes on different cases. Combining them with a meta-model catches errors that either alone would miss. Expected accuracy gain of 2-4% over best single model.

**Why ONNX for model export**
ONNX is runtime-agnostic. Train in Python, run in Node.js Lambda without any Python runtime. Clean separation between ML and application layers.

**Why ReAct pattern for agent**
Allows the agent to adapt its tool calling based on what it finds. If USPTO search reveals a troll attorney name, agent can immediately call check_attorney without being pre-programmed to do so. More robust than fixed tool call sequences.

**Why DynamoDB over PostgreSQL**
SafetyKit uses DynamoDB. Serverless, no connection pooling issues in Lambda, scales automatically, TTL built in for audit log expiry.

**Why Vercel for frontend + Lambda for backend**
Separates concerns. Frontend deploys in seconds on Vercel. Heavy agent work runs in Lambda with 60 second timeout. Next.js API routes proxy to Lambda.

**Why scale_pos_weight=3 in XGBoost**
Dataset has roughly 3 legitimate marks for every 1 fraud case. Without this, model learns to predict legitimate for everything and still gets 75% accuracy but misses all fraud.

---

## What goes on your resume when done

- Built trademark fraud detection system achieving 88%+ F1 on 80,000 USPTO cases using XGBoost + LightGBM stacking ensemble
- Engineered 12 features from USPTO bulk data including temporal signals, network-level troll detection, and commercial use proxies
- Built GPT-4o agentic pipeline with 4 live data tools evaluating 5 fraud policies, returning structured verdicts in under 8 seconds
- Deployed full serverless architecture on AWS Lambda + DynamoDB + CDK with GitHub Actions CI/CD
- Reduced LLM API costs by 60%+ using ML pre-classifier to gate GPT-4o on uncertain cases only
- Mirrors SafetyKit's three-layer architecture: real-time ML, agentic AI, and human review dashboard

---

## Testing rules — apply to every single phase

These are non-negotiable. Claude Code must follow all of these before marking any phase complete.

**Rule 1 — Test before moving on**
Every phase has a test gate at the end. Do not start the next phase until every test in the current phase passes and results are printed to the console.

**Rule 2 — Print a test report**
After running tests for each phase, print a clearly formatted report like this:

```
========================================
PHASE X TEST REPORT
========================================
[PASS] test name — what it checked
[PASS] test name — what it checked
[FAIL] test name — what went wrong
========================================
X/Y tests passed
DO NOT proceed to Phase X+1 until all tests pass.
========================================
```

**Rule 3 — Test real behavior, not just that code runs**
Do not write tests that just check "function exists" or "no exception thrown." Every test must assert on actual output values.

**Rule 4 — Test failure paths too**
Every phase must include at least one test for what happens when something goes wrong — bad input, API failure, missing data. The system must fail gracefully, never crash.

**Rule 5 — Keep a test log**
Append every phase test report to a file called `TEST_LOG.md` in the root of the project. This becomes your evidence of quality.

---

## Four critical testing moments — highest priority

These four moments are more important than any other tests in the project. Claude Code must treat these as hard stops — not suggestions.

---

### Critical moment 1 — After data cleaning

Run this immediately after clean.py completes. Do not proceed to features.py until every check below passes.

**Label sanity check**
```python
print(f"Total rows: {len(df)}")
print(f"Fraud (label=1): {df.label.sum()} ({df.label.mean()*100:.1f}%)")
print(f"Legitimate (label=0): {(df.label==0).sum()} ({(1-df.label.mean())*100:.1f}%)")
assert 0.10 <= df.label.mean() <= 0.45, "Label ratio outside expected range — check join logic"
assert len(df) >= 10000, "Dataset too small — check download and join"
```

**No data leakage**
```python
# Serial numbers in test set must not appear in train set
train_serials = set(train_df.serial_number)
test_serials = set(test_df.serial_number)
overlap = train_serials.intersection(test_serials)
assert len(overlap) == 0, f"DATA LEAKAGE: {len(overlap)} serial numbers appear in both train and test"
```

**Manual label spot check**
```python
# Print 10 fraud cases for human review
print("\n=== 10 RANDOM FRAUD CASES — VERIFY THESE LOOK LIKE TROLLS ===")
print(df[df.label==1].sample(10)[['serial_number','owner_name','filing_date','class_breadth']])

# Print 10 legitimate cases for human review  
print("\n=== 10 RANDOM LEGITIMATE CASES — VERIFY THESE LOOK LIKE REAL BUSINESSES ===")
print(df[df.label==0].sample(10)[['serial_number','owner_name','filing_date','section_8_filed']])
```

**Feature range validation**
```python
checks = {
    # Group 1 — owner behavior
    'owner_filing_count_2yr':       (1, 50000),   # minimum 1 (first filing)
    'owner_abandonment_rate':       (0.0, 1.0),   # 0 if only 1 filing or no abandonment
    'owner_is_individual':          (0, 1),
    # Group 2 — external signals
    'attorney_case_count':          (0, 100000),
    'owner_is_foreign':             (0, 1),
    # Group 3 — timing
    'days_domain_to_filing':        (-3650, 36500),  # -999 sentinel allowed
    'days_since_filing':            (0, 55000),
    'days_filing_to_registration':  (-1, 10000),     # -1 means never registered
    'days_first_use_to_filing':     (-3650, 36500),  # -999 sentinel allowed
    # Group 4 — USPTO examination
    'office_action_count':          (0, 50),
    'opposition_count':             (0, 20),
    'statement_of_use_filed':       (0, 1),
    # Group 5 — commercial legitimacy
    'section_8_filed':              (0, 1),
    'has_acquired_distinctiveness': (0, 1),
    'was_abandoned':                (0, 1),
    'is_currently_active':          (0, 1),
    # Group 6 — application scope
    'class_breadth':                (1, 45),
    'filing_basis':                 (0, 1),
    'specimen_type_encoded':        (0, 4),
}
for col, (min_val, max_val) in checks.items():
    assert df[col].min() >= min_val, f"{col} has values below {min_val}: {df[col].min()}"
    assert df[col].max() <= max_val, f"{col} has values above {max_val}: {df[col].max()}"
    assert df[col].isna().sum() == 0, f"{col} has {df[col].isna().sum()} null values — check missing value strategy"
    print(f"[PASS] {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}, nulls=0")
```

**Feature statistics by class**
```python
print("\n=== FEATURE MEANS: FRAUD vs LEGITIMATE ===")
feature_cols = list(checks.keys())
print(df.groupby('label')[feature_cols].mean().T.rename(columns={0:'legitimate', 1:'fraud'}))

# Top 3 strongest features must show clear separation between classes
for col in ['owner_filing_count_2yr', 'owner_abandonment_rate', 'office_action_count']:
    fraud_mean = df[df.label==1][col].mean()
    legit_mean = df[df.label==0][col].mean()
    assert fraud_mean != legit_mean, f"{col} has same mean for fraud and legitimate — feature is useless"
    print(f"[PASS] {col} differs: fraud={fraud_mean:.2f}, legitimate={legit_mean:.2f}")

# owner_abandonment_rate specific check
# Fraud cases must have significantly higher abandonment rate than legitimate
fraud_abandon = df[df.label==1]['owner_abandonment_rate'].mean()
legit_abandon = df[df.label==0]['owner_abandonment_rate'].mean()
assert fraud_abandon > legit_abandon, "Fraud abandonment rate should be higher than legitimate"
print(f"[PASS] owner_abandonment_rate: fraud={fraud_abandon:.2f}, legitimate={legit_abandon:.2f}")
```

**GATE: do not run features.py until all above pass**

---

### Critical moment 2 — After running each model

Run this immediately after each model finishes training. Print the full report. Never just look at accuracy.

```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pandas as pd

def full_model_report(model_name, y_true, y_pred, y_prob):
    print(f"\n{'='*50}")
    print(f"MODEL: {model_name}")
    print(f"{'='*50}")
    
    # Full classification report
    print(classification_report(y_true, y_pred, target_names=['legitimate', 'fraud']))
    
    # Confusion matrix in plain numbers
    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion matrix:")
    print(f"  True Negative  (caught legitimate correctly): {cm[0][0]}")
    print(f"  False Positive (flagged legitimate as fraud): {cm[0][1]}")
    print(f"  False Negative (missed actual fraud):         {cm[1][0]}")
    print(f"  True Positive  (caught fraud correctly):      {cm[1][1]}")
    
    # AUC-ROC
    auc = roc_auc_score(y_true, y_prob)
    print(f"AUC-ROC: {auc:.4f}")
    
    # Fraud-specific metrics (what actually matters)
    fraud_precision = cm[1][1] / (cm[0][1] + cm[1][1])
    fraud_recall = cm[1][1] / (cm[1][0] + cm[1][1])
    print(f"Fraud precision (of flagged, how many were real fraud): {fraud_precision:.4f}")
    print(f"Fraud recall (of all fraud, how many did we catch):     {fraud_recall:.4f}")
    
    # Warnings
    if fraud_recall < 0.85:
        print(f"[WARN] Recall {fraud_recall:.2f} below target 0.85 — model missing too much fraud")
    if fraud_precision < 0.80:
        print(f"[WARN] Precision {fraud_precision:.2f} below target 0.80 — too many false alarms")
    
    return {'model': model_name, 'auc': auc, 'recall': fraud_recall, 'precision': fraud_precision}

# Call this after every model:
# full_model_report("Logistic Regression", y_val, lr_pred, lr_prob)
# full_model_report("XGBoost", y_val, xgb_pred, xgb_prob)
# full_model_report("LightGBM", y_val, lgbm_pred, lgbm_prob)
# full_model_report("CatBoost", y_val, cat_pred, cat_prob)
```

**Feature importance check after each tree model**
```python
def check_feature_importance(model, model_name, feature_names):
    importance = pd.Series(model.feature_importances_, index=feature_names)
    importance = importance.sort_values(ascending=False)
    print(f"\n=== FEATURE IMPORTANCE: {model_name} ===")
    for feat, score in importance.items():
        print(f"  {feat}: {score:.4f}")
    
    # Top feature should make intuitive sense
    top_feature = importance.index[0]
    print(f"\nTop feature: {top_feature}")
    if top_feature not in ['owner_filing_count_2yr', 'section_8_filed', 'days_domain_to_filing', 'opposition_count']:
        print(f"[WARN] Unexpected top feature '{top_feature}' — review feature engineering")
    else:
        print(f"[PASS] Top feature makes intuitive sense for fraud detection")
```

**GATE: do not build ensemble until all 4 models have printed their full reports**

---

### Critical moment 3 — After comparing results

This is your decision point. Be honest about what the numbers say.

```python
def compare_all_models(results):
    """results is list of dicts from full_model_report"""
    print(f"\n{'='*70}")
    print(f"{'MODEL COMPARISON TABLE':^70}")
    print(f"{'='*70}")
    print(f"{'Model':<25} {'AUC-ROC':>10} {'Recall':>10} {'Precision':>10}")
    print(f"{'-'*70}")
    for r in sorted(results, key=lambda x: x['auc'], reverse=True):
        print(f"{r['model']:<25} {r['auc']:>10.4f} {r['recall']:>10.4f} {r['precision']:>10.4f}")
    print(f"{'='*70}")
    
    best = max(results, key=lambda x: x['auc'])
    print(f"\nBest single model: {best['model']} (AUC-ROC: {best['auc']:.4f})")
    
    # Check if models make different mistakes — if yes, ensembling will help
    print("\n[ACTION REQUIRED] Review above table and decide:")
    print("  - Which two models have highest AUC-ROC? Those go into ensemble.")
    print("  - Do the two best models have similar recall? If yes, ensemble may not help much.")
    print("  - Document your choice before proceeding to ensemble.py")

# After ensemble is built:
def verify_ensemble_beats_base(ensemble_auc, base_model_aucs):
    best_base = max(base_model_aucs)
    improvement = ensemble_auc - best_base
    if ensemble_auc > best_base:
        print(f"[PASS] Ensemble AUC {ensemble_auc:.4f} beats best base model {best_base:.4f} (+{improvement:.4f})")
    else:
        print(f"[FAIL] Ensemble AUC {ensemble_auc:.4f} does NOT beat best base model {best_base:.4f}")
        print(f"       Check stacking setup — meta-model may be overfitting")
        raise AssertionError("Ensemble must beat best base model before proceeding")
```

**GATE: do not proceed to threshold tuning until ensemble beats both base models on validation AUC-ROC**

---

### Critical moment 4 — After fine-tuning parameters

Parameter tuning must be systematic — one parameter at a time, documented.

```python
def tune_and_document(model_class, base_params, param_name, values_to_try, X_val, y_val):
    """Try one parameter at multiple values, print results"""
    print(f"\n=== TUNING {param_name} ===")
    results = []
    for value in values_to_try:
        params = {**base_params, param_name: value}
        model = model_class(**params)
        model.fit(X_train, y_train)
        auc = roc_auc_score(y_val, model.predict_proba(X_val)[:,1])
        results.append((value, auc))
        print(f"  {param_name}={value}: AUC-ROC={auc:.4f}")
    
    best_value, best_auc = max(results, key=lambda x: x[1])
    print(f"  Best: {param_name}={best_value} (AUC-ROC={best_auc:.4f})")
    return best_value

# Threshold tuning — must be done after model is fully trained
def tune_threshold(y_true, y_prob):
    print(f"\n=== THRESHOLD TUNING ===")
    print(f"{'Threshold':>12} {'Precision':>12} {'Recall':>10} {'F1':>10}")
    print(f"{'-'*50}")
    best_f1 = 0
    best_threshold = 0.5
    for threshold in [i/20 for i in range(4, 17)]:  # 0.20 to 0.80
        y_pred = (y_prob >= threshold).astype(int)
        from sklearn.metrics import precision_score, recall_score, f1_score
        p = precision_score(y_true, y_pred)
        r = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        marker = " <-- best" if f1 > best_f1 and r >= 0.85 else ""
        print(f"{threshold:>12.2f} {p:>12.4f} {r:>10.4f} {f1:>10.4f}{marker}")
        if f1 > best_f1 and r >= 0.85:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"\nChosen threshold: {best_threshold} (F1={best_f1:.4f}, recall>=0.85 maintained)")
    return best_threshold

# FINAL TEST SET EVALUATION — run exactly once, at the very end
def final_test_evaluation(model, X_test, y_test, threshold):
    print(f"\n{'='*60}")
    print(f"FINAL TEST SET EVALUATION — THIS NUMBER GOES IN YOUR README")
    print(f"{'='*60}")
    y_prob = model.predict_proba(X_test)[:,1]
    y_pred = (y_prob >= threshold).astype(int)
    full_model_report("Final Ensemble (Test Set)", y_test, y_pred, y_prob)
    print(f"\n[IMPORTANT] Do not tune further after seeing this number.")
    print(f"[IMPORTANT] This is your reported accuracy. Record it in TEST_LOG.md.")
```

**GATE: run test set evaluation exactly once. Record the number. Do not touch the model after this.**

---

## Phase test specifications

### Phase 1 — Python ML pipeline tests

Run after every file is complete, not just at the end of the phase.

**After download.py:**
```
[TEST] USPTO dataset downloaded — check file exists at ml/data/raw/
[TEST] File is not empty — check size > 10MB
[TEST] File is valid format — load first 100 rows with pandas, no errors
[TEST] Expected columns present — serial_number, filing_date, owner_name, status, attorney_name
[TEST] TTAB data accessible — make one request to ttabvue.uspto.gov, get 200 response
```

**After clean.py:**
```
[TEST] Output CSV exists at ml/data/processed/labeled.csv
[TEST] Label column present — column named 'label' exists with values 0 and 1 only
[TEST] No null labels — zero rows where label is NaN
[TEST] Class distribution — print fraud count and legitimate count, ratio between 1:2 and 1:5
[TEST] No duplicate serial numbers — assert len(df) == len(df.serial_number.unique())
[TEST] Date columns are datetime type — filing_date is not a string
[TEST] Minimum dataset size — at least 10,000 rows total
```

**After features.py:**
```
[TEST] All 12 features present — check column names match exactly
[TEST] No NaN values — zero nulls across all feature columns
[TEST] Feature ranges valid:
  - owner_filing_count_2yr >= 0
  - days_domain_to_filing between -365 and 3650
  - class_breadth between 1 and 45
  - opposition_count >= 0
  - office_action_count >= 0
  - section_8_filed is 0 or 1 only
  - owner_is_individual is 0 or 1 only
  - specimen_type_encoded between 0 and 3
[TEST] Print feature statistics — mean, std, min, max for each feature
[TEST] Fraud vs legitimate feature means differ — for top 3 features, fraud mean != legitimate mean
```

**After train.py:**
```
[TEST] Logistic Regression baseline trained — model file exists
[TEST] XGBoost trained — model file exists at ml/models/xgboost.joblib
[TEST] LightGBM trained — model file exists at ml/models/lightgbm.joblib
[TEST] XGBoost beats baseline — XGBoost F1 > Logistic Regression F1
[TEST] Validation metrics printed — precision, recall, F1, AUC-ROC for each model
[TEST] No data leakage — test set was never touched during training
[TEST] Single prediction works — feed one row, get back label (0 or 1) and confidence (0.0-1.0)
```

**After ensemble.py:**
```
[TEST] Ensemble beats best single model — ensemble F1 > max(xgboost_F1, lgbm_F1)
[TEST] ONNX export works — file exists at ml/models/ensemble.onnx
[TEST] ONNX inference matches joblib inference — run same 10 inputs through both, results match within 0.01
[TEST] ONNX loads in onnxruntime — no import errors, session created successfully
[TEST] Threshold tuning complete — print optimal threshold value and resulting precision/recall
```

**After evaluate.py:** (RANKER FRAMING — see TEST_LOG.md v4 for rationale)
```
[TEST] All Phase 1 gates met:
  - AUC-ROC         >= 0.85  (PASS/FAIL)
  - P@top-100       >= 0.85  (PASS/FAIL)
  - P@top-500       >= 0.80  (PASS/FAIL)
  - Review recall   >= 0.90  at score >= LOW_THRESHOLD  (PASS/FAIL)
[TEST] Confusion matrix PNG saved at ml/models/confusion_matrix.png
[TEST] Feature importance PNG saved at ml/models/feature_importance.png
[TEST] ROC curve PNG saved at ml/models/roc_curve.png
[TEST] Precision-recall curve PNG saved at ml/models/precision_recall_curve.png
[TEST] Full metrics printed to console in clean table format
```

**Phase 1 gate — do not proceed to Phase 2 until:**
- labeled.csv has at least 10,000 rows
- ensemble.onnx exists and loads without errors
- AUC-ROC >= 0.85, P@top-100 >= 0.85, P@top-500 >= 0.80, review recall >= 0.90
- All features have zero null values

**Why the ranker framing (replaces the old F1 >= 0.88 gate):**
The original F1 gate was set assuming clean fraud labels. Real TTAB-based labels are
noisier and the underlying signal doesn't support F1 0.88 as a binary classifier
without leakage (proven by ablation in v2 → v3). Reframing the ML layer as a
risk-score ranker lets the agent + human layers in Phases 4–6 make the actual
classification decision using richer context (LLM reasoning, domain lookups,
reverse image search). This matches how production fraud teams actually work.

---

### Phase 2 — TypeScript schemas tests

```
[TEST] package.json installs without errors — npm install exits 0
[TEST] TypeScript compiles — npx tsc --noEmit exits 0
[TEST] All Zod schemas parse valid input — run .parse() on example valid objects for each schema
[TEST] All Zod schemas reject invalid input — run .parse() on bad input, confirm ZodError thrown
[TEST] CheckRequestSchema rejects empty brand_name — z.parse({ brand_name: '' }) throws
[TEST] CheckRequestSchema rejects brand_name over 200 chars — throws ZodError
[TEST] VerdictSchema rejects confidence outside 0-1 — throws ZodError
[TEST] VerdictSchema rejects unknown verdict values — throws ZodError
[TEST] TypeScript types exported correctly — import and use all exported types without TS errors
```

**Phase 2 gate — do not proceed to Phase 3 until:**
- npm install succeeds
- tsc --noEmit exits 0 with zero errors
- all schema parse and reject tests pass

---

### Phase 3 — Agent tools tests

Run each tool independently before wiring them together. Use real API calls — not mocks.

**check_uspto_marks.ts:**
```
[TEST] Returns results for known mark — search "Nike" in class 025, expect at least 1 result
[TEST] Returns empty array for nonsense mark — search "xyzzy99999abc", expect 0 results
[TEST] Result shape matches Zod schema — parse each result through schema, no errors
[TEST] Handles API timeout gracefully — returns structured error, does not throw
[TEST] Handles network failure gracefully — returns structured error, does not throw
[TEST] Response time under 5 seconds — assert latency < 5000ms
```

**check_domain_age.ts:**
```
[TEST] Returns age for known domain — check "google.com", expect age > 9000 days
[TEST] Returns structured result for unknown domain — does not throw, returns not_found status
[TEST] Result shape matches Zod schema — parse result, no errors
[TEST] Handles API failure gracefully — returns structured error, does not throw
[TEST] Days calculation is correct — registration_date to today in days matches expected range
```

**web_search.ts:**
```
[TEST] Returns results for known brand — search "Nike shoes", expect at least 3 results
[TEST] Each result has url, title, snippet — assert all three fields present and non-empty
[TEST] Result shape matches Zod schema — parse each result, no errors
[TEST] Handles empty results gracefully — returns empty array, does not throw
[TEST] Handles API key invalid — returns structured error, does not throw
[TEST] Response time under 3 seconds — assert latency < 3000ms
```

**check_attorney.ts:**
```
[TEST] Returns result for known attorney — use a real USPTO bar number, expect active status
[TEST] Returns not_found for fake bar number — does not throw, returns structured not_found
[TEST] Result shape matches Zod schema — parse result, no errors
[TEST] Handles scraping failure gracefully — returns structured error, does not throw
```

**Phase 3 gate — do not proceed to Phase 4 until:**
- All 4 tools return valid data on happy path tests
- All 4 tools handle errors without throwing
- All results parse through their Zod schemas without errors

---

### Phase 4 — Agent core tests

**policies.ts:**
```
[TEST] All 5 policies exported — P1, P2, P3, P4, P5 all importable
[TEST] Each policy has name, description, tools array — assert all fields present
[TEST] Policy descriptions are non-empty strings — assert length > 50 chars each
```

**classifier.ts:**
```
[TEST] ONNX model loads without error — session created successfully
[TEST] Inference runs on valid input — feed 12 features, get back label and confidence
[TEST] Confidence is between 0 and 1 — assert 0 <= confidence <= 1
[TEST] Safe prediction for obviously legitimate mark:
  - owner_filing_count_2yr=1, section_8_filed=1, opposition_count=0
  - expect confidence < 0.4
[TEST] Fraud prediction for obvious troll:
  - owner_filing_count_2yr=300, section_8_filed=0, opposition_count=5
  - expect confidence > 0.7
[TEST] Handles missing features gracefully — returns error, does not crash Lambda
```

**agent.ts:**
```
[TEST] Agent runs end to end on "Nike Shoes" — returns valid verdict JSON
[TEST] Agent runs end to end on obviously fake brand "xyzzy troll mark" — returns verdict
[TEST] Output parses through VerdictSchema — no Zod errors
[TEST] All 5 policies present in output — P1-P5 all in response
[TEST] Each policy has triggered, confidence, reason — all fields present and correct types
[TEST] Agent handles tool failure mid-run — if one tool errors, agent continues and still returns verdict
[TEST] Agent completes within 30 seconds — assert total latency < 30000ms
[TEST] Token usage printed — log prompt tokens, completion tokens, estimated cost
```

**dynamo.ts:**
```
[TEST] Write succeeds — insert one record, get back the id
[TEST] Read by id succeeds — write then read, assert data matches
[TEST] List recent checks returns array — at least 0 items, no errors
[TEST] TTL is set correctly — ttl field is unix timestamp approximately 90 days from now
[TEST] Handles DynamoDB unavailable gracefully — returns structured error, does not crash
```

**Phase 4 gate — do not proceed to Phase 5 until:**
- Full agent run on "BrewBox Coffee" completes and returns valid verdict
- Output parses through VerdictSchema with zero errors
- DynamoDB write and read both succeed

---

### Phase 5 — API route tests

```
[TEST] POST /api/check with valid body returns 200 — { brand_name: "TestBrand" }
[TEST] Response body parses through VerdictSchema — no Zod errors
[TEST] POST with empty brand_name returns 400 — not 500
[TEST] POST with missing body returns 400 — not 500
[TEST] POST with brand_name over 200 chars returns 400 — not 500
[TEST] Response includes all required fields — verdict, confidence, policies, summary
[TEST] DynamoDB record created after successful check — query by brand name, record exists
[TEST] Concurrent requests do not interfere — send 3 requests simultaneously, all return valid responses
[TEST] Response time under 35 seconds for full agent run — assert latency < 35000ms
```

**Phase 5 gate — do not proceed to Phase 6 until:**
- POST /api/check returns 200 with valid verdict for "BrewBox Coffee"
- All error cases return 400, not 500
- DynamoDB record confirmed after check

---

### Phase 6 — Frontend tests

```
[TEST] npm run dev starts without errors — no compilation errors in console
[TEST] Page loads at localhost:3000 — HTTP 200 response
[TEST] CheckerForm renders — input field and submit button present in DOM
[TEST] Submitting "BrewBox Coffee" triggers API call — network request to /api/check visible
[TEST] AgentTrace shows tool names as they run — at least one tool name appears during run
[TEST] VerdictCard renders after completion — verdict badge visible (safe/review/high_risk)
[TEST] All 5 policy results displayed — P1-P5 all visible in UI
[TEST] Confidence bars render — 5 bars with values between 0 and 100%
[TEST] AuditLog shows recent checks — after one check, table has at least 1 row
[TEST] Clicking preset examples populates input — "NovaPay" pill click fills input field
[TEST] Error state renders — disconnect network, submit, confirm error message shown (not blank screen)
[TEST] Mobile viewport renders without overflow — test at 375px width, no horizontal scroll
```

**Phase 6 gate — do not proceed to Phase 7 until:**
- Full flow works end to end in browser: input → agent runs → verdict displayed → audit log updated
- No console errors during normal usage
- Error state shows message, not blank screen

---

### Phase 7 — Infrastructure tests

```
[TEST] CDK synth succeeds — cdk synth exits 0 with no errors
[TEST] CDK diff shows expected resources — Lambda, DynamoDB table, API Gateway all in diff
[TEST] Lambda function deploys — aws lambda get-function returns 200
[TEST] DynamoDB table exists — aws dynamodb describe-table returns 200
[TEST] Lambda invoke works — aws lambda invoke with test payload returns 200
[TEST] Lambda response parses through VerdictSchema — no Zod errors
[TEST] Environment variables set correctly — Lambda config shows all required env vars (values masked)
[TEST] Lambda timeout set to 60 seconds — assert configuration matches
[TEST] Lambda memory set to 1024MB — assert configuration matches
[TEST] DynamoDB TTL enabled — assert TTL attribute configured
[TEST] GitHub Actions workflow file valid YAML — yamllint passes
[TEST] IAM permissions correct — Lambda can write to DynamoDB (test with actual write)
```

**Phase 7 gate — do not proceed to Phase 8 until:**
- CDK synth exits 0
- Lambda invoke returns valid verdict
- DynamoDB write from Lambda succeeds

---

### Phase 8 — Evaluation harness tests

**ML eval:**
```
[TEST] cases.json loads without error — valid JSON, array of objects
[TEST] cases.json has at least 50 cases — assert length >= 50
[TEST] Each case has required fields — serial_number, brand_name, true_label, notes
[TEST] Model runs on all cases — no crashes on any case
[TEST] Metrics computed and printed — precision, recall, F1, AUC-ROC all in output
[TEST] Confusion matrix values sum to total case count — TP+TN+FP+FN = 50+
[TEST] Target metrics met or warning printed:
  - F1 >= 0.88 → PASS, else WARN with actual value
  - Recall >= 0.85 → PASS, else WARN
```

**Agent eval:**
```
[TEST] Agent eval runs on at least 20 cases without crashing
[TEST] Per-policy accuracy printed for each of P1-P5
[TEST] Average latency printed — in seconds
[TEST] Average cost printed — in USD
[TEST] Total cost for 20 cases under $1.00 — assert total_cost < 1.0
```

**Phase 8 gate — do not proceed to Phase 9 until:**
- ML eval completes on all cases
- Agent eval completes on at least 20 cases
- All metrics printed to console and appended to TEST_LOG.md

---

### Phase 9 — README tests

```
[TEST] README.md exists at root — file present
[TEST] Live demo link works — HTTP GET returns 200
[TEST] All metric claims match TEST_LOG.md — no made-up numbers
[TEST] Architecture diagram present — at least one diagram in README
[TEST] Setup instructions work — follow README from scratch in a new directory, app runs
```

---

## TEST_LOG.md format

Claude Code must create and maintain this file throughout the build. Append after every phase.

```markdown
# Test Log — TrademarkRisk

## Phase 1 — Python ML pipeline
Date: YYYY-MM-DD HH:MM
========================================
[PASS] USPTO dataset downloaded — 847MB file at ml/data/raw/
[PASS] File valid — loaded 12,742,819 rows successfully
[PASS] TTAB accessible — 200 response
[PASS] labeled.csv created — 82,441 rows (18,203 fraud, 64,238 legitimate)
[PASS] All 19 features present — zero nulls
[PASS] XGBoost F1: 0.871 beats LR baseline F1: 0.743
[PASS] Ensemble F1: 0.891 beats XGBoost: 0.871
[PASS] ONNX export validated
[PASS] Target metrics met — F1: 0.891, AUC-ROC: 0.934, Recall: 0.873, Precision: 0.910
========================================
9/9 tests passed. Proceeding to Phase 2.

## Phase 2 — TypeScript schemas
...
```

---

## Additional critical tests — gaps in existing coverage

---

### Feature engineering tests (add to Phase 1, after features.py)

```python
# Test 1 — all 19 features present with correct names
expected_features = [
    'owner_filing_count_2yr', 'owner_abandonment_rate', 'owner_is_individual',
    'attorney_case_count', 'owner_is_foreign',
    'days_domain_to_filing', 'days_since_filing',
    'days_filing_to_registration', 'days_first_use_to_filing',
    'office_action_count', 'opposition_count', 'statement_of_use_filed',
    'section_8_filed', 'has_acquired_distinctiveness', 'was_abandoned',
    'is_currently_active', 'class_breadth', 'filing_basis', 'specimen_type_encoded'
]
for feat in expected_features:
    assert feat in df.columns, f"Feature '{feat}' missing from dataframe"
print(f"[PASS] All 19 features present")

# Test 2 — owner_filing_count_2yr minimum is 1 not 0
# Every application counts itself so minimum is always 1
assert df.owner_filing_count_2yr.min() >= 1, "owner_filing_count_2yr cannot be 0 — every mark counts itself"
print(f"[PASS] owner_filing_count_2yr minimum is {df.owner_filing_count_2yr.min()} (expected 1)")

# Test 3 — owner_filing_count_2yr uses own_id not own_name
# own_id is numeric and more reliable than text for grouping
# Verify 2-year window is anchored to each row's filing_dt, not a fixed date
high_count_rows = df[df.owner_filing_count_2yr > 10]
print(f"Rows with owner_filing_count_2yr > 10: {len(high_count_rows)}")
print(f"Of these, fraud label: {high_count_rows.label.mean()*100:.1f}%")
print(f"[INFO] High count rows should have higher fraud rate than overall dataset")

# Test 4 — owner_abandonment_rate is between 0 and 1
assert df.owner_abandonment_rate.min() >= 0.0, "Abandonment rate cannot be negative"
assert df.owner_abandonment_rate.max() <= 1.0, "Abandonment rate cannot exceed 1.0"
# Owners with only 1 filing should have rate = 0 (unreliable sample)
single_filing_owners = df.groupby('own_id').filter(lambda x: len(x) == 1)
assert (single_filing_owners.owner_abandonment_rate == 0).all(), \
    "Single-filing owners should have abandonment_rate = 0"
print(f"[PASS] owner_abandonment_rate range valid and single-filing owners handled correctly")

# Test 5 — owner_abandonment_rate fraud vs legitimate separation
fraud_abandon = df[df.label==1]['owner_abandonment_rate'].mean()
legit_abandon = df[df.label==0]['owner_abandonment_rate'].mean()
assert fraud_abandon > legit_abandon, \
    f"Fraud abandonment ({fraud_abandon:.2f}) should exceed legitimate ({legit_abandon:.2f})"
print(f"[PASS] owner_abandonment_rate: fraud={fraud_abandon:.2f}, legitimate={legit_abandon:.2f}")

# Test 6 — days_domain_to_filing negative values are preserved
negative_cases = df[df.days_domain_to_filing < 0]
print(f"Cases where domain registered after filing: {len(negative_cases)}")
print(f"These are strong fraud signals — verify not being clipped to 0")

# Test 7 — section_8_filed maps correctly from use_afdv_acc_in
assert df.section_8_filed.isin([0, 1]).all(), "section_8_filed has values other than 0 and 1"
print(f"[PASS] section_8_filed values are all 0 or 1")

# Test 8 — was_abandoned computed correctly from abandon_dt
assert df.was_abandoned.isin([0, 1]).all(), "was_abandoned has values other than 0 and 1"
print(f"[PASS] was_abandoned: {df.was_abandoned.sum()} marks have abandon_dt present")

# Test 9 — class_breadth minimum is 1
assert df.class_breadth.min() >= 1, "class_breadth cannot be 0"
assert df.class_breadth.max() <= 45, "class_breadth above 45 — impossible"
print(f"[PASS] class_breadth range: min={df.class_breadth.min()}, max={df.class_breadth.max()}")

# Test 10 — known troll spot check
known_troll_serial = "YOUR_REAL_TROLL_SERIAL_HERE"
if known_troll_serial in df.serial_no.values:
    troll_row = df[df.serial_no == known_troll_serial].iloc[0]
    print(f"\n=== KNOWN TROLL FEATURE CHECK ===")
    print(f"serial_no:               {troll_row.serial_no}")
    print(f"owner_filing_count_2yr:  {troll_row.owner_filing_count_2yr} (expect 50+)")
    print(f"owner_abandonment_rate:  {troll_row.owner_abandonment_rate:.2f} (expect 0.7+)")
    print(f"section_8_filed:         {troll_row.section_8_filed} (expect 0)")
    print(f"opposition_count:        {troll_row.opposition_count} (expect 1+)")
    print(f"was_abandoned:           {troll_row.was_abandoned} (expect 1)")
    print(f"class_breadth:           {troll_row.class_breadth} (expect 3+)")
    print(f"label:                   {troll_row.label} (must be 1)")
    assert troll_row.label == 1, "Known troll not labeled as fraud — check TTAB join"
    print(f"[PASS] Known troll correctly labeled and feature values look reasonable")
```

---

### ONNX parity and speed tests (tighten existing, add speed check)

```python
# Tighten parity tolerance from 0.01 to 0.001
# A 0.01 difference near a 0.35 threshold can flip verdict — unacceptable
import time

test_inputs = X_test[:50]  # 50 samples

# Joblib inference
start = time.time()
joblib_probs = ensemble_model.predict_proba(test_inputs)[:,1]
joblib_time = time.time() - start

# ONNX inference
start = time.time()
onnx_probs = onnx_session.run(None, {'input': test_inputs.astype(np.float32)})[1][:,1]
onnx_time = time.time() - start

# Parity check — tight tolerance
max_diff = np.abs(joblib_probs - onnx_probs).max()
assert max_diff < 0.001, f"ONNX parity too loose: max difference {max_diff:.6f}"
print(f"[PASS] ONNX parity: max difference {max_diff:.6f} (threshold 0.001)")

# Speed check — ONNX must be faster
assert onnx_time < joblib_time, f"ONNX ({onnx_time:.3f}s) slower than joblib ({joblib_time:.3f}s)"
print(f"[PASS] ONNX faster: {onnx_time:.3f}s vs joblib {joblib_time:.3f}s")
```

---

### Cross-validation stability test (add to Phase 1, after train.py)

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for model_name, model in [("Logistic Regression", lr), ("XGBoost", xgb), ("LightGBM", lgbm)]:
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
    print(f"{model_name} CV F1: {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"  Fold scores: {[f'{s:.4f}' for s in scores]}")
    
    if scores.std() > 0.05:
        print(f"  [WARN] High variance ({scores.std():.4f}) — model unstable across folds")
        print(f"  Consider: more training data, simpler model, or better feature engineering")
    else:
        print(f"  [PASS] Stable across folds (std={scores.std():.4f} < 0.05)")
```

---

### Agent tool deep data shape tests (add to Phase 3)

```typescript
// check_uspto_marks — verify no null mark names
const usptoresult = await checkUsptoMarks('Nike', 25)
for (const mark of usptoresult.marks) {
  assert(typeof mark.name === 'string' && mark.name.length > 0,
    'Mark name is null or empty — will crash agent policy evaluation')
  assert(typeof mark.owner === 'string',
    'Mark owner is null — will crash agent')
  assert(typeof mark.status === 'string',
    'Mark status is null — will crash agent')
}
console.log('[PASS] All USPTO mark fields are non-null strings')

// check_domain_age — verify date is parseable, not "0000-00-00"
const whoisResult = await checkDomainAge('google.com')
if (whoisResult.status === 'found') {
  const date = new Date(whoisResult.registration_date)
  assert(!isNaN(date.getTime()), 
    `Registration date "${whoisResult.registration_date}" is not parseable`)
  assert(date.getFullYear() > 1990,
    `Registration date year ${date.getFullYear()} is unrealistic`)
  console.log('[PASS] Domain age date is valid and parseable')
}

// web_search — verify at least one result has non-empty snippet
const searchResult = await webSearch('Nike shoes buy online')
const nonEmptySnippets = searchResult.results.filter(r => r.snippet && r.snippet.length > 10)
assert(nonEmptySnippets.length > 0,
  'All snippets are empty — GPT-4o will have nothing to reason about for P5')
console.log(`[PASS] ${nonEmptySnippets.length} results have non-empty snippets`)

// All tools — verify payload size
for (const [name, payload] of [
  ['USPTO', usptoresult],
  ['Whois', whoisResult],
  ['Search', searchResult],
]) {
  const sizeKB = JSON.stringify(payload).length / 1024
  assert(sizeKB < 10, `${name} payload ${sizeKB.toFixed(1)}KB exceeds 10KB limit`)
  console.log(`[PASS] ${name} payload size: ${sizeKB.toFixed(1)}KB`)
}
```

---

### Agent policy consistency tests (add to Phase 4)

```typescript
function assertPolicyConsistency(verdict: Verdict, brandName: string): void {
  const triggered = Object.entries(verdict.policies)
    .filter(([, p]) => p.triggered)
  const triggeredCount = triggered.length
  const triggeredNames = triggered.map(([name]) => name)

  // Rule 1: no policies triggered → must be safe
  if (triggeredCount === 0) {
    console.assert(verdict.verdict === 'safe',
      `[INCONSISTENT] ${brandName}: 0 policies triggered but verdict is "${verdict.verdict}"`)
  }

  // Rule 2: 3+ policies triggered → must be high_risk
  if (triggeredCount >= 3) {
    console.assert(verdict.verdict === 'high_risk',
      `[INCONSISTENT] ${brandName}: ${triggeredCount} policies triggered but verdict is "${verdict.verdict}"`)
  }

  // Rule 3: P1 high confidence → cannot be safe
  if (verdict.policies.P1.triggered && verdict.policies.P1.confidence > 0.9) {
    console.assert(verdict.verdict !== 'safe',
      `[INCONSISTENT] ${brandName}: P1 triggered at ${verdict.policies.P1.confidence} but verdict is safe`)
  }

  // Rule 4: overall_confidence must align with verdict
  if (verdict.verdict === 'safe') {
    console.assert(verdict.overall_confidence < 0.5,
      `[INCONSISTENT] ${brandName}: verdict=safe but confidence=${verdict.overall_confidence}`)
  }
  if (verdict.verdict === 'high_risk') {
    console.assert(verdict.overall_confidence > 0.6,
      `[INCONSISTENT] ${brandName}: verdict=high_risk but confidence=${verdict.overall_confidence}`)
  }

  console.log(`[PASS] Policy consistency: ${brandName} — ${triggeredCount} policies triggered, verdict=${verdict.verdict}`)
}

// Run on every test case in eval harness
// Run on these specific known cases too:
const testCases = [
  { brand: 'Nike', expectedVerdict: 'high_risk' },       // famous mark — P1 certain
  { brand: 'xyzzy99999abc', expectedVerdict: 'safe' },   // nonsense — no policies
  { brand: 'BrewBox Coffee', expectedVerdict: 'review' }, // ambiguous case
]
```

---

### DynamoDB GSI test (add to Phase 4)

```typescript
// Write 3 records for same brand, query by brand using GSI, assert 3 results
const brandName = 'GSITestBrand_' + Date.now()

const ids = await Promise.all([
  writeToDynamo({ brand: brandName, verdict: 'safe', ...mockVerdict }),
  writeToDynamo({ brand: brandName, verdict: 'review', ...mockVerdict }),
  writeToDynamo({ brand: brandName, verdict: 'high_risk', ...mockVerdict }),
])

// Wait for GSI eventual consistency
await new Promise(resolve => setTimeout(resolve, 2000))

const results = await queryByBrand(brandName)
assert(results.length === 3,
  `GSI query returned ${results.length} results, expected 3 — GSI may be misconfigured`)
console.log('[PASS] DynamoDB GSI returns correct results for brand query')

// Cleanup
await Promise.all(ids.map(id => deleteFromDynamo(id)))
```

---

### API rate limit graceful handling test (add to Phase 3 and Phase 5)

```typescript
// Simulate 429 rate limit response for each tool
// Use dependency injection to pass a mock HTTP client

async function testRateLimitHandling() {
  // Mock Serper returning 429
  const mockSerper = { fetch: async () => ({ status: 429, body: 'rate limited' }) }
  const result = await webSearch('test query', mockSerper)
  
  assert(result.status === 'rate_limited', 'Tool should return rate_limited status')
  assert(typeof result.error === 'string', 'Tool should return error message')
  assert(!result.results, 'Tool should not return partial results on rate limit')
  console.log('[PASS] webSearch handles 429 gracefully')

  // Verify agent continues when one tool is rate limited
  const agentResult = await runAgent('TestBrand', 0.5, { serperMock: 'rate_limited' })
  assert(agentResult.verdict !== undefined, 'Agent should still return verdict when one tool fails')
  assert(agentResult.summary.includes('unavailable') || agentResult.tools_used.length < 4,
    'Agent should note that one tool was unavailable')
  console.log('[PASS] Agent continues with remaining tools when one is rate limited')
}
```

---

### ML vs agent agreement test (add to Phase 8 eval harness)

```python
def check_ml_agent_agreement(eval_results):
    """
    eval_results: list of dicts with ml_confidence, agent_verdict, brand_name, true_label
    """
    disagreements = []
    
    for r in eval_results:
        ml_verdict = (
            'high_risk' if r['ml_confidence'] > 0.85 
            else 'safe' if r['ml_confidence'] < 0.3 
            else 'review'
        )
        
        # Only flag strong disagreements — not review cases (those are expected to differ)
        if ml_verdict != 'review' and ml_verdict != r['agent_verdict']:
            disagreements.append({
                'brand': r['brand_name'],
                'ml_verdict': ml_verdict,
                'ml_confidence': r['ml_confidence'],
                'agent_verdict': r['agent_verdict'],
                'true_label': r['true_label']
            })
    
    disagreement_rate = len(disagreements) / len(eval_results)
    
    print(f"\n=== ML vs AGENT AGREEMENT ===")
    print(f"Total cases: {len(eval_results)}")
    print(f"Strong disagreements: {len(disagreements)} ({disagreement_rate*100:.1f}%)")
    
    if disagreements:
        print(f"\nDisagreement cases:")
        for d in disagreements:
            correct = 'ML' if (d['ml_verdict'] == 'high_risk') == (d['true_label'] == 1) else 'Agent'
            print(f"  {d['brand']}: ML={d['ml_verdict']}({d['ml_confidence']:.2f}), "
                  f"Agent={d['agent_verdict']}, True={d['true_label']}, Correct={correct}")
    
    if disagreement_rate > 0.20:
        print(f"[WARN] {disagreement_rate*100:.1f}% disagreement rate exceeds 20%")
        print(f"       ML model and agent may not be calibrated to same standard")
        print(f"       Review disagreement cases manually before reporting metrics")
    else:
        print(f"[PASS] Disagreement rate {disagreement_rate*100:.1f}% within acceptable range")
    
    return disagreements
```

---

### End-to-end latency breakdown test (add to Phase 5 and Phase 7)

```typescript
interface LatencyBreakdown {
  ml_inference: number
  check_uspto_marks: number
  check_domain_age: number
  web_search: number
  check_attorney: number
  gpt4o_reasoning: number
  dynamo_write: number
  total: number
}

async function measureLatencyBreakdown(brandName: string): Promise<LatencyBreakdown> {
  const timings: Partial<LatencyBreakdown> = {}
  const overall = Date.now()

  // ML inference
  let t = Date.now()
  await runMLClassifier(brandName)
  timings.ml_inference = Date.now() - t

  // Each tool separately
  t = Date.now()
  await checkUsptoMarks(brandName)
  timings.check_uspto_marks = Date.now() - t

  t = Date.now()
  await checkDomainAge(brandName + '.com')
  timings.check_domain_age = Date.now() - t

  t = Date.now()
  await webSearch(brandName)
  timings.web_search = Date.now() - t

  t = Date.now()
  await checkAttorney(brandName)
  timings.check_attorney = Date.now() - t

  timings.total = Date.now() - overall

  // Print breakdown
  console.log('\n=== LATENCY BREAKDOWN ===')
  for (const [key, ms] of Object.entries(timings)) {
    const bar = '█'.repeat(Math.round((ms as number) / 500))
    console.log(`${key.padEnd(22)}: ${((ms as number)/1000).toFixed(2)}s ${bar}`)
  }

  // Assert no single tool takes more than 10 seconds
  for (const [key, ms] of Object.entries(timings)) {
    if (key !== 'total') {
      assert((ms as number) < 10000,
        `${key} took ${ms}ms — exceeds 10 second limit`)
    }
  }
  console.log(`[PASS] All individual operations under 10 seconds`)
  
  return timings as LatencyBreakdown
}
```

---

### Lambda cold start test (add to Phase 7)

```bash
# Test cold start latency
# Step 1: invoke Lambda (warm it up)
aws lambda invoke \
  --function-name trademark-risk-agent \
  --payload '{"brand_name":"WarmupBrand"}' \
  /tmp/warmup_response.json

echo "Warm invocation complete. Waiting 15 minutes for cold start..."
sleep 900

# Step 2: invoke again after cold start
START=$(date +%s%3N)
aws lambda invoke \
  --function-name trademark-risk-agent \
  --payload '{"brand_name":"ColdStartTest"}' \
  /tmp/cold_response.json
END=$(date +%s%3N)

COLD_LATENCY=$((END - START))
echo "Cold start latency: ${COLD_LATENCY}ms"

# Assert cold start completes within timeout
if [ $COLD_LATENCY -lt 60000 ]; then
  echo "[PASS] Cold start completed in ${COLD_LATENCY}ms (under 60s timeout)"
else
  echo "[FAIL] Cold start took ${COLD_LATENCY}ms — exceeds Lambda timeout"
  echo "       Fix: cache ONNX session at module level, not inside handler"
fi

# Verify response is valid
cat /tmp/cold_response.json | python3 -c "
import json, sys
data = json.load(sys.stdin)
assert 'verdict' in data, 'Missing verdict in cold start response'
assert 'overall_confidence' in data, 'Missing confidence in cold start response'
print('[PASS] Cold start response is valid JSON with required fields')
"
```

---

## Notes for Claude Code

- Always use TypeScript strict mode
- All Zod schemas live in schemas/index.ts — never define types inline elsewhere
- Every tool function must have input validation with Zod before making any API call
- Every tool function must handle API failures gracefully and return a structured error
- The agent must handle tool failures without crashing — if one tool fails, continue with others
- DynamoDB writes must not block the API response — write async, return verdict immediately
- ONNX model file path must be configurable via environment variable for Lambda vs local
- All monetary values and confidence scores must be rounded to 2 decimal places before display
- Never log API keys or full user inputs to CloudWatch
- Never proceed to the next phase if the current phase gate conditions are not met
- Always append test results to TEST_LOG.md after each phase
- Print test reports in the exact format specified above — this is what goes in the README