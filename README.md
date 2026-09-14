# elai-platform-api

The ELAI confidence engine, exposed as an API. One deployable service; product
routers are isolated modules over one write model.

Every number returned carries four things: `value`, the `state` that governs it,
a pre-formatted `display` string, and the `rule_version` that produced it.
**Clients render `display`.** A client that formats numbers itself can print a
point value on a field nobody could see, which is the defect this exists to remove.

## Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
# http://localhost:8000/docs
```

## The two modes

```bash
ELAI_ESTIMATOR=demo_fitted uvicorn app.main:app    # radar model fitted and validated
ELAI_ESTIMATOR=unfitted    uvicorn app.main:app    # radar available, model unproven
```

`unfitted` is the honest starting position and sends every farm to Blind.
Availability of radar is not validation of radar. The difference between these two
runs is the whole Part 2b argument.

> The coefficients behind `demo_fitted` are **fabricated**, plausible in shape but
> not measured. `demo_fitted` appears in every reason line and ledger row so it is
> never mistaken for a real accuracy figure.

## Deploy to Railway

```bash
git init && git add -A && git commit -m "platform api"
gh repo create <org>/elai-platform-api --private --source=. --push

railway init
railway up
railway variables set CORS_ORIGINS=https://<your-app>.vercel.app
```

Health check is `/healthz`, wired in `railway.json`. It verifies the core loaded and
which rule version is live — a task serving the wrong rule version is worse than a
task that is down.

Before deploying, point the core dependency at your repo in `requirements.txt`:

```
elai-confidence-core @ git+https://github.com/<org>/elai-confidence-core@v0.1.0
```

## GenAI — how the chatbot is wired

`POST /v1/agent/explain` answers "why is this field flagged" for four audiences
from the same evidence.

**The model never decides what the evidence is.** Five tools assemble it —
`get_farm_context`, `get_latest_observation`, `get_observation_history`,
`get_cloud_state`, `get_pop_reference`. Claude phrases what they return. It
cannot read the database, cannot browse, and knows nothing about any farm from
training, so a hallucinated NDVI value is structurally impossible rather than
merely discouraged. `GET /v1/agent/tools` shows the whole surface.

**Groundedness is verified in code.** Every number in the generated answer must
appear in the tool results. If one does not, the answer is discarded and the
deterministic renderer serves instead, with `failure_reason: ungrounded` and the
offending tokens in the payload. This is the automated half of the evaluation
rubric — an agronomist panel judges usefulness; this makes an invented figure
unshippable.

**Guardrails are code paths, not prompt text.** Dosing and product questions are
refused before a token is spent. The Package of Practices reference contains no
chemical names and no rates by construction, and a test enforces that.

**The endpoint has no error state.** No API key, provider down, circuit open, or
groundedness rejected — all four produce the deterministic evidence chain. A
live demo cannot die on stage.

```bash
export ANTHROPIC_API_KEY=sk-ant-...        # optional; without it, deterministic
export ELAI_LLM_MODEL=claude-sonnet-4-6    # optional
```

The response carries `generator` (`claude` or `deterministic`) and an `llm` block
with the tools called, the round count, and the failure reason if it fell back.

## Endpoints

| Route | Screen |
|---|---|
| `GET /v1/farms` | S1 table, S2 map |
| `GET /v1/farms/{id}/state` | S4 farm detail |
| `GET /v1/farms/{id}/observations` | S7 history, ClaimChronos source |
| `GET /v1/farms/{id}/ledger` | audit trail |
| `GET /v1/portfolio` | S1 portfolio, confidence mix by area |
| `GET /v1/coverage` | blind-days, SLA, escalation rule |
| `POST /v1/agent/explain` | S8 explain panel |
| `GET /v1/agent/tools` | the model's entire tool surface |
| `POST /v1/feedback` | S9 ground-truth ingress |
| `GET /v1/lenderbook/book` | LenderBook L1 |
| `GET /v1/lenderbook/call-list` | LenderBook L3 |
| `GET /v1/trust/state/{id}` | TrustLayer |
| `POST /v1/trust/states:batch` | TrustLayer |

## Frontend contract

`openapi.json` is committed. Generate the client rather than hand-writing it:

```bash
npx openapi-typescript openapi.json -o packages/api-client/schema.d.ts
```

A schema change then breaks the build instead of breaking a client's screen.

## Not production yet

- In-memory store. Mongo and Postgres adapters slot in behind `Repository`.
- No auth. Paths A–D are specified in `IDENTITY_AND_ACCESS.md`, not implemented.
- The Claude tool-use loop is written but **has not been run against the live API
  from this machine** — the build environment cannot reach api.anthropic.com.
  Set `ANTHROPIC_API_KEY` and hit `/v1/agent/explain` to verify it end to end.
- Seed data is synthetic and labelled as such throughout.
- The yield estimate is a canopy-ratio stand-in, not a trained model.
