# AI Revenue Recovery Agent

A LangGraph agent that detects at-risk payments (failures, drop-offs, expired mandates), diagnoses the root cause using an LLM, and executes a bounded, auditable recovery action via Razorpay's test-mode API.

Built for **Track 03 — AI Revenue Recovery**: *"Don't just identify the problem. Show measured money recovered across a batch, with compliant escalation, stopping rules, and an audit trail."*

---

## The problem

Revenue loss from failed payments rarely happens in one clean step — a card gets declined, a UPI mandate expires, a customer drops off mid-checkout. Most systems either do nothing (silent loss) or retry blindly (annoying customers, wasting API calls, sometimes retrying a fraud-blocked payment). This agent closes that loop: **detect → diagnose → decide → act → audit**, with hard limits so it never acts recklessly.

## Architecture

```
                    ┌─────────────────────┐
  payment event ──▶ │ classify_root_cause │  LLM (Groq / Llama) reasons over
                    │                      │  the raw event → root cause +
                    └──────────┬───────────┘  confidence + justification
                               │
                    ┌──────────▼───────────┐
                    │ select_intervention  │  Deterministic policy maps
                    │                      │  root cause → action, gated by
                    └──────────┬───────────┘  amount cap + attempt cap
                               │
                    ┌──────────▼───────────┐
                    │ execute_bounded_     │  Real Razorpay test-mode API
                    │ action               │  call (payment link / status
                    └──────────┬───────────┘  fetch) or escalation
                               │
                    ┌──────────▼───────────┐
                    │ audit_log            │  Every decision, every
                    │                      │  reasoning step, every outcome
                    └──────────────────────┘  written before the loop ends
```

Each node is a LangGraph state transition. The full reasoning trace — classification, gating decision, and action outcome — is written to `logs/audit_log.jsonl` before the pipeline moves to the next event.

## The three gating rules (why the agent never acts recklessly)

1. **Amount cap** (`MAX_AUTO_ACTION_AMOUNT = ₹15,000`) — any at-risk payment above this is *always* escalated to a human, never auto-actioned. An autonomous agent should not have unlimited authority to act on customers' money.
2. **Attempt cap** (`MAX_AUTOMATED_ATTEMPTS = 2`) — if a payment has already failed more than twice, automation stops and a human takes over. Repeated automated nudges past this point looks like harassment, not recovery.
3. **Risk holds are never auto-actioned** — if the root cause is a fraud/risk-engine block, the agent always escalates. Automating around a risk block is exactly the failure mode a recovery agent must not have.

## What's real vs. simulated

| Component | Real | Notes |
|---|---|---|
| Root cause classification | ✅ Groq LLM call (`openai/gpt-oss-20b`) | Falls back to a rule-based lookup if no API key is set, so the pipeline never breaks |
| Recovery action (payment link / status check) | ✅ Razorpay test-mode API | Falls back to simulation if no API key is set |
| Whether the customer actually completes payment after a link is sent | ⚠️ Estimated | This happens asynchronously in the real world; reported separately from the API call outcome and labeled `estimated_outcome`, never conflated with "the API call succeeded" |

This separation matters: a payment link can be created successfully (real API success) while the customer still doesn't pay (a separate, later, asynchronous outcome). The audit log never confuses the two.

## Results (fill in from your own run)

```
Events processed: 60
Total at risk: ₹___,___
Recovered: ₹__,___  (__% of at-risk)
Escalated to human: __
Auto-recovery rate: __%
```

*(Run `python -m agent.graph` and paste the console output here.)*

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

cp .env.example .env           # then fill in real keys
python data/generate_events.py --n 60
python -m agent.graph
streamlit run ui/dashboard.py
```

Without API keys, everything runs in simulation mode — the pipeline, gating logic, and audit trail all work identically, just without live external calls. This is intentional: the architecture doesn't change based on whether keys are present.

### Getting API keys (both free)

- **Razorpay test-mode**: [dashboard.razorpay.com](https://dashboard.razorpay.com) → stay in Test Mode → Settings → API Keys → Generate Test Key
- **Groq**: [console.groq.com](https://console.groq.com) → API Keys → Create API Key

## Project structure

```
agent/
  graph.py            # LangGraph state machine — the four-node pipeline
  llm_classifier.py    # Root cause classification (Groq + rule-based fallback)
  razorpay_client.py   # Payment API wrapper (real + simulation fallback, retry/backoff)
data/
  generate_events.py   # Synthetic event generator (Razorpay-shaped webhook payloads)
ui/
  dashboard.py          # Streamlit dashboard — batch results, metrics, audit trail viewer
logs/
  audit_log.jsonl        # Append-only decision log (gitignored)
```

## What I'd build next

- Sample real API calls for a subset of a large batch rather than calling live for every event, to work within free-tier rate limits at scale
- Track false-positive interventions (cases where the agent acted but the customer would have completed payment anyway) as a separate honesty metric
- Move from a single deterministic policy table to a learned policy, using outcome data to improve intervention selection over time
