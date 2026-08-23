"""
LLM-backed root-cause classifier.

If GROQ_API_KEY is not set, falls back to the deterministic rule-based
lookup (ROOT_CAUSE_MAP in graph.py) so the pipeline never breaks without
a key.

Get a free key: console.groq.com -> API Keys -> Create API Key
Put it in .env (gitignored):

    GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
"""
import json
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_LIVE = bool(GROQ_API_KEY)

_client = None
if LLM_LIVE:
    from groq import Groq
    _client = Groq(api_key=GROQ_API_KEY)

MODEL = "openai/gpt-oss-20b"

VALID_ROOT_CAUSES = [
    "user_side_funds",
    "user_side_instrument",
    "user_side_mandate",
    "user_abandonment",
    "bank_side_transient",
    "gateway_side_transient",
    "risk_hold",
    "unknown",
]

SYSTEM_PROMPT = f"""You are a payments root-cause classifier for an Indian fintech revenue recovery system.

Given a failed/abandoned payment event, classify it into exactly one of these root causes:
- user_side_funds: customer lacks funds in their account
- user_side_instrument: customer's card/instrument was declined
- user_side_mandate: UPI autopay mandate expired or needs re-authorization
- user_abandonment: customer dropped off mid-checkout (didn't finish OTP, closed tab, session timeout)
- bank_side_transient: temporary bank-side issue (timeout), likely to succeed on retry
- gateway_side_transient: temporary payment gateway issue, likely to succeed on retry
- risk_hold: blocked by a fraud/risk engine — never safe to auto-retry
- unknown: doesn't clearly fit any of the above

Respond with ONLY a JSON object, no other text, no markdown fences:
{{"root_cause": "<one of: {', '.join(VALID_ROOT_CAUSES)}>", "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}}
"""


def is_live() -> bool:
    return LLM_LIVE


def classify_with_llm(event: dict) -> dict:
    """
    Calls Groq (Llama 3.3 70B) to classify root cause from the raw event.
    Falls back to a rule-based classification if no key configured or
    the call fails for any reason -- classification must never crash
    the pipeline.
    """
    if not LLM_LIVE:
        return _fallback(event, source="rule_based_no_key")

    user_prompt = json.dumps({
        "reason_code": event.get("reason_code"),
        "event_type": event.get("event_type"),
        "payment_method": event.get("payment_method"),
        "attempt_count": event.get("attempt_count"),
        "amount_inr": event.get("amount_inr"),
    })

    try:
        response = _client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=200,
        )
        raw = response.choices[0].message.content.strip()
        # strip accidental markdown fences if the model adds them
        if raw.startswith("```"):
            raw = raw.strip("`").replace("json", "", 1).strip()
        parsed = json.loads(raw)

        root_cause = parsed.get("root_cause")
        if root_cause not in VALID_ROOT_CAUSES:
            root_cause = "unknown"

        return {
            "root_cause": root_cause,
            "confidence": float(parsed.get("confidence", 0.5)),
            "reasoning": parsed.get("reasoning", ""),
            "source": "llm_groq",
        }
    except Exception as e:
        fallback = _fallback(event, source="rule_based_llm_error")
        fallback["error"] = str(e)
        return fallback


# Same taxonomy as the original rule-based version, kept here as a safety
# net so classification always succeeds even if the LLM call fails.
_RULE_MAP = {
    "insufficient_funds": "user_side_funds",
    "card_declined": "user_side_instrument",
    "upi_mandate_expired": "user_side_mandate",
    "otp_not_entered": "user_abandonment",
    "payment_method_not_selected": "user_abandonment",
    "session_timeout": "user_abandonment",
    "bank_timeout": "bank_side_transient",
    "gateway_error": "gateway_side_transient",
    "risk_engine_block": "risk_hold",
}


def _fallback(event: dict, source: str) -> dict:
    cause = _RULE_MAP.get(event.get("reason_code"), "unknown")
    return {
        "root_cause": cause,
        "confidence": 1.0,
        "reasoning": f"Rule-based lookup on reason_code='{event.get('reason_code')}'",
        "source": source,
    }
