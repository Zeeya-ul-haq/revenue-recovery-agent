"""
Thin wrapper around the Razorpay test-mode SDK.

If RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET are not set in the environment,
every function here falls back to a local simulation so the rest of the
pipeline (and the demo) still works without live credentials.

Get test keys: Razorpay Dashboard -> ensure "Test Mode" toggle is on
-> Settings -> API Keys -> Generate Test Key.
Never commit real keys. Put them in a local .env file (gitignored):

    RAZORPAY_KEY_ID=rzp_test_xxxxxxxxxxxx
    RAZORPAY_KEY_SECRET=xxxxxxxxxxxxxxxxxxxx
"""
import os
import random
import time
import uuid
from dotenv import load_dotenv

load_dotenv()

KEY_ID = os.getenv("RAZORPAY_KEY_ID")
KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET")

LIVE_MODE = bool(KEY_ID and KEY_SECRET)

_client = None
if LIVE_MODE:
    import razorpay
    _client = razorpay.Client(auth=(KEY_ID, KEY_SECRET))

MAX_RETRIES = 5
BASE_BACKOFF_SECONDS = 4


def _call_with_retry(fn, *args, **kwargs):
    """
    Calls the Razorpay SDK with retry + exponential backoff on rate-limit
    errors ("Too many requests"). Real payment infra always needs this --
    it's not a workaround, it's expected resilience for any system that
    calls a rate-limited external API in a batch.
    """
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs), None
        except Exception as e:
            last_error = e
            msg = str(e).lower()
            if "too many requests" in msg or "rate limit" in msg:
                wait = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))  # 4, 8, 16, 32, 64
                time.sleep(wait)
                continue
            # non-rate-limit error: don't retry, fail fast
            break
    return None, last_error


def is_live() -> bool:
    """Whether we're actually calling Razorpay's test-mode API vs. simulating."""
    return LIVE_MODE


def fetch_payment_status(payment_id: str) -> dict:
    """
    Fetch a payment's current status from Razorpay test mode.
    Falls back to a plausible simulated status if no keys configured.
    """
    if LIVE_MODE:
        result, error = _call_with_retry(_client.payment.fetch, payment_id)
        if error is None:
            return {
                "source": "razorpay_live_test_mode",
                "payment_id": payment_id,
                "status": result.get("status"),
                "raw": result,
            }
        return {
            "source": "razorpay_live_test_mode_error",
            "payment_id": payment_id,
            "status": "unknown",
            "error": str(error),
        }
    # simulation fallback
    status = random.choice(["failed", "created", "authorized"])
    return {"source": "simulated", "payment_id": payment_id, "status": status}


def create_recovery_payment_link(amount_inr: float, customer_id: str, order_id: str,
                                  description: str = "Complete your payment") -> dict:
    """
    Creates a real Razorpay test-mode payment link the customer could click
    and pay through Razorpay's test checkout, OR a simulated link if no
    keys are configured.
    """
    amount_paise = int(round(amount_inr * 100))  # Razorpay uses paise

    if LIVE_MODE:
        payload = {
            "amount": amount_paise,
            "currency": "INR",
            "description": f"{description} (order {order_id})",
            "reference_id": order_id,
            "notes": {"customer_id": customer_id, "order_id": order_id},
        }
        result, error = _call_with_retry(_client.payment_link.create, payload)
        if error is None:
            return {
                "source": "razorpay_live_test_mode",
                "status": "link_created",
                "short_url": result.get("short_url"),
                "payment_link_id": result.get("id"),
            }
        return {
            "source": "razorpay_live_test_mode_error",
            "status": "link_creation_failed",
            "error": str(error),
        }
    # simulation fallback
    return {
        "source": "simulated",
        "status": "link_created",
        "short_url": f"https://rzp.io/simulated/{uuid.uuid4().hex[:10]}",
        "payment_link_id": f"plink_sim_{uuid.uuid4().hex[:14]}",
    }
