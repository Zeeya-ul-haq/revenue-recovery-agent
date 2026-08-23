"""
Generates a synthetic batch of at-risk revenue events mimicking
Razorpay test-mode webhook payloads: failed payments, stalled
subscriptions, and checkout drop-offs.

Run: python data/generate_events.py --n 60 --out data/events.json
"""
import json
import random
import argparse
from datetime import datetime, timedelta

FAILURE_REASONS = [
    "insufficient_funds",
    "bank_timeout",
    "card_declined",
    "upi_mandate_expired",
    "otp_not_entered",       # -> checkout drop-off
    "gateway_error",
    "risk_engine_block",
]

EVENT_TYPES = ["payment.failed", "subscription.charged.failed", "checkout.abandoned"]

CUSTOMERS = [f"cust_{i:04d}" for i in range(1, 200)]


def random_amount():
    return round(random.uniform(199, 24999), 2)


def make_event(idx, base_time):
    event_type = random.choice(EVENT_TYPES)
    reason = random.choice(FAILURE_REASONS)
    # checkout.abandoned almost always maps to otp/user drop-off style reasons
    if event_type == "checkout.abandoned":
        reason = random.choice(["otp_not_entered", "payment_method_not_selected", "session_timeout"])

    ts = base_time + timedelta(minutes=random.randint(0, 60 * 24 * 3))

    return {
        "event_id": f"evt_{idx:05d}",
        "event_type": event_type,
        "order_id": f"order_{idx:05d}",
        "customer_id": random.choice(CUSTOMERS),
        "amount_inr": random_amount(),
        "reason_code": reason,
        "attempt_count": random.randint(1, 3),
        "created_at": ts.isoformat(),
        "payment_method": random.choice(["upi", "card", "netbanking", "wallet"]),
    }


def main(n, out_path):
    base_time = datetime(2026, 8, 1, 9, 0, 0)
    events = [make_event(i, base_time) for i in range(1, n + 1)]
    with open(out_path, "w") as f:
        json.dump(events, f, indent=2)
    print(f"Wrote {n} synthetic events to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--out", type=str, default="data/events.json")
    args = parser.parse_args()
    main(args.n, args.out)
