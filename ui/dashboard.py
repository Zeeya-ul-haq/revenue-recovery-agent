"""
Run: streamlit run ui/dashboard.py
"""
import json
import os
import sys
import pandas as pd
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from agent.graph import run_batch  # noqa: E402
from agent.razorpay_client import is_live as razorpay_is_live  # noqa: E402
from agent.llm_classifier import is_live as llm_is_live  # noqa: E402

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "events.json")

st.set_page_config(page_title="AI Revenue Recovery Agent", layout="wide")
st.title("💸 AI Revenue Recovery Agent")
st.caption("Detects at-risk payments → diagnoses root cause → executes a bounded, auditable recovery action")

badge_col1, badge_col2 = st.columns(2)
with badge_col1:
    if razorpay_is_live():
        st.success("🟢 Razorpay test-mode API connected", icon="✅")
    else:
        st.info("⚪ Razorpay: simulation mode (no keys)", icon="ℹ️")
with badge_col2:
    if llm_is_live():
        st.success("🟢 LLM classifier connected (Groq)", icon="✅")
    else:
        st.info("⚪ Classifier: rule-based fallback (no key)", icon="ℹ️")

if not os.path.exists(DATA_PATH):
    st.error("No event data found. Run `python data/generate_events.py` first.")
    st.stop()

with open(DATA_PATH) as f:
    events = json.load(f)

if st.button("▶ Run recovery batch", type="primary") or "outcomes" not in st.session_state:
    with st.spinner(f"Processing {len(events)} at-risk events..."):
        st.session_state["outcomes"] = run_batch(events)

outcomes = st.session_state["outcomes"]

rows = []
for o in outcomes:
    rows.append({
        "order_id": o["event"]["order_id"],
        "customer_id": o["event"]["customer_id"],
        "amount_inr": o["event"]["amount_inr"],
        "reason_code": o["event"]["reason_code"],
        "root_cause": o["root_cause"],
        "classifier": o.get("classification_source", "—"),
        "confidence": o.get("classification_confidence"),
        "intervention": o["intervention"],
        "escalated": o["escalated"],
        "recovered": o["recovered"],
    })
df = pd.DataFrame(rows)

total_at_risk = df["amount_inr"].sum()
total_recovered = df.loc[df["recovered"] == True, "amount_inr"].sum()
escalated_count = int(df["escalated"].sum())
auto_actioned = len(df) - escalated_count
recovery_rate = (df["recovered"] == True).sum() / max(auto_actioned, 1)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total at risk", f"₹{total_at_risk:,.0f}")
c2.metric("Recovered", f"₹{total_recovered:,.0f}", f"{total_recovered/total_at_risk:.1%} of at-risk")
c3.metric("Escalated to human", escalated_count)
c4.metric("Auto-recovery rate", f"{recovery_rate:.1%}")

st.subheader("Batch results")
st.dataframe(df, width="stretch")

st.subheader("Intervention breakdown")
st.bar_chart(df["intervention"].value_counts())

st.subheader("Audit trail (reasoning per decision)")
selected_order = st.selectbox("Inspect an order", df["order_id"])
match = next(o for o in outcomes if o["event"]["order_id"] == selected_order)
for step in match["reasoning"]:
    st.write("• " + step)
st.json(match["action_result"])
