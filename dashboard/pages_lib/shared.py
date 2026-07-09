"""Shared helpers and constants for all dashboard pages."""
import json
import pandas as pd
import streamlit as st
from pathlib import Path
import requests
import os

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
ASSETS = Path(__file__).parent.parent / "assets"

@st.cache_data
def load_json(name):
    return json.loads((ASSETS / name).read_text())

@st.cache_data
def load_csv(name):
    return pd.read_csv(ASSETS / name)

@st.cache_data
def load_parquet(name):
    return pd.read_parquet(ASSETS / name)

def esc(s):
    """Escape $ so Streamlit markdown doesn't render it as LaTeX."""
    return str(s).replace("$", "\\$")

FEATURE_LABELS = {
    "seq_days_since_last": "Days since last transaction",
    "autorenew_share": "Auto-renew share (history)",
    "last_auto_renew": "Auto-renew on (latest)",
    "cancel_count": "Total cancellations",
    "last_is_cancel": "Recently cancelled",
    "seq_cancel_last3": "Cancellations (last 3 txns)",
    "last_actual_paid": "Last amount paid (NT$)",
    "last_list_price": "Plan list price (NT$)",
    "last_plan_days": "Plan length (days)",
    "tenure_days": "Tenure (days)",
    "txn_count": "Transaction count",
    "secs_total": "Total listening (seconds)",
    "active_days": "Active listening days",
    "completion_ratio": "Song completion ratio",
    "seq_mean_gap_days": "Avg days between renewals",
    "seq_gap_widening": "Renewal gap widening",
    "seq_price_slope": "Price trajectory",
}

def build_customer_payload(seq_days, auto_renew, last_cancel, cancel_count, ar_share, paid,
                           tenure=800, txn_count=12):
    """Build a full 30-feature payload from the key inputs, defaults for the rest."""
    return {
        "city": 1.0, "bd": 30.0, "age_missing": 1, "gender": None, "registered_via": 7.0,
        "txn_count": float(txn_count), "cancel_count": float(cancel_count), "autorenew_share": ar_share,
        "last_plan_days": 30.0, "last_actual_paid": float(paid), "last_list_price": float(paid),
        "last_discount": 0.0, "last_auto_renew": float(auto_renew), "last_is_cancel": float(last_cancel),
        "payment_method_id": 41.0, "secs_total": 50000.0, "secs_mean": 4000.0, "unq_mean": 30.0,
        "active_days": 20.0, "plays_100": 200.0, "plays_25": 20.0, "completion_ratio": 9.5,
        "has_logs": 1, "tenure_days": float(tenure), "seq_days_since_last": float(seq_days),
        "seq_cancel_last3": 0.0, "seq_price_slope": 0.0, "seq_gap_widening": 0.0,
        "seq_mean_gap_days": 31.0, "seq_ar_dropped": 0,
    }

def call_api(endpoint, payload, timeout=10):
    """POST to the API; returns (data, error)."""
    try:
        r = requests.post(f"{API_URL}/{endpoint}", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)