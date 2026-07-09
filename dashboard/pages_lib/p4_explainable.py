"""Page 4: Explainable AI — SHAP + counterfactuals."""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import requests
from .shared import load_csv, load_json, load_parquet, ASSETS, API_URL, FEATURE_LABELS

def render():
    st.title("Explainable AI")
    st.caption("How the model reasons — globally and per customer — with plain-language factors")

    # ---- Global SHAP importance (native Plotly) ----
    st.subheader("What drives churn predictions")
    imp = load_csv("shap_importance.csv").head(12).copy()
    imp["label"] = imp["feature"].map(lambda f: FEATURE_LABELS.get(f, f))
    fig = px.bar(imp.sort_values("pct"), x="pct", y="label", orientation="h",
                 text="pct", title="Global feature importance (mean |SHAP|, %)",
                 color="pct", color_continuous_scale="Tealgrn")
    fig.update_traces(texttemplate="%{text:.1f}%")
    fig.update_layout(height=430, coloraxis_showscale=False, xaxis_title="Contribution (%)",
                      yaxis_title="", template="plotly_dark",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("No single feature exceeds ~11%; 22 features account for 90% of decisions — "
               "a well-distributed model, not one riding a single signal.")

    # ---- Global beeswarm (transparent PNG) ----
    st.subheader("SHAP beeswarm — direction & magnitude")
    bee = ASSETS / "shap_beeswarm.png"
    if bee.exists():
        _, mid, _ = st.columns([1, 4, 1])
        mid.image(str(bee), use_container_width=True)
    st.caption("Each dot is a customer. Right = pushed toward churn. Red = high feature value. "
               "E.g. high 'days since last transaction' (red, right) increases churn risk.")

    st.divider()

    st.subheader("Individual predictions — worked examples")
    st.caption("How the model decomposes two specific customers (SHAP waterfall)")
    w1, w2 = st.columns(2)
    with w1:
        st.markdown("**High-risk customer**")
        p = ASSETS / "shap_waterfall_high.png"
        if p.exists(): st.image(str(p), use_container_width=True)
        st.caption("Long inactivity, high plan cost, and lapsed auto-renew push risk up.")
    with w2:
        st.markdown("**Low-risk customer**")
        p = ASSETS / "shap_waterfall_low.png"
        if p.exists(): st.image(str(p), use_container_width=True)
        st.caption("Consistent auto-renewal and recent activity keep risk low.")