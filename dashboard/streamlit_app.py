import streamlit as st
st.set_page_config(page_title="Churn Intelligence Platform", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

from pages_lib import p1_executive, p2_data, p3_model, p4_explainable, p5_business, p6_monitoring, p7_live, p8_architecture, p9_experiments
# (import p3..p9 as we build them)

PAGES = {
    "1 · Executive Summary": p1_executive.render,
    "2 · Data & Features": p2_data.render,
    "3 · Model Development": p3_model.render,
    "4 · Explainable AI": p4_explainable.render,
    "5 · Business Strategy": p5_business.render,
    "6 · Model Monitoring": p6_monitoring.render,
    "7 · Live Prediction": p7_live.render,
    "8 · System Architecture": p8_architecture.render,
    "9 · Experiments & Insights": p9_experiments.render,
}

st.sidebar.title("📊 Churn Intelligence")
st.sidebar.caption("KKBox subscriber churn · end-to-end ML system")
choice = st.sidebar.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.caption("Model: LightGBM (calibrated) · PR-AUC 0.915")

render = PAGES[choice]
if render:
    render()
else:
    st.title(choice.split("·")[1].strip())
    st.info("🚧 Building this page next.")