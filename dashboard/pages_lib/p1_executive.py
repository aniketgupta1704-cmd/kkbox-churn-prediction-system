"""Page 1: Executive Summary."""
import streamlit as st
import plotly.express as px
from .shared import load_json, load_csv, esc

def render():
    kpi = load_json("kpi_summary.json")
    st.title("Executive Summary")
    st.caption("Subscriber churn prediction & retention intelligence — KKBox music streaming")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Subscribers", f"{kpi['dataset']['total_users']:,}")
    c2.metric("Modeled (Active)", f"{kpi['dataset']['modeled_population']:,}")
    c3.metric("Churn Rate", f"{kpi['dataset']['base_churn_rate']:.1%}")
    c4.metric("Model PR-AUC", f"{kpi['model']['pr_auc']:.3f}",
              help="Precision-Recall AUC on held-out test set (6.2% churn base rate)")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Log Loss", f"{kpi['model']['log_loss']:.3f}")
    c6.metric("Targeting Efficiency", "95% vs 6%",
              help="Actual churn rate among model-targeted vs random customers")
    c7.metric("Base-Case ROI", f"{kpi['business']['base_case_return_pct']}%",
              help=kpi['business']['assumptions'])
    c8.metric("Net Benefit", f"NT${kpi['business']['base_case_net_benefit_ntd']:,}")

    st.divider()
    st.subheader("Where the risk lives")
    seg = load_csv("risk_segments.csv")
    colA, colB = st.columns([3, 2])
    with colA:
        fig = px.bar(seg, x="risk_band", y="total_value_at_risk", color="risk_band",
                     text_auto=".2s", title="Value-at-Risk by Segment (NT$)",
                     color_discrete_sequence=px.colors.sequential.Reds_r)
        fig.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig, use_container_width=True)
    with colB:
        st.markdown("#### Key insight")
        st.info(esc(kpi['business']['risk_concentration']))
        st.markdown("**Actionable opportunity**")
        st.success(esc(kpi['business']['addressable_opportunity']))

    st.divider()
    st.subheader("System at a glance")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown("##### 🤖 Model")
        st.markdown(f"- **{kpi['model']['name']}**")
        st.markdown(f"- PR-AUC **{kpi['model']['pr_auc']:.3f}**, Brier **{kpi['model']['brier']}**")
        st.markdown(f"- {kpi['model']['n_features']} features")
        st.caption(esc(kpi['model']['calibration_note']))
    with m2:
        st.markdown("##### 💰 Business Impact")
        st.markdown(f"- {esc(kpi['business']['model_vs_random'])}")
        st.markdown(f"- {kpi['business']['targeting_multiple']} targeting efficiency")
        st.caption(esc(kpi['business']['assumptions']))
    with m3:
        st.markdown("##### 🔧 MLOps")
        st.markdown(f"- Drift: {kpi['mlops']['drift_features_flagged']}/{kpi['mlops']['drift_total_features']} features flagged")
        st.markdown(f"- Validation: {kpi['mlops']['synthetic_drift_validation']}")
        st.markdown(f"- {kpi['mlops']['counterfactual_rescuable_customers']} rescuable customers")