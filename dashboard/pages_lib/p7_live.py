"""Page 7: Live Prediction — polished single-customer inference."""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
from .shared import build_customer_payload, call_api, FEATURE_LABELS

def render():
    st.title("Live Prediction")
    st.caption("Score a subscriber in real time — churn risk, drivers, and the recommended retention action")

    st.markdown("Configure a customer profile and get a live prediction from the deployed model API.")

    # ---- Input form ----
    with st.container(border=True):
        st.markdown("##### Customer profile")
        r1c1, r1c2, r1c3 = st.columns(3)
        seq_days = r1c1.slider("Days since last transaction", 0, 400, 30)
        paid = r1c2.slider("Last amount paid (NT$)", 0, 500, 149)
        tenure = r1c3.slider("Tenure (days)", 0, 3000, 800)

        r2c1, r2c2, r2c3 = st.columns(3)
        auto_renew = r2c1.radio("Auto-renew on?", [1, 0], horizontal=True)
        last_cancel = r2c2.radio("Recently cancelled?", [0, 1], horizontal=True)
        cancel_count = r2c3.slider("Total cancellations", 0, 10, 0)

        ar_share = st.slider("Auto-renew share (history)", 0.0, 1.0, 1.0)

    predict = st.button("🔮 Predict churn risk", type="primary", use_container_width=True)

    if not predict:
        st.info("Set the profile above and click **Predict** to score this customer.")
        return

    payload = build_customer_payload(seq_days, auto_renew, last_cancel, cancel_count,
                                     ar_share, paid, tenure=tenure)
    expl, err = call_api("explain", payload)
    if err:
        st.error(f"⚠️ API not reachable — is the FastAPI server running on port 8000? ({err})")
        return
    cf, _ = call_api("counterfactual", payload)

    prob = expl["churn_probability"]
    risk = "HIGH" if prob >= 0.7 else ("MEDIUM" if prob >= 0.4 else "LOW")
    color = {"HIGH": "#ff4b4b", "MEDIUM": "#ffa500", "LOW": "#4bff9f"}[risk]

    st.divider()

    # ---- Result row: gauge + risk/confidence ----
    g1, g2, g3 = st.columns([2, 1, 2])
    with g1:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=prob*100,
            title={"text": "Churn Probability"},
            number={"suffix": "%", "font": {"size": 44}},
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": color},
                   "steps": [{"range": [0,40], "color": "#12271c"},
                             {"range": [40,70], "color": "#2a2410"},
                             {"range": [70,100], "color": "#2a1212"}],
                   "threshold": {"line": {"color": "white", "width": 3},
                                 "value": prob*100}}))
        gauge.update_layout(height=300, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(gauge, use_container_width=True)
    with g2:
        st.markdown("##### Risk level")
        st.markdown(f"<h1 style='color:{color}'>{risk}</h1>", unsafe_allow_html=True)
        # confidence = distance from decision boundary
        conf = abs(prob - 0.5) * 2
        st.metric("Confidence", f"{conf:.0%}",
                  help="How far the prediction is from the 50% decision boundary")
    with g3:
        st.markdown("##### Top contributing factors")
        for f in expl["top_factors"][:5]:
            arrow = "🔴" if "increases" in f["effect"] else "🟢"
            label = FEATURE_LABELS.get(f["feature"], f["feature"])
            st.markdown(f"{arrow} **{label}** — {f['effect'].replace(' churn risk','')}")

    st.divider()

    # ---- Recommended actions ----
    st.markdown("##### 💡 Recommended retention action")
    if cf:
        best = cf["interventions"][0]
        if best["reduction_pct"] > 1:
            st.success(f"**{best['intervention']}** → reduces churn risk from "
                       f"{cf['base_prob']:.0%} to {best['new_prob']:.0%} "
                       f"(**−{best['reduction_pct']:.0f}%**)")
        else:
            st.warning("No single intervention meaningfully reduces this customer's risk — "
                       "consider a bundled offer or escalation.")

        cfdf = pd.DataFrame(cf["interventions"])
        fig = px.bar(cfdf.sort_values("reduction_pct"), x="reduction_pct", y="intervention",
                     orientation="h", text="reduction_pct",
                     color="reduction_pct", color_continuous_scale="Greens",
                     title="All actions ranked by impact")
        fig.update_traces(texttemplate="−%{text:.0f}%")
        fig.update_layout(height=280, coloraxis_showscale=False, yaxis_title="",
                          xaxis_title="Risk reduction (%)", template="plotly_dark",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)