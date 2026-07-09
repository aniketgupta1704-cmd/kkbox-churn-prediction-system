"""Page 3: Model Development."""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
from .shared import load_csv, ASSETS, esc, load_json

def render():
    st.title("Model Development")
    st.caption("From four baselines to a calibrated, feature-engineered LightGBM — and the reasoning at each fork")

    # ---- Model comparison ----
    st.subheader("Model comparison")
    comp = load_csv("model_comparison.csv")
    cc1, cc2 = st.columns([3, 2])
    with cc1:
        fig = px.bar(comp.sort_values("pr_auc"), x="pr_auc", y="model", orientation="h",
                     text="pr_auc", title="PR-AUC by model (held-out test set)",
                     color="pr_auc", color_continuous_scale="Blues")
        fig.update_traces(texttemplate="%{text:.3f}")
        fig.update_layout(height=380, coloraxis_showscale=False,
                          xaxis_title="PR-AUC", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    with cc2:
        st.markdown("#### Why PR-AUC, not accuracy")
        st.markdown("""
        At a **6.2% churn rate**, a model predicting "nobody churns" scores 94% accuracy
        and is useless. **PR-AUC** focuses on the rare positive class, so it's the honest
        metric here. ROC-AUC (all models ~0.96–0.99) is too generous under imbalance —
        it masks the huge gap between LightGBM (0.91) and Logistic Regression (0.69).
        """)
        st.dataframe(comp[["model", "pr_auc", "roc_auc"]], use_container_width=True, hide_index=True)

    st.divider()

    # ---- The narrative: why advanced feature eng ----
    st.subheader("The pivotal decision: features over tuning")
    n1, n2 = st.columns(2)
    with n1:
        st.markdown("##### 1️⃣ Optuna tuning plateaued")
        st.markdown("""
        We ran **30 Optuna trials** on the baseline features. The result was a *non-result*:
        tuned PR-AUC **0.884** vs baseline **0.881** — statistically identical. Trial scores
        plateaued regardless of configuration.

        **The lesson:** hyperparameters were **not** the bottleneck. The predictive ceiling
        was set by the *features*, not the model config. This told us where to invest next.
        """)
    with n2:
        st.markdown("##### 2️⃣ So we engineered better features")
        st.markdown("""
        Instead of more tuning, we mined the **full transaction history** for sequence
        features. This moved PR-AUC **0.881 → 0.915** — a real gain, far larger than tuning
        produced.

        And critically, we'd **tested the engagement hypothesis first** (listening trends)
        and *rejected it* — churn here is renewal-driven. So we deliberately built a
        **transaction-heavy** model, which the data validated.
        """)
    st.info("**Key insight for reviewers:** the biggest gain came from feature engineering "
            "(+0.034 PR-AUC), not hyperparameter tuning (+0.003). Knowing *where* performance "
            "comes from — and acting on it — mattered more than optimizing any single model.")

    st.divider()

    # ---- Calibration ----
    st.subheader("Calibration: why 'no imbalance treatment' won")
    cal1, cal2 = st.columns([3, 2])
    with cal1:
        cal = load_json("calibration_data.json")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                 line=dict(dash="dash", color="gray"), name="Perfect"))
        fig.add_trace(go.Scatter(x=cal["none"]["mean_pred"], y=cal["none"]["frac_pos"],
                                 mode="lines+markers", name="No treatment (chosen)",
                                 line=dict(color="#4b9fff", width=3)))
        fig.add_trace(go.Scatter(x=cal["class_weight"]["mean_pred"], y=cal["class_weight"]["frac_pos"],
                                 mode="lines+markers", name="Class-weighted",
                                 line=dict(color="#ff8c42", width=3)))
        fig.update_layout(height=400, title="Calibration curve",
                          xaxis_title="Mean predicted probability",
                          yaxis_title="Actual churn fraction",
                          template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    with cal2:
        st.markdown("""
        We compared imbalance strategies. `scale_pos_weight` matched PR-AUC but **destroyed
        probability calibration** — it inflates predicted risk (orange line bows below the
        diagonal).

        Since the **business layer needs trustworthy probabilities** (dollar-at-risk =
        P(churn) × value), we chose **no imbalance treatment** — nearly identical ranking,
        far better calibration (**Brier 0.014 vs 0.037**). The blue line hugs the diagonal:
        70% predicted ≈ 70% actual.
        """)