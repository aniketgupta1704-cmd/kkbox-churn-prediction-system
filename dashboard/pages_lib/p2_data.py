"""Page 2: Data & Feature Engineering."""
import streamlit as st
import plotly.express as px
import pandas as pd
from .shared import load_parquet, FEATURE_LABELS
import plotly.graph_objects as go
def render():
    st.title("Data & Feature Engineering")
    st.caption("From three raw tables to a 30-feature modeling matrix — and the decisions along the way")

    st.subheader("Data pipeline")

    fig = go.Figure()

    # -------------------------
    # Helper functions
    # -------------------------

    def add_box(x0, y0, x1, y1, color, title, subtitle):
        fig.add_shape(
            type="rect",
            x0=x0, y0=y0,
            x1=x1, y1=y1,
            line=dict(color="#666", width=2),
            fillcolor=color,
            layer="below"
        )

        fig.add_annotation(
            x=(x0+x1)/2,
            y=(y0+y1)/2,
            text=f"<b>{title}</b><br><span style='font-size:11px'>{subtitle}</span>",
            showarrow=False,
            font=dict(color="white", size=13)
        )


    def add_arrow(x0, y0, x1, y1):
        fig.add_annotation(
            x=x1,
            y=y1,
            ax=x0,
            ay=y0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.2,
            arrowwidth=2,
            arrowcolor="#B0B0B0"
        )


    # -------------------------
    # Data source boxes
    # -------------------------

    add_box(
        0.0, 0.72, 0.22, 0.95,
        "#8B2635",
        "Transactions",
        "7.6M payments<br>renewals & cancellations"
    )

    add_box(
        0.39, 0.72, 0.61, 0.95,
        "#2D5A6B",
        "Members",
        "Demographics<br>registration"
    )

    add_box(
        0.78, 0.72, 1.00, 0.95,
        "#4A4A4A",
        "User Logs",
        "30GB listening logs<br>sampled subset"
    )

    # -------------------------
    # Feature engineering
    # -------------------------

    add_box(
        0.25, 0.30, 0.75, 0.56,
        "#1F6F54",
        "Feature Engineering",
        "Joins • Aggregations<br>Sequential Features<br>Leakage Control"
    )

    # -------------------------
    # Final dataset
    # -------------------------

    add_box(
        0.30, -0.02, 0.70, 0.18,
        "#2C3E50",
        "Final Modeling Matrix",
        "480,853 subscribers<br>30 engineered features"
    )

    # -------------------------
    # Arrows
    # -------------------------

    add_arrow(0.11,0.72,0.50,0.56)
    add_arrow(0.50,0.72,0.50,0.56)
    add_arrow(0.89,0.72,0.50,0.56)

    add_arrow(0.50,0.30,0.50,0.18)

    # -------------------------
    # Layout
    # -------------------------

    fig.update_layout(

        template="plotly_dark",

        height=520,

        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20
        ),

        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",

        xaxis=dict(
            visible=False,
            range=[-0.05,1.05]
        ),

        yaxis=dict(
            visible=False,
            range=[-0.10,1.05]
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Key engineering decisions")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("##### 📉 Why we subset the User Logs")
        st.markdown("""
        The full `user_logs.csv` is **~30GB / 392M rows** — impractical to load whole.
        We streamed it in **chunks**, keeping only active users and a 4-month window,
        reducing to **24.7M rows**.

        Then we **tested whether listening data helps** — it barely did (max churn
        correlation **0.035**). KKBox churn is defined by **renewal, not engagement**:
        auto-renewing users renew regardless of listening. So we leaned **transaction-heavy**.
        """)
    with d2:
        st.markdown("##### 🎯 Why transaction features won")
        st.markdown("""
        We mined the **full transaction history** (median 17 txns/user). Winner-style
        **sequence features** — days-since-last-transaction, cancellation recency, renewal
        cadence — drove the gains (PR-AUC **0.881 → 0.915**).

        We also **removed `total_paid`**: top feature by gain (32%) but an ablation showed
        it was a **redundant recency proxy** — removing it cost nothing (−0.004, noise) and
        improved interpretability.
        """)

    st.subheader("Feature categories")
    cats = pd.DataFrame([
        {"Category": "Transaction sequence", "Count": 6, "Examples": "days_since_last, cancel_last3, gap_widening", "Signal": "Strongest"},
        {"Category": "Transaction summary", "Count": 10, "Examples": "autorenew_share, cancel_count, last_plan_days", "Signal": "Strong"},
        {"Category": "Member attributes", "Count": 5, "Examples": "city, tenure_days, registered_via", "Signal": "Moderate"},
        {"Category": "Listening (subset)", "Count": 8, "Examples": "secs_total, active_days, completion_ratio", "Signal": "Weak (kept for completeness)"},
    ])
    st.dataframe(cats, use_container_width=True, hide_index=True)

    st.subheader("Interactive feature explorer")
    st.caption("Pick a feature to see its distribution and relationship to churn (10k stratified sample)")
    sample = load_parquet("features_sample.parquet")
    numeric_feats = [c for c in FEATURE_LABELS if c in sample.columns]
    choice = st.selectbox("Feature", numeric_feats,
                          format_func=lambda c: FEATURE_LABELS.get(c, c))

    e1, e2 = st.columns(2)
    with e1:
        fig = px.histogram(sample, x=choice, nbins=40, title=f"Distribution: {FEATURE_LABELS[choice]}")
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with e2:
        s = sample.copy()
        s["_bin"] = pd.qcut(s[choice].rank(method="first"), 8, labels=False)
        cbb = s.groupby("_bin").agg(churn=("is_churn","mean"), val=(choice,"mean")).reset_index()
        fig2 = px.line(cbb, x="val", y="churn", markers=True,
                       title=f"Churn rate vs {FEATURE_LABELS[choice]}")
        fig2.update_layout(height=350, yaxis_tickformat=".0%")
        st.plotly_chart(fig2, use_container_width=True)
    st.caption("Try 'Auto-renew share' or 'Days since last transaction' for sharp separation; "
               "listening features are notably flat.")