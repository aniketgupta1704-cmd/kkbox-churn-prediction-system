"""Page 5: Business Strategy — interactive retention allocator."""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from .shared import load_parquet, load_csv, esc

MONTHS_PROTECTED = 3  # a successful offer protects ~1 quarter of revenue

def render():
    st.title("Business Strategy")
    st.caption("Turn churn predictions into a budgeted, ROI-optimized retention campaign")

    df = load_parquet("business_sample.parquet").copy()

    # ---- Controls ----
    st.subheader("Campaign parameters")
    p1, p2, p3 = st.columns(3)
    budget = p1.slider("Retention budget (NT$)", 10_000, 500_000, 100_000, step=10_000)
    offer_cost = p2.slider("Offer cost per customer (NT$)", 5, 100, 30)
    success_rate = p3.slider("Retention success rate", 0.05, 0.80, 0.30, step=0.05,
                             help="Fraction of would-be churners the offer actually retains")

    # ---- Allocator (same logic as notebook) ----
    df["exp_gross"] = df["churn_prob"] * df["monthly_value"] * MONTHS_PROTECTED * success_rate
    cand = df[df["exp_gross"] > offer_cost].sort_values("exp_gross", ascending=False)
    n_afford = int(budget // offer_cost)
    targeted = cand.head(n_afford)
    spend = len(targeted) * offer_cost
    gross = targeted["exp_gross"].sum()
    net = gross - spend
    ret_pct = (100 * net / spend) if spend else 0

    # ---- KPI cards ----
    st.subheader("Campaign outcome")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Customers Targeted", f"{len(targeted):,}")
    k2.metric("Budget Spent", f"NT${spend:,.0f}")
    k3.metric("Expected Revenue Saved", f"NT${gross:,.0f}")
    k4.metric("Net Benefit", f"NT${net:,.0f}", delta=f"{ret_pct:.0f}% return")

    st.caption(esc(f"Assumes a successful offer protects ~{MONTHS_PROTECTED} months of revenue. "
                   f"Worth-targeting pool: {len(cand):,} customers; budget covers {n_afford:,}."))

    st.divider()

    # Budget utilization insight
    if len(targeted) < n_afford:
        unspent = budget - spend
        st.info(f"💡 Only {len(cand):,} customers are profitable to target at these settings. "
                f"The budget could afford {n_afford:,}, but targeting beyond the profitable pool "
                f"would lose money — so **NT${unspent:,.0f} is left unspent**. "
                f"The model tells you when to *stop* spending.")
    else:
        st.info(f"💡 Budget-constrained: {len(cand):,} customers are worth targeting, "
                f"but the budget only covers {n_afford:,}. More budget would capture more value.")

    # ---- Charts ----
    c1, c2 = st.columns(2)
    with c1:
        # cumulative net benefit as we target more customers
        t = targeted.reset_index(drop=True).copy()
        t["cum_net"] = (t["exp_gross"] - offer_cost).cumsum()
        t["rank"] = range(1, len(t) + 1)
        fig = px.area(t, x="rank", y="cum_net",
                      title="Cumulative net benefit vs customers targeted")
        fig.update_layout(height=340, xaxis_title="Customers targeted (by priority)",
                          yaxis_title="Cumulative net benefit (NT$)",
                          template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        # model vs random comparison
        rand = df.sample(len(targeted), random_state=1)
        rand_net = (rand["churn_prob"]*rand["monthly_value"]*MONTHS_PROTECTED*success_rate).sum() - len(rand)*offer_cost
        comp = pd.DataFrame({"strategy": ["Model-targeted", "Random"],
                             "net_benefit": [net, rand_net]})
        fig2 = px.bar(comp, x="strategy", y="net_benefit", color="strategy",
                      text="net_benefit", title="Model vs random targeting (same budget)",
                      color_discrete_map={"Model-targeted": "#1f6f54", "Random": "#8B2635"})
        fig2.update_traces(texttemplate="NT$%{text:,.0f}")
        fig2.update_layout(height=340, showlegend=False, yaxis_title="Net benefit (NT$)",
                           xaxis_title="", template="plotly_dark",
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

    # ---- ROI formula + assumptions ----
    with st.expander("📐 How the ROI is calculated (formula & assumptions)"):
        st.markdown("""
        **Per-customer expected saving:**
        expected_saving = P(churn) × monthly_value × months_protected × success_rate − offer_cost.
        The allocator targets every customer where this is positive, ranked by expected saving,
        until the budget is exhausted.

        **Assumptions (tunable above):**
        - *Success rate* — fraction of would-be churners the offer actually retains. A real team
          measures this via A/B test; we treat it as an input.
        - *Months protected* — a successful offer is assumed to protect ~3 months of revenue
          (deliberately conservative; not the full lifetime).
        - *Offer cost* — cost to extend one retention offer.

        The **dollar figures are assumption-dependent** (explore the sliders); the **model-vs-random
        gap is robust** — the model concentrates spend on genuine high-risk, high-value customers.
        """)

    st.divider()

    # ---- Targeting table (downloadable) ----
    st.subheader("Recommended targeting list (top 100)")
    show = targeted.head(100)[["churn_prob", "monthly_value", "exp_gross"]].copy()
    show.columns = ["Churn probability", "Monthly value (NT$)", "Expected saving (NT$)"]
    st.dataframe(show.round(2), use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download full targeting list (CSV)",
                       targeted[["churn_prob","monthly_value","exp_gross"]].to_csv(index=False),
                       "retention_targets.csv", "text/csv")