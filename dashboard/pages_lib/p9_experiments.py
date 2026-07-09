"""Page 9: Experiments & Insights."""
import streamlit as st

def render():
    st.title("Experiments & Insights")
    st.caption("The scientific journey — what we tried, what worked, what we rejected, and why")

    st.markdown("""
    This project's core lesson: **feature engineering — not hyperparameter tuning — drove the
    gains.** The experiments below document that reasoning, including the negative results,
    which were as informative as the successes.
    """)

    st.subheader("✅ What worked")
    w = [
        ("Transaction-sequence features", "PR-AUC 0.881 → 0.915",
         "Mining the full transaction history (recency, cancellation patterns, renewal cadence) "
         "produced the single biggest gain — far larger than any tuning."),
        ("No imbalance treatment + calibration", "Brier 0.014 vs 0.037",
         "Chose calibrated probabilities over class-weighting. Class weights matched ranking but "
         "destroyed calibration — and the business layer needs trustworthy probabilities."),
        ("Removing total_paid (ablation)", "−0.004 PR-AUC (noise)",
         "The top feature by gain was a redundant recency proxy. Removing it cost nothing and "
         "improved interpretability — verified by ablation, not assumed."),
    ]
    for title, metric, desc in w:
        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            c1.markdown(f"**{title}**")
            c1.caption(desc)
            c2.markdown("**Result**")
            c2.markdown(f"<span style='color:#4bff9f; font-size:1.1rem'>{metric}</span>",
                        unsafe_allow_html=True)

    st.subheader("❌ What we rejected (and why it mattered)")
    r = [
        ("Listening-trend features", "max corr 0.035",
         "Hypothesized engagement predicts churn. Tested it — it didn't. KKBox churn is "
         "renewal-driven, not engagement-driven. This negative result redirected the whole "
         "feature strategy toward transactions."),
        ("Aggressive hyperparameter tuning", "0.884 vs 0.881 baseline",
         "30 Optuna trials moved PR-AUC by +0.003 — statistically nothing. The finding: "
         "hyperparameters weren't the bottleneck, features were. This is why we pivoted to "
         "feature engineering."),
        ("SMOTE oversampling", "reduced performance",
         "Synthetic minority oversampling degraded results — tree models handle imbalance "
         "natively and don't benefit from it."),
    ]
    for title, metric, desc in r:
        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            c1.markdown(f"**{title}**")
            c1.caption(desc)
            c2.markdown("**Finding**")
            c2.markdown(f"<span style='color:#ff8c42; font-size:1.1rem'>{metric}</span>",
                        unsafe_allow_html=True)

    st.divider()
    st.info("**The meta-lesson:** knowing *where* performance comes from — and being willing to "
            "document what *didn't* work — mattered more than optimizing any single component. "
            "The biggest gain (+0.034 from features) dwarfed the tuning gain (+0.003), and the "
            "rejected experiments shaped the final design as much as the successful ones.")