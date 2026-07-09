"""Page 8: System Architecture."""
import streamlit as st

def render():
    st.title("System Architecture")
    st.caption("End-to-end pipeline — from raw KKBox tables to a deployed, monitored ML service")

    st.subheader("Pipeline")
    import plotly.graph_objects as go

    fig = go.Figure()

    def add_box(cx, cy, color, title, subtitle="", w=0.20, h=0.075):
        x0, x1 = cx - w/2, cx + w/2
        y0, y1 = cy - h/2, cy + h/2
        fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                      line=dict(color="#666", width=1.5), fillcolor=color, layer="below")
        txt = f"<b>{title}</b>"
        if subtitle:
            txt += f"<br><span style='font-size:10px'>{subtitle}</span>"
        fig.add_annotation(x=cx, y=cy, text=txt, showarrow=False, font=dict(color="white", size=12))
        return (cx, cy, h)  # return center + half-height for arrow anchoring

    def arrow_between(a, b):
        """Draw arrow from bottom of box a to top of box b (vertical)."""
        cx_a, cy_a, h_a = a
        cx_b, cy_b, h_b = b
        fig.add_annotation(x=cx_b, y=cy_b + h_b/2, ax=cx_a, ay=cy_a - h_a/2,
                           xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=3, arrowsize=1, arrowwidth=1.8,
                           arrowcolor="#aaa")

    # Vertical spine, centered
    raw   = add_box(0.5, 0.94, "#8B2635", "Raw KKBox Tables", "transactions · members · logs")
    fe    = add_box(0.5, 0.80, "#2D5A6B", "Feature Engineering", "src/churn/features.py")
    train = add_box(0.5, 0.66, "#1F6F54", "Training + Calibration", "LightGBM · Optuna · MLflow")
    model = add_box(0.5, 0.52, "#2C3E50", "LightGBM Model", "model_canonical.joblib")
    api   = add_box(0.5, 0.38, "#1F6F54", "FastAPI Service", "/predict /explain /counterfactual")
    docker= add_box(0.5, 0.24, "#2D5A6B", "Docker Container", "python:3.12 + libgomp1")
    ui    = add_box(0.5, 0.10, "#8B2635", "Streamlit Dashboard", "this app")

    # Side consumers of the model (SHAP, Evidently) — placed left, arrows from model
    shap  = add_box(0.16, 0.52, "#4A3A5A", "SHAP", "explainability", w=0.16)
    drift = add_box(0.84, 0.52, "#4A3A5A", "Evidently", "drift monitoring", w=0.16)

    # Vertical spine arrows
    arrow_between(raw, fe)
    arrow_between(fe, train)
    arrow_between(train, model)
    arrow_between(model, api)
    arrow_between(api, docker)
    arrow_between(docker, ui)

    # Horizontal arrows from model to side consumers
    fig.add_annotation(x=shap[0]+0.08, y=0.52, ax=model[0]-0.10, ay=0.52,
                       xref="x", yref="y", axref="x", ayref="y",
                       showarrow=True, arrowhead=3, arrowsize=1, arrowwidth=1.8, arrowcolor="#aaa")
    fig.add_annotation(x=drift[0]-0.08, y=0.52, ax=model[0]+0.10, ay=0.52,
                       xref="x", yref="y", axref="x", ayref="y",
                       showarrow=True, arrowhead=3, arrowsize=1, arrowwidth=1.8, arrowcolor="#aaa")

    fig.update_layout(
        template="plotly_dark", height=640,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0.02, 1.0]),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Technology stack")
    cols = st.columns(5)
    stack = [
        ("🌲 LightGBM", "Gradient-boosted model"),
        ("🔍 Optuna", "Hyperparameter search"),
        ("📊 SHAP", "Explainability"),
        ("📈 MLflow", "Experiment tracking"),
        ("⚡ FastAPI", "Model serving"),
        ("🐳 Docker", "Containerization"),
        ("📉 Evidently", "Drift monitoring"),
        ("🎨 Streamlit", "Dashboard"),
        ("🔀 GitHub Actions", "CI/CD"),
        ("🤗 HF Spaces", "Deployment"),
    ]
    for i, (name, desc) in enumerate(stack):
        with cols[i % 5]:
            st.markdown(f"**{name}**")
            st.caption(desc)

    st.divider()
    st.subheader("Deployment architecture")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("##### Microservice design")
        st.markdown("""
        The system deploys as **two independent services**:
        - **API** (Docker Space) — the model runtime: LightGBM, SHAP, all native
          dependencies frozen in a reproducible container.
        - **Dashboard** (Streamlit Space) — pure presentation, calls the API over HTTP.

        This separation means the dashboard stays lightweight (no ML dependencies), and the
        fragile ML-runtime environment is isolated in one place.
        """)
    with d2:
        st.markdown("##### Why containerize")
        st.markdown("""
        The container solves the exact environment problems encountered in development
        (Python version conflicts, the `libgomp` native library for LightGBM). Baked into
        the image once, they never recur — the API runs identically on any host.

        `Streamlit → HTTP → Dockerized FastAPI → model` is real microservice architecture,
        not a monolith.
        """)