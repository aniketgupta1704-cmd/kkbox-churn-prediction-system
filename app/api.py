"""FastAPI churn-prediction service."""
import os
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

import shap
import numpy as np

from app.schemas import ChurnRequest, ChurnResponse

MODEL_PATH = os.getenv("MODEL_PATH", "models/model_canonical.joblib")
THRESHOLD = float(os.getenv("CHURN_THRESHOLD", "0.5"))  # tune on Day 12

# Holds the loaded model; populated at startup via the lifespan handler
state = {"model": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at {MODEL_PATH}")
    state["model"] = joblib.load(MODEL_PATH)
    # Build SHAP explainer on the model's tree component
    prep = state["model"].named_steps["prep"]
    booster = state["model"].named_steps["m"]
    state["prep"] = prep
    state["explainer"] = shap.TreeExplainer(booster)
    state["feat_names"] = prep.get_feature_names_out()
    print(f"Model + SHAP explainer loaded from {MODEL_PATH}")
    yield
    state.clear()


app = FastAPI(
    title="KKBox Churn Prediction API",
    description="Scores subscriber churn risk from engineered features.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    """Redirect the bare root to the interactive docs."""
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": state["model"] is not None}


@app.post("/predict", response_model=ChurnResponse)
def predict(req: ChurnRequest):
    model = state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    row = pd.DataFrame([req.model_dump()])
    try:
        prob = float(model.predict_proba(row)[:, 1][0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    if prob >= max(THRESHOLD, 0.7):
        flag = "HIGH"
    elif prob >= THRESHOLD:
        flag = "MEDIUM"
    else:
        flag = "LOW"

    return ChurnResponse(churn_probability=round(prob, 4),
                         risk_flag=flag, threshold_used=THRESHOLD)


@app.post("/explain")
def explain(req: ChurnRequest):
    model = state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    row = pd.DataFrame([req.model_dump()])
    prob = float(model.predict_proba(row)[:, 1][0])

    # SHAP on the preprocessed row
    row_t = state["prep"].transform(row)
    shap_vals = state["explainer"].shap_values(row_t)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    shap_vals = np.array(shap_vals).reshape(-1)

    # Rank features by absolute contribution
    feat_names = state["feat_names"]
    contribs = sorted(zip(feat_names, shap_vals), key=lambda x: abs(x[1]), reverse=True)

    # Build plain-language reasons from the ORIGINAL (un-scaled) request values
    raw = req.model_dump()
    top = []
    for fname, sval in contribs[:6]:
        direction = "increases" if sval > 0 else "decreases"
        # strip the "num__" / "cat__" prefix for readability
        clean = fname.split("__", 1)[-1]
        top.append({
            "feature": clean,
            "effect": direction + " churn risk",
            "shap_value": round(float(sval), 3),
        })

    return {"churn_probability": round(prob, 4), "top_factors": top}

from typing import List

@app.post("/predict_batch")
def predict_batch(reqs: List[ChurnRequest]):
    model = state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    rows = pd.DataFrame([r.model_dump() for r in reqs])
    probs = model.predict_proba(rows)[:, 1]
    return {"probabilities": [round(float(p), 4) for p in probs]}

# Add near the other endpoints
LEVERS = {
    "Enable auto-renew":           {"last_auto_renew": 1, "autorenew_share": 1.0},
    "Reverse recent cancellation": {"last_is_cancel": 0, "seq_cancel_last3": 0},
    "Re-engage (recent activity)": {"seq_days_since_last": 5},
}

@app.post("/counterfactual")
def counterfactual(req: ChurnRequest):
    model = state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    import pandas as pd
    base_row = pd.DataFrame([req.model_dump()])
    base_prob = float(model.predict_proba(base_row)[:, 1][0])

    results = []
    # fixed levers
    for label, changes in LEVERS.items():
        cf = base_row.copy()
        for f, v in changes.items():
            if f in cf.columns:
                cf[f] = v
        new_prob = float(model.predict_proba(cf)[:, 1][0])
        results.append({"intervention": label, "new_prob": round(new_prob, 3),
                        "reduction_pct": round(100*(base_prob-new_prob)/base_prob, 1) if base_prob else 0})
    # discount lever (depends on list price)
    cf = base_row.copy()
    cf["last_discount"] = float(base_row["last_list_price"].iloc[0]) * 0.2
    new_prob = float(model.predict_proba(cf)[:, 1][0])
    results.append({"intervention": "Offer 20% loyalty discount", "new_prob": round(new_prob, 3),
                    "reduction_pct": round(100*(base_prob-new_prob)/base_prob, 1) if base_prob else 0})

    results.sort(key=lambda r: r["reduction_pct"], reverse=True)
    return {"base_prob": round(base_prob, 3), "interventions": results}