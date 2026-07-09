"""Pydantic request/response schemas for the churn API."""
from typing import Optional
from pydantic import BaseModel, Field


class ChurnRequest(BaseModel):
    """One customer's engineered features (computed upstream)."""
    # --- Member attributes (optional: ~11% of users lack a members row) ---
    city: Optional[float] = Field(None, description="City code")
    bd: Optional[float] = Field(None, description="Age (cleaned; NaN if missing)")
    age_missing: int = Field(0, ge=0, le=1, description="1 if age was missing/garbage")
    gender: Optional[str] = Field(None, description="'male'/'female'/None")
    registered_via: Optional[float] = Field(None, description="Registration channel code")

    # --- Transaction summary ---
    txn_count: float = Field(..., ge=0)
    cancel_count: float = Field(..., ge=0)
    autorenew_share: float = Field(..., ge=0, le=1)
    last_plan_days: float = Field(..., ge=0)
    last_actual_paid: float = Field(..., ge=0)
    last_list_price: float = Field(..., ge=0)
    last_discount: float
    last_auto_renew: float = Field(..., ge=0, le=1)
    last_is_cancel: float = Field(..., ge=0, le=1)
    payment_method_id: Optional[float] = Field(None)

    # --- Listening (March snapshot) ---
    secs_total: float = Field(..., ge=0)
    secs_mean: float = Field(..., ge=0)
    unq_mean: float = Field(..., ge=0)
    active_days: float = Field(..., ge=0)
    plays_100: float = Field(..., ge=0)
    plays_25: float = Field(..., ge=0)
    completion_ratio: float = Field(..., ge=0)
    has_logs: int = Field(..., ge=0, le=1)
    tenure_days: Optional[float] = Field(None)

    # --- Transaction-sequence features ---
    seq_days_since_last: float
    seq_cancel_last3: float = Field(..., ge=0)
    seq_price_slope: float
    seq_gap_widening: float
    seq_mean_gap_days: float = Field(..., ge=0)
    seq_ar_dropped: int = Field(..., ge=0, le=1)

    model_config = {
        "json_schema_extra": {
            "example": {
                "city": 13.0, "bd": 38.0, "age_missing": 0, "gender": "male",
                "registered_via": 7.0, "txn_count": 17, "cancel_count": 0,
                "autorenew_share": 1.0, "last_plan_days": 30, "last_actual_paid": 149,
                "last_list_price": 149, "last_discount": 0, "last_auto_renew": 1,
                "last_is_cancel": 0, "payment_method_id": 41.0, "secs_total": 50000,
                "secs_mean": 4000, "unq_mean": 30, "active_days": 20, "plays_100": 200,
                "plays_25": 20, "completion_ratio": 9.5, "has_logs": 1, "tenure_days": 2000,
                "seq_days_since_last": 5, "seq_cancel_last3": 0, "seq_price_slope": 0,
                "seq_gap_widening": 0.02, "seq_mean_gap_days": 30, "seq_ar_dropped": 0
            }
        }
    }


class ChurnResponse(BaseModel):
    churn_probability: float = Field(..., description="Calibrated churn probability [0,1]")
    risk_flag: str = Field(..., description="LOW / MEDIUM / HIGH")
    threshold_used: float = Field(..., description="Probability cutoff for flagging")