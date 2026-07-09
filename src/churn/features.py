"""Feature engineering for KKBox churn. Imported by notebooks AND the app
so training and serving compute identical features (no train/serve skew)."""
import numpy as np
import pandas as pd

REF_DATE = pd.Timestamp("2017-03-31")


def build_transaction_features(txn: pd.DataFrame) -> pd.DataFrame:
    txn = txn.sort_values(["msno", "transaction_date"])
    last = txn.groupby("msno").tail(1).set_index("msno")
    tf = pd.DataFrame(index=last.index)
    tf["txn_count"]        = txn.groupby("msno").size()
    tf["cancel_count"]     = txn.groupby("msno")["is_cancel"].sum()
    tf["autorenew_share"]  = txn.groupby("msno")["is_auto_renew"].mean()
    tf["total_paid"]       = txn.groupby("msno")["actual_amount_paid"].sum()
    tf["last_plan_days"]   = last["payment_plan_days"]
    tf["last_actual_paid"] = last["actual_amount_paid"]
    tf["last_list_price"]  = last["plan_list_price"]
    tf["last_discount"]    = last["plan_list_price"] - last["actual_amount_paid"]
    tf["last_auto_renew"]  = last["is_auto_renew"]
    tf["last_is_cancel"]   = last["is_cancel"]
    tf["payment_method_id"]= last["payment_method_id"]
    return tf.reset_index()


def build_log_features(logs: pd.DataFrame) -> pd.DataFrame:
    lf = logs.groupby("msno").agg(
        secs_total  = ("total_secs", "sum"),
        secs_mean   = ("total_secs", "mean"),
        unq_mean    = ("num_unq", "mean"),
        active_days = ("date", "nunique"),
        plays_100   = ("num_100", "sum"),
        plays_25    = ("num_25", "sum"),
    ).reset_index()
    lf["completion_ratio"] = lf["plays_100"] / (lf["plays_25"] + 1)
    return lf


def build_feature_matrix(labels, members, txn, logs) -> pd.DataFrame:
    members = members.copy()
    members["bd"] = members["bd"].where(members["bd"].between(10, 90))
    members["age_missing"] = members["bd"].isna().astype("int8")

    tf = build_transaction_features(txn)
    lf = build_log_features(logs)

    df = (labels[["msno", "is_churn"]]
          .merge(members[["msno","city","bd","age_missing","gender",
                          "registered_via","registration_init_time"]], on="msno", how="left")
          .merge(tf, on="msno", how="left")
          .merge(lf, on="msno", how="left"))

    df["has_txn"]  = df["txn_count"].notna().astype("int8")
    df["has_logs"] = df["secs_total"].notna().astype("int8")

    for c in ["txn_count","cancel_count","autorenew_share","total_paid","last_plan_days",
              "last_actual_paid","last_list_price","last_discount","last_auto_renew","last_is_cancel"]:
        df[c] = df[c].fillna(0)
    for c in ["secs_total","secs_mean","unq_mean","active_days","plays_100","plays_25","completion_ratio"]:
        df[c] = df[c].fillna(0)

    df["registration_init_time"] = pd.to_datetime(
        df["registration_init_time"], format="%Y%m%d", errors="coerce")
    df["tenure_days"] = (REF_DATE - df["registration_init_time"]).dt.days

    df["monthly_value"] = df["last_actual_paid"] * (30 / df["last_plan_days"].clip(lower=1))
    df["monthly_value"] = df["monthly_value"].replace([np.inf, -np.inf], 0).fillna(0)
    df["ltv"] = df["monthly_value"] * 24
    return df
