"""
Cloud Cost Anomaly Detection — REST API
FastAPI endpoints mounted onto the Dash app server.
Base URL: /api/v1/
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, date
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import warnings
warnings.filterwarnings("ignore")

api = FastAPI(
    title="Cloud Cost Anomaly Detection API",
    description="REST API for querying anomaly detection results, model metrics, and live predictions.",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

BASE = os.path.dirname(os.path.abspath(__file__))

# ── Cached data ────────────────────────────────────────────────
_cache = {}

def get_data(key, path, parse_dates=None):
    if key not in _cache:
        try:
            full = os.path.join(BASE, path)
            if path.endswith(".json"):
                _cache[key] = json.load(open(full))
            else:
                _cache[key] = pd.read_csv(full, parse_dates=parse_dates or [])
        except Exception as e:
            print(f"Cache load failed for {key}: {e}")
            return None
    return _cache[key]


# ── Pydantic models ────────────────────────────────────────────
class PredictRequest(BaseModel):
    provider:         str  = Field(..., example="AWS", description="Cloud provider: AWS / GCP / Azure")
    date:             str  = Field(..., example="2024-06-16", description="Date in YYYY-MM-DD")
    total_cost:       float= Field(..., example=7500.0,  description="Actual daily cost in USD")
    rolling_mean_7d:  float= Field(..., example=3100.0,  description="7-day rolling mean cost")
    rolling_std_7d:   float= Field(..., example=280.0,   description="7-day rolling std dev")
    rolling_mean_30d: float= Field(..., example=3050.0,  description="30-day rolling mean cost")
    rolling_std_30d:  float= Field(..., example=310.0,   description="30-day rolling std dev")
    z_score:          float= Field(..., example=15.8,    description="Z-score vs 7-day mean")

class AlertOut(BaseModel):
    id:                  str
    timestamp:           str
    date:                str
    provider:            str
    severity:            str
    ensemble_score:      float
    actual_cost:         float
    forecast_cost:       float
    pct_above_forecast:  float
    message:             str

class MetricOut(BaseModel):
    model:          str
    precision:      float
    recall:         float
    f1_score:       float
    auc_roc:        float
    true_positives: int
    false_positives:int
    false_negatives:int


# ── Routes ─────────────────────────────────────────────────────

@api.get("/api/v1/health", tags=["System"])
def health():
    """Health check — confirms API is running."""
    return {
        "status":    "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version":   "1.0.0",
        "models":    ["isolation_forest", "arima", "ensemble"],
    }


@api.get("/api/v1/metrics", tags=["Evaluation"], response_model=List[MetricOut])
def get_metrics():
    """Return performance metrics for all three models."""
    results = get_data("pipeline", "reports/pipeline_results.json")
    if not results:
        raise HTTPException(503, "Pipeline results not available")
    out = []
    for model_name, key in [
        ("Isolation Forest", "if_metrics"),
        ("ARIMA",            "arima_metrics"),
        ("Ensemble",         "ensemble_metrics"),
    ]:
        m = results.get(key, {})
        out.append(MetricOut(
            model=          model_name,
            precision=      round(m.get("precision",      0), 4),
            recall=         round(m.get("recall",         0), 4),
            f1_score=       round(m.get("f1_score",       0), 4),
            auc_roc=        round(m.get("auc_roc",        0), 4),
            true_positives= m.get("true_positives",  0),
            false_positives=m.get("false_positives", 0),
            false_negatives=m.get("false_negatives", 0),
        ))
    return out


@api.get("/api/v1/anomalies", tags=["Anomalies"])
def get_anomalies(
    provider:  Optional[str] = Query(None, description="Filter by provider: AWS/GCP/Azure"),
    severity:  Optional[str] = Query(None, description="Filter by severity: critical/high/medium/low"),
    start_date:Optional[str] = Query(None, description="YYYY-MM-DD"),
    end_date:  Optional[str] = Query(None, description="YYYY-MM-DD"),
    limit:     int           = Query(50,   description="Max records to return"),
):
    """Query detected anomalies from the test period."""
    preds = get_data("preds", "reports/full_predictions.csv", parse_dates=["date"])
    if preds is None:
        raise HTTPException(503, "Predictions not available")

    df = preds[preds["ensemble_prediction"] == 1].copy()

    if provider:
        df = df[df["provider"].str.upper() == provider.upper()]
    if severity:
        df = df[df["alert_severity"].str.lower() == severity.lower()]
    if start_date:
        df = df[df["date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["date"] <= pd.to_datetime(end_date)]

    df = df.sort_values("ensemble_score", ascending=False).head(limit)

    return {
        "count": len(df),
        "anomalies": df[["date","provider","total_cost","ensemble_score","alert_severity",
                          "arima_forecast","arima_upper_ci","has_anomaly"]].fillna(0).to_dict(orient="records"),
    }


@api.get("/api/v1/alerts", tags=["Alerts"], response_model=List[AlertOut])
def get_alerts(
    provider: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    limit:    int           = Query(20),
):
    """Return fired alerts from the alert log."""
    alerts = get_data("alerts", "reports/alert_log.json")
    if not alerts:
        raise HTTPException(503, "Alert log not available")

    filtered = alerts
    if provider:
        filtered = [a for a in filtered if a.get("provider","").upper() == provider.upper()]
    if severity:
        filtered = [a for a in filtered if a.get("severity","").lower() == severity.lower()]

    filtered = sorted(filtered, key=lambda x: x.get("ensemble_score", 0), reverse=True)[:limit]

    return [AlertOut(
        id=                  a.get("id", ""),
        timestamp=           a.get("timestamp", ""),
        date=                str(a.get("date", ""))[:10],
        provider=            a.get("provider", ""),
        severity=            a.get("severity", ""),
        ensemble_score=      round(a.get("ensemble_score", 0), 4),
        actual_cost=         round(a.get("actual_cost", 0), 2),
        forecast_cost=       round(a.get("forecast_cost", 0), 2),
        pct_above_forecast=  round(a.get("pct_above_forecast", 0), 1),
        message=             a.get("message", ""),
    ) for a in filtered]


@api.get("/api/v1/spending", tags=["Data"])
def get_spending(
    provider:   Optional[str] = Query(None,  description="AWS / GCP / Azure"),
    granularity:str           = Query("daily", description="daily | monthly"),
    start_date: Optional[str] = Query(None),
    end_date:   Optional[str] = Query(None),
):
    """Return aggregated spending data for charts."""
    daily = get_data("daily", "data/processed/daily_aggregated.csv", parse_dates=["date"])
    if daily is None:
        raise HTTPException(503, "Spending data not available")

    df = daily.copy()
    if provider:
        df = df[df["provider"].str.upper() == provider.upper()]
    if start_date:
        df = df[df["date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["date"] <= pd.to_datetime(end_date)]

    if granularity == "monthly":
        df["period"] = df["date"].dt.to_period("M").astype(str)
        df = df.groupby(["period","provider"]).agg(
            total_cost=("total_cost","sum"),
            anomaly_days=("has_anomaly","sum"),
        ).reset_index().rename(columns={"period":"date"})

    df["date"] = df["date"].astype(str)
    return {"count": len(df), "records": df.fillna(0).to_dict(orient="records")}


@api.post("/api/v1/predict", tags=["Prediction"])
def predict_anomaly(req: PredictRequest):
    """
    Live anomaly prediction for a single billing record.
    Computes rule-based ensemble score from the provided features.
    Returns: is_anomaly, severity, ensemble_score, explanation.
    """
    # Rule-based scoring (mirrors the trained ensemble logic)
    z     = abs(req.z_score)
    dev7  = (req.total_cost - req.rolling_mean_7d) / (req.rolling_std_7d + 1e-9)
    dev30 = (req.total_cost - req.rolling_mean_30d) / (req.rolling_std_30d + 1e-9)

    # IF-style score: deviation from rolling mean
    if_score = min(max(z / 20.0, 0), 1)

    # ARIMA-style score: distance above upper CI
    upper_ci = req.rolling_mean_7d + 2.5 * req.rolling_std_7d
    arima_flag  = req.total_cost > upper_ci
    arima_score = min(max((req.total_cost - upper_ci) / (upper_ci + 1e-9), 0), 1) if arima_flag else 0.0

    ensemble_score = round(0.45 * if_score + 0.55 * arima_score, 4)
    both_agree     = if_score >= 0.5 and arima_flag
    is_anomaly     = ensemble_score >= 0.60 or both_agree

    if   ensemble_score >= 0.90: severity = "CRITICAL"
    elif ensemble_score >= 0.75: severity = "HIGH"
    elif ensemble_score >= 0.60: severity = "MEDIUM"
    elif ensemble_score >= 0.40: severity = "LOW"
    else:                        severity = "NONE"

    pct_above = round((req.total_cost - req.rolling_mean_7d) / (req.rolling_mean_7d + 1e-9) * 100, 1)

    return {
        "input":          req.dict(),
        "is_anomaly":     is_anomaly,
        "severity":       severity if is_anomaly else "NONE",
        "ensemble_score": ensemble_score,
        "if_score":       round(if_score, 4),
        "arima_score":    round(arima_score, 4),
        "arima_flagged":  arima_flag,
        "pct_above_mean": pct_above,
        "upper_ci_bound": round(upper_ci, 2),
        "explanation": (
            f"Cost ${req.total_cost:,.0f} is {pct_above:+.0f}% vs 7-day mean ${req.rolling_mean_7d:,.0f}. "
            f"Z-score={z:.1f}. {'Exceeds 2.5σ upper bound ($' + f'{upper_ci:,.0f}).' if arima_flag else 'Within confidence interval.'} "
            f"Ensemble score={ensemble_score:.3f} → {severity}."
        ),
        "timestamp": datetime.utcnow().isoformat(),
    }


@api.get("/api/v1/summary", tags=["System"])
def get_summary():
    """High-level project summary — useful for external integrations."""
    results = get_data("pipeline", "reports/pipeline_results.json") or {}
    alerts  = get_data("alerts",   "reports/alert_log.json") or []
    em      = results.get("ensemble_metrics", {})
    return {
        "project": "AI-Driven Cloud Cost Anomaly Detection",
        "providers": ["AWS", "GCP", "Azure"],
        "date_range": results.get("data_summary", {}).get("date_range", "N/A"),
        "total_records": results.get("data_summary", {}).get("total_records", 0),
        "models": {
            "isolation_forest": results.get("if_metrics",       {}),
            "arima":            results.get("arima_metrics",     {}),
            "ensemble":         results.get("ensemble_metrics",  {}),
        },
        "alerts_total":    len(alerts),
        "false_positives": em.get("false_positives", 0),
        "ensemble_f1":     em.get("f1_score",  0),
        "ensemble_auc":    em.get("auc_roc",   0),
    }
