"""
Bring-Your-Own-Export Analysis Pipeline
Runs anomaly detection on user-uploaded billing data. Unlike main.py there
is no ground truth, so there are no metrics — the output is the findings:
flagged days, severity, forecast vs actual, and per-service root causes.
"""

import numpy as np
import pandas as pd

from src.preprocessing.pipeline     import BillingPreprocessor
from src.detection.isolation_forest import IsolationForestDetector
from src.detection.arima_detector   import ARIMADetector
from src.detection.ensemble         import EnsembleDetector
from src.alerting.root_cause        import RootCauseAnalyzer

MIN_HISTORY_DAYS = 60   # need enough history to learn a baseline
MAX_DETECT_DAYS  = 45   # cap the detection window to keep analysis fast


def analyze_billing_data(service_df: pd.DataFrame) -> dict:
    """
    Full label-free analysis of normalized billing data
    (date, provider, service, region, cost).

    Trains on all history except the most recent window, then runs the
    IF + ARIMA ensemble over that window with rolling one-step-ahead
    forecasts. Returns predictions, anomaly findings with root causes,
    and summary stats. Raises ValueError if there isn't enough history.
    """
    df = service_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    n_days = df["date"].nunique()
    if n_days < MIN_HISTORY_DAYS:
        raise ValueError(
            f"Need at least {MIN_HISTORY_DAYS} days of history to learn a "
            f"baseline — this file covers {n_days} days."
        )

    # Placeholder label so the shared preprocessing code runs; never used
    # for scoring since there is no ground truth here.
    df["is_anomaly"] = False

    # Daily provider-level aggregates with rolling features
    prep  = BillingPreprocessor(output_dir="data/processed")
    daily = prep.create_daily_aggregates(df)

    # Temporal split: detect on the most recent window, train on the rest
    dates        = sorted(daily["date"].unique())
    detect_days  = min(MAX_DETECT_DAYS, max(14, int(n_days * 0.25)))
    split_date   = dates[-detect_days]
    train_daily  = daily[daily["date"] <  split_date]
    detect_daily = daily[daily["date"] >= split_date]

    # Isolation Forest
    if_detector = IsolationForestDetector(contamination=0.05, n_estimators=200)
    if_detector.train(train_daily)
    if_preds = if_detector.predict(detect_daily)

    # ARIMA — rolling one-step-ahead (train and detect windows are
    # contiguous, so no warmup gap to bridge)
    arima_detector = ARIMADetector(order=(2, 1, 2), seasonal_order=(1, 0, 1, 7),
                                   sigma_threshold=2.5)
    arima_detector.train(train_daily)
    arima_preds = arima_detector.forecast_and_detect(detect_daily)

    # Ensemble
    ensemble = EnsembleDetector(if_weight=0.45, arima_weight=0.55,
                                score_threshold=0.60, require_both=True)
    combined = arima_preds.set_index(["date", "provider"])
    if_idx   = if_preds.set_index(["date", "provider"])
    for col in ["if_prediction", "if_score", "if_confidence"]:
        combined[col] = if_idx[col] if col in if_idx.columns else 0
    combined = ensemble.combine(combined.reset_index())

    # Root-cause attribution from the full service-level history
    rca = RootCauseAnalyzer(df, baseline_window=30, top_n=3)

    findings = []
    flagged = combined[combined["ensemble_prediction"] == 1].sort_values(
        "ensemble_score", ascending=False)
    for _, row in flagged.iterrows():
        drivers  = rca.explain(row["provider"], row["date"])
        forecast = float(row.get("arima_forecast", 0))
        actual   = float(row["total_cost"])
        findings.append({
            "date":          str(row["date"])[:10],
            "provider":      row["provider"],
            "severity":      row["alert_severity"],
            "ensemble_score": round(float(row["ensemble_score"]), 4),
            "actual_cost":   round(actual, 2),
            "forecast_cost": round(forecast, 2),
            "excess_usd":    round(actual - forecast, 2),
            "pct_above_forecast": round(100 * (actual - forecast) / (forecast + 1e-9), 1),
            "root_causes":   drivers,
        })

    summary = {
        "days_analyzed":   int(detect_daily["date"].nunique()),
        "training_days":   int(train_daily["date"].nunique()),
        "detect_window":   f"{str(split_date)[:10]} → {str(max(dates))[:10]}",
        "providers":       sorted(combined["provider"].unique()),
        "anomaly_days":    len(findings),
        "total_excess_usd": round(sum(max(0, f["excess_usd"]) for f in findings), 2),
        "total_cost_in_window": round(float(detect_daily["total_cost"].sum()), 2),
    }

    return {"summary": summary, "findings": findings, "predictions": combined}


if __name__ == "__main__":
    from src.ingestion.adapters import normalize_billing_export

    # End-to-end smoke test: simulated data disguised as a generic export
    raw = pd.read_csv("data/simulated/multicloud_billing.csv")
    export = raw.rename(columns={"cost": "Cost Amount", "service": "Product Name",
                                 "provider": "Cloud Vendor", "region": "Location"})
    export = export[["date", "Cloud Vendor", "Product Name", "Location", "Cost Amount"]]

    norm, info = normalize_billing_export(export)
    print(f"Detected format: {info['format']} | {info['days']} days | "
          f"{info['services']} services")

    results = analyze_billing_data(norm)
    s = results["summary"]
    print(f"\nDetect window : {s['detect_window']} ({s['days_analyzed']} days)")
    print(f"Anomaly days  : {s['anomaly_days']} | excess: ${s['total_excess_usd']:,.0f}")
    for f in results["findings"][:5]:
        print(f"\n  [{f['severity'].upper()}] {f['provider']} {f['date']} | "
              f"${f['actual_cost']:,.0f} ({f['pct_above_forecast']:+.0f}%)")
        for d in f["root_causes"]:
            print(f"    - {d['service']}: +${d['excess_usd']:,.0f} "
                  f"({d['share_of_excess_pct']:.0f}% of excess)")
