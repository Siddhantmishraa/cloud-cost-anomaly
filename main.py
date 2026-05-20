"""
Main Pipeline Orchestrator
Runs the complete cloud cost anomaly detection system end-to-end:
  1. Generate synthetic multi-cloud billing data
  2. Preprocess and feature engineer
  3. Train Isolation Forest + ARIMA
  4. Ensemble predictions
  5. Fire alerts
  6. Save all results for dashboard and reporting
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ingestion.data_simulator   import generate_full_dataset
from src.preprocessing.pipeline     import BillingPreprocessor
from src.detection.isolation_forest import IsolationForestDetector
from src.detection.arima_detector   import ARIMADetector
from src.detection.ensemble         import EnsembleDetector
from src.alerting.alert_engine      import AlertEngine


def run_pipeline(
    start_date:   str = "2023-01-01",
    end_date:     str = "2024-06-30",
    skip_simulate: bool = False,
) -> dict:
    """
    Full end-to-end pipeline. Returns a dict with all results and metrics.
    """
    print("\n" + "█" * 60)
    print("  CLOUD COST ANOMALY DETECTION SYSTEM")
    print("  Running full pipeline...")
    print("█" * 60)

    os.makedirs("data/simulated", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models",         exist_ok=True)
    os.makedirs("reports",        exist_ok=True)

    # ─────────────────────────────────────────────
    # STEP 1: Data Simulation
    # ─────────────────────────────────────────────
    print("\n[1/6] Generating multi-cloud billing data...")
    raw_path = "data/simulated/multicloud_billing.csv"

    if skip_simulate and os.path.exists(raw_path):
        print(f"  Skipping simulation, using existing {raw_path}")
        raw_df = pd.read_csv(raw_path, parse_dates=["date"])
    else:
        raw_df = generate_full_dataset(start_date, end_date, "data/simulated")

    # ─────────────────────────────────────────────
    # STEP 2: Preprocessing
    # ─────────────────────────────────────────────
    print("\n[2/6] Preprocessing pipeline...")
    preprocessor = BillingPreprocessor(output_dir="data/processed")
    prep_summary  = preprocessor.run(raw_path)

    daily_df  = pd.read_csv("data/processed/daily_aggregated.csv", parse_dates=["date"])
    daily_df  = daily_df.sort_values(["provider","date"]).reset_index(drop=True)

    # Temporal split for daily aggregates
    dates     = sorted(daily_df["date"].unique())
    n         = len(dates)
    train_end = dates[int(n * 0.70)]
    val_end   = dates[int(n * 0.85)]

    train_daily = daily_df[daily_df["date"] <= train_end]
    val_daily   = daily_df[(daily_df["date"] > train_end) & (daily_df["date"] <= val_end)]
    test_daily  = daily_df[daily_df["date"] > val_end]

    print(f"  Daily aggregate split: train={len(train_daily)}, "
          f"val={len(val_daily)}, test={len(test_daily)}")

    # ─────────────────────────────────────────────
    # STEP 3: Isolation Forest
    # ─────────────────────────────────────────────
    print("\n[3/6] Isolation Forest detection...")
    if_detector = IsolationForestDetector(contamination=0.05, n_estimators=200)
    if_results  = if_detector.run_full_pipeline(train_daily, test_daily)
    if_metrics  = if_results["metrics"]

    # ─────────────────────────────────────────────
    # STEP 4: ARIMA
    # ─────────────────────────────────────────────
    print("\n[4/6] ARIMA detection...")
    arima_detector = ARIMADetector(
        order=(2, 1, 2),
        seasonal_order=(1, 0, 1, 7),
        sigma_threshold=2.5,
    )
    arima_results = arima_detector.run_full_pipeline(train_daily, test_daily)
    arima_metrics = arima_results["metrics"]
    arima_preds   = arima_results["predictions"]

    # ─────────────────────────────────────────────
    # STEP 5: Ensemble
    # ─────────────────────────────────────────────
    print("\n[5/6] Ensemble combination...")
    ensemble = EnsembleDetector(
        if_weight=0.45, arima_weight=0.55,
        score_threshold=0.60, require_both=True,
    )

    # Merge IF and ARIMA predictions on test set
    if_preds = if_results["predictions"].copy()
    # Align columns: add IF predictions to ARIMA predictions dataframe
    arima_preds_indexed = arima_preds.set_index(["date","provider"])
    if_preds_indexed    = if_preds.set_index(["date","provider"])

    # Use ARIMA df as base, add IF columns
    combined = arima_preds_indexed.copy()
    for col in ["if_prediction", "if_score", "if_confidence"]:
        if col in if_preds_indexed.columns:
            combined[col] = if_preds_indexed[col]
        else:
            combined[col] = 0

    combined = combined.reset_index()
    combined  = ensemble.combine(combined)
    ens_metrics = ensemble.evaluate(combined)
    ensemble.save_results()

    # Model comparison
    print("\n  Model Comparison:")
    comparison = ensemble.compare_models(if_metrics, arima_metrics, ens_metrics)
    comparison.to_csv("reports/model_comparison.csv")

    # ─────────────────────────────────────────────
    # STEP 6: Alerting
    # ─────────────────────────────────────────────
    print("\n[6/6] Alert engine...")
    alert_engine = AlertEngine(
        cooldown_hours=0,   # no suppression in demo run
        min_severity="low",
    )
    alerts = alert_engine.run(combined)
    alert_summary = alert_engine.generate_alert_summary()

    # ─────────────────────────────────────────────
    # Save full results package
    # ─────────────────────────────────────────────
    combined.to_csv("reports/full_predictions.csv", index=False)
    raw_df.to_csv("reports/raw_with_ground_truth.csv", index=False)

    pipeline_results = {
        "run_timestamp": datetime.now().isoformat(),
        "data_summary":  prep_summary,
        "if_metrics":    if_metrics,
        "arima_metrics": arima_metrics,
        "ensemble_metrics": ens_metrics,
        "alert_summary": alert_summary,
        "total_days_analyzed": len(test_daily),
        "anomaly_days_detected": int(combined["ensemble_prediction"].sum()),
    }

    with open("reports/pipeline_results.json", "w") as f:
        json.dump(pipeline_results, f, indent=2, default=str)

    print("\n" + "█" * 60)
    print("  PIPELINE COMPLETE")
    print(f"  Results saved to reports/")
    print("█" * 60)
    print(f"\n  Ensemble Performance:")
    print(f"    F1 Score  : {ens_metrics.get('f1_score', 0):.4f}")
    print(f"    Precision : {ens_metrics.get('precision', 0):.4f}")
    print(f"    Recall    : {ens_metrics.get('recall', 0):.4f}")
    print(f"    AUC-ROC   : {ens_metrics.get('auc_roc', 0):.4f}")

    return pipeline_results


if __name__ == "__main__":
    results = run_pipeline(
        start_date="2023-01-01",
        end_date="2024-06-30",
    )
