# ☁️ AI-Driven Cloud Cost Anomaly Detection System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikitlearn)
![AWS](https://img.shields.io/badge/AWS-Billing%20API-FF9900?logo=amazonaws)

> An AI-powered system that monitors AWS billing patterns, detects spending spikes using Isolation Forest + ARIMA ensemble models, and dispatches automated severity-tiered alerts.

## Results

Evaluated with rolling one-step-ahead forecasts and training-set-only score
normalization — no test-set statistics leak into predictions or scores.

| Model | Precision | Recall | F1 Score | AUC-ROC | False Pos |
|-------|-----------|--------|----------|---------|-----------|
| Isolation Forest | 0.2973 | 0.8462 | 0.4400 | 0.9554 | 26 |
| ARIMA | 1.0000 | 0.6154 | 0.7619 | 0.8963 | 0 |
| **Ensemble** | **1.0000** | **0.6923** | **0.8182** | **0.9396** | **0** |

## Quick Start

```bash
git clone https://github.com/Siddhantmishraa/cloud-cost-anomaly.git
cd cloud-cost-anomaly-detection
pip install -r requirements.txt
python main.py                          # Run full pipeline
python dashboard/visualize.py           # Generate all charts
python src/reporting/report_generator.py  # Generate PDF report
```

## Bring Your Own Export

Drop a billing CSV from your cloud console straight into the dashboard — no
API keys needed. The app auto-detects the format (AWS Cost & Usage Report,
AWS Cost Explorer CSV, GCP billing export, Azure cost analysis, or any
generic CSV with date/service/cost columns), trains on your history, and
flags anomalous days in the most recent window with per-service root-cause
attribution. Requires 60+ days of history. Data is analyzed in-memory and
never stored.

## Structure

```
src/ingestion/data_simulator.py    # AWS billing data simulator
src/ingestion/adapters.py          # BYOE: billing export format adapters
src/byoe_pipeline.py               # BYOE: label-free analysis pipeline
src/preprocessing/pipeline.py      # Feature engineering (24 features)
src/detection/isolation_forest.py  # Isolation Forest model
src/detection/arima_detector.py    # ARIMA/SARIMA model  
src/detection/ensemble.py          # Weighted ensemble combiner
src/alerting/alert_engine.py       # Slack/Email/PagerDuty alerts
src/alerting/root_cause.py         # Per-service root-cause attribution
src/reporting/report_generator.py  # PDF monthly report
dashboard/visualize.py             # 7 charts + interactive HTML
notebooks/cloud_anomaly_detection.ipynb  # Full walkthrough
```

## Alert Severity Tiers

| Severity | Score | Action | SLA |
|---------|-------|--------|-----|
| CRITICAL | >= 0.90 | PagerDuty + Slack + Email | < 15 min |
| HIGH | >= 0.75 | Slack + Email | < 1 hour |
| MEDIUM | >= 0.60 | Slack | < 4 hours |
| LOW | >= 0.40 | Dashboard only | Next business day |

## Traffic Surge Simulation

Controlled 5.2x compute spike injected June 16-18, 2024 (decaying 5.2x → 4.1x → 2.3x).
Detection result: **8 of 9 provider-days flagged**, with per-service root-cause
attribution (e.g. "EC2 +$3,089, 77% of excess"). The single miss is the day-3
decayed 2.3x tail on GCP, which scored just below the alert threshold.
