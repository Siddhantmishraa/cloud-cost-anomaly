"""
Multi-Cloud Billing Data Simulator
Generates realistic historical billing data for AWS, GCP, and Azure
with seasonal patterns, trends, and injected anomalies.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import json
import os

np.random.seed(42)
random.seed(42)


CLOUD_PROVIDERS = {
    "AWS": {
        "services": {
            "EC2":       {"base": 1200, "weight": 0.35, "seasonal": True},
            "S3":        {"base": 180,  "weight": 0.08, "seasonal": False},
            "RDS":       {"base": 450,  "weight": 0.15, "seasonal": True},
            "Lambda":    {"base": 80,   "weight": 0.04, "seasonal": False},
            "CloudFront":{"base": 120,  "weight": 0.05, "seasonal": True},
            "EKS":       {"base": 320,  "weight": 0.12, "seasonal": True},
            "Redshift":  {"base": 280,  "weight": 0.10, "seasonal": False},
            "ElastiCache":{"base": 150, "weight": 0.06, "seasonal": False},
            "SageMaker": {"base": 95,   "weight": 0.05, "seasonal": False},
        }
    },
    "GCP": {
        "services": {
            "Compute Engine": {"base": 950,  "weight": 0.32, "seasonal": True},
            "Cloud Storage":  {"base": 140,  "weight": 0.07, "seasonal": False},
            "BigQuery":       {"base": 380,  "weight": 0.16, "seasonal": False},
            "GKE":            {"base": 290,  "weight": 0.13, "seasonal": True},
            "Cloud SQL":      {"base": 220,  "weight": 0.10, "seasonal": True},
            "Cloud Run":      {"base": 75,   "weight": 0.05, "seasonal": False},
            "Vertex AI":      {"base": 185,  "weight": 0.09, "seasonal": False},
            "Pub/Sub":        {"base": 55,   "weight": 0.04, "seasonal": False},
            "Cloud CDN":      {"base": 90,   "weight": 0.04, "seasonal": True},
        }
    },
    "Azure": {
        "services": {
            "Virtual Machines":  {"base": 1050, "weight": 0.33, "seasonal": True},
            "Blob Storage":      {"base": 160,  "weight": 0.07, "seasonal": False},
            "Azure SQL":         {"base": 400,  "weight": 0.14, "seasonal": True},
            "AKS":               {"base": 300,  "weight": 0.12, "seasonal": True},
            "Azure Functions":   {"base": 65,   "weight": 0.04, "seasonal": False},
            "Cosmos DB":         {"base": 250,  "weight": 0.10, "seasonal": False},
            "Azure Databricks":  {"base": 210,  "weight": 0.09, "seasonal": False},
            "App Service":       {"base": 130,  "weight": 0.06, "seasonal": True},
            "Azure AI Services": {"base": 85,   "weight": 0.05, "seasonal": False},
        }
    }
}

ANOMALY_TYPES = [
    {"name": "traffic_surge",        "multiplier": (3.5, 7.0),  "duration_days": (1, 3),  "services": ["EC2","Compute Engine","Virtual Machines","EKS","GKE","AKS"]},
    {"name": "misconfiguration",     "multiplier": (4.0, 12.0), "duration_days": (1, 2),  "services": ["S3","Cloud Storage","Blob Storage","RDS","Cloud SQL","Azure SQL"]},
    {"name": "crypto_mining_attack", "multiplier": (6.0, 15.0), "duration_days": (1, 1),  "services": ["EC2","Compute Engine","Virtual Machines"]},
    {"name": "runaway_job",          "multiplier": (5.0, 10.0), "duration_days": (2, 5),  "services": ["Redshift","BigQuery","Azure Databricks","SageMaker","Vertex AI"]},
    {"name": "ddos_amplification",   "multiplier": (3.0, 8.0),  "duration_days": (1, 2),  "services": ["CloudFront","Cloud CDN","Azure Functions","Lambda"]},
]


def generate_base_cost(service_config: dict, date: datetime, provider: str) -> float:
    """Generate daily base cost with trend, seasonality, and noise."""
    base = service_config["base"]

    # Long-term upward trend (5% monthly growth)
    days_from_start = (date - datetime(2023, 1, 1)).days
    trend = 1 + (days_from_start / 365) * 0.05

    # Weekly seasonality (lower on weekends)
    dow = date.weekday()
    weekly_factor = 0.70 if dow >= 5 else (1.0 + 0.08 * (dow / 4))

    # Monthly seasonality (end-of-month spikes for reporting)
    day_of_month = date.day
    monthly_factor = 1.0
    if service_config.get("seasonal"):
        if day_of_month >= 28:
            monthly_factor = 1.15
        elif day_of_month <= 3:
            monthly_factor = 0.88

    # Annual seasonality (Q4 higher, Q1/Q3 lower)
    month = date.month
    annual_factors = {1:0.88, 2:0.90, 3:0.95, 4:0.98, 5:1.00,
                      6:1.02, 7:0.97, 8:0.96, 9:1.01, 10:1.06,
                      11:1.10, 12:1.15}
    annual_factor = annual_factors.get(month, 1.0) if service_config.get("seasonal") else 1.0

    # Gaussian noise (5% std dev)
    noise = np.random.normal(1.0, 0.05)

    cost = base * trend * weekly_factor * monthly_factor * annual_factor * noise
    return max(cost * 0.1, cost)  # floor at 10% of base


def inject_anomalies(df: pd.DataFrame, num_anomalies: int = 35) -> pd.DataFrame:
    """Inject realistic anomaly events into the billing data."""
    df = df.copy()
    df["is_anomaly"] = False
    df["anomaly_type"] = None
    df["anomaly_severity"] = None

    date_range = df["date"].unique()
    # Spread anomalies: at least 5 days apart
    anomaly_dates = sorted(np.random.choice(date_range[10:-10], size=num_anomalies, replace=False))

    for i, anom_date in enumerate(anomaly_dates):
        anom_type = random.choice(ANOMALY_TYPES)
        multiplier = random.uniform(*anom_type["multiplier"])
        duration = random.randint(*anom_type["duration_days"])
        severity = "critical" if multiplier > 8 else ("high" if multiplier > 4 else "medium")

        # Pick affected provider & service
        provider = random.choice(list(CLOUD_PROVIDERS.keys()))
        eligible = [s for s in anom_type["services"]
                    if s in CLOUD_PROVIDERS[provider]["services"]]
        if not eligible:
            eligible = list(CLOUD_PROVIDERS[provider]["services"].keys())[:2]
        service = random.choice(eligible)

        # Apply spike across duration
        for d in range(duration):
            target_date = pd.Timestamp(anom_date) + timedelta(days=d)
            mask = (df["date"] == target_date) & \
                   (df["provider"] == provider) & \
                   (df["service"] == service)
            if mask.any():
                # Gradual decay for multi-day anomalies
                decay = 1.0 / (1 + d * 0.4)
                df.loc[mask, "cost"] = df.loc[mask, "cost"] * multiplier * decay
                df.loc[mask, "is_anomaly"] = True
                df.loc[mask, "anomaly_type"] = anom_type["name"]
                df.loc[mask, "anomaly_severity"] = severity

    return df


def simulate_traffic_surge(df: pd.DataFrame, surge_date: str = None) -> pd.DataFrame:
    """Simulate a specific, controlled traffic surge event for demo/testing."""
    df = df.copy()
    if surge_date is None:
        # Default: 2 weeks before end of dataset
        last_date = df["date"].max()
        surge_date = last_date - timedelta(days=14)

    surge_date = pd.Timestamp(surge_date)
    surge_duration = 3  # days

    print(f"  Injecting traffic surge starting {surge_date.date()} for {surge_duration} days")

    for d in range(surge_duration):
        target = surge_date + timedelta(days=d)
        multiplier = [5.2, 4.1, 2.3][d]  # spike then decay

        # Hit compute services across all clouds
        compute_services = ["EC2", "Compute Engine", "Virtual Machines", "EKS", "GKE", "AKS"]
        mask = (df["date"] == target) & (df["service"].isin(compute_services))
        df.loc[mask, "cost"] *= multiplier
        df.loc[mask, "is_anomaly"] = True
        df.loc[mask, "anomaly_type"] = "simulated_traffic_surge"
        df.loc[mask, "anomaly_severity"] = "critical"

    return df


def generate_full_dataset(
    start_date: str = "2023-01-01",
    end_date: str = "2024-06-30",
    output_dir: str = "data/simulated"
) -> pd.DataFrame:
    """Generate the complete multi-cloud billing dataset."""
    print("=" * 60)
    print("  Multi-Cloud Billing Data Simulator")
    print("=" * 60)

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end   = datetime.strptime(end_date,   "%Y-%m-%d")
    dates = pd.date_range(start, end, freq="D")

    print(f"  Date range : {start_date} → {end_date} ({len(dates)} days)")
    print(f"  Providers  : {', '.join(CLOUD_PROVIDERS.keys())}")

    records = []
    for provider, config in CLOUD_PROVIDERS.items():
        for service, svc_config in config["services"].items():
            for date in dates:
                cost = generate_base_cost(svc_config, date, provider)
                records.append({
                    "date":       date,
                    "provider":   provider,
                    "service":    service,
                    "region":     _pick_region(provider),
                    "cost":       round(cost, 4),
                    "currency":   "USD",
                    "is_anomaly": False,
                    "anomaly_type": None,
                    "anomaly_severity": None,
                })

    df = pd.DataFrame(records)
    total_services = sum(len(c["services"]) for c in CLOUD_PROVIDERS.values())
    print(f"  Records    : {len(df):,} ({total_services} services × {len(dates)} days)")

    # Inject random anomalies
    print("\n  Injecting anomalies...")
    df = inject_anomalies(df, num_anomalies=40)
    anom_count = df["is_anomaly"].sum()
    print(f"  Anomalies injected: {anom_count} records")

    # Inject controlled traffic surge
    df = simulate_traffic_surge(df)

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    main_path = os.path.join(output_dir, "multicloud_billing.csv")
    df.to_csv(main_path, index=False)

    # Also save per-provider
    for provider in CLOUD_PROVIDERS:
        prov_df = df[df["provider"] == provider]
        prov_df.to_csv(os.path.join(output_dir, f"{provider.lower()}_billing.csv"), index=False)

    # Daily aggregated view
    daily = df.groupby(["date","provider"]).agg(
        total_cost=("cost","sum"),
        service_count=("service","nunique"),
        has_anomaly=("is_anomaly","any")
    ).reset_index()
    daily.to_csv(os.path.join(output_dir, "daily_aggregated.csv"), index=False)

    # Save metadata
    meta = {
        "generated_at": datetime.now().isoformat(),
        "date_range": {"start": start_date, "end": end_date, "days": len(dates)},
        "providers": list(CLOUD_PROVIDERS.keys()),
        "total_records": len(df),
        "total_anomalies": int(df["is_anomaly"].sum()),
        "anomaly_rate_pct": round(df["is_anomaly"].mean() * 100, 2),
        "total_cost_usd": round(df["cost"].sum(), 2),
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Dataset saved to: {output_dir}/")
    print(f"  Total cost simulated: ${meta['total_cost_usd']:,.2f}")
    print(f"  Anomaly rate: {meta['anomaly_rate_pct']}%")
    print("=" * 60)

    return df


def _pick_region(provider: str) -> str:
    regions = {
        "AWS":   ["us-east-1","us-west-2","eu-west-1","ap-southeast-1"],
        "GCP":   ["us-central1","us-east1","europe-west1","asia-southeast1"],
        "Azure": ["East US","West US 2","West Europe","Southeast Asia"],
    }
    return random.choice(regions.get(provider, ["global"]))


if __name__ == "__main__":
    df = generate_full_dataset(
        start_date="2023-01-01",
        end_date="2024-06-30",
        output_dir="data/simulated"
    )
    print("\nSample records:")
    print(df[df["is_anomaly"]].head(5)[["date","provider","service","cost","anomaly_type","anomaly_severity"]])
