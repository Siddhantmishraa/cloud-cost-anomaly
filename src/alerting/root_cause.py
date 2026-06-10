"""
Root Cause Analyzer for Cloud Cost Anomalies
When a provider-day is flagged, attributes the spike to the specific
services (and regions) that drove it, by comparing each service's cost
that day against its own trailing baseline.
"""

import numpy as np
import pandas as pd


class RootCauseAnalyzer:
    """
    Built once from service-level billing data (date, provider, service,
    region, cost). For any flagged (provider, date), `explain()` returns the
    services ranked by excess dollars over their trailing-window baseline.

    The baseline for day D uses only days before D (shifted rolling window),
    so attribution never peeks at the day being explained.
    """

    def __init__(self, service_df: pd.DataFrame,
                 baseline_window: int = 30,
                 top_n: int = 3):
        self.top_n = top_n

        df = service_df[["date", "provider", "service", "region", "cost"]].copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["provider", "service", "date"]).reset_index(drop=True)

        grouped = df.groupby(["provider", "service"])["cost"]
        df["baseline_mean"] = grouped.transform(
            lambda x: x.shift(1).rolling(baseline_window, min_periods=7).mean()
        )
        df["baseline_std"] = grouped.transform(
            lambda x: x.shift(1).rolling(baseline_window, min_periods=7).std()
        )
        self.df = df

    def explain(self, provider: str, date) -> list:
        """
        Return the top services driving cost above baseline for one
        (provider, date), ranked by excess dollars. Each entry:
          service, region, actual_cost, baseline_cost, excess_usd,
          pct_above_baseline, z_score, share_of_excess_pct
        """
        date = pd.Timestamp(str(date)[:10])
        day = self.df[(self.df["provider"] == provider) &
                      (self.df["date"] == date)].copy()
        day = day.dropna(subset=["baseline_mean"])
        if day.empty:
            return []

        day["excess_usd"] = day["cost"] - day["baseline_mean"]
        day["pct_above"]  = 100 * day["excess_usd"] / (day["baseline_mean"] + 1e-9)
        day["z_score"]    = day["excess_usd"] / (day["baseline_std"] + 1e-9)

        total_excess = day["excess_usd"].clip(lower=0).sum() + 1e-9
        drivers = (day[day["excess_usd"] > 0]
                   .sort_values("excess_usd", ascending=False)
                   .head(self.top_n))

        return [{
            "service":             r["service"],
            "region":              r["region"],
            "actual_cost":         round(float(r["cost"]), 2),
            "baseline_cost":       round(float(r["baseline_mean"]), 2),
            "excess_usd":          round(float(r["excess_usd"]), 2),
            "pct_above_baseline":  round(float(r["pct_above"]), 1),
            "z_score":             round(float(r["z_score"]), 1),
            "share_of_excess_pct": round(100 * float(r["excess_usd"]) / total_excess, 1),
        } for _, r in drivers.iterrows()]

    @staticmethod
    def format_drivers(drivers: list) -> str:
        """One-line human-readable summary for alert messages."""
        if not drivers:
            return "Root cause: no single service stands out above baseline."
        parts = [
            f"{d['service']} ({d['region']}) +${d['excess_usd']:,.0f} "
            f"[{d['pct_above_baseline']:+.0f}% vs baseline, "
            f"{d['share_of_excess_pct']:.0f}% of excess]"
            for d in drivers
        ]
        return "Top drivers: " + " · ".join(parts)


if __name__ == "__main__":
    raw = pd.read_csv("data/simulated/multicloud_billing.csv", parse_dates=["date"])
    rca = RootCauseAnalyzer(raw)

    # Explain a known injected anomaly day
    anomaly_days = raw[raw["is_anomaly"]][["date", "provider"]].drop_duplicates().tail(3)
    for _, row in anomaly_days.iterrows():
        drivers = rca.explain(row["provider"], row["date"])
        print(f"\n{row['provider']} {row['date'].date()}:")
        print(" ", RootCauseAnalyzer.format_drivers(drivers))
