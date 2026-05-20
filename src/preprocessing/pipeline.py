"""
Preprocessing Pipeline for Cloud Billing Time-Series Data
Handles cleaning, feature engineering, normalization, and train/test splitting.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats
import joblib
import os
import warnings
warnings.filterwarnings("ignore")


class BillingPreprocessor:
    """
    Complete preprocessing pipeline for multi-cloud billing data.
    Prepares features for both Isolation Forest and ARIMA models.
    """

    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = output_dir
        self.scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        os.makedirs(output_dir, exist_ok=True)

    def load_raw_data(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path, parse_dates=["date"])
        df = df.sort_values(["provider", "service", "date"]).reset_index(drop=True)
        print(f"  Loaded {len(df):,} records from {path}")
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicates, handle missing values, cap extreme outliers."""
        initial = len(df)

        # Drop exact duplicates
        df = df.drop_duplicates(subset=["date","provider","service"])

        # Fill missing costs with forward-fill then median
        df["cost"] = df.groupby(["provider","service"])["cost"].transform(
            lambda x: x.ffill().fillna(x.median())
        )

        # Floor negative costs
        df["cost"] = df["cost"].clip(lower=0)

        print(f"  Cleaned: {initial:,} → {len(df):,} records "
              f"({initial - len(df)} duplicates removed)")
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create rich time-series features for anomaly detection.
        Features are computed per (provider, service) group.
        """
        print("  Engineering features...")
        df = df.copy()

        # Calendar features
        df["day_of_week"]  = df["date"].dt.dayofweek
        df["day_of_month"] = df["date"].dt.day
        df["month"]        = df["date"].dt.month
        df["quarter"]      = df["date"].dt.quarter
        df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
        df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

        # Rolling statistics per group (key anomaly signals)
        for window in [7, 14, 30]:
            col = f"rolling_mean_{window}d"
            std_col = f"rolling_std_{window}d"
            df[col] = df.groupby(["provider","service"])["cost"].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            df[std_col] = df.groupby(["provider","service"])["cost"].transform(
                lambda x: x.rolling(window, min_periods=1).std().fillna(0)
            )

        # Cost deviation from rolling mean (z-score style)
        df["deviation_7d"]  = (df["cost"] - df["rolling_mean_7d"]) / (df["rolling_std_7d"] + 1e-9)
        df["deviation_30d"] = (df["cost"] - df["rolling_mean_30d"]) / (df["rolling_std_30d"] + 1e-9)

        # Day-over-day and week-over-week growth rates
        df["cost_lag_1d"] = df.groupby(["provider","service"])["cost"].shift(1)
        df["cost_lag_7d"] = df.groupby(["provider","service"])["cost"].shift(7)
        df["growth_1d"]   = (df["cost"] - df["cost_lag_1d"]) / (df["cost_lag_1d"] + 1e-9)
        df["growth_7d"]   = (df["cost"] - df["cost_lag_7d"]) / (df["cost_lag_7d"] + 1e-9)

        # Cumulative monthly cost (budget tracking)
        df["month_key"] = df["date"].dt.to_period("M")
        df["cumulative_monthly_cost"] = df.groupby(
            ["provider","service","month_key"]
        )["cost"].cumsum()

        # Percentile rank within each service (0=cheapest day, 1=most expensive)
        df["cost_percentile"] = df.groupby(["provider","service"])["cost"].transform(
            lambda x: x.rank(pct=True)
        )

        # Provider-level daily total (cross-service context)
        daily_provider = df.groupby(["date","provider"])["cost"].sum().rename("provider_daily_total")
        df = df.merge(daily_provider, on=["date","provider"], how="left")
        df["service_share"] = df["cost"] / (df["provider_daily_total"] + 1e-9)

        print(f"  Features created: {len([c for c in df.columns if c not in ['date','provider','service','region','currency','is_anomaly','anomaly_type','anomaly_severity']])} feature columns")
        return df

    def create_daily_aggregates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate to provider-level daily totals for time-series modeling."""
        daily = df.groupby(["date","provider"]).agg(
            total_cost          = ("cost",           "sum"),
            service_count       = ("service",         "nunique"),
            max_service_cost    = ("cost",            "max"),
            cost_std            = ("cost",            "std"),
            has_anomaly         = ("is_anomaly",      "any"),
            anomaly_count       = ("is_anomaly",      "sum"),
        ).reset_index()

        # Rolling features on aggregated data
        for window in [7, 14, 30]:
            daily[f"rolling_mean_{window}d"] = daily.groupby("provider")["total_cost"].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            daily[f"rolling_std_{window}d"] = daily.groupby("provider")["total_cost"].transform(
                lambda x: x.rolling(window, min_periods=1).std().fillna(0)
            )

        daily["upper_bound_2sigma"] = daily["rolling_mean_7d"] + 2 * daily["rolling_std_7d"]
        daily["lower_bound_2sigma"] = daily["rolling_mean_7d"] - 2 * daily["rolling_std_7d"]
        daily["z_score"]            = (daily["total_cost"] - daily["rolling_mean_7d"]) / (daily["rolling_std_7d"] + 1e-9)

        return daily

    def train_test_split_temporal(
        self,
        df: pd.DataFrame,
        test_ratio: float = 0.20,
        val_ratio:  float = 0.10
    ) -> tuple:
        """
        Time-series aware split: train | validation | test (chronological order).
        Never shuffles — avoids data leakage.
        """
        dates = sorted(df["date"].unique())
        n = len(dates)
        train_end = dates[int(n * (1 - test_ratio - val_ratio))]
        val_end   = dates[int(n * (1 - test_ratio))]

        train = df[df["date"] <= train_end]
        val   = df[(df["date"] > train_end) & (df["date"] <= val_end)]
        test  = df[df["date"] > val_end]

        print(f"  Train: {len(train):,} records (up to {train_end})")
        print(f"  Val  : {len(val):,} records ({train_end} → {val_end})")
        print(f"  Test : {len(test):,} records (after {val_end})")
        return train, val, test

    def get_feature_matrix(self, df: pd.DataFrame) -> tuple:
        """Extract the feature matrix X and labels y for Isolation Forest."""
        feature_cols = [
            "cost", "deviation_7d", "deviation_30d",
            "growth_1d", "growth_7d", "cost_percentile",
            "service_share", "day_of_week", "is_weekend",
            "day_of_month", "month", "rolling_mean_7d",
            "rolling_std_7d", "rolling_mean_30d",
        ]
        available = [c for c in feature_cols if c in df.columns]
        X = df[available].fillna(0).values
        y = df["is_anomaly"].astype(int).values if "is_anomaly" in df.columns else None
        return X, y, available

    def normalize(self, X_train, X_test):
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled  = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def save_artifacts(self, train, val, test, daily):
        """Persist processed datasets."""
        train.to_csv(f"{self.output_dir}/train.csv", index=False)
        val.to_csv(f"{self.output_dir}/val.csv",   index=False)
        test.to_csv(f"{self.output_dir}/test.csv",  index=False)
        daily.to_csv(f"{self.output_dir}/daily_aggregated.csv", index=False)
        joblib.dump(self.scaler, f"{self.output_dir}/scaler.pkl")
        print(f"  Artifacts saved to {self.output_dir}/")

    def run(self, raw_path: str) -> dict:
        """Full preprocessing pipeline."""
        print("\n" + "=" * 60)
        print("  Preprocessing Pipeline")
        print("=" * 60)

        df    = self.load_raw_data(raw_path)
        df    = self.clean_data(df)
        df    = self.engineer_features(df)
        daily = self.create_daily_aggregates(df)

        print("\n  Temporal split:")
        train, val, test = self.train_test_split_temporal(df)

        self.save_artifacts(train, val, test, daily)

        summary = {
            "total_records": len(df),
            "feature_count": len(self.get_feature_matrix(df)[2]),
            "train_size": len(train),
            "val_size":   len(val),
            "test_size":  len(test),
            "anomaly_rate": round(df["is_anomaly"].mean() * 100, 2),
            "date_range":  f"{df['date'].min().date()} → {df['date'].max().date()}",
        }
        print(f"\n  Done. {summary['feature_count']} features, "
              f"{summary['anomaly_rate']}% anomaly rate")
        return summary


if __name__ == "__main__":
    preprocessor = BillingPreprocessor(output_dir="data/processed")
    summary = preprocessor.run("data/simulated/multicloud_billing.csv")
    print("\nSummary:", summary)
