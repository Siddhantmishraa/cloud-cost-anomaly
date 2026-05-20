"""
ARIMA-based Anomaly Detector for Cloud Billing Time-Series
Forecasts expected spending and flags deviations outside confidence intervals.
Uses auto_arima for automatic order selection (p, d, q).
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
import warnings
import json
import os
warnings.filterwarnings("ignore")


class ARIMADetector:
    """
    ARIMA-based anomaly detector.
    For each cloud provider, fits an ARIMA model on historical daily costs,
    forecasts forward, and flags days where actual > forecast ± k*sigma.
    """

    def __init__(self,
                 order: tuple = (2, 1, 2),
                 seasonal_order: tuple = (1, 0, 1, 7),
                 confidence_level: float = 0.95,
                 sigma_threshold: float = 2.5,
                 output_dir: str = "models"):
        self.order            = order
        self.seasonal_order   = seasonal_order
        self.confidence_level = confidence_level
        self.sigma_threshold  = sigma_threshold
        self.output_dir       = output_dir
        self.fitted_models    = {}
        self.forecasts        = {}
        self.results          = {}
        os.makedirs(output_dir, exist_ok=True)

    def check_stationarity(self, series: pd.Series) -> dict:
        """ADF test for stationarity."""
        result = adfuller(series.dropna(), autolag="AIC")
        return {
            "adf_statistic": round(result[0], 4),
            "p_value":       round(result[1], 4),
            "is_stationary": result[1] < 0.05,
        }

    def fit_provider(self, provider: str, train_series: pd.Series) -> dict:
        """Fit ARIMA model for a single provider's daily cost series."""
        stationarity = self.check_stationarity(train_series)

        # Auto-select differencing order
        d = 0 if stationarity["is_stationary"] else 1

        try:
            # Try SARIMA first (captures weekly seasonality)
            model = ARIMA(
                train_series,
                order=(self.order[0], d, self.order[2]),
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted = model.fit(method_kwargs={"warn_convergence": False})
            model_type = "SARIMA"
        except Exception:
            # Fallback to simple ARIMA
            try:
                model = ARIMA(train_series, order=(2, d, 2))
                fitted = model.fit()
                model_type = "ARIMA"
            except Exception as e:
                print(f"    Warning: ARIMA failed for {provider}: {e}")
                return None

        aic  = round(fitted.aic, 2)
        rmse = round(np.sqrt(np.mean(fitted.resid**2)), 2)
        print(f"    {provider}: {model_type}({self.order[0]},{d},{self.order[2]}) "
              f"| AIC={aic} | Train RMSE={rmse}")

        return {
            "fitted": fitted,
            "model_type": model_type,
            "aic": aic,
            "train_rmse": rmse,
            "stationarity": stationarity,
        }

    def train(self, train_df: pd.DataFrame):
        """Fit ARIMA models for all providers."""
        print("\n  Fitting ARIMA models...")
        providers = train_df["provider"].unique()

        for provider in providers:
            prov_data = train_df[train_df["provider"] == provider].copy()
            prov_data = prov_data.sort_values("date").set_index("date")
            series = prov_data["total_cost"].asfreq("D").ffill()

            result = self.fit_provider(provider, series)
            if result:
                self.fitted_models[provider] = result

    def forecast_and_detect(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate forecasts for test period and detect anomalies via
        confidence interval violations.
        """
        all_predictions = []

        for provider, artefact in self.fitted_models.items():
            prov_test = test_df[test_df["provider"] == provider].copy()
            prov_test = prov_test.sort_values("date")

            if len(prov_test) == 0:
                continue

            fitted  = artefact["fitted"]
            n_steps = len(prov_test)

            try:
                # Rolling one-step-ahead forecast
                forecast_obj = fitted.get_forecast(steps=n_steps)
                forecast_mean = forecast_obj.predicted_mean.values
                forecast_ci   = forecast_obj.conf_int(alpha=1 - self.confidence_level)
                upper_ci      = forecast_ci.iloc[:, 1].values
                lower_ci      = forecast_ci.iloc[:, 0].values
            except Exception:
                # Fallback: use residual std for CI
                resid_std    = np.std(fitted.resid)
                last_val     = fitted.fittedvalues.iloc[-1]
                forecast_mean = np.full(n_steps, last_val)
                upper_ci     = forecast_mean + self.sigma_threshold * resid_std
                lower_ci     = forecast_mean - self.sigma_threshold * resid_std

            actuals   = prov_test["total_cost"].values
            residuals = actuals - forecast_mean[:len(actuals)]
            resid_std = np.std(residuals) + 1e-9

            # Anomaly: actual exceeds upper CI OR deviates > sigma_threshold
            arima_pred = (
                (actuals > upper_ci[:len(actuals)]) |
                (np.abs(residuals) > self.sigma_threshold * resid_std)
            ).astype(int)

            # Anomaly score: normalized residual magnitude
            arima_score = np.abs(residuals) / (np.max(np.abs(residuals)) + 1e-9)

            prov_result = prov_test.copy()
            prov_result["arima_forecast"]  = forecast_mean[:len(actuals)]
            prov_result["arima_upper_ci"]  = upper_ci[:len(actuals)]
            prov_result["arima_lower_ci"]  = lower_ci[:len(actuals)]
            prov_result["arima_residual"]  = residuals
            prov_result["arima_prediction"] = arima_pred
            prov_result["arima_score"]     = arima_score

            all_predictions.append(prov_result)

        if not all_predictions:
            return test_df

        return pd.concat(all_predictions, ignore_index=True)

    def evaluate(self, predicted_df: pd.DataFrame) -> dict:
        """Evaluate ARIMA anomaly detection against ground truth."""
        y_true  = predicted_df["has_anomaly"].astype(int).values
        y_pred  = predicted_df["arima_prediction"].values
        y_score = predicted_df["arima_score"].values

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        metrics = {
            "precision":     round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall":        round(recall_score(y_true, y_pred, zero_division=0), 4),
            "f1_score":      round(f1_score(y_true, y_pred, zero_division=0), 4),
            "auc_roc":       round(roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.0, 4),
            "true_positives":  int(tp),
            "false_positives": int(fp),
            "true_negatives":  int(tn),
            "false_negatives": int(fn),
            "false_positive_rate": round(fp / (fp + tn + 1e-9), 4),
            "false_negative_rate": round(fn / (fn + tp + 1e-9), 4),
            "total_anomalies_detected": int(y_pred.sum()),
            "total_actual_anomalies":   int(y_true.sum()),
            "confidence_level":  self.confidence_level,
            "sigma_threshold":   self.sigma_threshold,
            "arima_order":       str(self.order),
            "seasonal_order":    str(self.seasonal_order),
        }

        # Per-provider AIC
        for p, a in self.fitted_models.items():
            metrics[f"{p}_aic"] = a.get("aic", "N/A")

        self.results = metrics
        return metrics

    def save_metrics(self, name: str = "arima"):
        path = os.path.join(self.output_dir, f"{name}_metrics.json")
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"  ARIMA metrics saved to {path}")

    def run_full_pipeline(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
        print("\n" + "=" * 60)
        print("  ARIMA Detection Pipeline")
        print("=" * 60)

        self.train(train_df)
        predicted = self.forecast_and_detect(test_df)
        metrics   = self.evaluate(predicted)
        self.save_metrics()

        print(f"\n  Results:")
        print(f"    Precision : {metrics['precision']:.4f}")
        print(f"    Recall    : {metrics['recall']:.4f}")
        print(f"    F1 Score  : {metrics['f1_score']:.4f}")
        print(f"    AUC-ROC   : {metrics['auc_roc']:.4f}")
        print(f"    TP={metrics['true_positives']}  FP={metrics['false_positives']}  "
              f"FN={metrics['false_negatives']}  TN={metrics['true_negatives']}")

        return {"predictions": predicted, "metrics": metrics}


if __name__ == "__main__":
    daily = pd.read_csv("data/processed/daily_aggregated.csv", parse_dates=["date"])
    daily = daily.sort_values(["provider","date"])
    split = int(len(daily) * 0.8)
    train = daily.iloc[:split]
    test  = daily.iloc[split:]

    detector = ARIMADetector(sigma_threshold=2.5)
    results  = detector.run_full_pipeline(train, test)
