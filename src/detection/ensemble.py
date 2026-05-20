"""
Ensemble Anomaly Detector
Combines Isolation Forest and ARIMA signals for higher precision anomaly detection.
Only fires alerts when both models agree (intersection strategy).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
import json
import os
import warnings
warnings.filterwarnings("ignore")


class EnsembleDetector:
    """
    Combines IF and ARIMA predictions using a weighted ensemble score.
    Strategy: alert when ensemble_score > threshold OR both models agree.
    """

    def __init__(self,
                 if_weight: float = 0.45,
                 arima_weight: float = 0.55,
                 score_threshold: float = 0.60,
                 require_both: bool = True,
                 output_dir: str = "models"):
        self.if_weight       = if_weight
        self.arima_weight    = arima_weight
        self.score_threshold = score_threshold
        self.require_both    = require_both
        self.output_dir      = output_dir
        self.results         = {}
        os.makedirs(output_dir, exist_ok=True)

    def combine(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Combine IF and ARIMA predictions into ensemble signal.
        Expects columns: if_prediction, if_score, arima_prediction, arima_score
        """
        df = df.copy()

        # Weighted ensemble score
        df["ensemble_score"] = (
            self.if_weight    * df.get("if_score",    pd.Series(0, index=df.index)) +
            self.arima_weight * df.get("arima_score", pd.Series(0, index=df.index))
        )

        # Both-agree strategy (lowest false positives)
        both_agree = (
            df.get("if_prediction",    pd.Series(0, index=df.index)) &
            df.get("arima_prediction", pd.Series(0, index=df.index))
        )

        # Score-threshold strategy
        score_flag = df["ensemble_score"] >= self.score_threshold

        if self.require_both:
            df["ensemble_prediction"] = (both_agree | score_flag).astype(int)
        else:
            df["ensemble_prediction"] = score_flag.astype(int)

        # Severity tiers
        df["alert_severity"] = "none"
        df.loc[df["ensemble_score"] >= 0.40, "alert_severity"] = "low"
        df.loc[df["ensemble_score"] >= 0.60, "alert_severity"] = "medium"
        df.loc[df["ensemble_score"] >= 0.75, "alert_severity"] = "high"
        df.loc[df["ensemble_score"] >= 0.90, "alert_severity"] = "critical"

        return df

    def evaluate(self, df: pd.DataFrame) -> dict:
        """Full evaluation of ensemble predictions."""
        if "has_anomaly" not in df.columns:
            return {}

        y_true  = df["has_anomaly"].astype(int).values
        y_pred  = df["ensemble_prediction"].values
        y_score = df["ensemble_score"].values

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        metrics = {
            "model": "Ensemble (IF + ARIMA)",
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
            "if_weight":       self.if_weight,
            "arima_weight":    self.arima_weight,
            "score_threshold": self.score_threshold,
            "strategy":        "both_agree + score_threshold" if self.require_both else "score_threshold",
        }
        self.results = metrics
        return metrics

    def compare_models(self, if_metrics: dict, arima_metrics: dict, ensemble_metrics: dict) -> pd.DataFrame:
        """Side-by-side comparison table of all three models."""
        rows = []
        for name, m in [("Isolation Forest", if_metrics),
                         ("ARIMA",            arima_metrics),
                         ("Ensemble",          ensemble_metrics)]:
            rows.append({
                "Model":         name,
                "Precision":     m.get("precision",     0),
                "Recall":        m.get("recall",        0),
                "F1 Score":      m.get("f1_score",      0),
                "AUC-ROC":       m.get("auc_roc",       0),
                "True Pos":      m.get("true_positives",  0),
                "False Pos":     m.get("false_positives", 0),
                "False Neg":     m.get("false_negatives", 0),
                "FP Rate":       m.get("false_positive_rate", 0),
                "FN Rate":       m.get("false_negative_rate", 0),
            })
        comparison = pd.DataFrame(rows).set_index("Model")

        path = os.path.join(self.output_dir, "model_comparison.csv")
        comparison.to_csv(path)
        print(f"\n  Model comparison saved to {path}")
        print("\n" + comparison.to_string())
        return comparison

    def save_results(self):
        path = os.path.join(self.output_dir, "ensemble_metrics.json")
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2)
