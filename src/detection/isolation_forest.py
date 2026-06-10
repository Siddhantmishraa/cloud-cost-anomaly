"""
Isolation Forest Anomaly Detector for Cloud Billing Data
Unsupervised ML model that isolates anomalies by building random isolation trees.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, precision_score, recall_score, f1_score
)
import joblib
import json
import os
import warnings
warnings.filterwarnings("ignore")


class IsolationForestDetector:
    """
    Isolation Forest based cloud cost anomaly detector.
    Works per (provider, service) pair for precision, and also on daily aggregates.
    """

    def __init__(self, contamination: float = 0.05, n_estimators: int = 200,
                 random_state: int = 42, output_dir: str = "models"):
        self.contamination  = contamination
        self.n_estimators   = n_estimators
        self.random_state   = random_state
        self.output_dir     = output_dir
        self.models         = {}     # one model per provider
        self.results        = {}
        os.makedirs(output_dir, exist_ok=True)

    def _get_feature_cols(self, df: pd.DataFrame) -> list:
        candidates = [
            "total_cost", "cost_std", "max_service_cost",
            "rolling_mean_7d", "rolling_std_7d",
            "rolling_mean_14d", "rolling_std_14d",
            "rolling_mean_30d", "rolling_std_30d",
            "z_score",
        ]
        return [c for c in candidates if c in df.columns]

    def train(self, train_df: pd.DataFrame):
        """Train one Isolation Forest model per cloud provider."""
        print("\n  Training Isolation Forest models...")
        providers = train_df["provider"].unique()

        for provider in providers:
            prov_data = train_df[train_df["provider"] == provider]
            feat_cols = self._get_feature_cols(prov_data)
            X = prov_data[feat_cols].fillna(0).values

            model = IsolationForest(
                contamination = self.contamination,
                n_estimators  = self.n_estimators,
                max_features  = min(len(feat_cols), 8),
                random_state  = self.random_state,
                n_jobs        = -1,
            )
            model.fit(X)
            # Score range from TRAINING data only, so test-set scores don't
            # depend on other test-set values (no leakage)
            train_scores = model.score_samples(X)
            self.models[provider] = {
                "model": model,
                "feature_cols": feat_cols,
                "score_min": float(train_scores.min()),
                "score_max": float(train_scores.max()),
            }
            print(f"    {provider}: trained on {len(prov_data):,} records "
                  f"({len(feat_cols)} features)")

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict anomalies. Returns df with added columns:
          - if_prediction  : 1=anomaly, 0=normal
          - if_score       : anomaly score (higher = more anomalous)
          - if_confidence  : 0-1 confidence
        """
        df = df.copy()
        df["if_prediction"] = 0
        df["if_score"]      = 0.0

        for provider, artefact in self.models.items():
            mask = df["provider"] == provider
            prov_data = df[mask]
            if len(prov_data) == 0:
                continue
            feat_cols = artefact["feature_cols"]
            X = prov_data[feat_cols].fillna(0).values

            # Isolation Forest: -1=anomaly, 1=normal
            raw_pred  = artefact["model"].predict(X)
            raw_score = artefact["model"].score_samples(X)  # more negative = more anomalous

            df.loc[mask, "if_prediction"] = (raw_pred == -1).astype(int)
            # Normalize to 0-1 using the TRAINING score range (higher = more
            # anomalous). Test scores beyond the training range clip to 0/1.
            score_min = artefact["score_min"]
            score_max = artefact["score_max"]
            norm_score = 1 - (raw_score - score_min) / (score_max - score_min + 1e-9)
            df.loc[mask, "if_score"] = np.clip(norm_score, 0.0, 1.0)

        df["if_confidence"] = df["if_score"]
        return df

    def evaluate(self, predicted_df: pd.DataFrame) -> dict:
        """
        Compute full evaluation metrics against ground truth.
        predict() copies its input, so the ground-truth label travels on the
        same row as the prediction — no positional alignment needed.
        """
        y_true  = predicted_df["has_anomaly"].astype(int).values
        y_pred  = predicted_df["if_prediction"].values
        y_score = predicted_df["if_score"].values

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
            "contamination_param": self.contamination,
            "n_estimators": self.n_estimators,
        }

        self.results = metrics
        return metrics

    def save(self, name: str = "isolation_forest"):
        """Persist trained models and metrics."""
        path = os.path.join(self.output_dir, f"{name}.pkl")
        joblib.dump(self.models, path)
        metrics_path = os.path.join(self.output_dir, f"{name}_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"  Model saved to {path}")

    def load(self, name: str = "isolation_forest"):
        path = os.path.join(self.output_dir, f"{name}.pkl")
        self.models = joblib.load(path)
        print(f"  Model loaded from {path}")

    def run_full_pipeline(self, train_df, test_df) -> dict:
        """Train, predict, evaluate in one call."""
        print("\n" + "=" * 60)
        print("  Isolation Forest Detection Pipeline")
        print("=" * 60)

        self.train(train_df)
        predicted = self.predict(test_df)
        metrics   = self.evaluate(predicted)
        self.save()

        print(f"\n  Results:")
        print(f"    Precision : {metrics['precision']:.4f}")
        print(f"    Recall    : {metrics['recall']:.4f}")
        print(f"    F1 Score  : {metrics['f1_score']:.4f}")
        print(f"    AUC-ROC   : {metrics['auc_roc']:.4f}")
        print(f"    TP={metrics['true_positives']}  FP={metrics['false_positives']}  "
              f"FN={metrics['false_negatives']}  TN={metrics['true_negatives']}")

        return {"predictions": predicted, "metrics": metrics}


if __name__ == "__main__":
    import sys
    sys.path.append(".")

    train_df = pd.read_csv("data/processed/daily_aggregated.csv", parse_dates=["date"])
    # Simple temporal split for quick test
    split = int(len(train_df) * 0.8)
    train = train_df.iloc[:split]
    test  = train_df.iloc[split:]

    detector = IsolationForestDetector(contamination=0.05)
    results  = detector.run_full_pipeline(train, test)
    print("\nMetrics:", results["metrics"])
