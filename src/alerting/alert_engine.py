"""
Cloud Cost Alert Engine
Generates, deduplicates, and dispatches alerts for detected anomalies.
Supports Slack (webhook), email (SMTP), and console logging.
"""

import json
import os
import smtplib
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from collections import defaultdict
import pandas as pd
import requests
import warnings
warnings.filterwarnings("ignore")


SEVERITY_COLORS = {
    "critical": "#FF0000",
    "high":     "#FF6600",
    "medium":   "#FFAA00",
    "low":      "#0099FF",
    "none":     "#AAAAAA",
}

SEVERITY_EMOJI = {
    "critical": "🚨",
    "high":     "⚠️",
    "medium":   "⚡",
    "low":      "ℹ️",
}


class AlertEngine:
    """
    Manages alert lifecycle:
      1. Generate alerts from anomaly predictions
      2. Deduplicate (suppress repeat alerts within cooldown window)
      3. Dispatch to configured channels (Slack / email / console)
      4. Log all alerts for audit trail
    """

    def __init__(self,
                 slack_webhook_url: str = None,
                 email_config: dict = None,
                 cooldown_hours: int = 4,
                 min_severity: str = "low",
                 root_cause_analyzer=None,
                 output_dir: str = "reports"):
        self.slack_webhook   = slack_webhook_url
        self.email_config    = email_config or {}
        self.cooldown_hours  = cooldown_hours
        self.min_severity    = min_severity
        self.root_cause      = root_cause_analyzer
        self.output_dir      = output_dir
        self._alert_log      = []
        self._last_alerted   = defaultdict(lambda: datetime.min)
        self._severity_order = ["none", "low", "medium", "high", "critical"]
        os.makedirs(output_dir, exist_ok=True)

    def _severity_rank(self, severity: str) -> int:
        return self._severity_order.index(severity) if severity in self._severity_order else 0

    def _should_suppress(self, key: str) -> bool:
        """Return True if the alert for `key` is within cooldown."""
        last = self._last_alerted.get(key, datetime.min)
        return (datetime.now() - last).total_seconds() < self.cooldown_hours * 3600

    def generate_alerts(self, predictions_df: pd.DataFrame) -> list:
        """Convert anomaly predictions into structured alert objects."""
        alerts = []
        anomalies = predictions_df[predictions_df["ensemble_prediction"] == 1].copy()

        for _, row in anomalies.iterrows():
            severity = row.get("alert_severity", "medium")
            if self._severity_rank(severity) < self._severity_rank(self.min_severity):
                continue

            provider = row.get("provider", "Unknown")
            date_str = str(row.get("date", ""))[:10]
            cost     = row.get("total_cost", 0)
            forecast = row.get("arima_forecast", cost * 0.6)
            score    = row.get("ensemble_score", 0)

            pct_above = ((cost - forecast) / (forecast + 1e-9)) * 100

            # Attribute the spike to the services that drove it
            root_causes = []
            if self.root_cause is not None:
                root_causes = self.root_cause.explain(provider, date_str)

            alert = {
                "id":           f"{provider}_{date_str}_{severity}",
                "timestamp":    datetime.now().isoformat(),
                "date":         date_str,
                "provider":     provider,
                "severity":     severity,
                "ensemble_score": round(float(score), 4),
                "actual_cost":  round(float(cost), 2),
                "forecast_cost": round(float(forecast), 2),
                "pct_above_forecast": round(float(pct_above), 1),
                "root_causes":  root_causes,
                "message":      self._format_message(provider, date_str, severity,
                                                      cost, forecast, pct_above,
                                                      root_causes),
                "suppressed":   self._should_suppress(f"{provider}_{severity}"),
            }
            alerts.append(alert)

            if not alert["suppressed"]:
                self._last_alerted[f"{provider}_{severity}"] = datetime.now()

        return alerts

    def _format_message(self, provider, date, severity, actual, forecast,
                        pct_above, root_causes=None):
        emoji = SEVERITY_EMOJI.get(severity, "⚠️")
        msg = (
            f"{emoji} [{severity.upper()}] Cloud cost anomaly detected\n"
            f"Provider: {provider} | Date: {date}\n"
            f"Actual: ${actual:,.2f} | Forecast: ${forecast:,.2f}\n"
            f"Deviation: +{pct_above:.1f}% above expected"
        )
        if root_causes:
            driver_lines = [
                f"  • {d['service']} ({d['region']}): ${d['actual_cost']:,.0f} "
                f"vs ${d['baseline_cost']:,.0f} baseline "
                f"({d['pct_above_baseline']:+.0f}%, {d['share_of_excess_pct']:.0f}% of excess)"
                for d in root_causes
            ]
            msg += "\nTop drivers:\n" + "\n".join(driver_lines)
        return msg

    def dispatch_console(self, alerts: list):
        """Print alerts to console (always enabled)."""
        active = [a for a in alerts if not a["suppressed"]]
        print(f"\n  Alert Engine — {len(active)} active alerts "
              f"({len(alerts) - len(active)} suppressed)")
        for a in active:
            sev_upper = a["severity"].upper().ljust(8)
            print(f"  [{sev_upper}] {a['provider']} {a['date']} | "
                  f"${a['actual_cost']:,.0f} ({a['pct_above_forecast']:+.0f}%) | "
                  f"score={a['ensemble_score']:.2f}")
            for d in a.get("root_causes", []):
                print(f"             ↳ {d['service']} ({d['region']}): "
                      f"+${d['excess_usd']:,.0f} ({d['pct_above_baseline']:+.0f}% vs baseline, "
                      f"{d['share_of_excess_pct']:.0f}% of excess)")

    def dispatch_slack(self, alerts: list):
        """Send alert to Slack via webhook (if configured)."""
        if not self.slack_webhook:
            return
        active = [a for a in alerts if not a["suppressed"]]
        for alert in active:
            payload = {
                "attachments": [{
                    "color": SEVERITY_COLORS.get(alert["severity"], "#AAAAAA"),
                    "title": f"☁️ Cloud Cost Anomaly — {alert['provider']}",
                    "text": alert["message"],
                    "fields": [
                        {"title": "Score",    "value": str(alert["ensemble_score"]), "short": True},
                        {"title": "Severity", "value": alert["severity"].upper(),    "short": True},
                    ],
                    "footer": "Cloud Cost Anomaly Detection System",
                    "ts": int(datetime.now().timestamp()),
                }]
            }
            try:
                r = requests.post(self.slack_webhook, json=payload, timeout=5)
                if r.status_code == 200:
                    print(f"  Slack alert sent for {alert['provider']} {alert['date']}")
            except Exception as e:
                print(f"  Slack dispatch failed: {e}")

    def dispatch_email(self, alerts: list):
        """Send summary email (if SMTP config provided)."""
        if not self.email_config.get("smtp_host"):
            return
        active = [a for a in alerts if not a["suppressed"]
                  and self._severity_rank(a["severity"]) >= self._severity_rank("high")]
        if not active:
            return

        body_lines = ["Cloud Cost Anomaly Alert Summary\n", "=" * 50]
        for a in active:
            body_lines.append(a["message"])
            body_lines.append("-" * 40)

        msg = MIMEMultipart()
        msg["From"]    = self.email_config.get("from_addr", "alerts@example.com")
        msg["To"]      = self.email_config.get("to_addr",   "ops@example.com")
        msg["Subject"] = f"🚨 Cloud Cost Alert — {len(active)} anomalies detected"
        msg.attach(MIMEText("\n".join(body_lines), "plain"))

        try:
            with smtplib.SMTP(self.email_config["smtp_host"],
                              self.email_config.get("smtp_port", 587)) as server:
                server.starttls()
                server.login(self.email_config.get("user",""),
                             self.email_config.get("password",""))
                server.send_message(msg)
            print(f"  Email sent for {len(active)} high/critical alerts")
        except Exception as e:
            print(f"  Email dispatch failed (expected in demo): {e}")

    def save_alert_log(self, alerts: list, filename: str = "alert_log.json"):
        """Persist full alert log for audit trail."""
        self._alert_log.extend(alerts)
        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as f:
            json.dump(self._alert_log, f, indent=2, default=str)
        print(f"  Alert log saved: {len(self._alert_log)} total alerts → {path}")

    def generate_alert_summary(self) -> dict:
        """Statistical summary of all alerts fired."""
        if not self._alert_log:
            return {}
        df = pd.DataFrame(self._alert_log)
        summary = {
            "total_alerts":     len(df),
            "suppressed":       int(df["suppressed"].sum()),
            "active_alerts":    int((~df["suppressed"]).sum()),
            "by_severity":      df["severity"].value_counts().to_dict(),
            "by_provider":      df["provider"].value_counts().to_dict(),
            "avg_pct_above":    round(df["pct_above_forecast"].mean(), 1),
            "max_pct_above":    round(df["pct_above_forecast"].max(), 1),
            "avg_score":        round(df["ensemble_score"].mean(), 4),
        }
        return summary

    def run(self, predictions_df: pd.DataFrame) -> list:
        """Main entry: generate + dispatch + log all alerts."""
        alerts = self.generate_alerts(predictions_df)
        self.dispatch_console(alerts)
        self.dispatch_slack(alerts)
        self.dispatch_email(alerts)
        self.save_alert_log(alerts)
        return alerts


if __name__ == "__main__":
    # Demo with mock predictions
    mock_data = pd.DataFrame({
        "date": pd.date_range("2024-05-01", periods=10),
        "provider": ["AWS"] * 5 + ["GCP"] * 5,
        "total_cost": [1200, 1180, 4800, 1150, 1220, 900, 870, 910, 5200, 880],
        "arima_forecast": [1150] * 10,
        "ensemble_prediction": [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
        "ensemble_score": [0.2, 0.15, 0.88, 0.18, 0.22, 0.19, 0.17, 0.21, 0.91, 0.16],
        "alert_severity": ["none","none","critical","none","none",
                           "none","none","none","critical","none"],
    })

    engine = AlertEngine(cooldown_hours=0)
    alerts = engine.run(mock_data)
    summary = engine.generate_alert_summary()
    print("\nAlert summary:", summary)
