"""
Dashboard Visualization Generator
Creates all charts for the project report and interactive dashboard.
Generates static PNG files + an interactive HTML dashboard.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import json
import os
import warnings
warnings.filterwarnings("ignore")

COLORS = {
    "AWS":   "#FF9900",
    "GCP":   "#4285F4",
    "Azure": "#00A4EF",
    "anomaly":  "#E74C3C",
    "normal":   "#2ECC71",
    "forecast": "#9B59B6",
    "upper_ci": "#E8D5F0",
    "bg":       "#F8F9FA",
}

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      10,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linestyle":     "--",
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
})


def load_data():
    daily   = pd.read_csv("data/processed/daily_aggregated.csv",    parse_dates=["date"])
    preds   = pd.read_csv("reports/full_predictions.csv",           parse_dates=["date"])
    raw     = pd.read_csv("data/simulated/multicloud_billing.csv",  parse_dates=["date"])
    metrics = json.load(open("reports/pipeline_results.json"))
    return daily, preds, raw, metrics


def plot_spending_trends(daily: pd.DataFrame, preds: pd.DataFrame, out_dir: str):
    """Multi-provider spending trends with anomaly overlay."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle("Multi-Cloud Daily Spending Trends with Anomaly Detection",
                 fontsize=14, fontweight="bold", y=1.01)

    providers = ["AWS", "GCP", "Azure"]
    for ax, provider in zip(axes, providers):
        pdata = daily[daily["provider"] == provider].sort_values("date")
        pred_p = preds[preds["provider"] == provider].sort_values("date")

        ax.plot(pdata["date"], pdata["total_cost"],
                color=COLORS[provider], linewidth=1.2, label="Actual cost", alpha=0.9)

        if "arima_forecast" in pred_p.columns:
            ax.plot(pred_p["date"], pred_p["arima_forecast"],
                    color=COLORS["forecast"], linewidth=1.5, linestyle="--",
                    label="ARIMA forecast", alpha=0.85)
            if "arima_upper_ci" in pred_p.columns:
                ax.fill_between(pred_p["date"],
                                pred_p["arima_lower_ci"].clip(lower=0),
                                pred_p["arima_upper_ci"],
                                alpha=0.15, color=COLORS["forecast"],
                                label="95% CI")

        # Anomaly markers
        anom = pdata[pdata["has_anomaly"]]
        if len(anom):
            ax.scatter(anom["date"], anom["total_cost"],
                       color=COLORS["anomaly"], s=80, zorder=5,
                       label=f"Anomalies ({len(anom)})", marker="^")

        ax.set_ylabel(f"{provider}\nDaily Cost ($)", fontsize=9)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"${x:,.0f}")
        )
        ax.legend(fontsize=8, loc="upper left")
        ax.set_facecolor("#FAFAFA")

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    path = os.path.join(out_dir, "01_spending_trends.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_anomaly_heatmap(raw: pd.DataFrame, out_dir: str):
    """Heatmap of anomaly frequency by provider and month."""
    raw["month"] = raw["date"].dt.to_period("M")
    anom = raw[raw["is_anomaly"]].groupby(["provider","month"]).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(anom.values, cmap="YlOrRd", aspect="auto")

    ax.set_yticks(range(len(anom.index)))
    ax.set_yticklabels(anom.index)
    ax.set_xticks(range(len(anom.columns)))
    ax.set_xticklabels([str(m) for m in anom.columns], rotation=45, ha="right", fontsize=8)

    for i in range(anom.shape[0]):
        for j in range(anom.shape[1]):
            val = anom.values[i, j]
            if val > 0:
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=8, fontweight="bold",
                        color="white" if val > anom.values.max() * 0.5 else "black")

    plt.colorbar(im, ax=ax, label="Anomaly count")
    ax.set_title("Anomaly Frequency Heatmap by Provider and Month", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "02_anomaly_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_model_comparison(metrics: dict, out_dir: str):
    """Side-by-side model performance comparison bar chart."""
    models    = ["Isolation Forest", "ARIMA", "Ensemble"]
    metric_keys = ["precision", "recall", "f1_score", "auc_roc"]
    labels    = ["Precision", "Recall", "F1 Score", "AUC-ROC"]
    data_keys = ["if_metrics", "arima_metrics", "ensemble_metrics"]

    values = {m_label: [] for m_label in labels}
    for key in data_keys:
        m = metrics.get(key, {})
        for mk, ml in zip(metric_keys, labels):
            values[ml].append(m.get(mk, 0))

    x       = np.arange(len(models))
    n_metrics = len(labels)
    width   = 0.18
    colors  = ["#3498DB", "#2ECC71", "#E74C3C", "#F39C12"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (label, color) in enumerate(zip(labels, colors)):
        bars = ax.bar(x + i * width - (n_metrics/2 - 0.5) * width,
                      values[label], width, label=label, color=color, alpha=0.85)
        for bar, val in zip(bars, values[label]):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_ylim(0, 1.12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Performance Comparison: Isolation Forest vs ARIMA vs Ensemble",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.axhline(y=0.8, color="gray", linestyle=":", alpha=0.5, label="0.8 threshold")

    plt.tight_layout()
    path = os.path.join(out_dir, "03_model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_confusion_matrices(metrics: dict, out_dir: str):
    """3-panel confusion matrix comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Confusion Matrices — Anomaly Detection Results",
                 fontsize=12, fontweight="bold")

    model_names = ["Isolation Forest", "ARIMA", "Ensemble"]
    data_keys   = ["if_metrics", "arima_metrics", "ensemble_metrics"]
    cmaps       = ["Blues", "Greens", "Purples"]

    for ax, name, key, cmap in zip(axes, model_names, data_keys, cmaps):
        m  = metrics.get(key, {})
        tp = m.get("true_positives",  0)
        fp = m.get("false_positives", 0)
        fn = m.get("false_negatives", 0)
        tn = m.get("true_negatives",  0)
        cm = np.array([[tn, fp], [fn, tp]])

        im = ax.imshow(cm, cmap=cmap, vmin=0, vmax=max(cm.max(), 1))
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Predicted\nNormal", "Predicted\nAnomaly"])
        ax.set_yticklabels(["Actual\nNormal", "Actual\nAnomaly"])

        labels_text = [["TN", "FP"], ["FN", "TP"]]
        for i in range(2):
            for j in range(2):
                v   = cm[i, j]
                lbl = labels_text[i][j]
                ax.text(j, i, f"{lbl}\n{v}",
                        ha="center", va="center", fontsize=12, fontweight="bold",
                        color="white" if v > cm.max() * 0.5 else "black")

        f1  = m.get("f1_score",  0)
        prec = m.get("precision", 0)
        ax.set_title(f"{name}\nF1={f1:.3f}  Prec={prec:.3f}", fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    path = os.path.join(out_dir, "04_confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_cost_distribution(raw: pd.DataFrame, out_dir: str):
    """Cost distribution by provider + anomaly type breakdown."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Box plot per provider
    providers = ["AWS", "GCP", "Azure"]
    data_per_provider = [raw[raw["provider"] == p]["cost"].values for p in providers]
    bp = ax1.boxplot(data_per_provider, labels=providers, patch_artist=True,
                     showfliers=True, flierprops=dict(marker=".", markersize=2, alpha=0.3))
    for patch, color in zip(bp["boxes"], [COLORS["AWS"], COLORS["GCP"], COLORS["Azure"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax1.set_title("Daily Service Cost Distribution by Provider", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Daily Cost ($)")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # Anomaly type breakdown
    anom_df = raw[raw["is_anomaly"] & raw["anomaly_type"].notna()]
    if len(anom_df):
        type_counts = anom_df["anomaly_type"].value_counts()
        wedge_colors = ["#E74C3C","#E67E22","#F1C40F","#1ABC9C","#3498DB","#9B59B6"]
        ax2.pie(type_counts.values, labels=type_counts.index,
                colors=wedge_colors[:len(type_counts)],
                autopct="%1.1f%%", startangle=140,
                wedgeprops={"edgecolor":"white","linewidth":1.5})
        ax2.set_title("Anomaly Distribution by Type", fontsize=11, fontweight="bold")
    else:
        ax2.text(0.5, 0.5, "No anomalies", ha="center", va="center", transform=ax2.transAxes)

    plt.tight_layout()
    path = os.path.join(out_dir, "05_cost_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_traffic_surge(daily: pd.DataFrame, preds: pd.DataFrame, out_dir: str):
    """Zoom-in on the simulated traffic surge event."""
    surge_start = pd.Timestamp("2024-06-10")
    surge_end   = pd.Timestamp("2024-06-25")

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle("Simulated Traffic Surge Event — Detection Analysis",
                 fontsize=13, fontweight="bold")

    for ax, provider in zip(axes, ["AWS", "GCP", "Azure"]):
        pdata = daily[(daily["provider"] == provider) &
                      (daily["date"] >= surge_start) &
                      (daily["date"] <= surge_end)].sort_values("date")
        pred_p = preds[(preds["provider"] == provider) &
                       (preds["date"] >= surge_start) &
                       (preds["date"] <= surge_end)].sort_values("date")

        ax.plot(pdata["date"], pdata["total_cost"],
                color=COLORS[provider], linewidth=2, marker="o", markersize=5,
                label="Actual cost")

        if "arima_forecast" in pred_p.columns and len(pred_p):
            ax.plot(pred_p["date"], pred_p["arima_forecast"],
                    color=COLORS["forecast"], linewidth=1.5, linestyle="--",
                    label="ARIMA forecast")
            ax.fill_between(pred_p["date"],
                            pred_p.get("arima_lower_ci", pred_p["arima_forecast"] * 0.85),
                            pred_p.get("arima_upper_ci", pred_p["arima_forecast"] * 1.15),
                            alpha=0.2, color=COLORS["forecast"])

        # Shade surge period
        ax.axvspan(pd.Timestamp("2024-06-16"), pd.Timestamp("2024-06-18"),
                   alpha=0.15, color="red", label="Surge period")

        # Mark detected anomalies
        if "ensemble_prediction" in pred_p.columns:
            detected = pred_p[pred_p["ensemble_prediction"] == 1]
            ax.scatter(detected["date"], detected["total_cost"],
                       color="#E74C3C", s=120, zorder=6, marker="^",
                       label=f"Detected ({len(detected)})")

        ax.set_ylabel(f"{provider}\nCost ($)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.legend(fontsize=8, loc="upper left")

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha="right")
    plt.tight_layout()

    path = os.path.join(out_dir, "06_traffic_surge_detection.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_monthly_cost_summary(daily: pd.DataFrame, out_dir: str):
    """Stacked monthly cost bar chart by provider."""
    daily["month"] = daily["date"].dt.to_period("M")
    monthly = daily.groupby(["month","provider"])["total_cost"].sum().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(14, 6))
    x       = range(len(monthly))
    bottoms = np.zeros(len(monthly))
    providers_in_data = monthly.columns.tolist()

    for provider in providers_in_data:
        color = COLORS.get(provider, "#888")
        ax.bar(x, monthly[provider], bottom=bottoms, label=provider,
               color=color, alpha=0.85, width=0.7)
        bottoms += monthly[provider].values

    ax.set_xticks(range(len(monthly)))
    ax.set_xticklabels([str(m) for m in monthly.index], rotation=45, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}K"))
    ax.set_ylabel("Monthly Cost ($)", fontsize=11)
    ax.set_title("Monthly Cloud Spend by Provider (Jan 2023 – Jun 2024)", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left")
    plt.tight_layout()

    path = os.path.join(out_dir, "07_monthly_cost_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def generate_html_dashboard(daily, preds, raw, metrics, out_dir: str):
    """Generate a self-contained interactive HTML dashboard using Plotly."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.io as pio
    except ImportError:
        print("  Plotly not available, skipping HTML dashboard")
        return

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            "AWS Daily Cost + Anomalies", "GCP Daily Cost + Anomalies",
            "Azure Daily Cost + Anomalies", "Model Performance Comparison",
            "Monthly Cost by Provider",    "Alert Severity Distribution",
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    provider_positions = [("AWS", 1, 1), ("GCP", 1, 2), ("Azure", 2, 1)]
    for provider, row, col in provider_positions:
        pdata  = daily[daily["provider"] == provider].sort_values("date")
        pred_p = preds[preds["provider"] == provider].sort_values("date")

        fig.add_trace(go.Scatter(
            x=pdata["date"], y=pdata["total_cost"],
            name=f"{provider} cost", line=dict(color=COLORS[provider], width=1.5),
            showlegend=(row == 1 and col == 1),
        ), row=row, col=col)

        if "arima_forecast" in pred_p.columns:
            fig.add_trace(go.Scatter(
                x=pred_p["date"], y=pred_p["arima_forecast"],
                name="ARIMA forecast", line=dict(color="#9B59B6", dash="dash", width=1),
                showlegend=(row == 1 and col == 1),
            ), row=row, col=col)

        anom = pdata[pdata["has_anomaly"]]
        fig.add_trace(go.Scatter(
            x=anom["date"], y=anom["total_cost"],
            mode="markers", marker=dict(color="red", size=8, symbol="triangle-up"),
            name="Anomaly", showlegend=(row == 1 and col == 1),
        ), row=row, col=col)

    # Model comparison bar chart
    model_names = ["Isolation Forest", "ARIMA", "Ensemble"]
    data_keys   = ["if_metrics", "arima_metrics", "ensemble_metrics"]
    for metric_name, color in [("f1_score","#3498DB"), ("precision","#2ECC71"), ("recall","#E74C3C")]:
        vals = [metrics.get(k, {}).get(metric_name, 0) for k in data_keys]
        fig.add_trace(go.Bar(
            x=model_names, y=vals, name=metric_name.replace("_"," ").title(),
            marker_color=color, opacity=0.85,
        ), row=2, col=2)

    # Monthly cost
    daily["month_str"] = daily["date"].dt.to_period("M").astype(str)
    monthly = daily.groupby(["month_str","provider"])["total_cost"].sum().reset_index()
    for provider in ["AWS","GCP","Azure"]:
        mp = monthly[monthly["provider"] == provider]
        fig.add_trace(go.Bar(
            x=mp["month_str"], y=mp["total_cost"],
            name=f"{provider} monthly", marker_color=COLORS[provider], opacity=0.85,
        ), row=3, col=1)

    # Alert severity
    try:
        alerts_df = pd.read_json("reports/alert_log.json")
        if len(alerts_df):
            sev_counts = alerts_df["severity"].value_counts()
            fig.add_trace(go.Pie(
                labels=sev_counts.index, values=sev_counts.values,
                marker=dict(colors=["#E74C3C","#FF6600","#FFAA00","#0099FF"]),
                showlegend=False,
            ), row=3, col=2)
    except Exception:
        pass

    fig.update_layout(
        title=dict(text="☁️ Cloud Cost Anomaly Detection — Interactive Dashboard",
                   font=dict(size=18)),
        height=1000,
        barmode="stack",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    )

    path = os.path.join(out_dir, "dashboard.html")
    pio.write_html(fig, path, include_plotlyjs=True, full_html=True)
    print(f"  Saved: {path}")


def run_all(out_dir: str = "dashboard"):
    os.makedirs(out_dir, exist_ok=True)
    print("\n" + "=" * 60)
    print("  Generating Visualizations")
    print("=" * 60)

    daily, preds, raw, metrics = load_data()

    plot_spending_trends(daily, preds, out_dir)
    plot_anomaly_heatmap(raw, out_dir)
    plot_model_comparison(metrics, out_dir)
    plot_confusion_matrices(metrics, out_dir)
    plot_cost_distribution(raw, out_dir)
    plot_traffic_surge(daily, preds, out_dir)
    plot_monthly_cost_summary(daily, out_dir)
    generate_html_dashboard(daily, preds, raw, metrics, out_dir)

    print(f"\n  All visualizations saved to {out_dir}/")


if __name__ == "__main__":
    run_all()
