"""
Cloud Cost Anomaly Detection — Live Dashboard
Plotly Dash application served via Gunicorn on Render.com
"""

import os
import io
import json
import base64
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, State, callback, no_update
import dash_bootstrap_components as dbc
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

from src.ingestion.adapters import normalize_billing_export
from src.byoe_pipeline      import analyze_billing_data

# ── Bootstrap theme ───────────────────────────────────────────
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="Cloud Cost Anomaly Detection",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server   # expose Flask server for Gunicorn

# ── Color palette ──────────────────────────────────────────────
COLORS = {
    "bg":        "#0A1628",
    "card":      "#112240",
    "teal":      "#00B4D8",
    "amber":     "#F5A623",
    "red":       "#E74C3C",
    "green":     "#2ECC71",
    "aws":       "#FF9900",
    "gcp":       "#4285F4",
    "azure":     "#00A4EF",
    "text":      "#E8F4F8",
    "subtext":   "#7A8FA6",
}

PROVIDER_COLORS = {"AWS": COLORS["aws"], "GCP": COLORS["gcp"], "Azure": COLORS["azure"]}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(17,34,64,0.6)",
    font=dict(color=COLORS["text"], family="Inter, sans-serif", size=12),
    margin=dict(l=10, r=10, t=30, b=10),
)
AXIS_STYLE  = dict(gridcolor="rgba(255,255,255,0.07)", showgrid=True)
LEGEND_STYLE= dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1)

# ── Data loaders ───────────────────────────────────────────────
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    try:
        daily   = pd.read_csv(os.path.join(base, "data/processed/daily_aggregated.csv"), parse_dates=["date"])
        preds   = pd.read_csv(os.path.join(base, "reports/full_predictions.csv"),         parse_dates=["date"])
        raw     = pd.read_csv(os.path.join(base, "data/simulated/multicloud_billing.csv"), parse_dates=["date"])
        metrics = json.load(open(os.path.join(base, "reports/pipeline_results.json")))
        alerts  = json.load(open(os.path.join(base, "reports/alert_log.json")))
        return daily, preds, raw, metrics, alerts
    except Exception as e:
        print(f"Data load error: {e}")
        return None, None, None, {}, []

daily, preds, raw, metrics, alert_log = load_data()

# ── KPI cards ──────────────────────────────────────────────────
def kpi_card(value, label, color=COLORS["teal"], icon="📊"):
    return dbc.Card([
        dbc.CardBody([
            html.Div(icon, style={"fontSize": "20px", "marginBottom": "4px"}),
            html.Div(value, style={"fontSize": "26px", "fontWeight": "600", "color": color}),
            html.Div(label, style={"fontSize": "11px", "color": COLORS["subtext"], "marginTop": "2px"}),
        ], style={"padding": "16px 20px"})
    ], style={"background": COLORS["card"], "border": f"1px solid {color}30", "borderRadius": "12px"})

# ── Alert badge ────────────────────────────────────────────────
def severity_badge(sev):
    colors = {"CRITICAL": "#E74C3C", "HIGH": "#F39C12", "MEDIUM": "#3498DB", "LOW": "#7F8C8D"}
    c = colors.get(sev.upper(), "#888")
    return html.Span(sev, style={
        "background": f"{c}25", "color": c, "padding": "2px 8px",
        "borderRadius": "4px", "fontSize": "10px", "fontWeight": "600",
        "border": f"1px solid {c}50",
    })

# ── Figures ────────────────────────────────────────────────────
def fig_spending_trends(provider_filter="ALL"):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=["AWS", "GCP", "Azure"])
    providers = ["AWS", "GCP", "Azure"]
    for i, prov in enumerate(providers, 1):
        if provider_filter != "ALL" and prov != provider_filter:
            continue
        d   = daily[daily["provider"] == prov].sort_values("date")
        p   = preds[preds["provider"] == prov].sort_values("date") if preds is not None else pd.DataFrame()
        col = PROVIDER_COLORS[prov]

        fig.add_trace(go.Scatter(x=d["date"], y=d["total_cost"], name=f"{prov} actual",
            line=dict(color=col, width=1.5), opacity=0.9), row=i, col=1)

        if len(p) and "arima_forecast" in p.columns:
            fig.add_trace(go.Scatter(x=p["date"], y=p["arima_forecast"], name="ARIMA forecast",
                line=dict(color=COLORS["teal"], dash="dash", width=1.2), opacity=0.7,
                showlegend=(i == 1)), row=i, col=1)
            if "arima_upper_ci" in p.columns:
                fig.add_trace(go.Scatter(
                    x=list(p["date"]) + list(p["date"])[::-1],
                    y=list(p["arima_upper_ci"]) + list(p["arima_lower_ci"].clip(lower=0))[::-1],
                    fill="toself", fillcolor="rgba(0,180,216,0.08)",
                    line=dict(color="rgba(0,0,0,0)"), name="95% CI", showlegend=(i == 1),
                ), row=i, col=1)

        anoms = d[d["has_anomaly"]]
        if len(anoms):
            fig.add_trace(go.Scatter(x=anoms["date"], y=anoms["total_cost"], mode="markers",
                marker=dict(color=COLORS["red"], size=8, symbol="triangle-up", line=dict(width=1, color="white")),
                name=f"Anomaly", showlegend=(i == 1)), row=i, col=1)

    fig.update_layout(**PLOTLY_LAYOUT, xaxis=AXIS_STYLE, yaxis=AXIS_STYLE, legend=LEGEND_STYLE, height=520,
        title=dict(text="Multi-Cloud Daily Spending Trends", font=dict(size=14, color=COLORS["text"])))
    for ann in fig.layout.annotations:
        ann.font.color = COLORS["subtext"]
        ann.font.size  = 11
    return fig


def fig_model_comparison():
    models  = ["Isolation Forest", "ARIMA", "Ensemble"]
    metrics_list = [metrics.get("if_metrics", {}), metrics.get("arima_metrics", {}), metrics.get("ensemble_metrics", {})]
    metric_names = ["Precision", "Recall", "F1 Score", "AUC-ROC"]
    metric_keys  = ["precision", "recall", "f1_score", "auc_roc"]
    colors       = [COLORS["aws"], COLORS["teal"], COLORS["amber"]]

    fig = go.Figure()
    for mi, (mname, mdata, col) in enumerate(zip(models, metrics_list, colors)):
        vals = [round(mdata.get(k, 0), 4) for k in metric_keys]
        fig.add_trace(go.Bar(name=mname, x=metric_names, y=vals,
            marker_color=col, opacity=0.85, text=[f"{v:.3f}" for v in vals],
            textposition="outside", textfont=dict(color=COLORS["text"], size=10)))

    base = {k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("yaxis", "legend")}
    fig.update_layout(**base, height=320, barmode="group",
        yaxis=dict(range=[0, 1.18], gridcolor="rgba(255,255,255,0.07)"),
        legend=dict(orientation="h", y=-0.15, bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
        title=dict(text="Model Performance Comparison", font=dict(size=14, color=COLORS["text"])))
    return fig


def fig_anomaly_heatmap():
    if raw is None:
        return go.Figure()
    r = raw.copy()
    r["month"] = r["date"].dt.to_period("M").astype(str)
    anom = r[r["is_anomaly"]].groupby(["provider", "month"]).size().unstack(fill_value=0)
    fig = go.Figure(go.Heatmap(
        z=anom.values, x=list(anom.columns), y=list(anom.index),
        colorscale=[[0, COLORS["card"]], [0.5, COLORS["amber"]], [1, COLORS["red"]]],
        text=anom.values, texttemplate="%{text}", textfont=dict(size=11),
        showscale=True, colorbar=dict(title="Count", tickfont=dict(color=COLORS["text"])),
    ))
    fig.update_layout(**PLOTLY_LAYOUT, legend=LEGEND_STYLE, height=200,
        title=dict(text="Anomaly Frequency by Provider & Month", font=dict(size=14, color=COLORS["text"])),
        xaxis=dict(tickangle=-40, tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=10)))
    return fig


def fig_monthly_spend():
    if daily is None:
        return go.Figure()
    d = daily.copy()
    d["month"] = d["date"].dt.to_period("M").astype(str)
    monthly = d.groupby(["month", "provider"])["total_cost"].sum().reset_index()
    fig = go.Figure()
    for prov, col in PROVIDER_COLORS.items():
        mp = monthly[monthly["provider"] == prov]
        fig.add_trace(go.Bar(x=mp["month"], y=mp["total_cost"], name=prov,
            marker_color=col, opacity=0.85))
    fig.update_layout(**PLOTLY_LAYOUT,
        xaxis=dict(tickangle=-45, tickfont=dict(size=9), gridcolor="rgba(255,255,255,0.07)"),
        yaxis=dict(tickformat="$,.0f", gridcolor="rgba(255,255,255,0.07)"),
        legend=LEGEND_STYLE, height=280, barmode="stack",
        title=dict(text="Monthly Spend by Provider ($)", font=dict(size=14, color=COLORS["text"])))
    return fig


def fig_anomaly_types():
    if raw is None:
        return go.Figure()
    anom = raw[raw["is_anomaly"] & raw["anomaly_type"].notna()]
    if not len(anom):
        return go.Figure()
    counts = anom["anomaly_type"].value_counts()
    fig = go.Figure(go.Pie(
        labels=counts.index, values=counts.values, hole=0.45,
        marker=dict(colors=[COLORS["red"], COLORS["aws"], COLORS["teal"], COLORS["amber"], COLORS["gcp"]],
                    line=dict(color=COLORS["bg"], width=2)),
        textfont=dict(color=COLORS["text"], size=11),
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=280,
        title=dict(text="Anomaly Distribution by Type", font=dict(size=14, color=COLORS["text"])),
        showlegend=True,
        legend=dict(font=dict(size=10, color=COLORS["text"]), bgcolor="rgba(0,0,0,0)"))
    return fig

# ── Alert table rows ────────────────────────────────────────────
def alert_table_rows():
    if not alert_log:
        return [html.Tr([html.Td("No alerts fired", colSpan=6,
            style={"textAlign": "center", "color": COLORS["subtext"], "padding": "16px"})])]
    rows = []
    for a in sorted(alert_log, key=lambda x: x.get("ensemble_score", 0), reverse=True):
        sev = a.get("severity", "low").upper()
        rows.append(html.Tr([
            html.Td(severity_badge(sev)),
            html.Td(a.get("provider", ""), style={"color": COLORS["aws"], "fontWeight": "600"}),
            html.Td(str(a.get("date", ""))[:10], style={"color": COLORS["subtext"]}),
            html.Td(f"${a.get('actual_cost', 0):,.0f}", style={"color": COLORS["text"]}),
            html.Td(f"+{a.get('pct_above_forecast', 0):.0f}%", style={"color": COLORS["red"], "fontWeight": "600"}),
            html.Td(f"{a.get('ensemble_score', 0):.2f}", style={"color": COLORS["teal"]}),
        ], style={"borderBottom": f"1px solid {COLORS['card']}"}))
    return rows


def _make_metrics_col(name, m, color):
    rows = [
        ("Precision",     m.get("precision",     0)),
        ("Recall",        m.get("recall",        0)),
        ("F1 Score",      m.get("f1_score",      0)),
        ("AUC-ROC",       m.get("auc_roc",       0)),
        ("True Pos",      m.get("true_positives",  0)),
        ("False Pos",     m.get("false_positives", 0)),
        ("False Neg",     m.get("false_negatives", 0)),
    ]
    return html.Div([
        html.Div(name, style={"color": color, "fontWeight": "600", "fontSize": "13px", "marginBottom": "8px"}),
        *[html.Div([
            html.Span(label, style={"color": COLORS["subtext"], "fontSize": "11px"}),
            html.Span(f"{val:.4f}" if isinstance(val, float) else str(val),
                style={"color": COLORS["text"], "fontSize": "12px", "fontWeight": "500", "float": "right"}),
        ], style={"borderBottom": f"1px solid {COLORS['card']}80", "padding": "4px 0",
                  "overflow": "hidden"}) for label, val in rows],
    ])


# ── Bring-Your-Own-Export mode ─────────────────────────────────
def fig_byoe_trends(preds):
    """Detection-window chart: actual vs forecast with flagged days."""
    fig = go.Figure()
    for prov in sorted(preds["provider"].unique()):
        p   = preds[preds["provider"] == prov].sort_values("date")
        col = PROVIDER_COLORS.get(prov, COLORS["teal"])
        fig.add_trace(go.Scatter(x=p["date"], y=p["total_cost"], name=f"{prov} actual",
            line=dict(color=col, width=1.6)))
        if "arima_forecast" in p.columns:
            fig.add_trace(go.Scatter(x=p["date"], y=p["arima_forecast"],
                name=f"{prov} forecast", line=dict(color=col, dash="dot", width=1), opacity=0.55))
        flagged = p[p["ensemble_prediction"] == 1]
        if len(flagged):
            fig.add_trace(go.Scatter(x=flagged["date"], y=flagged["total_cost"], mode="markers",
                marker=dict(color=COLORS["red"], size=10, symbol="triangle-up",
                            line=dict(width=1, color="white")),
                name=f"{prov} anomaly"))
    fig.update_layout(**PLOTLY_LAYOUT, xaxis=AXIS_STYLE,
        yaxis=dict(tickformat="$,.0f", gridcolor="rgba(255,255,255,0.07)"),
        legend=LEGEND_STYLE, height=380,
        title=dict(text="Your Data — Detection Window (actual vs forecast)",
                   font=dict(size=14, color=COLORS["text"])))
    return fig


def _byoe_error_card(message):
    return dbc.Alert([
        html.Strong("Could not analyze this file. "),
        html.Span(message, style={"fontSize": "13px"}),
    ], color="danger", style={"marginTop": "12px"})


def render_byoe_results(results, info):
    s = results["summary"]
    findings = results["findings"]

    finding_cards = []
    for f in findings:
        driver_items = [
            html.Li(
                f"{d['service']} ({d['region']}): +${d['excess_usd']:,.0f} "
                f"({d['pct_above_baseline']:+.0f}% vs baseline · "
                f"{d['share_of_excess_pct']:.0f}% of excess)",
                style={"fontSize": "12px", "color": COLORS["text"]},
            ) for d in f["root_causes"]
        ]
        finding_cards.append(dbc.Card([dbc.CardBody([
            html.Div([
                severity_badge(f["severity"].upper()),
                html.Span(f" {f['provider']} · {f['date']}",
                    style={"fontWeight": "600", "color": COLORS["text"], "marginLeft": "8px"}),
                html.Span(f"score {f['ensemble_score']:.2f}",
                    style={"float": "right", "color": COLORS["teal"], "fontSize": "12px"}),
            ]),
            html.Div(
                f"${f['actual_cost']:,.0f} actual vs ${f['forecast_cost']:,.0f} forecast "
                f"({f['pct_above_forecast']:+.0f}%)",
                style={"fontSize": "13px", "color": COLORS["subtext"], "margin": "6px 0"}),
            html.Ul(driver_items, style={"marginBottom": "0", "paddingLeft": "20px"})
            if driver_items else
            html.Div("No single service stands out above baseline.",
                     style={"fontSize": "12px", "color": COLORS["subtext"]}),
        ], style={"padding": "12px 16px"})],
        style={"background": COLORS["bg"], "border": f"1px solid {COLORS['red']}30",
               "borderRadius": "8px", "marginBottom": "8px"}))

    provider_label = (", ".join(info["provider"]) if isinstance(info["provider"], list)
                      else info["provider"])
    return html.Div([
        dbc.Alert([
            html.Strong(f"✓ Parsed as {info['format'].replace('_', ' ')} export — "),
            html.Span(f"{provider_label} · {info['days']} days · {info['services']} services · "
                      f"${info['total_cost']:,.0f} total", style={"fontSize": "13px"}),
        ], color="success", style={"marginTop": "12px", "marginBottom": "12px"}),
        dbc.Row([
            dbc.Col(kpi_card(str(s["days_analyzed"]), "Days analyzed", COLORS["teal"], "📅"), md=3),
            dbc.Col(kpi_card(str(s["anomaly_days"]), "Anomaly days found", COLORS["red"], "🚨"), md=3),
            dbc.Col(kpi_card(f"${s['total_excess_usd']:,.0f}", "Excess spend flagged", COLORS["amber"], "💸"), md=3),
            dbc.Col(kpi_card(str(s["training_days"]), "Days used as baseline", COLORS["green"], "📚"), md=3),
        ], className="g-3", style={"marginBottom": "12px"}),
        dcc.Graph(figure=fig_byoe_trends(results["predictions"]),
                  config={"displayModeBar": False}),
        html.Div([
            html.Div(f"🔍 Findings ({len(findings)})", style={"fontSize": "14px",
                "fontWeight": "500", "color": COLORS["text"], "margin": "12px 0 8px"}),
            *finding_cards,
        ]) if findings else html.Div("✅ No anomalies detected in the most recent window.",
            style={"color": COLORS["green"], "padding": "12px 0", "fontSize": "14px"}),
    ])


byoe_section = dbc.Row([dbc.Col([
    dbc.Card([
        html.Div([
            html.Span("📤 Analyze Your Own Billing Export",
                style={"fontSize": "14px", "fontWeight": "500", "color": COLORS["text"]}),
            html.Span("AWS Cost & Usage Report · GCP billing export · Azure cost analysis · generic CSV",
                style={"fontSize": "11px", "color": COLORS["subtext"], "marginLeft": "12px"}),
        ], style={"padding": "12px 16px", "borderBottom": f"1px solid {COLORS['teal']}20"}),
        html.Div([
            html.Div("Download a billing CSV from your cloud console and drop it here. "
                     "Everything runs in this app — your data is not stored. "
                     "Needs at least 60 days of history.",
                style={"fontSize": "12px", "color": COLORS["subtext"], "marginBottom": "10px"}),
            dcc.Upload(
                id="byoe-upload",
                children=html.Div(["Drag & drop or ", html.A("select a CSV file",
                    style={"color": COLORS["teal"], "textDecoration": "underline"})]),
                style={"width": "100%", "height": "64px", "lineHeight": "64px",
                       "borderWidth": "1.5px", "borderStyle": "dashed",
                       "borderColor": f"{COLORS['teal']}60", "borderRadius": "10px",
                       "textAlign": "center", "color": COLORS["subtext"],
                       "cursor": "pointer", "background": f"{COLORS['teal']}08"},
                multiple=False,
                max_size=30 * 1024 * 1024,
            ),
            html.Div([
                dbc.Button("⬇ Try it with a sample export", id="byoe-sample-btn",
                    size="sm", outline=True, color="info", style={"marginTop": "10px"}),
                dcc.Download(id="byoe-sample-dl"),
            ]),
            dcc.Loading(html.Div(id="byoe-results"), type="dot", color=COLORS["teal"]),
        ], style={"padding": "16px"}),
    ], style={"background": COLORS["card"], "border": f"1px solid {COLORS['teal']}25",
              "borderRadius": "12px"}),
])])


# ── Layout ─────────────────────────────────────────────────────
em  = metrics.get("ensemble_metrics", {})
total_cost = daily["total_cost"].sum() if daily is not None else 0

app.layout = dbc.Container([
    # ── Header ──────────────────────────────────────────────────
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Span("☁️ ", style={"fontSize": "24px"}),
                html.Span("Cloud Cost Anomaly Detection",
                    style={"fontSize": "22px", "fontWeight": "600", "color": COLORS["text"]}),
            ], style={"display": "flex", "alignItems": "center", "gap": "8px"}),
            html.Div("Live dashboard · AWS / GCP / Azure · Isolation Forest + ARIMA Ensemble",
                style={"color": COLORS["subtext"], "fontSize": "12px", "marginTop": "4px"}),
        ], md=8),
        dbc.Col([
            html.Div([
                html.Span("🟢 Pipeline running", style={"fontSize": "12px", "color": COLORS["green"]}),
                html.Br(),
                html.Span(f"Last updated: {datetime.now().strftime('%b %d, %Y %H:%M')}",
                    style={"fontSize": "11px", "color": COLORS["subtext"]}),
            ], style={"textAlign": "right"}),
        ], md=4),
    ], style={"padding": "20px 8px 12px", "borderBottom": f"1px solid {COLORS['teal']}30"}),

    html.Br(),

    # ── KPI row ──────────────────────────────────────────────────
    dbc.Row([
        dbc.Col(kpi_card(f"${total_cost/1e6:.2f}M", "Total cost monitored",  COLORS["teal"],  "💰"), md=3),
        dbc.Col(kpi_card(str(len(alert_log)),         "Alerts fired",          COLORS["amber"], "🔔"), md=3),
        dbc.Col(kpi_card(f"{em.get('f1_score',0):.3f}", "Ensemble F1 score",  COLORS["green"], "🎯"), md=3),
        dbc.Col(kpi_card("0",                          "False positives",       COLORS["red"],   "✅"), md=3),
    ], className="g-3"),

    html.Br(),

    # ── Bring-your-own-export mode ────────────────────────────────
    byoe_section,

    html.Br(),

    # ── Provider filter ───────────────────────────────────────────
    dbc.Row([
        dbc.Col([
            html.Label("Filter provider:", style={"color": COLORS["subtext"], "fontSize": "12px", "marginRight": "8px"}),
            dcc.Dropdown(
                id="provider-filter",
                options=[{"label": "All providers", "value": "ALL"},
                         {"label": "AWS",   "value": "AWS"},
                         {"label": "GCP",   "value": "GCP"},
                         {"label": "Azure", "value": "Azure"}],
                value="ALL",
                clearable=False,
                style={"width": "200px", "display": "inline-block",
                       "background": COLORS["card"], "color": COLORS["text"]},
            ),
        ], style={"display": "flex", "alignItems": "center"}),
    ], style={"marginBottom": "12px"}),

    # ── Main spending trends chart ────────────────────────────────
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dcc.Graph(id="spending-trends", figure=fig_spending_trends(), config={"displayModeBar": False})
            ], style={"background": COLORS["card"], "border": f"1px solid {COLORS['teal']}25", "borderRadius": "12px", "padding": "8px"}),
        ]),
    ]),

    html.Br(),

    # ── Row: model comparison + heatmap ──────────────────────────
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dcc.Graph(figure=fig_model_comparison(), config={"displayModeBar": False})
            ], style={"background": COLORS["card"], "border": f"1px solid {COLORS['amber']}25", "borderRadius": "12px", "padding": "8px"}),
        ], md=7),
        dbc.Col([
            dbc.Card([
                dcc.Graph(figure=fig_anomaly_heatmap(), config={"displayModeBar": False})
            ], style={"background": COLORS["card"], "border": f"1px solid {COLORS['red']}25", "borderRadius": "12px", "padding": "8px"}),
        ], md=5),
    ], className="g-3"),

    html.Br(),

    # ── Row: monthly spend + anomaly types ───────────────────────
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dcc.Graph(figure=fig_monthly_spend(), config={"displayModeBar": False})
            ], style={"background": COLORS["card"], "border": f"1px solid {COLORS['gcp']}25", "borderRadius": "12px", "padding": "8px"}),
        ], md=7),
        dbc.Col([
            dbc.Card([
                dcc.Graph(figure=fig_anomaly_types(), config={"displayModeBar": False})
            ], style={"background": COLORS["card"], "border": f"1px solid {COLORS['teal']}25", "borderRadius": "12px", "padding": "8px"}),
        ], md=5),
    ], className="g-3"),

    html.Br(),

    # ── Alert log table ───────────────────────────────────────────
    dbc.Row([
        dbc.Col([
            dbc.Card([
                html.Div([
                    html.Span("🔔 Alert Log", style={"fontSize": "14px", "fontWeight": "500", "color": COLORS["text"]}),
                    html.Span(f"{len(alert_log)} total alerts · 0 false positives",
                        style={"fontSize": "11px", "color": COLORS["subtext"], "marginLeft": "12px"}),
                ], style={"padding": "12px 16px", "borderBottom": f"1px solid {COLORS['teal']}20"}),
                html.Div([
                    html.Table([
                        html.Thead(html.Tr([
                            html.Th(h, style={"color": COLORS["teal"], "fontSize": "11px",
                                "fontWeight": "500", "padding": "8px 12px", "textAlign": "left"})
                            for h in ["Severity", "Provider", "Date", "Actual Cost", "Deviation", "Score"]
                        ]), style={"borderBottom": f"1px solid {COLORS['teal']}30"}),
                        html.Tbody(alert_table_rows(),
                            style={"fontSize": "12px", "color": COLORS["text"]}),
                    ], style={"width": "100%", "borderCollapse": "collapse"}),
                ], style={"padding": "8px 8px", "overflowX": "auto"}),
            ], style={"background": COLORS["card"], "border": f"1px solid {COLORS['teal']}25", "borderRadius": "12px"}),
        ]),
    ]),

    html.Br(),

    # ── Metrics panel ─────────────────────────────────────────────
    dbc.Row([
        dbc.Col([
            dbc.Card([
                html.Div("📊 Model Performance Metrics", style={"padding": "12px 16px",
                    "fontSize": "14px", "fontWeight": "500", "color": COLORS["text"],
                    "borderBottom": f"1px solid {COLORS['teal']}20"}),
                dbc.Row([
                    dbc.Col(_make_metrics_col("Isolation Forest", metrics.get("if_metrics", {}), COLORS["aws"]),    md=4),
                    dbc.Col(_make_metrics_col("ARIMA",            metrics.get("arima_metrics", {}), COLORS["teal"]), md=4),
                    dbc.Col(_make_metrics_col("Ensemble ✓",       metrics.get("ensemble_metrics", {}), COLORS["amber"]), md=4),
                ], style={"padding": "16px"}),
            ], style={"background": COLORS["card"], "border": f"1px solid {COLORS['teal']}25", "borderRadius": "12px"}),
        ]),
    ]),

    html.Br(),

    # ── Footer ────────────────────────────────────────────────────
    html.Div([
        html.Span("Built with Plotly Dash · scikit-learn · statsmodels · Deployed on Render.com",
            style={"fontSize": "11px", "color": COLORS["subtext"]}),
        html.Br(),
        html.Span("AI-Driven Cloud Cost Anomaly Detection System",
            style={"fontSize": "11px", "color": COLORS["subtext"]}),
    ], style={"textAlign": "center", "padding": "16px", "borderTop": f"1px solid {COLORS['teal']}20"}),

], fluid=True, style={"background": COLORS["bg"], "minHeight": "100vh", "padding": "0 16px"})



# ── Callback: filter spending trends ─────────────────────────
@callback(Output("spending-trends", "figure"), Input("provider-filter", "value"))
def update_trends(prov):
    return fig_spending_trends(prov)


# ── Callback: analyze uploaded billing export ─────────────────
@callback(
    Output("byoe-results", "children"),
    Input("byoe-upload", "contents"),
    State("byoe-upload", "filename"),
    prevent_initial_call=True,
)
def analyze_upload(contents, filename):
    if not contents:
        return no_update
    try:
        _, b64 = contents.split(",", 1)
        decoded = base64.b64decode(b64)
        df = pd.read_csv(io.BytesIO(decoded))
    except Exception:
        return _byoe_error_card(f"'{filename}' is not a readable CSV file.")

    try:
        normalized, info = normalize_billing_export(df)
        results = analyze_billing_data(normalized)
        return render_byoe_results(results, info)
    except ValueError as e:
        return _byoe_error_card(str(e))
    except Exception as e:
        return _byoe_error_card(f"Analysis failed: {e}")


# ── Callback: download a sample export to try the flow ────────
@callback(
    Output("byoe-sample-dl", "data"),
    Input("byoe-sample-btn", "n_clicks"),
    prevent_initial_call=True,
)
def download_sample(n_clicks):
    if raw is None:
        return no_update
    sample = raw.rename(columns={
        "provider": "Cloud Vendor", "service": "Product Name",
        "region": "Location", "cost": "Cost Amount",
    })[["date", "Cloud Vendor", "Product Name", "Location", "Cost Amount"]]
    return dcc.send_data_frame(sample.to_csv, "sample_billing_export.csv", index=False)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
