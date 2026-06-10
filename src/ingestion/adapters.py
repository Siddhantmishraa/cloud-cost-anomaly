"""
Billing Export Adapters — "Bring Your Own Export" mode
Normalizes billing CSV exports from AWS, GCP, and Azure (plus a generic
fallback) into the pipeline's canonical schema:

    date, provider, service, region, cost

No cloud API access is needed: every provider lets users download these
exports from their billing console (AWS Cost & Usage Report / Cost Explorer
CSV, GCP billing export, Azure cost analysis export).
"""

import re
import numpy as np
import pandas as pd

CANONICAL_COLUMNS = ["date", "provider", "service", "region", "cost"]


def _norm(col: str) -> str:
    """'lineItem/UsageStartDate' -> 'lineitemusagestartdate'"""
    return re.sub(r"[^a-z0-9]", "", str(col).lower())


# Known export formats. For each canonical field, the first matching
# column (after normalization) wins. `signature` columns identify the format.
FORMAT_SPECS = {
    "aws_cur": {
        "provider_name": "AWS",
        "signature": ["lineitemusagestartdate"],
        "date":    ["lineitemusagestartdate", "usagestartdate"],
        "service": ["productproductname", "lineitemproductcode", "productservicename"],
        "cost":    ["lineitemunblendedcost", "lineitemblendedcost", "lineitemnetunblendedcost"],
        "region":  ["productregion", "productregioncode", "lineitemavailabilityzone"],
    },
    "aws_cost_explorer": {
        "provider_name": "AWS",
        "signature": ["service", "unblendedcost"],
        "date":    ["startdate", "start", "date", "usagestartdate"],
        "service": ["service", "servicename", "dimension"],
        "cost":    ["unblendedcost", "amortizedcost", "netunblendedcost", "cost", "costusd"],
        "region":  ["region"],
    },
    "gcp_billing": {
        "provider_name": "GCP",
        "signature": ["servicedescription"],
        "date":    ["usagestartdate", "usagestarttime", "date", "day"],
        "service": ["servicedescription", "service", "skudescription"],
        "cost":    ["cost", "costusd", "totalcost"],
        "region":  ["locationregion", "location", "region"],
    },
    "azure_cost": {
        "provider_name": "Azure",
        "signature": ["metercategory"],
        "date":    ["usagedatetime", "date", "usagedate", "billingperiodstartdate"],
        "service": ["metercategory", "servicename", "consumedservice"],
        "cost":    ["pretaxcost", "costinbillingcurrency", "cost", "actualcost"],
        "region":  ["resourcelocation", "location", "region"],
    },
}

# Generic fallback: keyword search over normalized column names
GENERIC_KEYWORDS = {
    "date":     ["date", "day", "time", "period"],
    "service":  ["service", "product", "meter", "sku", "category", "resource"],
    "cost":     ["cost", "amount", "spend", "charge", "price", "total"],
    "region":   ["region", "location", "zone"],
    "provider": ["provider", "cloud", "vendor", "platform"],
}


def detect_format(df: pd.DataFrame) -> str:
    """Identify which export format a dataframe is, by signature columns."""
    norm_cols = {_norm(c) for c in df.columns}
    for fmt, spec in FORMAT_SPECS.items():
        if all(sig in norm_cols for sig in spec["signature"]):
            return fmt
    return "generic"


def _find_column(df: pd.DataFrame, candidates: list) -> str:
    """Return the original column name whose normalized form matches first."""
    lookup = {_norm(c): c for c in df.columns}
    for cand in candidates:
        if cand in lookup:
            return lookup[cand]
    return None


def _find_generic_column(df: pd.DataFrame, keywords: list) -> str:
    """Keyword-based fallback matching for unknown export layouts."""
    for kw in keywords:
        for col in df.columns:
            if kw in _norm(col):
                return col
    return None


def normalize_billing_export(df: pd.DataFrame,
                             default_provider: str = None) -> tuple:
    """
    Convert any supported billing export into the canonical schema.
    Returns (normalized_df, info) where info describes what was detected.
    Raises ValueError with a user-readable message when required columns
    (date, cost) cannot be found.
    """
    fmt = detect_format(df)

    if fmt != "generic":
        spec = FORMAT_SPECS[fmt]
        date_col    = _find_column(df, spec["date"])
        service_col = _find_column(df, spec["service"])
        cost_col    = _find_column(df, spec["cost"])
        region_col  = _find_column(df, spec["region"])
        provider    = spec["provider_name"]
        provider_col = None
    else:
        date_col    = _find_generic_column(df, GENERIC_KEYWORDS["date"])
        service_col = _find_generic_column(df, GENERIC_KEYWORDS["service"])
        cost_col    = _find_generic_column(df, GENERIC_KEYWORDS["cost"])
        region_col  = _find_generic_column(df, GENERIC_KEYWORDS["region"])
        provider_col = _find_generic_column(df, GENERIC_KEYWORDS["provider"])
        provider    = default_provider or "Cloud"

    if date_col is None or cost_col is None:
        raise ValueError(
            "Could not find a date and cost column in this file. "
            f"Columns seen: {list(df.columns)[:12]}. Expected something like "
            "date/UsageStartDate and cost/UnblendedCost/PreTaxCost."
        )

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    out["cost"] = pd.to_numeric(df[cost_col], errors="coerce")
    out["service"] = (df[service_col].astype(str).str.strip()
                      if service_col else "All services")
    out["region"] = (df[region_col].astype(str).str.strip().replace({"nan": "unknown", "": "unknown"})
                     if region_col else "unknown")
    out["provider"] = (df[provider_col].astype(str).str.strip()
                       if fmt == "generic" and provider_col else provider)

    bad_rows = int(out["date"].isna().sum() + out["cost"].isna().sum())
    out = out.dropna(subset=["date", "cost"])
    out = out[out["cost"] >= 0]

    if out.empty:
        raise ValueError("No usable rows after parsing dates and costs.")

    # Exports are usually line-item level — collapse to one row per
    # (date, provider, service, region)
    out = (out.groupby(["date", "provider", "service", "region"], as_index=False)
              ["cost"].sum())
    out = out[CANONICAL_COLUMNS].sort_values(["provider", "service", "date"])

    info = {
        "format": fmt,
        "provider": provider if fmt != "generic" else sorted(out["provider"].unique()),
        "rows": len(out),
        "dropped_rows": bad_rows,
        "services": int(out["service"].nunique()),
        "date_range": f"{out['date'].min().date()} → {out['date'].max().date()}",
        "days": int(out["date"].nunique()),
        "total_cost": round(float(out["cost"].sum()), 2),
        "mapping": {"date": date_col, "service": service_col,
                    "cost": cost_col, "region": region_col},
    }
    return out.reset_index(drop=True), info


if __name__ == "__main__":
    # Smoke test: disguise simulated data as an Azure cost export
    raw = pd.read_csv("data/simulated/multicloud_billing.csv")
    azure = raw[raw["provider"] == "Azure"].rename(columns={
        "date": "UsageDateTime", "service": "MeterCategory",
        "cost": "PreTaxCost", "region": "ResourceLocation",
    })[["UsageDateTime", "MeterCategory", "PreTaxCost", "ResourceLocation"]]

    norm, info = normalize_billing_export(azure)
    print("Detected:", info["format"], "| provider:", info["provider"])
    print("Days:", info["days"], "| services:", info["services"],
          "| total: $", info["total_cost"])
    print(norm.head())
