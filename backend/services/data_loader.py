import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/processed/coe_clean.csv")

def load_coe_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["month"])
    return df

def get_latest(df=None):
    if df is None:
        df = load_coe_data()
    latest_month = df["month"].max()
    result = []
    for cat, group in df[df["month"] == latest_month].groupby("vehicle_class"):
        row = group.iloc[0]
        quota = int(row["quota"]) if pd.notna(row.get("quota")) else 0
        bids = int(str(row.get("bids_received", 0)).replace(",", "")) if pd.notna(row.get("bids_received")) else 0
        result.append({
            "category": cat,
            "premium": float(row["premium"]),
            "quota": quota,
            "bids_received": bids,
            "month": latest_month.strftime("%Y-%m"),
        })
    return sorted(result, key=lambda x: x["category"])

def get_history(df=None, category=None, start_year=None, end_year=None):
    if df is None:
        df = load_coe_data()
    if category:
        df = df[df["vehicle_class"] == category]
    if start_year:
        df = df[df["month"].dt.year >= int(start_year)]
    if end_year:
        df = df[df["month"].dt.year <= int(end_year)]
    df = df.sort_values("month").reset_index(drop=True)
    df = df.rename(columns={"vehicle_class": "category"})
    return df

def get_stats(df=None):
    if df is None:
        df = load_coe_data()
    result = []
    for cat, group in df.groupby("vehicle_class"):
        result.append({
            "category": cat,
            "total_records": len(group),
            "min_premium": float(group["premium"].min()),
            "max_premium": float(group["premium"].max()),
            "avg_premium": float(group["premium"].mean()),
            "date_range": f"{group['month'].min().strftime('%Y-%m')} to {group['month'].max().strftime('%Y-%m')}",
        })
    return sorted(result, key=lambda x: x["category"])