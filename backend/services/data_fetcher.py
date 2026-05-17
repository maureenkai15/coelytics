import requests
import pandas as pd
from pathlib import Path

# Paths
RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Correct dataset ID from data.gov.sg (LTA COE Bidding Results)
DATASET_ID = "d_69b3380ad7e51aff3a7dcc84eba52b8a"
BASE_URL = f"https://data.gov.sg/api/action/datastore_search?resource_id={DATASET_ID}"

def fetch_coe_data():
    print("Fetching COE data from data.gov.sg...")
    all_records = []
    offset = 0
    limit = 1000

    while True:
        url = f"{BASE_URL}&limit={limit}&offset={offset}"
        response = requests.get(url)
        response.raise_for_status()
        result = response.json()["result"]
        records = result["records"]
        if not records:
            break
        all_records.extend(records)
        offset += limit
        print(f"  Fetched {len(all_records)} records so far...")
        if len(records) < limit:
            break

    df = pd.DataFrame(all_records)
    print(f"\nTotal records fetched: {len(df)}")
    print("Columns:", df.columns.tolist())
    print(df.head(3))

    raw_path = RAW_DATA_DIR / "coe_raw.csv"
    df.to_csv(raw_path, index=False)
    print(f"\nRaw data saved to {raw_path}")
    return df

def clean_coe_data(df):
    print("\nCleaning data...")
    df.columns = [c.lower().strip() for c in df.columns]
    print("Columns after lowercase:", df.columns.tolist())

    df["premium"] = pd.to_numeric(df["premium"], errors="coerce")
    df["quota"] = pd.to_numeric(df["quota"], errors="coerce")
    df["month"] = pd.to_datetime(df["month"], format="%Y-%m")

    df = df.dropna(subset=["premium", "month", "vehicle_class"])
    df = df.sort_values("month").reset_index(drop=True)

    print(f"Clean records: {len(df)}")

    processed_path = PROCESSED_DATA_DIR / "coe_clean.csv"
    df.to_csv(processed_path, index=False)
    print(f"Clean data saved to {processed_path}")
    return df

if __name__ == "__main__":
    df_raw = fetch_coe_data()
    df_clean = clean_coe_data(df_raw)
    print("\nLatest 10 records:")
    print(df_clean[["month", "vehicle_class", "premium", "quota"]].tail(10))