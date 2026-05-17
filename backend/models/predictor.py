import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

DATA_PATH  = Path("data/processed/coe_clean.csv")
MODEL_DIR  = Path("ml_models")
MODEL_DIR.mkdir(exist_ok=True)

CATEGORIES = ["Category A", "Category B", "Category C", "Category D", "Category E"]

def load_and_prepare(category: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["month"])
    df = df[df["vehicle_class"] == category].copy()
    df = df.sort_values("month").reset_index(drop=True)

    # ── Time features ──────────────────────────────────────
    df["year"]    = df["month"].dt.year
    df["month_n"] = df["month"].dt.month
    df["quarter"] = df["month"].dt.quarter
    df["month_sin"] = np.sin(2 * np.pi * df["month_n"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_n"] / 12)

    # ── Lag features ───────────────────────────────────────
    for lag in [1, 2, 3, 6, 12]:
        df[f"lag_{lag}"] = df["premium"].shift(lag)

    # ── Rolling features ───────────────────────────────────
    df["roll_3"]  = df["premium"].shift(1).rolling(3).mean()
    df["roll_6"]  = df["premium"].shift(1).rolling(6).mean()
    df["roll_12"] = df["premium"].shift(1).rolling(12).mean()
    df["roll_std_3"] = df["premium"].shift(1).rolling(3).std()

    # ── Demand feature ─────────────────────────────────────
    if "bids_received" in df.columns and "quota" in df.columns:
        df["demand_ratio"] = pd.to_numeric(df["bids_received"], errors="coerce") / \
                             pd.to_numeric(df["quota"], errors="coerce").replace(0, np.nan)
        df["demand_ratio"] = df["demand_ratio"].fillna(1.0)
    else:
        df["demand_ratio"] = 1.0

    # ── Trend features ─────────────────────────────────────
    df["pct_change_1"] = df["premium"].pct_change(1)
    df["pct_change_3"] = df["premium"].pct_change(3)

    df = df.dropna()
    return df

FEATURES = [
    "year", "month_n", "quarter", "month_sin", "month_cos",
    "lag_1", "lag_2", "lag_3", "lag_6", "lag_12",
    "roll_3", "roll_6", "roll_12", "roll_std_3",
    "demand_ratio", "pct_change_1", "pct_change_3",
]

def train_category(category: str):
    print(f"\n{'='*50}")
    print(f"Training: {category}")
    print(f"{'='*50}")

    df = load_and_prepare(category)
    X  = df[FEATURES]
    y  = df["premium"]

    print(f"  Samples: {len(df)} | Features: {len(FEATURES)}")
    print(f"  Date range: {df['month'].min().strftime('%Y-%m')} → {df['month'].max().strftime('%Y-%m')}")

    # ── Train/test split (last 12 months = test) ───────────
    split = len(df) - 12
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # ── XGBoost model ──────────────────────────────────────
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        early_stopping_rounds=30,
        eval_metric="mae",
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # ── Evaluate ───────────────────────────────────────────
    preds = model.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    mape  = np.mean(np.abs((y_test - preds) / y_test)) * 100

    print(f"  MAE  : ${mae:,.0f}")
    print(f"  RMSE : ${rmse:,.0f}")
    print(f"  MAPE : {mape:.1f}%")

    # ── Feature importance ─────────────────────────────────
    importance = pd.Series(model.feature_importances_, index=FEATURES)
    top5 = importance.nlargest(5)
    print(f"  Top features: {', '.join(top5.index.tolist())}")

    # ── Save model ─────────────────────────────────────────
    cat_slug = category.replace(" ", "_").lower()
    model_path = MODEL_DIR / f"xgb_{cat_slug}.pkl"
    meta = {
        "model":    model,
        "features": FEATURES,
        "category": category,
        "mae":      mae,
        "rmse":     rmse,
        "mape":     mape,
        "trained_on": pd.Timestamp.now().isoformat(),
        "last_date":  df["month"].max().isoformat(),
        "last_premium": float(df["premium"].iloc[-1]),
    }
    with open(model_path, "wb") as f:
        pickle.dump(meta, f)
    print(f"  Saved → {model_path}")

    return meta, df, preds, y_test

def predict_future(category: str, months_ahead: int = 6):
    cat_slug   = category.replace(" ", "_").lower()
    model_path = MODEL_DIR / f"xgb_{cat_slug}.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"No model found for {category}. Run training first.")

    with open(model_path, "rb") as f:
        meta = pickle.load(f)

    model   = meta["model"]
    df      = load_and_prepare(category)
    history = df.copy()

    predictions = []
    future_df   = history.copy()

    for i in range(months_ahead):
        last_row  = future_df.iloc[-1]
        next_month = last_row["month"] + pd.DateOffset(months=1)

        row = {
            "year":      next_month.year,
            "month_n":   next_month.month,
            "quarter":   next_month.quarter,
            "month_sin": np.sin(2 * np.pi * next_month.month / 12),
            "month_cos": np.cos(2 * np.pi * next_month.month / 12),
            "lag_1":     future_df["premium"].iloc[-1],
            "lag_2":     future_df["premium"].iloc[-2],
            "lag_3":     future_df["premium"].iloc[-3],
            "lag_6":     future_df["premium"].iloc[-6],
            "lag_12":    future_df["premium"].iloc[-12],
            "roll_3":    future_df["premium"].iloc[-3:].mean(),
            "roll_6":    future_df["premium"].iloc[-6:].mean(),
            "roll_12":   future_df["premium"].iloc[-12:].mean(),
            "roll_std_3":future_df["premium"].iloc[-3:].std(),
            "demand_ratio": future_df["demand_ratio"].iloc[-3:].mean(),
            "pct_change_1": (future_df["premium"].iloc[-1] - future_df["premium"].iloc[-2]) / future_df["premium"].iloc[-2],
            "pct_change_3": (future_df["premium"].iloc[-1] - future_df["premium"].iloc[-4]) / future_df["premium"].iloc[-4],
        }

        X_pred  = pd.DataFrame([row])[meta["features"]]
        pred    = float(model.predict(X_pred)[0])
        predictions.append({"month": next_month, "predicted_premium": round(pred, 0), "category": category})

        new_row         = pd.Series(row)
        new_row["month"]   = next_month
        new_row["premium"] = pred
        future_df = pd.concat([future_df, new_row.to_frame().T], ignore_index=True)

    return pd.DataFrame(predictions)

if __name__ == "__main__":
    print("COElytics — XGBoost Model Training")
    print("="*50)

    all_results = {}
    for cat in CATEGORIES:
        meta, df, preds, y_test = train_category(cat)
        all_results[cat] = meta

    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    for cat, m in all_results.items():
        print(f"  {cat:<14} MAE=${m['mae']:>8,.0f}  MAPE={m['mape']:>5.1f}%")

    print("\nGenerating 6-month forecasts...")
    for cat in CATEGORIES:
        fc = predict_future(cat, months_ahead=6)
        print(f"\n{cat}:")
        for _, row in fc.iterrows():
            print(f"  {row['month'].strftime('%Y-%m')}  →  ${row['predicted_premium']:>10,.0f}")

    print("\nAll models trained and saved to ml_models/")