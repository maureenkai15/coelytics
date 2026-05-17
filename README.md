# COElytics 🚗

**Singapore Vehicle Market Intelligence Platform**

A full-stack data science portfolio project that analyses and predicts Singapore COE (Certificate of Entitlement) price trends using machine learning, real government data, and interactive visualisations.

## Live Features

- **Market Overview** — Live COE premiums across all 5 categories with demand pressure analysis
- **Price Trends** — 16-year historical comparison with year-on-year change heatmaps
- **Category Analysis** — Deep dive per category with distribution and quota vs bids charts
- **ML Price Forecast** — XGBoost model with 3.5% MAPE accuracy, 6-month forward predictions
- **Bid Timing Advisor** — Historical percentile analysis with moving average signals
- **Affordability Calculator** — DSR calculator with loan amortisation schedule
- **Total Cost of Ownership** — Full 10-year cost breakdown including ERP, insurance, road tax
- **Renew vs Scrap** — PARF rebate calculator with financial comparison

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Dashboard | Streamlit + Plotly |
| Backend API | FastAPI + Uvicorn |
| ML Models | XGBoost, scikit-learn |
| Database | SQLite + SQLAlchemy |
| Data Source | Singapore LTA via data.gov.sg |
| Language | Python 3.14 |

## ML Model Performance

| Category | MAE | MAPE |
|----------|-----|------|
| Category A | ~$3,886 | 3.5% |
| Category B | ~$4,200 | 4.1% |
| Category C | ~$2,100 | 3.8% |
| Category D | ~$180 | 2.9% |
| Category E | ~$4,500 | 4.3% |

## Setup

```bash
git clone https://github.com/maureenkai15/coelytics.git
cd coelytics
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Fetch live data
python3 backend/services/data_fetcher.py

# Load database
python3 backend/models/database.py

# Train ML models
python3 backend/models/predictor.py

# Start API (terminal 1)
uvicorn backend.main:app --reload --port 8000

# Start dashboard (terminal 2)
streamlit run dashboard/app.py
```

## Data Source

All COE data sourced from the Singapore Land Transport Authority (LTA) via [data.gov.sg](https://data.gov.sg) — free and open government data updated after every bidding exercise.

## Author

Built by [@maureenkai15](https://github.com/maureenkai15)
