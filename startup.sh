#!/bin/bash
python3 backend/services/data_fetcher.py
python3 backend/models/database.py
python3 backend/models/predictor.py
uvicorn backend.main:app --host 0.0.0.0 --port $PORT