"""
daily_predict.py -- Daily prediction pipeline for DeepTime Forecasting.

This script is designed to run in a GitHub Actions environment (or locally).
It fetches the latest 60 trading days of OHLCV data from Twelve Data API,
converts to percentage returns, runs inference through the trained model,
and converts the predicted return back to a dollar price.

Usage:
    python scripts/daily_predict.py            # normal daily run
    python scripts/daily_predict.py --seed     # seed with historical test data
"""

import os
import sys
import json
import math
import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import joblib

# ---------------------------------------------------------------------------
# Path setup -- works from repo root or from scripts/ directory
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(os.path.join(BASE_DIR, "src"))

from model import CNN_RNN_AttnModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TICKERS = ["AAPL", "JNJ", "JPM", "MSFT", "PEP"]
FEATURES = ["Close", "High", "Low", "Open", "Volume"]
SEQ_LENGTH = 60
PREDICTIONS_FILE = os.path.join(BASE_DIR, "data", "predictions.json")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pth")
FEATURE_SCALER_PATH = os.path.join(BASE_DIR, "data", "feature_scaler.pkl")
TARGET_SCALER_PATH = os.path.join(BASE_DIR, "data", "target_scaler.pkl")


def _sanitize_value(v):
    """Replace NaN/Inf floats with None (JSON null)."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _sanitize_dict(d):
    """Sanitize all values in a dict, replacing NaN/Inf with None."""
    return {k: _sanitize_value(v) for k, v in d.items()}


def _sanitize_entry(entry):
    """Sanitize a single prediction entry."""
    if "predictions" in entry and isinstance(entry["predictions"], dict):
        entry["predictions"] = _sanitize_dict(entry["predictions"])
    if "actuals" in entry and isinstance(entry["actuals"], dict):
        entry["actuals"] = _sanitize_dict(entry["actuals"])
    return entry


def load_model():
    """Load the trained model and target scaler."""
    model = CNN_RNN_AttnModel()
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=torch.device("cpu"), weights_only=True)
    )
    model.eval()
    target_scaler = joblib.load(TARGET_SCALER_PATH)
    return model, target_scaler


def fetch_market_data(days_back=120):
    """
    Fetch recent OHLCV data from Twelve Data API.
    We request extra days to account for weekends/holidays
    and ensure we get at least SEQ_LENGTH+1 trading days (need +1 for returns).
    """
    sys.path.insert(0, os.path.join(BASE_DIR, "src"))
    from market_data import fetch_historical_ohlcv

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    print(f"Fetching data from {start_date.date()} to {end_date.date()} ...")
    df = fetch_historical_ohlcv(
        TICKERS,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
    )
    df = df.ffill().bfill()

    if len(df) < SEQ_LENGTH + 2:
        raise ValueError(
            f"Only got {len(df)} trading days, need at least {SEQ_LENGTH + 2}. "
            "Try increasing days_back."
        )
    return df


def prepare_input(df):
    """
    Build the (1, SEQ_LENGTH, 25) input tensor from raw OHLCV data.

    Converts raw prices to percentage returns and scales using the
    SAME StandardScaler that was fit during training.

    Returns:
        X_tensor: input tensor for the model
        last_close: dict of {ticker: close_price} for the last trading day
    """
    # Compute percentage returns
    returns_df = df.pct_change().iloc[1:]

    # Build feature matrix from returns
    feature_list = []
    for ticker in TICKERS:
        for feature in FEATURES:
            feature_list.append(returns_df[feature][ticker].values)
    X_returns = np.column_stack(feature_list)
    X_returns = np.nan_to_num(X_returns, nan=0.0, posinf=0.0, neginf=0.0)

    # Scale with the TRAINING scaler
    feature_scaler = joblib.load(FEATURE_SCALER_PATH)
    X_scaled = feature_scaler.transform(X_returns)

    # Take the last SEQ_LENGTH days as input
    X_input = X_scaled[-SEQ_LENGTH:]
    X_tensor = torch.tensor(X_input[np.newaxis, :, :], dtype=torch.float32)

    # Get the last close prices (base for return -> price conversion)
    last_close = {}
    latest = df["Close"].iloc[-1]
    for ticker in TICKERS:
        last_close[ticker] = float(latest[ticker])

    return X_tensor, last_close


def get_actual_prices(df):
    """Get the most recent actual closing prices from the dataframe."""
    latest = df["Close"].iloc[-1]
    actuals = {}
    for ticker in TICKERS:
        val = float(latest[ticker])
        actuals[ticker] = None if (math.isnan(val) or math.isinf(val)) else round(val, 2)
    return actuals


def load_predictions():
    """Load existing predictions.json, sanitizing any NaN values."""
    if os.path.exists(PREDICTIONS_FILE):
        with open(PREDICTIONS_FILE, "r") as f:
            raw = f.read()
        raw = raw.replace("NaN", "null").replace("Infinity", "null").replace("-Infinity", "null")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            print("  [WARN] predictions.json is corrupted, starting fresh")
            return []
        return [_sanitize_entry(e) for e in data]
    return []


def save_predictions(data):
    """Save predictions to JSON file (NaN-safe)."""
    os.makedirs(os.path.dirname(PREDICTIONS_FILE), exist_ok=True)
    data = [_sanitize_entry(e) for e in data]
    with open(PREDICTIONS_FILE, "w") as f:
        json.dump(data, f, indent=2, allow_nan=False)
    print(f"Saved {len(data)} entries to {PREDICTIONS_FILE}")


def run_daily_prediction():
    """Main daily prediction pipeline."""
    print("=" * 60)
    print("DeepTime Forecasting -- Daily Prediction Pipeline")
    print("=" * 60)

    # Load model
    print("\n[1/4] Loading model and scaler...")
    model, target_scaler = load_model()

    # Fetch market data
    print("\n[2/4] Fetching market data...")
    df = fetch_market_data()
    print(f"  Got {len(df)} trading days")

    # Prepare input and run inference
    print("\n[3/4] Running inference...")
    X_input, last_close = prepare_input(df)

    with torch.no_grad():
        y_pred = model(X_input)

    # Inverse-transform to get predicted returns
    pred_returns = target_scaler.inverse_transform(y_pred.numpy())[0]

    # Convert returns to prices: predicted_price = last_close * (1 + return)
    predictions = {}
    for i, ticker in enumerate(TICKERS):
        pred_price = float(last_close[ticker] * (1.0 + float(pred_returns[i])))
        predictions[ticker] = None if math.isnan(pred_price) else round(pred_price, 2)

    # Get actual closing prices
    actuals = get_actual_prices(df)

    # Determine the date
    last_date = df.index[-1]
    prediction_date = last_date.strftime("%Y-%m-%d")

    print(f"\n  Prediction date: {prediction_date}")
    print(f"  Predictions: {predictions}")
    print(f"  Actuals:     {actuals}")

    # Append to predictions.json
    print("\n[4/4] Saving results...")
    all_predictions = load_predictions()

    new_entry = {
        "date": prediction_date,
        "predictions": predictions,
        "actuals": actuals,
    }

    existing_idx = next(
        (i for i, e in enumerate(all_predictions) if e["date"] == prediction_date),
        None,
    )
    if existing_idx is not None:
        all_predictions[existing_idx] = new_entry
        print(f"  Updated existing entry for {prediction_date}")
    else:
        all_predictions.append(new_entry)
        print(f"  Added new entry for {prediction_date}")

    all_predictions.sort(key=lambda x: x["date"])
    save_predictions(all_predictions)

    print("\n[OK] Daily prediction complete!")
    return new_entry


def seed_with_test_data():
    """
    Seed predictions.json with historical test-set predictions.
    This gives the charts data to display immediately.
    
    Uses base_closes_test.npy to convert predicted returns back to
    dollar prices for display.
    """
    print("=" * 60)
    print("Seeding predictions.json with historical test data...")
    print("=" * 60)

    model, target_scaler = load_model()

    # Load test data
    X_test = np.load(os.path.join(BASE_DIR, "data", "X_test.npy"))
    y_test = np.load(os.path.join(BASE_DIR, "data", "y_test.npy"))
    base_closes = np.load(os.path.join(BASE_DIR, "data", "base_closes_test.npy"))

    # Run predictions on all test samples
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        y_pred = model(X_tensor)

    # Inverse transform to get returns
    pred_returns = target_scaler.inverse_transform(y_pred.numpy())
    actual_returns = target_scaler.inverse_transform(y_test)

    # Convert returns to prices using base closes
    pred_prices = base_closes * (1.0 + pred_returns)
    actual_prices = base_closes * (1.0 + actual_returns)

    # Generate dates (test set is roughly the last 15% of 2018-2026 data)
    num_test = len(X_test)
    base_date = datetime(2023, 6, 1)

    entries = []
    current_date = base_date
    for i in range(num_test):
        while current_date.weekday() >= 5:
            current_date += timedelta(days=1)

        preds = {}
        acts = {}
        for j, ticker in enumerate(TICKERS):
            preds[ticker] = round(float(pred_prices[i][j]), 2)
            acts[ticker] = round(float(actual_prices[i][j]), 2)

        entries.append({
            "date": current_date.strftime("%Y-%m-%d"),
            "predictions": preds,
            "actuals": acts,
        })
        current_date += timedelta(days=1)

    save_predictions(entries)
    print(f"\n[OK] Seeded {len(entries)} historical entries!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepTime daily prediction pipeline")
    parser.add_argument(
        "--seed",
        action="store_true",
        help="Seed predictions.json with historical test-set data",
    )
    args = parser.parse_args()

    if args.seed:
        seed_with_test_data()
    else:
        run_daily_prediction()
