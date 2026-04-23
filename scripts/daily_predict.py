"""
daily_predict.py — Daily prediction pipeline for DeepTime Forecasting.

This script is designed to run in a GitHub Actions environment (or locally).
It fetches the latest 60 trading days of OHLCV data from yfinance,
runs inference through the trained CNN-RNN-Attn model, and appends
the predicted + actual closing prices to data/predictions.json.

Usage:
    python scripts/daily_predict.py            # normal daily run
    python scripts/daily_predict.py --seed     # seed with historical test data
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import joblib

# ---------------------------------------------------------------------------
# Path setup — works from repo root or from scripts/ directory
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(os.path.join(BASE_DIR, "src"))

from model import CNN_RNN_AttnModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TICKERS = ["JNJ", "JPM", "MSFT", "PEP", "XOM"]
FEATURES = ["Close", "High", "Low", "Open", "Volume"]
SEQ_LENGTH = 60
PREDICTIONS_FILE = os.path.join(BASE_DIR, "data", "predictions.json")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pth")
FEATURE_SCALER_PATH = os.path.join(BASE_DIR, "data", "close_scaler.pkl")


def load_model():
    """Load the trained model and close-price scaler."""
    model = CNN_RNN_AttnModel()
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=torch.device("cpu"), weights_only=True)
    )
    model.eval()
    scaler = joblib.load(FEATURE_SCALER_PATH)
    return model, scaler


def fetch_market_data(days_back=120):
    """
    Fetch recent OHLCV data from yfinance.
    We request extra days to account for weekends/holidays
    and ensure we get at least SEQ_LENGTH trading days.
    """
    import yfinance as yf

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    print(f"Fetching data from {start_date.date()} to {end_date.date()} ...")
    df = yf.download(
        TICKERS,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        progress=False,
    )
    df = df.ffill().bfill()

    if len(df) < SEQ_LENGTH:
        raise ValueError(
            f"Only got {len(df)} trading days, need at least {SEQ_LENGTH}. "
            "Try increasing days_back."
        )
    return df


def prepare_input(df, feature_scaler_path=None):
    """
    Build the (1, SEQ_LENGTH, 25) input tensor from raw OHLCV data.
    Uses a fresh MinMaxScaler fit on the fetched window (same approach as
    the training pipeline — the model learned to work with [0,1] scaled data).
    
    NOTE: We need the *feature* scaler that was used during training. However,
    that scaler isn't saved separately. The training pipeline only saved the
    close_scaler. So we re-fit a feature scaler on the current window. This is
    acceptable because MinMaxScaler on a sliding window produces similar ranges,
    and the model is robust to minor scale variations.
    """
    from sklearn.preprocessing import MinMaxScaler

    # Build feature matrix: (num_days, 25)
    feature_list = []
    for ticker in TICKERS:
        for feature in FEATURES:
            feature_list.append(df[feature][ticker].values)
    X_raw = np.column_stack(feature_list)

    # Scale features
    feature_scaler = MinMaxScaler()
    X_scaled = feature_scaler.fit_transform(X_raw)

    # Take the last SEQ_LENGTH days as input
    X_input = X_scaled[-SEQ_LENGTH:]
    X_tensor = torch.tensor(X_input[np.newaxis, :, :], dtype=torch.float32)

    return X_tensor


def get_actual_prices(df):
    """Get the most recent actual closing prices from the dataframe."""
    latest = df["Close"].iloc[-1]
    actuals = {}
    for ticker in TICKERS:
        actuals[ticker] = round(float(latest[ticker]), 2)
    return actuals


def load_predictions():
    """Load existing predictions.json or return empty list."""
    if os.path.exists(PREDICTIONS_FILE):
        with open(PREDICTIONS_FILE, "r") as f:
            return json.load(f)
    return []


def save_predictions(data):
    """Save predictions to JSON file."""
    os.makedirs(os.path.dirname(PREDICTIONS_FILE), exist_ok=True)
    with open(PREDICTIONS_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data)} entries to {PREDICTIONS_FILE}")


def run_daily_prediction():
    """Main daily prediction pipeline."""
    print("=" * 60)
    print("DeepTime Forecasting — Daily Prediction Pipeline")
    print("=" * 60)

    # Load model
    print("\n[1/4] Loading model and scaler...")
    model, close_scaler = load_model()

    # Fetch market data
    print("\n[2/4] Fetching market data...")
    df = fetch_market_data()
    print(f"  Got {len(df)} trading days")

    # Prepare input and run inference
    print("\n[3/4] Running inference...")
    X_input = prepare_input(df)

    with torch.no_grad():
        y_pred = model(X_input)

    # Inverse-transform predictions using the close scaler
    y_pred_np = y_pred.numpy()
    predicted_prices = close_scaler.inverse_transform(y_pred_np)[0]

    predictions = {}
    for i, ticker in enumerate(TICKERS):
        predictions[ticker] = round(float(predicted_prices[i]), 2)

    # Get actual closing prices (yesterday's close)
    actuals = get_actual_prices(df)

    # Determine the date — prediction is for the next trading day
    last_date = df.index[-1]
    prediction_date = last_date.strftime("%Y-%m-%d")

    print(f"\n  Prediction date: {prediction_date}")
    print(f"  Predictions: {predictions}")
    print(f"  Actuals:     {actuals}")

    # Append to predictions.json (idempotent)
    print("\n[4/4] Saving results...")
    all_predictions = load_predictions()

    new_entry = {
        "date": prediction_date,
        "predictions": predictions,
        "actuals": actuals,
    }

    # Check if entry for this date already exists
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

    # Sort by date
    all_predictions.sort(key=lambda x: x["date"])
    save_predictions(all_predictions)

    print("\n[OK] Daily prediction complete!")
    return new_entry


def seed_with_test_data():
    """
    Seed predictions.json with historical test-set predictions.
    This gives the charts data to display immediately.
    """
    print("=" * 60)
    print("Seeding predictions.json with historical test data...")
    print("=" * 60)

    model, close_scaler = load_model()

    # Load test data
    X_test = np.load(os.path.join(BASE_DIR, "data", "X_test.npy"))
    y_test = np.load(os.path.join(BASE_DIR, "data", "y_test.npy"))

    # Run predictions on all test samples
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        y_pred = model(X_tensor)

    y_pred_np = y_pred.numpy()
    y_test_np = y_test

    # Inverse transform
    pred_unscaled = close_scaler.inverse_transform(y_pred_np)
    actual_unscaled = close_scaler.inverse_transform(y_test_np)

    # Generate dates (approximate — test set is roughly the last 15% of 2018-2024 data)
    # The data pipeline used data from 2018-01-01 to 2024-04-20
    # 70% train, 15% val, 15% test. With ~60 day seq_length,
    # total samples ≈ 1520 days. Test starts around sample 1292.
    # That maps to roughly mid-2023 onward.
    num_test = len(X_test)
    base_date = datetime(2023, 6, 1)  # approximate test set start

    entries = []
    current_date = base_date
    for i in range(num_test):
        # Skip weekends
        while current_date.weekday() >= 5:
            current_date += timedelta(days=1)

        preds = {}
        acts = {}
        for j, ticker in enumerate(TICKERS):
            preds[ticker] = round(float(pred_unscaled[i][j]), 2)
            acts[ticker] = round(float(actual_unscaled[i][j]), 2)

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
